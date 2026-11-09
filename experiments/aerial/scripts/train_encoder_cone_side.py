"""Train the image encoder and depth decoder so left/right cones follow the facade.

Holds out six route-0 frames from the turnfix flight. Those frames are the
acceptance check: the posterior-mode depth map must change with the picture,
and the open side must match ground truth where the two cones clearly differ.

Does not update the actor, the RSSM, the reward head, or the collision head.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from experiments.aerial.rl.depth_geometry import cone_clearances
from experiments.aerial.rl.dynamics_torch import _cone_side_loss, _symlog
from experiments.aerial.rl.train_rl import load_torch_dynamics

ROOT = Path("/home/yao/aerial-wam-v2")
CFG = ROOT / "configs/_tmp_feasible_side_h125.yaml"
WM = ROOT / "experiments/aerial/rl/artifacts/wm_ckpt_soft_init_antisat_cost/wm_soft_init.pt"
TRAJ = ROOT / "experiments/aerial/rl/artifacts/feasible_side_turnfix_20260928/eval_noshield/traj"
OUT_DIR = ROOT / "experiments/aerial/rl/artifacts/wm_ckpt_cone_side_encoder_20260928"
PAIRS = OUT_DIR / "pairs.npz"
CKPT = OUT_DIR / "wm_encoder_depth.pt"
REPORT = OUT_DIR / "holdout_mode.json"

HOLDOUT = (0, 34, 43, 69, 125, 152)
HOLDOUT_RADIUS = 6
YAW_OFFSETS_DEG = (-80.0, -40.0, 0.0, 40.0, 80.0)
CLEAR_RATIO = 1.25
PIXEL_WEIGHT = 0.05
STEPS = 400
BATCH = 16
LR = 1e-4


def _rows(name: str):
    path = TRAJ / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _train_indices(n: int, route: str, stride: int):
    banned = set()
    if route == "route00.jsonl":
        for h in HOLDOUT:
            for d in range(-HOLDOUT_RADIUS, HOLDOUT_RADIUS + 1):
                banned.add(h + d)
    return [i for i in range(0, n, stride) if i not in banned]


def _grab(client, airsim):
    rs = client.simGetImages(
        [
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False),
        ]
    )
    bgr = np.frombuffer(rs[0].image_data_uint8, dtype=np.uint8).reshape(rs[0].height, rs[0].width, 3)
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    depth = np.array(
        airsim.list_to_2d_float_array(rs[1].image_data_float, rs[1].width, rs[1].height),
        dtype=np.float32,
    )
    depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0).astype(np.float32)
    depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb, dtype=np.uint8), np.ascontiguousarray(depth, dtype=np.float32)


def _set_pose(client, airsim, pos, yaw):
    q = airsim.to_quaternion(0.0, 0.0, float(yaw))
    ned = airsim.Vector3r(float(pos[0]), float(pos[1]), float(-pos[2]))
    client.simSetVehiclePose(airsim.Pose(ned, q), ignore_collision=True)
    time.sleep(0.22)


def collect() -> None:
    import airsim

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    rgbs, depths, proprios = [], [], []
    left_far = right_far = tie = 0
    try:
        for name, stride in (("route00.jsonl", 7), ("route01.jsonl", 5)):
            rows = _rows(name)
            for i in _train_indices(len(rows), name, stride):
                row = rows[i]
                pos = np.asarray(row["pos"], dtype=np.float64)
                yaw0 = float(np.radians(float(row["yaw_deg"])))
                for off in YAW_OFFSETS_DEG:
                    yaw = yaw0 + float(np.radians(off))
                    _set_pose(client, airsim, pos, yaw)
                    rgb, depth = _grab(client, airsim)
                    cones = cone_clearances(depth)
                    if abs(cones["left"] - cones["right"]) < 1.0:
                        tie += 1
                    elif cones["left"] > cones["right"]:
                        left_far += 1
                    else:
                        right_far += 1
                    rgbs.append(rgb)
                    depths.append(depth)
                    proprios.append(np.array([pos[0], pos[1], pos[2], yaw], np.float32))
    finally:
        try:
            client.enableApiControl(False)
        except Exception:
            pass
    np.savez_compressed(
        PAIRS,
        rgb=np.stack(rgbs),
        depth=np.stack(depths),
        proprio=np.stack(proprios),
    )
    print(
        f"collected {len(rgbs)} left_farther={left_far} right_farther={right_far} tie={tie}",
        flush=True,
    )
    if min(left_far, right_far) < 10:
        raise SystemExit("side labels are still one-sided; not training")


def _mode_z(dyn, embed: torch.Tensor) -> torch.Tensor:
    b = embed.shape[0]
    h = dyn.rssm.initial_h(b, dyn.device, dyn.torch_dtype)
    post = dyn.rssm.post_probs(h, embed)
    classes = int(dyn.rssm.stoch_classes)
    flat = post.reshape(-1, classes)
    idx = flat.argmax(dim=-1)
    onehot = F.one_hot(idx, classes).to(dtype=post.dtype).view_as(post)
    z = (onehot + post - post.detach()).reshape(b, -1)
    return torch.cat([h, z], dim=-1)


def _depth_sym(dyn, feature: torch.Tensor) -> torch.Tensor:
    pred = dyn.depth_decoder(feature).squeeze(1)
    return pred


def train() -> None:
    data = np.load(PAIRS)
    rgb = data["rgb"]
    depth = data["depth"]
    proprio = data["proprio"]
    n = rgb.shape[0]
    raw = yaml.safe_load(CFG.read_text())
    dyn, _ = load_torch_dynamics(raw.get("world_model") or {}, str(WM), device="cuda", freeze=False)
    for p in dyn.parameters():
        p.requires_grad_(False)
    for p in list(dyn.encoder.parameters()) + list(dyn.depth_decoder.parameters()):
        p.requires_grad_(True)
    dyn.depth_decoder_trained = True
    opt = torch.optim.AdamW(
        [p for p in dyn.parameters() if p.requires_grad],
        lr=LR,
        betas=(0.9, 0.95),
    )
    dyn.train()
    rng = np.random.default_rng(0)
    print(f"trainable {sum(p.numel() for p in dyn.parameters() if p.requires_grad)} n={n}", flush=True)
    for step in range(STEPS):
        ix = rng.integers(0, n, size=BATCH)
        xb = torch.from_numpy(np.ascontiguousarray(rgb[ix])).to(dyn.device)
        db = torch.from_numpy(np.ascontiguousarray(depth[ix])).to(dyn.device, dyn.torch_dtype)
        pb = torch.from_numpy(np.ascontiguousarray(proprio[ix])).to(dyn.device, dyn.torch_dtype)
        flip = torch.rand(BATCH, device=dyn.device) < 0.5
        if bool(flip.any()):
            xb = xb.clone()
            db = db.clone()
            xb[flip] = xb[flip].flip(2)
            db[flip] = db[flip].flip(2)
        embed = dyn._embed(xb, pb)
        pred = _depth_sym(dyn, _mode_z(dyn, embed))
        if pred.shape[-2:] != db.shape[-2:]:
            pred = F.interpolate(pred.unsqueeze(1), size=db.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        side = _cone_side_loss(pred, db)
        valid = torch.isfinite(db) & (db > 0.0) & (db < 200.0)
        if bool(valid.any()):
            pixel = F.mse_loss(pred[valid], _symlog(db)[valid])
        else:
            pixel = pred.new_zeros(())
        loss = side + PIXEL_WEIGHT * pixel
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in dyn.parameters() if p.requires_grad], 100.0)
        opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            with torch.no_grad():
                meters = torch.sign(pred) * torch.expm1(pred.abs())
                spread = float(meters.reshape(BATCH, -1).std(dim=0).mean())
            print(
                f"step {step} side={float(side):.3f} pixel={float(pixel):.3f} map_std={spread:.3f}",
                flush=True,
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dyn.save_checkpoint(str(CKPT), optimizer=opt, step=STEPS)
    print(f"saved {CKPT}", flush=True)


def evaluate() -> None:
    import airsim

    raw = yaml.safe_load(CFG.read_text())
    dyn, _ = load_torch_dynamics(raw.get("world_model") or {}, str(CKPT), device="cuda", freeze=True)
    dyn.eval()
    rows = _rows("route00.jsonl")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    records = []
    try:
        for step in HOLDOUT:
            row = rows[step]
            pos = np.asarray(row["pos"], dtype=np.float64)
            yaw = float(np.radians(float(row["yaw_deg"])))
            _set_pose(client, airsim, pos, yaw)
            rgb, _depth_small = _grab(client, airsim)
            rs = client.simGetImages(
                [airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)]
            )
            full = np.array(
                airsim.list_to_2d_float_array(rs[0].image_data_float, rs[0].width, rs[0].height),
                dtype=np.float32,
            )
            gt = cone_clearances(np.where(np.isfinite(full) & (full > 0), full, 0.0))
            rgb_t = torch.from_numpy(rgb).unsqueeze(0).to(dyn.device)
            proprio = torch.tensor([[pos[0], pos[1], pos[2], yaw]], device=dyn.device, dtype=dyn.torch_dtype)
            with torch.no_grad():
                pred = _depth_sym(dyn, _mode_z(dyn, dyn._embed(rgb_t, proprio))).squeeze(0)
                meters = (torch.sign(pred) * torch.expm1(pred.abs())).float().cpu().numpy()
            wm = cone_clearances(meters)
            ratio = max(gt["left"], gt["right"]) / max(min(gt["left"], gt["right"]), 1e-3)
            clear = bool(ratio >= CLEAR_RATIO)
            same = (gt["left"] >= gt["right"]) == (wm["left"] >= wm["right"])
            rec = {
                "step": step,
                "gt": [round(float(gt[k]), 2) for k in ("forward", "left", "right")],
                "wm": [round(float(wm[k]), 2) for k in ("forward", "left", "right")],
                "clear": clear,
                "same_side": bool(same),
            }
            records.append(rec)
            print("step", step, "gt", rec["gt"], "wm", rec["wm"], "clear", clear, "same", same, flush=True)
    finally:
        try:
            client.enableApiControl(False)
        except Exception:
            pass
    diffs = []
    for a in records:
        for b in records:
            diffs.append(max(abs(x - y) for x, y in zip(a["wm"], b["wm"])))
    spread = float(max(diffs) if diffs else 0.0)
    clear_rows = [r for r in records if r["clear"]]
    signs_ok = bool(clear_rows) and all(r["same_side"] for r in clear_rows)
    moved = spread >= 1.0
    passed = bool(moved and signs_ok)
    report = {"passed": passed, "map_spread_m": round(spread, 3), "frames": records}
    REPORT.write_text(json.dumps(report, indent=2))
    print(
        f"HOLD OUT spread={spread:.2f}m clear={len(clear_rows)} signs_ok={signs_ok} passed={passed}",
        flush=True,
    )


def main() -> None:
    collect()
    train()
    evaluate()


if __name__ == "__main__":
    main()
