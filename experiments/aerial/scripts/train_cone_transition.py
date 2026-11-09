"""Train a next-cone predictor on image-depth cones, then check the six holdout frames.

The frozen encoder does not separate left from right. Current cones come from
the image depth head. This script learns how those cones move after a yaw,
labeled by the same head on the next view. It does not train the actor.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.cone_transition import ConeTransition
from experiments.aerial.rl.depth_geometry import cone_clearances
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation

ROOT = Path("/home/yao/aerial-wam-v2")
PAIRS = ROOT / "experiments/aerial/rl/artifacts/wm_ckpt_cone_side_encoder_20260928/pairs.npz"
DEPTH_CKPT = (
    ROOT
    / "experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"
)
TRAJ = (
    ROOT
    / "experiments/aerial/rl/artifacts/feasible_side_turnfix_20260928/eval_noshield/traj/route00.jsonl"
)
OUT = ROOT / "experiments/aerial/rl/artifacts/cone_transition_20260929"
HOLDOUT = (0, 34, 43, 69, 125, 152)
CLEAR_RATIO = 1.25


def _cones_of(depth: np.ndarray) -> np.ndarray:
    c = cone_clearances(np.asarray(depth, dtype=np.float32))
    return np.asarray(
        [float(c["forward"]), float(c["left"]), float(c["right"])], dtype=np.float64
    )


def _da3_cones(pred: DepthMinPredictor, rgb: np.ndarray) -> np.ndarray:
    pred.reset()
    obs = Observation(rgb=np.asarray(rgb, dtype=np.uint8), state=np.zeros(7, dtype=np.float32))
    _mn, cones = pred.predict_min_and_cones(obs)
    if cones is None:
        return np.full(3, np.nan)
    return np.asarray(
        [float(cones["forward"]), float(cones["left"]), float(cones["right"])],
        dtype=np.float64,
    )


def _side(cones: np.ndarray) -> tuple[float, bool]:
    left, right = float(cones[1]), float(cones[2])
    if not (np.isfinite(left) and np.isfinite(right) and min(left, right) > 0.2):
        return 0.0, False
    delta = math.log(left) - math.log(right)
    return delta, abs(delta) >= math.log(CLEAR_RATIO)


def _groups(proprio: np.ndarray) -> list[np.ndarray]:
    n = int(proprio.shape[0])
    if n % 5 != 0:
        raise RuntimeError(f"expected groups of 5 yaw views, got {n}")
    groups = []
    for i in range(0, n, 5):
        block = proprio[i : i + 5, :3]
        spread = float(np.max(np.linalg.norm(block - block[:1], axis=1)))
        if spread > 0.05:
            raise RuntimeError(f"yaw group at {i} is not one pose (spread {spread:.3f})")
        groups.append(np.arange(i, i + 5))
    return groups


def _pairs(cones: np.ndarray, proprio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src, dst, act = [], [], []
    for idx in _groups(proprio):
        for a in idx:
            for b in idx:
                yaw_a = float(proprio[int(a), 3])
                yaw_b = float(proprio[int(b), 3])
                dyaw = (yaw_b - yaw_a + math.pi) % (2.0 * math.pi) - math.pi
                src.append(cones[int(a)])
                dst.append(cones[int(b)])
                act.append([0.0, 0.0, 0.0, dyaw])
    return (
        np.asarray(src, dtype=np.float32),
        np.asarray(dst, dtype=np.float32),
        np.asarray(act, dtype=np.float32),
    )


def _batch_loss(pred: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    log_p = torch.log1p(pred)
    log_d = torch.log1p(dst.clamp(min=0))
    mse = torch.mean((log_p - log_d) ** 2)
    gap = log_d[:, 1] - log_d[:, 2]
    pred_gap = log_p[:, 1] - log_p[:, 2]
    clear = gap.abs() >= math.log(CLEAR_RATIO)
    if bool(clear.any()):
        order = torch.relu(math.log(CLEAR_RATIO) - gap.sign() * pred_gap)[clear].mean()
    else:
        order = pred.new_zeros(())
    return mse + order


def _train(src: np.ndarray, dst: np.ndarray, act: np.ndarray) -> ConeTransition:
    stay = np.isclose(act[:, 3], 0.0)
    # Cross-yaw pairs outnumber staying put. Repeat the stay rows so a zero
    # action keeps the cones it was given.
    repeats = max(1, int(np.count_nonzero(~stay) // max(1, int(np.count_nonzero(stay)))))
    src = np.concatenate([src, np.repeat(src[stay], repeats, axis=0)], axis=0)
    dst = np.concatenate([dst, np.repeat(dst[stay], repeats, axis=0)], axis=0)
    act = np.concatenate([act, np.repeat(act[stay], repeats, axis=0)], axis=0)
    model = ConeTransition().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    n = int(src.shape[0])
    rng = np.random.default_rng(0)
    src_t = torch.from_numpy(src).cuda()
    dst_t = torch.from_numpy(dst).cuda()
    act_t = torch.from_numpy(act).cuda()
    stay_t = torch.isclose(act_t[:, 3], torch.zeros((), device="cuda"))
    model.train()
    for step in range(2000):
        pick = torch.as_tensor(rng.integers(0, n, size=128), device="cuda")
        pred = model(src_t[pick], act_t[pick])
        loss = _batch_loss(pred, dst_t[pick])
        opt.zero_grad()
        loss.backward()
        nn_clip(model)
        opt.step()
        if step % 400 == 0 or step == 1999:
            with torch.no_grad():
                stay_loss = _batch_loss(model(src_t[stay_t][:256], act_t[stay_t][:256]), dst_t[stay_t][:256])
                yaw_loss = _batch_loss(model(src_t[~stay_t][:256], act_t[~stay_t][:256]), dst_t[~stay_t][:256])
            print(
                f"step {step} loss {float(loss.detach()):.4f} "
                f"stay {float(stay_loss):.4f} yaw {float(yaw_loss):.4f}",
                flush=True,
            )
    model.eval()
    return model


def nn_clip(model: ConeTransition) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)


def _pose(rec: dict) -> tuple[float, float, float, float]:
    if "pos" in rec:
        x, y, z = (float(v) for v in rec["pos"][:3])
    else:
        x, y, z = float(rec["x"]), float(rec["y"]), float(rec["z"])
    yaw = float(rec.get("yaw", rec.get("yaw_deg")))
    yaw_deg = math.degrees(yaw) if abs(yaw) <= 6.5 else yaw
    return x, y, z, yaw_deg


def _grab(client, x: float, y: float, z_up: float, yaw_deg: float):
    import airsim

    quat = airsim.to_quaternion(0.0, 0.0, math.radians(yaw_deg))
    pose = airsim.Pose(airsim.Vector3r(x, y, -z_up), quat)
    client.simSetVehiclePose(pose, True)
    time.sleep(0.3)
    responses = client.simGetImages(
        [
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False),
        ]
    )
    scene, depth_resp = responses
    rgb_bgr = np.frombuffer(scene.image_data_uint8, dtype=np.uint8).reshape(
        scene.height, scene.width, 3
    )
    rgb = rgb_bgr[:, :, ::-1].copy()
    depth = airsim.list_to_2d_float_array(
        depth_resp.image_data_float, depth_resp.width, depth_resp.height
    )
    return rgb, np.asarray(depth, dtype=np.float32)


def _holdout(pred: DepthMinPredictor, model: ConeTransition) -> dict:
    import airsim

    rows = [json.loads(line) for line in TRAJ.read_text().splitlines() if line.strip()]
    client = airsim.MultirotorClient()
    client.confirmConnection()
    report = []
    current_ok = 0
    current_clear = 0
    yaw_ok = 0
    yaw_clear = 0
    zero_keeps_left = 0
    left_open = 0
    for step in HOLDOUT:
        x, y, z, yaw_deg = _pose(rows[step])
        views = {}
        for off in (0, -40, 40):
            rgb, depth = _grab(client, x, y, z, yaw_deg + off)
            views[off] = {
                "da3": _da3_cones(pred, rgb),
                "gt": _cones_of(depth),
            }
        base = views[0]["da3"]
        gt = views[0]["gt"]
        d_da3, clear_da3 = _side(base)
        d_gt, clear_gt = _side(gt)
        same = (not clear_gt) or (clear_da3 and (d_da3 > 0) == (d_gt > 0))
        if clear_gt:
            current_clear += 1
            current_ok += int(same and (d_da3 > 0) == (d_gt > 0))
        zero = model.predict_np(base, np.zeros(4, dtype=np.float64))
        d_zero, clear_zero = _side(zero)
        if clear_gt and d_gt > 0:
            left_open += 1
            zero_keeps_left += int(clear_zero and d_zero > 0)
        yaw_rows = []
        for off in (-40, 40):
            target = views[off]["da3"]
            action = np.array([0.0, 0.0, 0.0, math.radians(off)], dtype=np.float64)
            nxt = model.predict_np(base, action)
            d_hat, clear_hat = _side(nxt)
            d_tgt, clear_tgt = _side(target)
            match = bool(clear_tgt and clear_hat and (d_hat > 0) == (d_tgt > 0))
            if clear_tgt:
                yaw_clear += 1
                yaw_ok += int(match)
            yaw_rows.append(
                {
                    "offset_deg": off,
                    "pred": [round(float(v), 2) for v in nxt],
                    "da3": [round(float(v), 2) for v in target],
                    "match": match,
                    "target_clear": clear_tgt,
                }
            )
        line = {
            "step": step,
            "da3": [round(float(v), 2) for v in base],
            "gt": [round(float(v), 2) for v in gt],
            "da3_follows": bool(same and ((not clear_gt) or (d_da3 > 0) == (d_gt > 0))),
            "zero_pred": [round(float(v), 2) for v in zero],
            "yaw": yaw_rows,
        }
        report.append(line)
        print(
            f"step {step} da3 L{base[1]:.1f} R{base[2]:.1f} gt L{gt[1]:.1f} R{gt[2]:.1f} "
            f"follows {line['da3_follows']} zero L{zero[1]:.1f} R{zero[2]:.1f}",
            flush=True,
        )
    passed = current_clear > 0 and current_ok == current_clear and zero_keeps_left == left_open
    summary = {
        "current_clear_match": f"{current_ok}/{current_clear}",
        "zero_keeps_left_open": f"{zero_keeps_left}/{left_open}",
        "yaw_clear_match": f"{yaw_ok}/{yaw_clear}",
        "current_follows_building": passed,
        "frames": report,
    }
    print(
        f"CURRENT {summary['current_clear_match']} "
        f"ZERO_LEFT {summary['zero_keeps_left_open']} "
        f"YAW {summary['yaw_clear_match']} PASS {passed}",
        flush=True,
    )
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(PAIRS)
    pred = DepthMinPredictor.from_checkpoint(DEPTH_CKPT, device="cuda")
    cached = OUT / "pair_da3_cones.npz"
    if cached.exists():
        cones = np.load(cached)["cones"].astype(np.float32)
        print(f"loaded da3 cones {cones.shape[0]}", flush=True)
    else:
        rgb = data["rgb"]
        cones = np.zeros((rgb.shape[0], 3), dtype=np.float32)
        for i in range(rgb.shape[0]):
            cones[i] = _da3_cones(pred, rgb[i])
            if i % 30 == 0:
                print(f"da3 {i}/{rgb.shape[0]}", flush=True)
        np.savez_compressed(cached, cones=cones)
    src, dst, act = _pairs(cones, data["proprio"])
    print(f"transition pairs {src.shape[0]}", flush=True)
    model = _train(src, dst, act)
    torch.save({"model": model.state_dict()}, OUT / "cone_transition.pt")
    summary = _holdout(pred, model)
    (OUT / "holdout_cones.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
