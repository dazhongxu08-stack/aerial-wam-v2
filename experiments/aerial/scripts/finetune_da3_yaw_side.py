"""Teach the image depth head to keep left/right after a yaw.

The DINOv2 encoder stays frozen. The DPT decoder is updated only on the
left/right log-ratio, not on absolute meters. The six route-0 poses are not
in the training pairs. No policy update.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.depth_geometry import cone_clearances, cone_mins_torch
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
OUT = ROOT / "experiments/aerial/rl/artifacts/da3_yaw_side_20260929"
HOLDOUT = (0, 34, 43, 69, 125, 152)
CLEAR = math.log(1.25)
OFFSETS = (-80, -40, 0, 40, 80)


def _soft_sides(depth: torch.Tensor, temp: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmin of the center-band left and right halves, in meters."""
    b, h, w = depth.shape
    dh = max(1, int(h * 0.5))
    r0 = (h - dh) // 2
    band = depth[:, r0 : r0 + dh, :]
    mid = w // 2

    def softmin(crop: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(crop) & (crop > 0.05)
        neg = (-crop / temp).masked_fill(~valid, -1.0e4)
        return -temp * torch.logsumexp(neg.reshape(b, -1), dim=-1)

    return softmin(band[:, :, :mid]), softmin(band[:, :, mid:])


def _gap_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    gt_f, gt_l, gt_r = cone_mins_torch(gt)
    pr_l, pr_r = _soft_sides(pred)
    gap_p = torch.log(pr_l.clamp(min=0.05)) - torch.log(pr_r.clamp(min=0.05))
    gap_g = torch.log(gt_l.clamp(min=0.05)) - torch.log(gt_r.clamp(min=0.05))
    clear = gap_g.abs() >= CLEAR
    if not bool(clear.any()):
        return gap_p.sum() * 0.0
    # Push the predicted side to agree with the building. Frames that already
    # agree contribute nothing, so the update does not chase the meter scale.
    signed = gap_g.detach().sign() * gap_p
    return torch.relu(CLEAR - signed)[clear].mean()


def _side_np(left: float, right: float) -> tuple[float, bool]:
    if not (np.isfinite(left) and np.isfinite(right) and min(left, right) > 0.2):
        return 0.0, False
    delta = math.log(left) - math.log(right)
    return delta, abs(delta) >= CLEAR


def _score_batch(model, rgb: np.ndarray, depth: np.ndarray, device: str) -> tuple[int, int]:
    model.eval()
    ok = n = 0
    with torch.no_grad():
        for i in range(0, rgb.shape[0], 8):
            batch = torch.from_numpy(rgb[i : i + 8]).to(device)
            window = batch.unsqueeze(1)
            pred, _ = model.predict_from_window(window)
            gt = torch.from_numpy(depth[i : i + 8]).to(device)
            _, gl, gr = cone_mins_torch(gt)
            _, pl, pr = cone_mins_torch(pred)
            for a, b, c, d in zip(gl.tolist(), gr.tolist(), pl.tolist(), pr.tolist()):
                dg, cg = _side_np(a, b)
                dp, cp = _side_np(c, d)
                if cg:
                    n += 1
                    ok += int(cp and (dp > 0) == (dg > 0))
    return ok, n


def _by_offset(model, rgb, depth, device: str) -> None:
    for k, name in enumerate(OFFSETS):
        idx = np.arange(k, rgb.shape[0], 5)
        ok, n = _score_batch(model, rgb[idx], depth[idx], device)
        print(f"  offset {name:+d} {ok}/{n}", flush=True)


def _train(model, rgb: np.ndarray, depth: np.ndarray) -> None:
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.decoder.parameters(), lr=1e-6)
    n = rgb.shape[0]
    rng = np.random.default_rng(0)
    for step in range(120):
        model.decoder.train()
        pick = rng.integers(0, n, size=4)
        images = rgb[pick].copy()
        gt = depth[pick].copy()
        if rng.random() < 0.5:
            images = images[:, :, ::-1, :].copy()
            gt = gt[:, :, ::-1].copy()
        pred, _ = model.predict_from_window(torch.from_numpy(images).cuda().unsqueeze(1))
        loss = _gap_loss(pred, torch.from_numpy(gt).cuda())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 40 == 0 or step == 119:
            print(f"step {step} gap {float(loss.detach()):.4f}", flush=True)
    model.eval()


def _grab(client, x, y, z_up, yaw_deg):
    import airsim

    quat = airsim.to_quaternion(0.0, 0.0, math.radians(yaw_deg))
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, -z_up), quat), True)
    time.sleep(0.3)
    scene, depth_resp = client.simGetImages(
        [
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False),
        ]
    )
    bgr = np.frombuffer(scene.image_data_uint8, dtype=np.uint8).reshape(scene.height, scene.width, 3)
    depth = airsim.list_to_2d_float_array(
        depth_resp.image_data_float, depth_resp.width, depth_resp.height
    )
    return bgr[:, :, ::-1].copy(), np.asarray(depth, dtype=np.float32)


def _holdout(pred: DepthMinPredictor) -> dict:
    import airsim

    rows = [json.loads(line) for line in TRAJ.read_text().splitlines() if line.strip()]
    client = airsim.MultirotorClient()
    client.confirmConnection()
    heading_ok = heading_n = yaw_ok = yaw_n = 0
    frames = []
    for step in HOLDOUT:
        rec = rows[step]
        x, y, z = (float(v) for v in rec["pos"][:3])
        yaw = float(rec["yaw_deg"])
        views = {}
        for off in (0, -40, 40):
            rgb, gt = _grab(client, x, y, z, yaw + off)
            pred.reset()
            obs = Observation(rgb=rgb, state=np.zeros(7, dtype=np.float32))
            hat = pred.predict_depth(obs)
            c_hat = cone_clearances(hat)
            c_gt = cone_clearances(gt)
            views[off] = (
                np.asarray([c_hat["forward"], c_hat["left"], c_hat["right"]]),
                np.asarray([c_gt["forward"], c_gt["left"], c_gt["right"]]),
            )
        da3, gt0 = views[0]
        d_hat, c_hat = _side_np(da3[1], da3[2])
        d_gt, c_gt = _side_np(gt0[1], gt0[2])
        if c_gt:
            heading_n += 1
            heading_ok += int(c_hat and (d_hat > 0) == (d_gt > 0))
        print(
            f"step {step} da3 L{da3[1]:.1f} R{da3[2]:.1f} gt L{gt0[1]:.1f} R{gt0[2]:.1f}",
            flush=True,
        )
        yaw_rows = []
        for off in (-40, 40):
            da3_y, gt_y = views[off]
            dp, cp = _side_np(da3_y[1], da3_y[2])
            dt, ct = _side_np(gt_y[1], gt_y[2])
            match = bool(ct and cp and (dp > 0) == (dt > 0))
            if ct:
                yaw_n += 1
                yaw_ok += int(match)
            yaw_rows.append(
                {
                    "offset": off,
                    "da3": [round(float(v), 2) for v in da3_y],
                    "gt": [round(float(v), 2) for v in gt_y],
                    "match": match,
                    "clear": ct,
                }
            )
            print(
                f"  {off:+d} da3 L{da3_y[1]:.1f} R{da3_y[2]:.1f} "
                f"gt L{gt_y[1]:.1f} R{gt_y[2]:.1f} match {match}",
                flush=True,
            )
        frames.append({"step": step, "yaw": yaw_rows})
    summary = {
        "heading": f"{heading_ok}/{heading_n}",
        "yaw": f"{yaw_ok}/{yaw_n}",
        "frames": frames,
    }
    print(f"HEADING {summary['heading']} YAW {summary['yaw']}", flush=True)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(PAIRS)
    rgb = data["rgb"]
    depth = data["depth"]
    pred = DepthMinPredictor.from_checkpoint(DEPTH_CKPT, device="cuda")
    model = pred._model
    print("before", flush=True)
    _by_offset(model, rgb, depth, "cuda")
    _train(model, rgb, depth)
    print("after", flush=True)
    _by_offset(model, rgb, depth, "cuda")
    payload = torch.load(DEPTH_CKPT, map_location="cpu")
    payload["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload["yaw_side_finetune"] = True
    torch.save(payload, OUT / "depth_yaw_side.pt")
    summary = _holdout(pred)
    (OUT / "holdout.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
