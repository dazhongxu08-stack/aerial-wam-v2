#!/usr/bin/env python3
"""Fit the office consequence head to the chassis frames.

Depth images come from the real-camera head. Labels come from RealSense
after the same floor hide the car uses. A new checkpoint is written.
The real depth file, the sim gap head, and the car are not replaced.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.aerial.rl.consequence_model import (  # noqa: E402
    depth_input,
    goal_features,
    load_checkpoint,
    save_checkpoint,
)
from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402
from experiments.aerial.rl.env.obs import Observation  # noqa: E402
from office_bands import cones3_forward  # noqa: E402

ROOT = Path("/data/simworld/run/keep_20261104")
ANCHORS = Path("/data/simworld/run/office_real_20261104/anchors.npz")
ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
DEPTH = ART / "da3_office_real_20261103/depth_office_cone_real.pt"
INIT = ART / "indoor_office_gap_20261104/c_s218/consequence_best.pt"
OUT = ART / "indoor_office_real_20261104/c_s218"
FY, CY, SRC_H = 386.88, 246.954, 480.0
GROUND_FRAC = 0.42
INVALID_M = 30.0
YAW = 0.3141592653589793
STEPS = 80


def hide_ground(depth: np.ndarray) -> np.ndarray:
    out = np.array(depth, np.float32, copy=True)
    focal = 0.5 * out.shape[0] / math.tan(math.radians(45.0))
    angle = math.atan((GROUND_FRAC * SRC_H - CY) / FY)
    row = int(np.clip(math.floor(out.shape[0] / 2.0 + focal * math.tan(angle)), 0, out.shape[0] - 1))
    out[row:, :] = INVALID_M
    return out


def pool_min(depth: np.ndarray, out: int = 56) -> np.ndarray:
    d = np.asarray(depth, np.float32)
    h, w = d.shape
    fh, fw = h // out, w // out
    d = d[: fh * out, : fw * out].reshape(out, fh, out, fw)
    return d.min(axis=(1, 3)).astype(np.float32)


def metric_scale(pred: np.ndarray, real: np.ndarray, seen: np.ndarray) -> float:
    model = np.where(seen, pred, np.inf).astype(np.float32)
    sensor = np.where(np.isfinite(real) & (real > 0), real, np.inf).astype(np.float32)
    a = pool_min(model)[8:29]
    b = pool_min(sensor)[8:29]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0.2) & (a < 1e3) & (b > 0.4) & (b < 6.0)
    if int(ok.sum()) < 40:
        return 1.0
    ratio = float(np.median(b[ok] / a[ok]))
    if not 0.05 < ratio < 2.0:
        return 1.0
    return ratio


def encode() -> None:
    if ANCHORS.is_file():
        return
    frames = sorted(ROOT.glob("frame_*.npz"))
    if not frames:
        raise SystemExit(f"no frames in {ROOT}")
    depth_model = DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    packed_d, cones, da_cones, goals, pose = [], [], [], [], []
    state = np.array([0.0, 0.0, 0.34, 0, 0, 0, 0.0], np.float64)
    goal = np.array([8.0, 0.0, 0.34], np.float64)
    goal_v = goal_features(state, goal)
    scales = []
    for path in frames:
        z = np.load(path)
        rgb = np.ascontiguousarray(z["rgb"])
        real = z["depth"].astype(np.float32)
        real[real <= 0] = np.nan
        seen = z["seen"].astype(bool)
        if seen.shape != real.shape:
            seen = np.isfinite(real)
        depth_model.reset()
        pred = depth_model.predict_depth(Observation(
            rgb=rgb, state=np.zeros(7, np.float32), collided=False,
        ))
        pred = np.asarray(pred, np.float32)
        if pred.shape[:2] != real.shape[:2]:
            import cv2
            pred = cv2.resize(pred, (real.shape[1], real.shape[0]), interpolation=cv2.INTER_LINEAR)
        ratio = metric_scale(pred, real, seen)
        scales.append(ratio)
        da56 = pool_min(hide_ground(np.where(seen, pred * ratio, INVALID_M)))
        rs56 = pool_min(hide_ground(np.where(np.isfinite(real), real, INVALID_M)))
        g = cones3_forward(rs56)
        d = cones3_forward(da56)
        packed_d.append(depth_input(da56))
        cones.append(np.asarray(g, np.float32))
        da_cones.append(np.asarray(d, np.float32))
        goals.append(goal_v)
        pose.append(np.asarray(z["pose"], np.float32))
        print(
            f"{path.name} scale={ratio:.3f} rs={g[0]:.2f},{g[1]:.2f},{g[2]:.2f} "
            f"da={d[0]:.2f},{d[1]:.2f},{d[2]:.2f}",
            flush=True,
        )
    ANCHORS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        ANCHORS,
        depth=np.stack(packed_d).astype(np.float32),
        cones=np.stack(cones).astype(np.float32),
        da=np.stack(da_cones).astype(np.float32),
        goal=np.stack(goals).astype(np.float32),
        pose=np.stack(pose).astype(np.float32),
    )
    print(f"SCALE med {float(np.median(scales)):.3f} n {len(frames)}", flush=True)
    del depth_model
    torch.cuda.empty_cache()


def pack(depth, goal, device):
    d = torch.as_tensor(depth, dtype=torch.float32, device=device)
    g = torch.as_tensor(goal, dtype=torch.float32, device=device)
    prev = torch.zeros(d.shape[0], 4, device=device)
    return d, g, prev


def logits(model, depth, goal, prev, dx, dyaw):
    actions = torch.zeros(depth.shape[0], 10, 4, device=depth.device)
    actions[:, 0, 0] = dx
    actions[:, 0, 3] = dyaw
    return model(depth, goal, prev, actions)["hazard_logit"][:, 0]


def main() -> int:
    encode()
    data = np.load(ANCHORS)
    dev = torch.device("cuda")
    depth, goal, prev = pack(data["depth"], data["goal"], dev)
    cones = torch.as_tensor(data["cones"], dtype=torch.float32, device=dev)
    fwd, left, right = cones[:, 0], cones[:, 1], cones[:, 2]
    side_left = (left < 1.0) & (fwd > 2.0) & (right > 1.2)
    side_right = (right < 1.0) & (fwd > 2.0) & (left > 1.2)
    side = side_left | side_right
    safe = (fwd > 2.0) & ~side
    opened = (fwd > 2.5) & (left > 1.5) & (right > 1.5)
    hit = fwd < 1.15
    print(
        f"n {len(fwd)} safe {int(safe.sum())} open {int(opened.sum())} "
        f"hit {int(hit.sum())} side_l {int(side_left.sum())} side_r {int(side_right.sum())}",
        flush=True,
    )
    model = load_checkpoint(str(INIT), "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    def report(step: int) -> tuple[float, float, float, float, float]:
        model.eval()
        with torch.no_grad():
            zf = logits(model, depth, goal, prev, 0.8, 0.0)
            zl = logits(model, depth, goal, prev, 0.0, -YAW)
            zr = logits(model, depth, goal, prev, 0.0, YAW)
            sf = 1.0 - torch.sigmoid(zf)
            sl = 1.0 - torch.sigmoid(zl)
            sr = 1.0 - torch.sigmoid(zr)
        open_fwd = float(sf[opened].min()) if bool(opened.any()) else 0.0
        open_spin = float(torch.maximum(sl, sr)[opened].max()) if bool(opened.any()) else 1.0
        hit_fwd = float(sf[hit].max()) if bool(hit.any()) else 1.0
        side_fwd = float(sf[side].max()) if bool(side.any()) else 1.0
        away = []
        if bool(side_left.any()):
            away.append(sr[side_left])
        if bool(side_right.any()):
            away.append(sl[side_right])
        side_away = float(torch.cat(away).min()) if away else 0.0
        print(
            f"step {step} open_fwd_min {open_fwd:.3f} open_spin_max {open_spin:.3f} "
            f"hit_fwd_max {hit_fwd:.3f} side_fwd_max {side_fwd:.3f} side_away_min {side_away:.3f}",
            flush=True,
        )
        return open_fwd, open_spin, hit_fwd, side_fwd, side_away

    OUT.mkdir(parents=True, exist_ok=True)
    report(0)
    best = None
    for step in range(1, STEPS + 1):
        model.train()
        z_fwd = logits(model, depth, goal, prev, 0.8, 0.0)
        z_left = logits(model, depth, goal, prev, 0.0, -YAW)
        z_right = logits(model, depth, goal, prev, 0.0, YAW)
        loss = torch.zeros((), device=dev)
        if bool(safe.any()):
            loss = loss + torch.relu(z_fwd[safe] + 2.0).pow(2).mean()
        if bool(opened.any()):
            loss = loss + torch.relu(z_fwd[opened] + 3.0).pow(2).mean()
            loss = loss + torch.relu(2.0 - z_left[opened]).pow(2).mean()
            loss = loss + torch.relu(2.0 - z_right[opened]).pow(2).mean()
        if bool(hit.any()):
            loss = loss + 4.0 * torch.relu(2.0 - z_fwd[hit]).pow(2).mean()
        if bool(side.any()):
            loss = loss + 4.0 * torch.relu(2.0 - z_fwd[side]).pow(2).mean()
        if bool(side_left.any()):
            loss = loss + 4.0 * torch.relu(z_right[side_left] + 2.0).pow(2).mean()
        if bool(side_right.any()):
            loss = loss + 4.0 * torch.relu(z_left[side_right] + 2.0).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == STEPS:
            stats = report(step)
            ok = stats[0] >= 0.95 and stats[1] <= 0.5 and stats[2] <= 0.2 and stats[3] <= 0.3 and stats[4] >= 0.95
            save_checkpoint(OUT / "consequence_last.pt", model, {"step": step})
            if ok and (best is None or stats[0] > best):
                best = stats[0]
                save_checkpoint(OUT / "consequence_best.pt", model, {"step": step, "open_fwd": stats[0]})
                print("SAVE", step, flush=True)
    if best is None:
        save_checkpoint(OUT / "consequence_best.pt", model, {"step": STEPS, "fallback": True})
        print("SAVE_LAST", flush=True)
    print("BEST", best, "OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
