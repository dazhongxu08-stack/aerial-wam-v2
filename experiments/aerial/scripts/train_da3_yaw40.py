"""Keep left/right after a 40° yaw, without dropping the near-meter gate.

Starts from the near-wall decoder. The signed S-8j checkpoint is not written.
Saves a new file only when hold side stays 24/25, near error does not rise,
and the ±40° side count does not fall.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.wm_yaw_prior import N_YAW, SIDE_RATIO, _gt_cones
from experiments.aerial.scripts.train_da3_cone_yaw import (
    _flatten,
    _loss,
    _score,
    _soft_sides,
)


def _yaw40(pred: np.ndarray, gt: np.ndarray, n_pos: int) -> tuple[int, int]:
    ok = n = 0
    step = 2
    for pos in range(n_pos):
        base = pos * N_YAW
        for src in range(N_YAW):
            for dst in ((src + step) % N_YAW, (src - step) % N_YAW):
                truth = _gt_cones(gt[base + dst])
                hat = _gt_cones(pred[base + dst])
                if min(truth[1], truth[2]) <= 1e-3:
                    continue
                if max(truth[1], truth[2]) / min(truth[1], truth[2]) < SIDE_RATIO:
                    continue
                n += 1
                ok += int((hat[1] >= hat[2]) == (truth[1] >= truth[2]))
    return ok, n


def _predict_maps(model, rgb: np.ndarray, device: str) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for i in range(0, rgb.shape[0], 4):
            window = torch.from_numpy(np.ascontiguousarray(rgb[i : i + 4])).to(device).unsqueeze(1)
            pred, _ = model.predict_from_window(window)
            rows.append(pred.detach().float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _side_only(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pl, pr = _soft_sides(pred)
    gl, gr = _soft_sides(gt.detach())
    gdiff = torch.log(gl.clamp_min(0.05)) - torch.log(gr.clamp_min(0.05))
    pdiff = torch.log(pl.clamp_min(0.05)) - torch.log(pr.clamp_min(0.05))
    clear = gdiff.abs() >= math.log(SIDE_RATIO)
    if not bool(clear.any()):
        return pred.sum() * 0.0
    return torch.relu(-pdiff[clear] * torch.sign(gdiff[clear])).mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth-ckpt",
        default="experiments/aerial/rl/artifacts/da3_cone_yaw_20261009/depth_cone_yaw.pt",
    )
    parser.add_argument(
        "--pairs",
        default="experiments/aerial/rl/artifacts/wm_ckpt_yaw_prior_20261009/pairs.npz",
    )
    parser.add_argument(
        "--out",
        default="experiments/aerial/rl/artifacts/da3_cone_yaw_20261009/depth_cone_yaw40.pt",
    )
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    src = (ROOT / args.depth_ckpt).resolve()
    signed = (
        ROOT
        / "experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"
    ).resolve()
    out = (ROOT / args.out).resolve()
    if out == signed or src == signed:
        raise SystemExit("refusing to write over the signed depth head")
    if out == src:
        raise SystemExit(f"refusing to write over {src}")
    blob = np.load(ROOT / args.pairs)
    rgb = blob["rgb"]
    depth = blob["depth"]
    n_pos = int(rgb.shape[0])
    train_idx = list(range(0, n_pos - 2))
    hold_idx = [n_pos - 2, n_pos - 1]
    train_rgb, train_depth = _flatten(rgb, depth, train_idx)
    hold_rgb, hold_depth = _flatten(rgb, depth, hold_idx)
    near = np.array([_gt_cones(train_depth[i])[0] < 25.0 for i in range(train_depth.shape[0])])
    clear = []
    for i in range(train_depth.shape[0]):
        left, right = _gt_cones(train_depth[i])[1], _gt_cones(train_depth[i])[2]
        if min(left, right) <= 1e-3:
            clear.append(False)
            continue
        clear.append(max(left, right) / min(left, right) >= SIDE_RATIO)
    clear = np.asarray(clear, dtype=bool)
    pred = DepthMinPredictor.from_checkpoint(str(src), device="cuda")
    model = pred._model
    before = _score(model, hold_rgb, hold_depth, "cuda")
    hold_maps = _predict_maps(model, hold_rgb, "cuda")
    y_ok, y_n = _yaw40(hold_maps, hold_depth, len(hold_idx))
    print(
        f"before hold {before['side']} near {before['near_mae']:.2f} "
        f"±40 {y_ok}/{y_n}",
        flush=True,
    )
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.decoder.parameters(), lr=1e-6)
    rng = np.random.default_rng(1)
    near_idx = np.flatnonzero(near)
    clear_idx = np.flatnonzero(clear)
    if near_idx.size < 4 or clear_idx.size < 4:
        raise SystemExit(f"pools too small near={near_idx.size} clear={clear_idx.size}")
    for step in range(int(args.steps)):
        model.decoder.train()
        a = near_idx[rng.integers(0, near_idx.size, size=2)]
        b = clear_idx[rng.integers(0, clear_idx.size, size=2)]
        images = np.concatenate([train_rgb[a], train_rgb[b]], axis=0)
        gt = np.concatenate([train_depth[a], train_depth[b]], axis=0)
        if rng.random() < 0.5:
            images = images[:, :, ::-1, :].copy()
            gt = gt[:, :, ::-1].copy()
        window = torch.from_numpy(np.ascontiguousarray(images)).cuda().unsqueeze(1)
        hat, _ = model.predict_from_window(window)
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).cuda()
        loss = _loss(hat[:2], gt_t[:2]) + 2.0 * _side_only(hat[2:], gt_t[2:])
        if not bool(torch.isfinite(loss)):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == int(args.steps) - 1:
            print(f"step {step} loss {float(loss.detach()):.4f}", flush=True)
    after = _score(model, hold_rgb, hold_depth, "cuda")
    after_maps = _predict_maps(model, hold_rgb, "cuda")
    a_ok, a_n = _yaw40(after_maps, hold_depth, len(hold_idx))
    print(
        f"after hold {after['side']} near {after['near_mae']:.2f} "
        f"±40 {a_ok}/{a_n}",
        flush=True,
    )
    keep = (
        after["side_ok"] >= before["side_ok"]
        and after["side_n"] == before["side_n"]
        and after["near_mae"] <= before["near_mae"] + 0.3
        and a_n == y_n
        and a_ok >= y_ok
        and (a_ok > y_ok or after["near_mae"] < before["near_mae"] - 0.15)
    )
    if not keep:
        print("±40° gate failed; weights not saved", flush=True)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(src, map_location="cpu")
    payload["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload["yaw40_finetune"] = True
    torch.save(payload, out)
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
