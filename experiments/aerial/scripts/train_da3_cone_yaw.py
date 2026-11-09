"""Update the image depth head so the picture sets forward, left, and right.

The DINOv2 encoder stays frozen. The DPT decoder is trained on route-0 yaw
fans. The signed checkpoint is read only. A new file is written only when the
held-out frames are at least as good on side sign and forward error.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.wm_yaw_prior import CENTER_FRAC, SIDE_RATIO, _gt_cones


def _symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _band_mean(depth: torch.Tensor) -> torch.Tensor:
    batch, height, width = depth.shape
    dh = max(1, int(height * CENTER_FRAC))
    dw = max(1, int(width * CENTER_FRAC))
    r0 = (height - dh) // 2
    c0 = (width - dw) // 2
    crop = depth[:, r0 : r0 + dh, c0 : c0 + dw]
    valid = torch.isfinite(crop) & (crop > 0.5) & (crop < 80.0)
    means = []
    for i in range(int(batch)):
        values = crop[i][valid[i]]
        if int(values.numel()) < 4:
            means.append(depth.new_zeros(()))
        else:
            means.append(values.mean())
    return torch.stack(means)


def _soft_sides(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, height, width = depth.shape
    dh = max(1, int(height * CENTER_FRAC))
    r0 = (height - dh) // 2
    band = depth[:, r0 : r0 + dh, :]
    mid = width // 2

    def softmin(crop: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(crop) & (crop > 0.05)
        neg = (-crop / 2.0).masked_fill(~valid, -1.0e4)
        return -2.0 * torch.logsumexp(neg.reshape(batch, -1), dim=-1)

    return softmin(band[:, :, :mid]), softmin(band[:, :, mid:])


def _loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred_f = _band_mean(pred)
    gt_f = _band_mean(gt)
    pl, pr = _soft_sides(pred)
    # Ground-truth sides from the same band mins, computed outside, passed as
    # the gt depth map. Use hard mins via a detached softmin at low temperature.
    gl, gr = _soft_sides(gt.detach())
    gdiff = torch.log(gl.clamp_min(0.05)) - torch.log(gr.clamp_min(0.05))
    pdiff = torch.log(pl.clamp_min(0.05)) - torch.log(pr.clamp_min(0.05))
    clear = gdiff.abs() >= math.log(SIDE_RATIO)
    if bool(clear.any()):
        order = torch.relu(-pdiff[clear] * torch.sign(gdiff[clear])).mean()
    else:
        order = pred.sum() * 0.0
    finite = torch.isfinite(gt_f) & (gt_f > 0.5) & torch.isfinite(pred_f)
    agree = ((torch.sign(pdiff) == torch.sign(gdiff)) | ~clear).detach()
    use = finite & agree
    if bool(use.any()):
        meters = F.mse_loss(_symlog(pred_f[use]), _symlog(gt_f[use]))
    else:
        meters = pred.sum() * 0.0
    return 4.0 * order + meters


def _flatten(rgb: np.ndarray, depth: np.ndarray, indices: list[int]):
    images = np.concatenate([rgb[i] for i in indices], axis=0)
    maps = np.concatenate([depth[i] for i in indices], axis=0)
    return images, maps


def _score(model, rgb: np.ndarray, depth: np.ndarray, device: str) -> dict:
    model.eval()
    preds = []
    gts = []
    side_ok = side_n = 0
    with torch.no_grad():
        for i in range(0, rgb.shape[0], 4):
            window = torch.from_numpy(np.ascontiguousarray(rgb[i : i + 4])).to(device).unsqueeze(1)
            pred, _ = model.predict_from_window(window)
            gt = depth[i : i + 4]
            pred_np = pred.detach().float().cpu().numpy()
            for k in range(pred_np.shape[0]):
                hat = _gt_cones(pred_np[k])
                truth = _gt_cones(gt[k])
                preds.append(hat)
                gts.append(truth)
                if (
                    min(truth[1], truth[2]) > 1e-3
                    and max(truth[1], truth[2]) / min(truth[1], truth[2]) >= SIDE_RATIO
                ):
                    side_n += 1
                    side_ok += int((hat[1] >= hat[2]) == (truth[1] >= truth[2]))
    pred_a = np.stack(preds)
    gt_a = np.stack(gts)
    finite = np.isfinite(pred_a[:, 0]) & np.isfinite(gt_a[:, 0]) & (gt_a[:, 0] > 0.5)
    mae = float(np.mean(np.abs(pred_a[finite, 0] - gt_a[finite, 0]))) if finite.any() else float("inf")
    std = float(np.std(pred_a[finite, 0])) if finite.sum() > 1 else 0.0
    ratio = pred_a[finite, 0] / np.clip(gt_a[finite, 0], 1e-3, None)
    med = float(np.median(ratio)) if finite.any() else float("nan")
    near = finite & (gt_a[:, 0] < 25.0)
    near_mae = (
        float(np.mean(np.abs(pred_a[near, 0] - gt_a[near, 0]))) if near.any() else mae
    )
    return {
        "side": f"{side_ok}/{side_n}",
        "side_ok": int(side_ok),
        "side_n": int(side_n),
        "side_rate": (side_ok / side_n) if side_n else 0.0,
        "fwd_mae": mae,
        "fwd_std": std,
        "fwd_ratio": med,
        "near_mae": near_mae,
        "near_n": int(near.sum()),
    }


def _better(after: dict, before: dict) -> bool:
    return (
        after["side_ok"] >= before["side_ok"]
        and after["side_n"] == before["side_n"]
        and after["near_mae"] < before["near_mae"] - 0.3
        and after["fwd_std"] > 1.0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth-ckpt",
        default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt",
    )
    parser.add_argument(
        "--pairs",
        default="experiments/aerial/rl/artifacts/wm_ckpt_yaw_prior_20261009/pairs.npz",
    )
    parser.add_argument(
        "--out",
        default="experiments/aerial/rl/artifacts/da3_cone_yaw_20261009/depth_cone_yaw.pt",
    )
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    src = (ROOT / args.depth_ckpt).resolve()
    out = (ROOT / args.out).resolve()
    if out == src:
        raise SystemExit(f"refusing to write over {src}")
    blob = np.load(ROOT / args.pairs)
    rgb = blob["rgb"]
    depth = blob["depth"]
    train_idx = list(range(0, rgb.shape[0] - 2))
    hold_idx = [rgb.shape[0] - 2, rgb.shape[0] - 1]
    train_rgb, train_depth = _flatten(rgb, depth, train_idx)
    hold_rgb, hold_depth = _flatten(rgb, depth, hold_idx)
    near_mask = np.array([_gt_cones(train_depth[i])[0] < 25.0 for i in range(train_depth.shape[0])])
    if int(near_mask.sum()) >= 8:
        train_rgb = train_rgb[near_mask]
        train_depth = train_depth[near_mask]
    print(f"train near frames {train_rgb.shape[0]}", flush=True)
    pred = DepthMinPredictor.from_checkpoint(str(src), device="cuda")
    model = pred._model
    before = _score(model, hold_rgb, hold_depth, "cuda")
    print(
        f"before hold side {before['side']} mae {before['fwd_mae']:.2f} "
        f"near {before['near_mae']:.2f} std {before['fwd_std']:.2f} ratio {before['fwd_ratio']:.2f}",
        flush=True,
    )
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.decoder.parameters(), lr=1e-6)
    rng = np.random.default_rng(0)
    n = train_rgb.shape[0]
    for step in range(int(args.steps)):
        model.decoder.train()
        pick = rng.integers(0, n, size=2)
        images = train_rgb[pick].copy()
        gt = train_depth[pick].copy()
        if rng.random() < 0.5:
            images = images[:, :, ::-1, :].copy()
            gt = gt[:, :, ::-1].copy()
        window = torch.from_numpy(np.ascontiguousarray(images)).cuda().unsqueeze(1)
        hat, _ = model.predict_from_window(window)
        loss = _loss(hat, torch.from_numpy(np.ascontiguousarray(gt)).cuda())
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
    print(
        f"after hold side {after['side']} mae {after['fwd_mae']:.2f} "
        f"near {after['near_mae']:.2f} std {after['fwd_std']:.2f} ratio {after['fwd_ratio']:.2f}",
        flush=True,
    )
    if not _better(after, before):
        print(
            f"holdout did not improve side {before['side']}->{after['side']} "
            f"near {before['near_mae']:.2f}->{after['near_mae']:.2f}; weights not saved",
            flush=True,
        )
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(src, map_location="cpu")
    payload["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload["cone_yaw_finetune"] = True
    torch.save(payload, out)
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
