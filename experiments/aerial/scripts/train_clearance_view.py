"""Fit forward / left / right clearance on the current picture.

The route 0 canyon stays out of the loss. Nothing is flown.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.clearance_view import ClearanceView
from experiments.aerial.rl.dataset import load_episode
import torch.nn.functional as F

from experiments.aerial.scripts.train_clearance_code import _metrics, _mirror_batch
from experiments.aerial.scripts.train_posterior_gate import _batch, _clear


def _depth(obs) -> np.ndarray:
    depth = np.asarray(obs.depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[0]
    return depth


def _center_stats(depth: np.ndarray, center_frac: float = 0.12) -> tuple[float, float]:
    """Mean and 10th percentile of the same center crop as approach_clearances."""
    height, width = depth.shape
    band_h = max(1, int(round(height * center_frac)))
    band_w = max(1, int(round(width * center_frac)))
    row0 = (height - band_h) // 2
    col0 = (width - band_w) // 2
    crop = depth[row0 : row0 + band_h, col0 : col0 + band_w]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    if valid.size == 0:
        return 80.0, 80.0
    return float(min(valid.mean(), 80.0)), float(min(np.percentile(valid, 10), 80.0))


def _clip_log(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        value = 80.0
    return float(np.log1p(min(value, 80.0)))


def _label_row(tr) -> tuple[list[float], float]:
    mean_m, p10 = _center_stats(_depth(tr.obs))
    _fwd, left, right = _clear(tr.obs)
    uniform_near = mean_m < 15.0 and abs(mean_m - p10) < 4.0
    return [_clip_log(mean_m), _clip_log(left), _clip_log(right)], (4.0 if uniform_near else 1.0)


def _raw_rows(frames) -> np.ndarray:
    rows = []
    for tr in frames:
        mean_m, _p10 = _center_stats(_depth(tr.obs))
        _fwd, left, right = _clear(tr.obs)
        rows.append([mean_m, left, right])
    return np.asarray(rows, dtype=np.float32)


def _weighted_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per = ((pred - target) ** 2).mean(dim=-1)
    mse = (per * weight).sum() / weight.sum().clamp_min(1.0)
    meter_gap = torch.expm1(target[:, 1]) - torch.expm1(target[:, 2])
    big = meter_gap.abs() >= 8.0
    loss = mse
    if bool(big.any()):
        sign = torch.where(
            meter_gap[big] >= 0, torch.ones_like(meter_gap[big]), -torch.ones_like(meter_gap[big])
        )
        side = pred[big, 1] - pred[big, 2]
        loss = loss + 2.0 * F.relu(0.35 - sign * side).mean()
    true_m = torch.expm1(target[:, 0])
    close = true_m < 12.0
    if bool(close.any()):
        loss = loss + 4.0 * F.relu(pred[close, 0] - target[close, 0]).mean()
    return loss


def _held_keys(frames):
    keys = []
    for tr in frames:
        pos = np.asarray(tr.obs.position)
        keys.append((float(pos[0]), float(pos[1]), float(tr.obs.yaw)))
    return keys


def _leaks(tr, keys) -> bool:
    pos = np.asarray(tr.obs.position)
    yaw = float(tr.obs.yaw)
    for hx, hy, hyaw in keys:
        dyaw = (yaw - hyaw + np.pi) % (2 * np.pi) - np.pi
        if (float(pos[0]) - hx) ** 2 + (float(pos[1]) - hy) ** 2 < 4.0 and abs(dyaw) < 0.25:
            return True
    return False


def _load_split(street: Path, passage: Path, extra: list[str]):
    street_ep = load_episode(street)
    train, held = [], []
    for tr in street_ep:
        if float(np.asarray(tr.obs.position)[1]) <= 50.0:
            held.append(tr)
        else:
            train.append(tr)
    train.extend(load_episode(passage))
    keys = _held_keys(held)
    extra_n = 0
    for item in extra:
        path = Path(item)
        files = sorted(path.glob("episode_*.npz")) if path.is_dir() else [path]
        for ep_path in files:
            for tr in load_episode(ep_path):
                if getattr(tr.obs, "depth", None) is None or _leaks(tr, keys):
                    continue
                train.append(tr)
                extra_n += 1
    return train, held, extra_n


def _eval(model, frames):
    preds, raws = [], []
    for start in range(0, len(frames), 64):
        chunk = frames[start : start + 64]
        rgb, _prop = _batch(chunk, None)
        with torch.no_grad():
            pred = model(rgb.to(next(model.parameters()).device))
        preds.append(pred.cpu().numpy())
        raws.append(_raw_rows(chunk))
    return np.concatenate(preds), np.concatenate(raws)


def _report(tag: str, pred: np.ndarray, raw: np.ndarray) -> tuple[float, float, int]:
    acc, corr, sign_n = _metrics(pred, raw)
    gap = raw[:, 1] - raw[:, 2]
    pred_gap = np.expm1(pred[:, 1]) - np.expm1(pred[:, 2])
    big = np.abs(gap) >= 8.0
    left = big & (gap > 0)
    right = big & (gap < 0)
    mae = float(np.mean(np.abs(np.expm1(pred[:, 0]) - raw[:, 0])))
    print(
        f"{tag} sign_acc={acc:.2f} n={sign_n} "
        f"left={int(((pred_gap > 0) & left).sum())}/{int(left.sum())} "
        f"right={int(((pred_gap < 0) & right).sum())}/{int(right.sum())} "
        f"fwd_corr={corr:.2f} fwd_mae_m={mae:.1f}"
    )
    return acc, corr, sign_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--street", required=True)
    parser.add_argument("--passage", required=True)
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init", default="")
    args = parser.parse_args()

    train, held, extra_n = _load_split(Path(args.street), Path(args.passage), list(args.extra))
    print(f"train={len(train)} extra={extra_n} held_canyon={len(held)}")
    if len(train) < 20 or len(held) < 10:
        raise SystemExit("refuse: not enough frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClearanceView().to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
        print(f"init={args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    rgb, prop = _batch(train, None)
    target = torch.as_tensor([row for row, _weight in (_label_row(tr) for tr in train)], dtype=torch.float32)
    batch = max(1, int(args.batch))
    model.train()
    for step in range(int(args.steps)):
        opt.zero_grad()
        pick = torch.randint(0, rgb.shape[0], (min(batch, rgb.shape[0]),))
        rgb_s, _prop_s, target_s = _mirror_batch(rgb[pick], prop[pick], target[pick])
        pred = model(rgb_s.to(device))
        loss = _weighted_loss(pred, target_s.to(device), torch.ones(target_s.shape[0], device=device))
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 250 == 0:
            print(f"step={step} loss={float(loss.detach()):.4f}")

    model.eval()
    train_pred, train_raw = _eval(model, train[::6])
    _report("train", train_pred, train_raw)
    pred, raw = _eval(model, held)
    acc, corr, sign_n = _report("held", pred, raw)
    pairs = " ".join(
        f"{float(np.expm1(p)):.0f}/{float(t):.0f}" for p, t in zip(pred[:, 0], raw[:, 0])
    )
    print(f"held_fwd_pred/true {pairs}")
    close_frames = []
    for tr in train:
        mean_m, _p10 = _center_stats(_depth(tr.obs))
        _fwd, left, right = _clear(tr.obs)
        if mean_m < 12.0 and max(float(left), float(right)) < mean_m + 8.0:
            close_frames.append(tr)
    if close_frames:
        close_pred, close_raw = _eval(model, close_frames[:240])
        print(
            f"close n={len(close_frames)} "
            f"pred_med={float(np.median(np.expm1(close_pred[:, 0]))):.1f} "
            f"true_med={float(np.median(close_raw[:, 0])):.1f}"
        )
    ok = acc >= 0.8 and corr >= 0.5 and sign_n >= 10
    out = Path(args.out)
    if ok:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, str(out))
        print(f"PASS saved {out}")
    else:
        print("FAIL clearance view")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
