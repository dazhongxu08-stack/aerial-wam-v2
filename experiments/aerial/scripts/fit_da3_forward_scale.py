"""Fit one positive scale on the image depth head.

Left and right stay in the same order because every pixel is multiplied by the
same number. The signed checkpoint is not rewritten. A sidecar is written only
when the held-out side count is unchanged and the forward error drops.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.wm_yaw_prior import SIDE_RATIO, _gt_cones


def _predict(model, rgb: np.ndarray, device: str) -> np.ndarray:
    rows = []
    for i in range(0, rgb.shape[0], 4):
        window = torch.from_numpy(np.ascontiguousarray(rgb[i : i + 4])).to(device).unsqueeze(1)
        pred, _ = model.predict_from_window(window)
        rows.append(pred.detach().float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _pack(pred: np.ndarray, gt: np.ndarray) -> dict:
    hats = np.stack([_gt_cones(pred[i]) for i in range(pred.shape[0])])
    truth = np.stack([_gt_cones(gt[i]) for i in range(gt.shape[0])])
    finite = np.isfinite(hats[:, 0]) & np.isfinite(truth[:, 0]) & (truth[:, 0] > 0.5) & (hats[:, 0] > 0.5)
    side_ok = side_n = 0
    for hat, true in zip(hats, truth):
        if min(true[1], true[2]) <= 1e-3:
            continue
        if max(true[1], true[2]) / min(true[1], true[2]) < SIDE_RATIO:
            continue
        side_n += 1
        side_ok += int((hat[1] >= hat[2]) == (true[1] >= true[2]))
    ratio = hats[finite, 0] / np.clip(truth[finite, 0], 1e-3, None)
    mae = float(np.mean(np.abs(hats[finite, 0] - truth[finite, 0]))) if finite.any() else float("inf")
    return {
        "side_ok": int(side_ok),
        "side_n": int(side_n),
        "fwd_mae": mae,
        "fwd_ratio_med": float(np.median(ratio)) if finite.any() else float("nan"),
        "n_finite": int(finite.sum()),
        "hats": hats,
        "truth": truth,
        "finite": finite,
    }


def main() -> int:
    src = ROOT / "experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"
    pairs = ROOT / "experiments/aerial/rl/artifacts/wm_ckpt_yaw_prior_20261009/pairs.npz"
    out_dir = ROOT / "experiments/aerial/rl/artifacts/da3_cone_yaw_20261009"
    blob = np.load(pairs)
    rgb = blob["rgb"]
    depth = blob["depth"]
    n_pos = int(rgb.shape[0])
    train_idx = list(range(0, n_pos - 2))
    hold_idx = [n_pos - 2, n_pos - 1]
    pred = DepthMinPredictor.from_checkpoint(str(src), device="cuda")
    model = pred._model
    model.eval()

    def flat(indices):
        return (
            np.concatenate([rgb[i] for i in indices], axis=0),
            np.concatenate([depth[i] for i in indices], axis=0),
        )

    train_rgb, train_gt = flat(train_idx)
    hold_rgb, hold_gt = flat(hold_idx)
    print(f"predict train {train_rgb.shape[0]} hold {hold_rgb.shape[0]}", flush=True)
    train_hat = _predict(model, train_rgb, "cuda")
    hold_hat = _predict(model, hold_rgb, "cuda")
    train = _pack(train_hat, train_gt)
    hold = _pack(hold_hat, hold_gt)
    near = train["finite"] & (train["truth"][:, 0] < 25.0)
    if int(near.sum()) < 8:
        near = train["finite"]
    hat_n = train["hats"][near, 0]
    gt_n = train["truth"][near, 0]
    near_ratio = float(np.median(hat_n / np.clip(gt_n, 1e-3, None)))
    grid = np.linspace(0.9, 2.4, 31)
    maes = [float(np.mean(np.abs(s * hat_n - gt_n))) for s in grid]
    scale = float(grid[int(np.argmin(maes))])
    print(
        f"train near n={int(near.sum())} ratio {near_ratio:.3f} "
        f"mae_scale {scale:.3f} mae {min(maes):.2f}",
        flush=True,
    )
    print(
        f"train side {train['side_ok']}/{train['side_n']} mae {train['fwd_mae']:.2f} "
        f"ratio {train['fwd_ratio_med']:.3f} scale {scale:.3f}",
        flush=True,
    )
    print(
        f"hold raw side {hold['side_ok']}/{hold['side_n']} mae {hold['fwd_mae']:.2f} "
        f"ratio {hold['fwd_ratio_med']:.3f}",
        flush=True,
    )
    scaled = _pack(hold_hat * scale, hold_gt)
    print(
        f"hold scaled side {scaled['side_ok']}/{scaled['side_n']} mae {scaled['fwd_mae']:.2f} "
        f"ratio {scaled['fwd_ratio_med']:.3f}",
        flush=True,
    )
    log_p = np.log(train["hats"][near, 0])
    log_g = np.log(train["truth"][near, 0])
    exponent, log_scale = np.linalg.lstsq(
        np.stack([log_p, np.ones_like(log_p)], axis=1), log_g, rcond=None
    )[0]
    exponent = float(exponent)
    power_scale = float(np.exp(log_scale))
    print(f"power exponent {exponent:.3f} scale {power_scale:.3f}", flush=True)
    if exponent <= 0.0:
        powered = scaled
        print("power exponent is not positive", flush=True)
    else:
        powered_maps = power_scale * np.power(np.clip(hold_hat, 1e-3, None), exponent)
        powered = _pack(powered_maps, hold_gt)
    print(
        f"hold power side {powered['side_ok']}/{powered['side_n']} mae {powered['fwd_mae']:.2f} "
        f"ratio {powered['fwd_ratio_med']:.3f}",
        flush=True,
    )
    n_yaw = 18
    step = 2
    for name, maps in (("raw", hold_hat), ("power", powered_maps if exponent > 0.0 else hold_hat * scale)):
        ok40 = n40 = 0
        hats = np.stack([_gt_cones(maps[i]) for i in range(maps.shape[0])])
        truth = np.stack([_gt_cones(hold_gt[i]) for i in range(hold_gt.shape[0])])
        for pos in range(len(hold_idx)):
            base = pos * n_yaw
            for src in range(n_yaw):
                for dst in ((src + step) % n_yaw, (src - step) % n_yaw):
                    t = truth[base + dst]
                    h = hats[base + dst]
                    if min(t[1], t[2]) <= 1e-3:
                        continue
                    if max(t[1], t[2]) / min(t[1], t[2]) < SIDE_RATIO:
                        continue
                    n40 += 1
                    ok40 += int((h[1] >= h[2]) == (t[1] >= t[2]))
        print(f"{name} after ±40° side {ok40}/{n40}", flush=True)
    hold_near = hold["finite"] & (hold["truth"][:, 0] < 25.0)

    def _near_stats(maps: np.ndarray) -> tuple:
        hats = np.stack([_gt_cones(maps[i]) for i in range(maps.shape[0])])
        if not bool(hold_near.any()):
            return float("inf"), float("nan")
        mae = float(np.mean(np.abs(hats[hold_near, 0] - hold["truth"][hold_near, 0])))
        ratio = float(
            np.median(hats[hold_near, 0] / np.clip(hold["truth"][hold_near, 0], 1e-3, None))
        )
        return mae, ratio

    raw_near_mae, raw_near_ratio = _near_stats(hold_hat)
    scaled_near_mae, scaled_near_ratio = _near_stats(hold_hat * scale)
    power_maps = (
        power_scale * np.power(np.clip(hold_hat, 1e-3, None), exponent)
        if exponent > 0.0
        else hold_hat * scale
    )
    power_near_mae, power_near_ratio = _near_stats(power_maps)
    print(
        f"hold near n={int(hold_near.sum())} raw mae {raw_near_mae:.2f} ratio {raw_near_ratio:.3f} "
        f"scaled mae {scaled_near_mae:.2f} ratio {scaled_near_ratio:.3f} "
        f"power mae {power_near_mae:.2f} ratio {power_near_ratio:.3f}",
        flush=True,
    )
    use_power = (
        exponent > 0.0
        and 0.8 <= power_near_ratio <= 1.25
        and power_near_mae <= scaled_near_mae
    )
    chosen_mae = power_near_mae if use_power else scaled_near_mae
    chosen_ratio = power_near_ratio if use_power else scaled_near_ratio
    chosen_side = powered if use_power else scaled
    same_side = chosen_side["side_ok"] == hold["side_ok"] and chosen_side["side_n"] == hold["side_n"]
    ratio_ok = 0.8 <= chosen_ratio <= 1.25
    better = same_side and ratio_ok and chosen_mae < raw_near_mae - 0.5
    report_scale = power_scale if use_power else scale
    report_exp = exponent if use_power else 1.0
    report = {
        "scale": report_scale,
        "exponent": report_exp,
        "train_side": f"{train['side_ok']}/{train['side_n']}",
        "train_mae": train["fwd_mae"],
        "train_ratio": train["fwd_ratio_med"],
        "hold_side": f"{hold['side_ok']}/{hold['side_n']}",
        "hold_mae": hold["fwd_mae"],
        "hold_ratio": hold["fwd_ratio_med"],
        "hold_scaled_side": f"{scaled['side_ok']}/{scaled['side_n']}",
        "hold_scaled_mae": scaled["fwd_mae"],
        "hold_scaled_ratio": scaled["fwd_ratio_med"],
        "hold_power_side": f"{powered['side_ok']}/{powered['side_n']}",
        "hold_power_mae": powered["fwd_mae"],
        "hold_power_ratio": powered["fwd_ratio_med"],
        "saved": bool(better),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "forward_scale_report.json").write_text(json.dumps(report, indent=2))
    if not better:
        print("scale did not pass; weights not saved", flush=True)
        return 1
    dest = out_dir / "depth_cone_yaw.pt"
    if dest.resolve() == src.resolve():
        raise SystemExit(f"refusing to write over {src}")
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.link(str(src), str(dest))
    sidecar = Path(str(dest) + ".forward_scale.json")
    sidecar.write_text(
        json.dumps(
            {
                "scale": report["scale"],
                "exponent": report["exponent"],
                "fit": "route0 yaw-fan train positions",
            },
            indent=2,
        )
    )
    print(f"saved {dest} + {sidecar.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
