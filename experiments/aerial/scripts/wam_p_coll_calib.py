#!/usr/bin/env python3
"""e8: calibrate WM ``coll_head`` (p_coll) against GT collided labels + near-depth.

Reports AUC / Brier / ECE and a depth-binned mean p_coll, so we can tell whether
the collision signal that imagination RL actually trains on is informative.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    """Mann-Whitney / Wilcoxon-Mann-Whitney AUC; None if one class missing."""
    y = y.astype(bool)
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        return None
    # Pairwise: fraction of (pos,neg) pairs with pos>neg (+0.5 ties).
    # For modest n this O(n_pos*n_neg) is fine (max_samples~2k).
    gt = 0.0
    for pv in pos:
        gt += float(np.sum(neg < pv)) + 0.5 * float(np.sum(neg == pv))
    return gt / (pos.size * neg.size)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if not np.any(m):
            continue
        ece += (m.sum() / n) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--wm-ckpt", required=True)
    ap.add_argument("--config", default="configs/aerial_rl.yaml")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/wam_p_coll_calib.json")
    args = ap.parse_args()

    import torch

    from experiments.aerial.rl import dataset as ds
    from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics
    from experiments.aerial.rl.latent_encode_probe import (
        encode_window_packed,
        forward_depth_m,
    )

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    wm_cfg = dict(cfg.get("world_model") or {})
    wm_cfg["device"] = str(args.device)

    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = root / dataset
    ckpt = Path(args.wm_ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt

    print(f"[p_coll_calib] loading dataset {dataset}")
    episodes = ds.load_dataset(dataset, skip_quarantined=True)
    print(f"[p_coll_calib] {len(episodes)} episodes")

    print(f"[p_coll_calib] loading WM {ckpt}")
    dyn = TorchRSSMDynamics.from_config(wm_cfg)
    meta = dyn.load_checkpoint(str(ckpt))
    dyn.eval()
    print(f"[p_coll_calib] ckpt step={meta.get('step')}")

    ys: List[float] = []
    ps: List[float] = []
    dfs: List[Optional[float]] = []
    n_taken = 0
    for ep in episodes:
        if n_taken >= int(args.max_samples):
            break
        if len(ep) < 2:
            continue
        for t in range(0, len(ep), max(1, int(args.stride))):
            if n_taken >= int(args.max_samples):
                break
            tr = ep[t]
            obs = tr.obs
            collided = bool(getattr(obs, "collided", False))
            try:
                feat = encode_window_packed(dyn, ep, t, window=int(args.window))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] encode fail ep t={t}: {exc}")
                continue
            with torch.no_grad():
                f = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(dyn.device)
                # coll_head expects feature_dim = h+z packed
                logit = dyn.coll_head(f.to(dyn.torch_dtype)).squeeze()
                p = float(torch.sigmoid(logit).item())
            ys.append(1.0 if collided else 0.0)
            ps.append(p)
            dfs.append(forward_depth_m(obs))
            n_taken += 1

    y = np.asarray(ys, dtype=np.float64)
    p = np.asarray(ps, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    auc = _auc(y, p)
    brier = float(np.mean((p - y) ** 2))
    ece = _ece(y, p)
    # threshold sweep at max_p_coll=0.5 (shield emergency latch)
    thr = 0.5
    pred = p >= thr
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None

    # depth-binned mean p_coll (near should be high if head is geometry-aware)
    depth_bins = [(0, 3), (3, 6), (6, 10), (10, 20), (20, 200)]
    depth_rows = []
    for lo, hi in depth_bins:
        idx = [
            i
            for i, d in enumerate(dfs)
            if d is not None and np.isfinite(d) and lo <= float(d) < hi
        ]
        if not idx:
            depth_rows.append({"band": [lo, hi], "n": 0})
            continue
        depth_rows.append(
            {
                "band": [lo, hi],
                "n": len(idx),
                "mean_p_coll": round(float(p[idx].mean()), 4),
                "frac_collided": round(float(y[idx].mean()), 4),
            }
        )

    out: Dict[str, Any] = {
        "wm_ckpt": str(ckpt),
        "dataset": str(dataset),
        "n": int(len(y)),
        "n_pos_collided": n_pos,
        "n_neg": n_neg,
        "pos_rate": round(n_pos / max(1, len(y)), 4),
        "mean_p_coll": round(float(p.mean()), 4),
        "mean_p_coll_on_pos": round(float(p[y == 1].mean()), 4) if n_pos else None,
        "mean_p_coll_on_neg": round(float(p[y == 0].mean()), 4) if n_neg else None,
        "auc": None if auc is None else round(float(auc), 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "at_thr_0.5": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": None if prec is None else round(prec, 4),
            "recall": None if rec is None else round(rec, 4),
        },
        "p_coll_by_fwd_depth": depth_rows,
        "verdict": (
            "informative"
            if (auc is not None and auc >= 0.7 and n_pos >= 20)
            else (
                "weak"
                if (auc is not None and auc >= 0.55)
                else "uninformative_or_degenerate"
            )
        ),
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"[p_coll_calib] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
