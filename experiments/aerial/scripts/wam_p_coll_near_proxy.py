#!/usr/bin/env python3
"""e8b: p_coll vs near-depth proxy (d_fwd < 3m) for depthaux + baseline WMs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch

    from experiments.aerial.rl import dataset as ds
    from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics
    from experiments.aerial.rl.latent_encode_probe import (
        encode_window_packed,
        forward_depth_m,
    )

    cfg = yaml.safe_load((root / "configs/aerial_rl.yaml").read_text()) or {}
    wm_cfg = dict(cfg.get("world_model") or {})
    wm_cfg["device"] = "cuda"
    episodes = ds.load_dataset(
        root / "experiments/aerial/rl/artifacts/dataset_v0_d_full_20260828",
        skip_quarantined=True,
    )

    def auc(y: np.ndarray, s: np.ndarray):
        yb = y.astype(bool)
        pos, neg = s[yb], s[~yb]
        if pos.size == 0 or neg.size == 0:
            return None
        gt = 0.0
        for pv in pos:
            gt += float(np.sum(neg < pv)) + 0.5 * float(np.sum(neg == pv))
        return gt / (pos.size * neg.size)

    def probe(ckpt: str, max_n: int = 1500, stride: int = 8, near_m: float = 3.0):
        dyn = TorchRSSMDynamics.from_config(wm_cfg)
        dyn.load_checkpoint(str(root / ckpt) if not Path(ckpt).is_absolute() else ckpt)
        dyn.eval()
        ys, ps = [], []
        n = 0
        for ep in episodes:
            if n >= max_n:
                break
            if len(ep) < 2:
                continue
            for t in range(0, len(ep), stride):
                if n >= max_n:
                    break
                obs = ep[t].obs
                d = forward_depth_m(obs)
                if d is None:
                    continue
                feat = encode_window_packed(dyn, ep, t, window=8)
                with torch.no_grad():
                    f = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(
                        dyn.device, dyn.torch_dtype
                    )
                    p = float(torch.sigmoid(dyn.coll_head(f)).item())
                ys.append(1.0 if float(d) < near_m else 0.0)
                ps.append(p)
                n += 1
        y = np.asarray(ys)
        p = np.asarray(ps)
        p_near = float(p[y == 1].mean()) if (y == 1).any() else None
        p_far = float(p[y == 0].mean()) if (y == 0).any() else None
        a = auc(y, p)
        return {
            "ckpt": ckpt,
            "n": int(len(y)),
            "n_near_lt3": int(y.sum()),
            "auc_vs_near3m": None if a is None else round(float(a), 4),
            "mean_p_near": None if p_near is None else round(p_near, 4),
            "mean_p_far": None if p_far is None else round(p_far, 4),
            "imag_penalty_near_w10": None if p_near is None else round(10.0 * p_near, 4),
            "frac_p_ge_0.5": round(float((p >= 0.5).mean()), 4),
            "p99": round(float(np.quantile(p, 0.99)), 4),
            "p_max": round(float(p.max()), 4),
        }

    outs = {}
    for name, ckpt in [
        (
            "depthaux_8500",
            "experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt",
        ),
        (
            "baseline_3500",
            "experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt",
        ),
    ]:
        print(f"probing {name}...", flush=True)
        outs[name] = probe(ckpt)
        print(json.dumps(outs[name], indent=2), flush=True)

    out_path = root / "artifacts/wam_p_coll_near_proxy.json"
    out_path.write_text(json.dumps(outs, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
