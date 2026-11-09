"""Calibration of the ensemble's 10-step survival on branch probes: real no-collision rate per predicted bin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.collect_consequence_dataset import pool_min
from experiments.aerial.rl.consequence_model import goal_features, history_inputs, load_checkpoint, survival
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.precompute_da3_consequence import decode_frames

BINS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 1.01]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dirs", required=True)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--depth-ckpt", default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep-routes", default="", help="comma list of route ids; empty keeps all")
    args = ap.parse_args()
    keep = set(filter(None, args.keep_routes.split(",")))
    dev = args.device
    models = [load_checkpoint(Path(p), dev).eval() for p in args.ckpts.split(",")]
    pred = DepthMinPredictor.from_checkpoint(Path(args.depth_ckpt), device=dev)
    surv_all, alive_all, prog_all = [], [], []
    for bd in [Path(p) for p in args.probe_dirs.split(",")]:
        review = json.loads((bd / "AUTO_REVIEW.json").read_text())
        for rec in review["probes"]:
            if keep and rec["route_id"] not in keep:
                continue
            z = np.load(bd / "probes" / rec["file"])
            if "step_depth56" not in z.files:
                continue
            frames = decode_frames({"jpeg": z["hist_jpeg"], "jpeg_offsets": z["hist_jpeg_offsets"]})
            pred.reset()
            d56 = [pool_min(pred.predict_depth(Observation(rgb=np.ascontiguousarray(f), state=np.zeros(7, np.float32),
                                                           collided=False))) for f in frames]
            hist_a = np.asarray(z["hist_action"], np.float32)
            cands = np.asarray(z["candidates"], np.float32)[:, :10]
            M = len(cands)
            coll = np.asarray(z["coll_step"])
            prev = hist_a[-2] if len(hist_a) >= 2 else np.zeros(4, np.float32)
            g = torch.as_tensor(np.repeat(goal_features(z["state"], z["goal"])[None], M, 0), device=dev)
            pa = torch.as_tensor(np.repeat(prev[None], M, 0), device=dev)
            a = torch.as_tensor(cands, device=dev)
            s_l, p_l = [], []
            with torch.no_grad():
                for m in models:
                    d_k, a_k = history_inputs(d56, list(hist_a[:len(d56) - 1]), m.frame_offsets, m.hist_actions)
                    d_k = d_k[0] if len(m.frame_offsets) == 1 else d_k
                    d = torch.as_tensor(np.repeat(d_k[None], M, 0), device=dev)
                    ha = torch.as_tensor(np.repeat(a_k[None], M, 0), device=dev) if m.hist_actions else None
                    o = m(d, g, pa, a, ha)
                    s_l.append(survival(o["hazard_logit"])[..., -1].cpu().numpy())
                    p_l.append((-o["dgoal"][..., -1]).cpu().numpy())
            surv_all.append(np.mean(s_l, 0))
            prog_all.append(np.mean(p_l, 0))
            alive_all.append(~((coll >= 0) & (coll < 10)))
    s = np.concatenate(surv_all)
    al = np.concatenate(alive_all)
    print(f"candidates {len(s)}  overall real survival {al.mean():.3f}")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (s >= lo) & (s < hi)
        if m.any():
            print(f"  predicted [{lo:.2f},{min(hi, 1.0):.2f})  n={int(m.sum()):5d}  mean_pred={s[m].mean():.3f}  real={al[m].mean():.3f}")
    pr = np.concatenate(prog_all)
    print("candidates with predicted progress > 1 m:")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (s >= lo) & (s < hi) & (pr > 1.0)
        if m.any():
            print(f"  predicted [{lo:.2f},{min(hi, 1.0):.2f})  n={int(m.sum()):5d}  mean_pred={s[m].mean():.3f}  real={al[m].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
