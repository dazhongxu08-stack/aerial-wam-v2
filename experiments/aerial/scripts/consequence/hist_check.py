"""Does a history model rely on real history? Score branch probes with real vs truncated history.

Variants feed the same probe with its full history, only the last 6 / 2 frames
(as in the older onpolicy / branch probes), or the current frame alone; missing
past frames are clamped to the oldest kept one, exactly as in training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.collect_consequence_dataset import pool_min
from experiments.aerial.rl.consequence_model import (
    goal_features, history_inputs, load_checkpoint, select_candidate, survival)
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.precompute_da3_consequence import decode_frames


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--models", required=True, help="name=ckpt1,ckpt2;name2=...")
    ap.add_argument("--keep", default="21,6,2,1", help="history frames kept per variant")
    ap.add_argument("--depth-ckpt", default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--surv-min", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = args.device
    groups = {}
    for part in args.models.split(";"):
        name, paths = part.split("=")
        groups[name] = [load_checkpoint(Path(p), dev).eval() for p in paths.split(",")]
    keeps = [int(k) for k in args.keep.split(",")]
    pred = DepthMinPredictor.from_checkpoint(Path(args.depth_ckpt), device=dev)
    bd = Path(args.probe_dir)
    review = json.loads((bd / "AUTO_REVIEW.json").read_text(encoding="utf-8"))

    rows = {f"{n}/k{k}": {"surv": [], "prog": [], "pick": []} for n in groups for k in keeps}
    hit_all, dg_all = [], []
    for rec in review["probes"]:
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
        hit = (coll >= 0) & (coll < 10)
        dist = np.asarray(z["dist"], np.float64)
        dg = np.where(np.isfinite(dist[:, 10]), dist[:, 10] - dist[:, 0], np.nan)
        hit_all.append(hit)
        dg_all.append(dg)
        prev = hist_a[-2] if len(hist_a) >= 2 else np.zeros(4, np.float32)
        g = torch.as_tensor(np.repeat(goal_features(z["state"], z["goal"])[None], M, 0), device=dev)
        pa = torch.as_tensor(np.repeat(prev[None], M, 0), device=dev)
        a = torch.as_tensor(cands, device=dev)
        for k in keeps:
            seq = d56[-k:]
            acts = list(hist_a[:len(d56) - 1])[-(len(seq) - 1):] if len(seq) > 1 else []
            for n, ms in groups.items():
                s_l, p_l = [], []
                with torch.no_grad():
                    for m in ms:
                        d_k, a_k = history_inputs(seq, acts, m.frame_offsets, m.hist_actions)
                        d_k = d_k[0] if len(m.frame_offsets) == 1 else d_k
                        d = torch.as_tensor(np.repeat(d_k[None], M, 0), device=dev)
                        ha = torch.as_tensor(np.repeat(a_k[None], M, 0), device=dev) if m.hist_actions else None
                        o = m(d, g, pa, a, ha)
                        s_l.append(survival(o["hazard_logit"])[..., -1].cpu().numpy())
                        p_l.append((-o["dgoal"][..., -1]).cpu().numpy())
                s, p = np.mean(s_l, 0), np.mean(p_l, 0)
                r = rows[f"{n}/k{k}"]
                r["surv"].append(s)
                r["prog"].append(p)
                r["pick"].append(select_candidate(s, p, np.ones(M, bool), args.surv_min))

    hit = np.concatenate(hit_all)
    dg = np.concatenate(dg_all)
    n_probe = len(hit_all)
    out = {"probes": n_probe, "cands": int(len(hit)), "hit_rate": float(hit.mean()), "variants": {}}
    for key, r in rows.items():
        s = np.concatenate(r["surv"])
        p = np.concatenate(r["prog"])
        ref = rows[key.split("/")[0] + f"/k{keeps[0]}"]
        s_ref = np.concatenate(ref["surv"])
        picks_hit = [bool(h[i]) for h, i in zip(hit_all, r["pick"])]
        picks_prog = [float(-d[i]) if np.isfinite(d[i]) else 0.0 for d, i in zip(dg_all, r["pick"])]
        same = np.mean([i == j for i, j in zip(r["pick"], ref["pick"])])
        ok = np.isfinite(dg)
        out["variants"][key] = {
            "pred_hazard_mean": float((1 - s).mean()),
            "brier": float((((1 - s) - hit) ** 2).mean()),
            "auc_hazard": auc(1 - s, hit),
            "confident_hit_frac": float(((s >= args.surv_min) & hit).sum() / max(1, hit.sum())),
            "prog_mae_m": float(np.abs(p[ok] - (-dg[ok])).mean()),
            "pick_hit_rate": float(np.mean(picks_hit)),
            "pick_real_progress_m": float(np.mean(picks_prog)),
            "mean_abs_dsurv_vs_full": float(np.abs(s - s_ref).mean()),
            "pick_same_as_full": float(same),
        }
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"probes={n_probe} cands={len(hit)} hit_rate={hit.mean():.3f}")
    for key, v in out["variants"].items():
        print(key, " ".join(f"{a}={b:.3f}" for a, b in v.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
