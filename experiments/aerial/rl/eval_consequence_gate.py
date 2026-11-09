#!/usr/bin/env python3
"""Stage 2 offline gate: does the predictor rank candidates like the simulator?

Inputs per probe are deploy-time only (depth head on the real flight history,
goal in body frame, previous action). Ground truth is the counterfactual
rollout of every candidate from the same pose.

Metrics on probes that have both a safe candidate and a colliding one:
* pairwise: share of (safe, collided) candidate pairs the model orders correctly
* top1_safe: the model's highest-scored candidate did not collide
* top1_progress: it did not collide and moved toward the goal (> 0.5 m)
Baselines: always the straight-forward candidate, uniform random, oracle.
Choices are made among candidates that pass the hold-altitude rule
(``altitude_allowed``); the unrestricted numbers are reported alongside.
Gate (restricted): pairwise >= 0.7, top1_safe >= 0.7, top1_safe at least
the exact uniform-random expectation, and top1_progress at least 0.15 above it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--depth-ckpt", default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt")
    ap.add_argument("--collision-cost-m", type=float, default=20.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--input-depth", default="da3", choices=("da3", "gt"),
                    help="gt = diagnostic only; must match how the checkpoint was trained")
    ap.add_argument("--routes", default="", help="comma list; evaluate only probes on these routes")
    ap.add_argument("--select", default="score", choices=("score", "filter"),
                    help="score: argmax expected progress minus collision price; "
                         "filter: most predicted progress among candidates with survival >= --surv-min "
                         "(pairwise then ranks by survival)")
    ap.add_argument("--surv-min", type=float, default=0.7)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch

    from experiments.aerial.rl.collect_consequence_dataset import pool_min
    from experiments.aerial.rl.consequence_model import (
        altitude_allowed, goal_features, history_inputs, load_checkpoint, score_candidates, select_candidate, survival,
    )
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation
    from experiments.aerial.rl.precompute_da3_consequence import decode_frames
    from experiments.aerial.rl.precompute_env_consequence import env_pair

    dev = torch.device(args.device)
    model = load_checkpoint(root / args.ckpt, device=args.device)
    pred = DepthMinPredictor.from_checkpoint(root / args.depth_ckpt, device=args.device)
    probe_dir = root / args.probe_dir
    review = json.loads((probe_dir / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    rows = []
    only = set(r for r in args.routes.split(",") if r)
    for rec in review["probes"]:
        if only and rec["route_id"] not in only:
            continue
        z = np.load(probe_dir / "probes" / rec["file"])
        if args.input_depth == "gt":
            seq = [z["gt_depth56"].astype(np.float32)]
        else:
            frames = decode_frames({"jpeg": z["hist_jpeg"], "jpeg_offsets": z["hist_jpeg_offsets"]})
            pred.reset()
            seq = []
            for rgb in frames:
                d = pred.predict_depth(Observation(rgb=np.ascontiguousarray(rgb), state=np.zeros(7, np.float32), collided=False))
                seq.append(pool_min(d))
        hist_a = z["hist_action"]
        prev = hist_a[-2] if len(hist_a) >= 2 else np.zeros(4, np.float32)
        cands = z["candidates"]
        M = len(cands)
        d_k, a_k = history_inputs(seq, list(hist_a[:len(seq) - 1]), model.frame_offsets, model.hist_actions)
        d_k = d_k[0] if len(model.frame_offsets) == 1 else d_k
        depth = torch.as_tensor(np.repeat(d_k[None], M, 0), device=dev)
        goal = torch.as_tensor(np.repeat(goal_features(z["state"], z["goal"])[None], M, 0), device=dev)
        prev_t = torch.as_tensor(np.repeat(prev[None], M, 0).astype(np.float32), device=dev)
        ha = torch.as_tensor(np.repeat(a_k[None], M, 0), device=dev) if model.hist_actions else None
        env = None
        if model.env_dim:
            last = decode_frames({"jpeg": z["hist_jpeg"], "jpeg_offsets": z["hist_jpeg_offsets"]})[-1:]
            e, _ = env_pair(pred._model, last)
            env = torch.as_tensor(np.repeat(e[:1].astype(np.float32), M, 0), device=dev)
        with torch.no_grad():
            out = model(depth, goal, prev_t, torch.as_tensor(cands, device=dev), ha, env=env)
            score = score_candidates(out, args.collision_cost_m).cpu().numpy()
            surv = survival(out["hazard_logit"])[..., -1].cpu().numpy()
            pred_prog = (-out["dgoal"][..., -1]).cpu().numpy()
        safe = z["coll_step"] < 0
        prog = np.where(safe, z["dist"][:, 0] - z["dist"][:, -1], np.nan)
        allowed = altitude_allowed(cands, z["state"], z["goal"])
        rows.append({"probe": rec["probe"], "route_id": rec["route_id"], "families": rec["families"],
                     "score": score.round(3).tolist(), "surv": surv.round(4).tolist(),
                     "pred_prog": pred_prog.round(3).tolist(), "safe": safe.tolist(), "allowed": allowed.tolist(),
                     "progress": [None if not s else round(float(p), 2) for s, p in zip(safe, prog)]})

    def summarize(choose, restricted: bool) -> dict:
        """Mixed probes only. ``restricted`` = choose among altitude-allowed candidates."""
        n = top_safe = top_prog = 0
        pairs = correct = 0
        picked: Dict[str, int] = {}
        for r in rows:
            keep = np.asarray(r["allowed"]) if restricted else np.ones(len(r["safe"]), bool)
            safe = np.asarray(r["safe"])
            if safe[keep].all() or (~safe[keep]).all():
                continue
            n += 1
            rank_key = r["surv"] if args.select == "filter" else r["score"]
            score = np.where(keep, np.asarray(rank_key), -np.inf)
            prog = np.asarray([np.nan if p is None else p for p in r["progress"]], dtype=np.float64)
            c = choose(r, score, safe & keep, prog, keep)
            picked[r["families"][c]] = picked.get(r["families"][c], 0) + 1
            top_safe += int(safe[c])
            top_prog += int(safe[c] and prog[c] > 0.5)
            for i in np.flatnonzero(safe & keep):
                for j in np.flatnonzero(~safe & keep):
                    pairs += 1
                    correct += int(score[i] > score[j])
        return {"n_mixed_probes": n,
                "top1_safe": round(top_safe / max(1, n), 3),
                "top1_progress": round(top_prog / max(1, n), 3),
                "pairwise": round(correct / max(1, pairs), 3),
                "picked_families": picked}

    def choose_model(r, s, safe, p, keep) -> int:
        if args.select == "filter":
            return select_candidate(r["surv"], r["pred_prog"], keep, args.surv_min)
        return int(np.argmax(s))

    def evaluate_all(restricted: bool) -> dict:
        rng = np.random.default_rng(0)
        res = {
            "model": summarize(choose_model, restricted),
            "baseline_always_forward": summarize(lambda r, s, safe, p, keep: r["families"].index("fwd"), restricted),
            "baseline_random": summarize(lambda r, s, safe, p, keep: int(rng.choice(np.flatnonzero(keep))), restricted),
            "oracle": summarize(lambda r, s, safe, p, keep: int(np.nanargmax(np.where(safe, p, -1e9))), restricted),
        }
        for k in ("baseline_always_forward", "baseline_random", "oracle"):
            res[k].pop("pairwise")
        return res

    def random_expected(restricted: bool) -> dict:
        """Exact uniform-choice expectation over the same probes (no sampling luck)."""
        n, safe_sum, prog_sum = 0, 0.0, 0.0
        for r in rows:
            keep = np.asarray(r["allowed"]) if restricted else np.ones(len(r["safe"]), bool)
            safe = np.asarray(r["safe"])
            if safe[keep].all() or (~safe[keep]).all():
                continue
            prog = np.asarray([np.nan if p is None else p for p in r["progress"]], dtype=np.float64)
            n += 1
            safe_sum += float(safe[keep].mean())
            prog_sum += float(np.mean(safe[keep] & (np.nan_to_num(prog[keep], nan=-1.0) > 0.5)))
        return {"top1_safe": round(safe_sum / max(1, n), 3), "top1_progress": round(prog_sum / max(1, n), 3)}

    unrestricted = evaluate_all(False)
    restricted = evaluate_all(True)
    unrestricted["random_expected"] = random_expected(False)
    restricted["random_expected"] = random_expected(True)
    model_res = restricted["model"]
    rnd = restricted["random_expected"]
    n_all = len(rows)
    fam_safe = {}
    for r in rows:
        for f, s in zip(r["families"], r["safe"]):
            fam_safe.setdefault(f, []).append(bool(s))
    res = {
        "n_probes": n_all,
        "rule": "hold altitude: no net climb unless goal >1 m above, no net descent unless >1 m below (|net dz|<=0.5 m)",
        "restricted": restricted,
        "unrestricted": unrestricted,
        "gt_safe_rate_by_family": {f: round(float(np.mean(v)), 3) for f, v in fam_safe.items()},
        "gate": "restricted: pairwise>=0.7, top1_safe>=0.7, top1_safe>=random, top1_progress>=random+0.15",
        "gate_checks": {
            "pairwise>=0.7": bool(model_res["pairwise"] >= 0.7),
            "top1_safe>=0.7": bool(model_res["top1_safe"] >= 0.7),
            "top1_safe>=random": bool(model_res["top1_safe"] >= rnd["top1_safe"]),
            "top1_progress>=random+0.15": bool(model_res["top1_progress"] >= rnd["top1_progress"] + 0.15),
        },
        "gate_pass": bool(
            model_res["pairwise"] >= 0.7
            and model_res["top1_safe"] >= 0.7
            and model_res["top1_safe"] >= rnd["top1_safe"]
            and model_res["top1_progress"] >= rnd["top1_progress"] + 0.15
        ),
        "ckpt": args.ckpt,
        "select": args.select if args.select == "score" else f"filter(surv>={args.surv_min})",
        "rows": rows,
    }
    text = json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1)
    print(text)
    out = Path(args.out) if args.out else probe_dir / f"GATE_{Path(args.ckpt).parent.name}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0 if res["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
