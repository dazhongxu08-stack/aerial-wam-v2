#!/usr/bin/env python3
"""Counterfactual probe set on held-out routes (ground truth for the Stage 2 gate).

For each probe: teleport to a random point on a held-out route with a random
heading, fly forward until a facade sits 3-8 m ahead, inside the ~9 m a
10-step straight press covers (so the depth head's
frame history is real flight), then from that exact pose execute every
candidate action sequence in the simulator, re-placing the drone before each.
Outcomes per candidate: collision step, distance to goal and GT cones after
each step. Shield off. Candidates come from the same random macro families
as the collector (fixed family list, random parameters and signs).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("probe_counterfactual")

FAMILIES = (
    "fwd", "yaw_fwd", "yaw_fwd", "yaw_fwd", "yaw_fwd", "goal_turn_fwd",
    "strafe", "strafe", "climb", "smooth_random", "yaw_only",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation", default="experiments/aerial/phase3_unified/annotations/outdoor_complex_od_holdout.json")
    ap.add_argument("--split", default="holdout", choices=("holdout", "train"),
                    help="holdout = gate ground truth; train = branching training data")
    ap.add_argument("--config", default="configs/_tmp_feasible_side_h125.yaml")
    ap.add_argument("--out-dir", default="experiments/aerial/rl/artifacts/consequence_probe_holdout_20261004")
    ap.add_argument("--n-probes", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--spawn-z", type=float, default=24.0)
    ap.add_argument("--facade-max-m", type=float, default=8.0)
    ap.add_argument("--facade-min-m", type=float, default=3.0)
    ap.add_argument("--approach-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fwd-test", default=None, choices=("box", "corridor"),
                    help="facade test; default box for holdout, corridor for train")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import yaml

    from experiments.aerial.rl.collect_consequence_dataset import (
        CONES, EpisodeBuffer, encode_jpeg, goal_bearing_body, pool_min, sample_macro,
    )
    from experiments.aerial.rl.depth_geometry import cone_clearances
    from experiments.aerial.rl.train_rl import _build_env

    routes = json.loads((root / args.annotation).read_text(encoding="utf-8"))["episodes"]
    if any(r.get("complexity_meta", {}).get("split") != args.split for r in routes):
        raise SystemExit(f"refusing: annotation has routes outside split={args.split}")
    cfg = yaml.safe_load((root / args.config).read_text()) or {}
    env_cfg = dict(cfg.get("env") or {})
    env_cfg.update({"backend": "airsim", "step_hz": 5.0, "grab_depth": True})
    env = _build_env(env_cfg)
    out = root / args.out_dir
    (out / "probes").mkdir(parents=True, exist_ok=True)
    review_path = out / "AUTO_REVIEW.json"
    review: Dict[str, Any] = (json.loads(review_path.read_text()) if review_path.is_file()
                              else {"probes": [], "attempts": 0, "families": list(FAMILIES), "args": vars(args)})
    rng = np.random.default_rng(args.seed + review["attempts"])
    H = int(args.horizon)

    def fwd_box(obs: Any) -> float:
        return float(cone_clearances(np.asarray(obs.depth, np.float32))["forward"])

    def fwd_corridor(obs: Any, r: float = 1.5) -> float:
        d = np.asarray(obs.depth, np.float64)
        h, w = d.shape
        f = (w / 2.0) / math.tan(math.radians(45.0))
        lat = np.abs((np.arange(w) + 0.5 - w / 2.0) / f)[None, :] * d
        ver = np.abs((np.arange(h) + 0.5 - h / 2.0) / f)[:, None] * d
        m = np.isfinite(d) & (d > 0.05) & (lat <= r) & (ver <= r)
        return float(d[m].min()) if m.any() else float("inf")

    # Held-out probes keep the original box test so the gate set stays comparable.
    # Train branching uses the flight corridor: the box often sees a facade off the path.
    fwd_test = args.fwd_test or ("box" if args.split == "holdout" else "corridor")
    fwd_cone = fwd_box if fwd_test == "box" else fwd_corridor

    while len(review["probes"]) < int(args.n_probes):
        review["attempts"] += 1
        r = routes[int(rng.integers(0, len(routes)))]
        pos = np.asarray(r["pos"], dtype=np.float64)
        goal = pos[-1].copy()
        i0 = int(rng.integers(0, max(1, len(pos) - 2)))
        p0 = pos[i0].copy()
        p0[2] = float(args.spawn_z)
        yaw0 = float(rng.uniform(-math.pi, math.pi))
        try:
            obs = env.reset({"pos": [p0.tolist(), goal.tolist()], "yaw": [yaw0, yaw0]})
        except RuntimeError as exc:
            logger.warning("spawn reset failed, skipping: %s", exc)
            continue
        if bool(obs.collided) or fwd_cone(obs) < float(args.facade_min_m):
            continue
        hist = EpisodeBuffer()
        hist.add(obs, np.zeros(4), -1, -1)
        ok = False
        for _ in range(int(args.approach_steps)):
            if float(args.facade_min_m) <= fwd_cone(obs) <= float(args.facade_max_m):
                ok = True
                break
            a = np.array([float(rng.uniform(0.6, 1.0)), 0.0, 0.0, 0.0])
            hist.action[-1] = a.astype(np.float32)
            obs, _ = env.step(a)
            hist.add(obs, np.zeros(4), -1, -1)
            if bool(obs.collided):
                break
        if not ok or bool(obs.collided):
            continue
        s0 = np.asarray(obs.state, dtype=np.float64)
        pose = {"pos": [s0[:3].tolist(), goal.tolist()], "yaw": [float(s0[6]), float(s0[6])]}
        bearing = goal_bearing_body(s0[:3], float(s0[6]), goal)
        cands, names = [], []
        for fam in FAMILIES:
            name, seq = sample_macro(rng, H, bearing, name=fam)
            if fam == "fwd":
                seq[:, 0] = 0.8
            cands.append(seq)
            names.append(name)
        cands = np.stack(cands).astype(np.float32)
        M = len(cands)
        coll_step = np.full(M, -1, np.int32)
        dist = np.full((M, H + 1), np.nan, np.float32)
        cones = np.full((M, H + 1, 5), np.nan, np.float32)
        replace_err = np.zeros(M, np.float32)
        step_depth = np.full((M, H + 1, 56, 56), np.nan, np.float16)
        failed = False
        for m in range(M):
            try:
                o = env.reset(pose)
            except RuntimeError as exc:
                logger.warning("re-place failed on candidate %d, dropping probe: %s", m, exc)
                failed = True
                break
            replace_err[m] = float(np.linalg.norm(np.asarray(o.position) - s0[:3]))
            dist[m, 0] = float(np.linalg.norm(np.asarray(o.position) - goal))
            cc = cone_clearances(np.asarray(o.depth, np.float32))
            cones[m, 0] = [cc[k] for k in CONES]
            step_depth[m, 0] = pool_min(np.asarray(o.depth, np.float32))
            for k in range(H):
                o, _ = env.step(cands[m, k])
                dist[m, k + 1] = float(np.linalg.norm(np.asarray(o.position) - goal))
                cc = cone_clearances(np.asarray(o.depth, np.float32))
                cones[m, k + 1] = [cc[c] for c in CONES]
                step_depth[m, k + 1] = pool_min(np.asarray(o.depth, np.float32))
                if bool(o.collided):
                    coll_step[m] = k
                    break
        if failed:
            review.setdefault("dropped", []).append({"attempt": review["attempts"], "route_id": r["route_id"], "reason": "replace_failed"})
            review_path.write_text(json.dumps(review, indent=1), encoding="utf-8")
            continue
        idx = len(review["probes"])
        fname = f"probe_{idx:04d}.npz"
        offsets = np.cumsum([0] + [len(b) for b in hist.jpeg]).astype(np.int64)
        np.savez(
            out / "probes" / fname,
            hist_jpeg=np.frombuffer(b"".join(hist.jpeg), dtype=np.uint8),
            hist_jpeg_offsets=offsets,
            hist_action=np.stack(hist.action),
            gt_depth56=pool_min(np.asarray(obs.depth, np.float32)),
            state=s0.astype(np.float32),
            goal=goal.astype(np.float32),
            candidates=cands,
            coll_step=coll_step,
            dist=dist,
            cones=cones,
            step_depth56=step_depth,
            replace_err=replace_err,
        )
        rec = {
            "probe": idx, "file": fname, "route_id": r["route_id"],
            "fwd_cone_m": round(fwd_cone(obs), 2),
            "families": names,
            "collided": [int(c >= 0) for c in coll_step],
            "progress_m": [None if coll_step[m] >= 0 else round(float(dist[m, 0] - dist[m, H]), 2)
                           for m in range(M)],
            "max_replace_err_m": round(float(replace_err.max()), 2),
        }
        review["probes"].append(rec)
        review_path.write_text(json.dumps(review, indent=1), encoding="utf-8")
        logger.info("probe %d %s fwd=%.1f collided=%s replace_err=%.2f",
                    idx, r["route_id"], rec["fwd_cone_m"], rec["collided"], rec["max_replace_err_m"])
    logger.info("DONE probes=%d attempts=%d", len(review["probes"]), review["attempts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
