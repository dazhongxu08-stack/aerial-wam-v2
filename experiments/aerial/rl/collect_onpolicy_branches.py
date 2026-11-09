#!/usr/bin/env python3
"""Branching probes on states the current MPC actually visits (train routes, shield off).

Fly the consequence-model MPC (same candidates, ensemble and selection rule as
``mpc_consequence_eval``) on a train route. Branch states: the states 3/6/10/15
steps before a collision, plus one random state per flight. From each, re-place
the drone and roll out every candidate family in the simulator, exactly like
``probe_counterfactual_airsim``; output files use the same format so
``train_consequence --branch-dirs`` reads them unchanged.

``--branch-mode near_goal`` starts on a ring around the goal instead, keeps flying
after arrival, and branches only states near the goal. Goals inside a collision
volume (fresh contact timestamp at the goal), near held-out goals, or outside the
inland interior are dropped before any flight; tight and open goals are balanced.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("onpolicy_branches")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation", default="experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json")
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--depth-ckpt", default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt")
    ap.add_argument("--config", default="configs/_tmp_feasible_side_h125.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-states", type=int, default=150)
    ap.add_argument("--before-collision", default="3,6,10,15")
    ap.add_argument("--hist-frames", type=int, default=6)
    ap.add_argument("--surv-min", type=float, default=0.7)
    ap.add_argument("--replan-every", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--branch-mode", default="collision", choices=("collision", "lowclear", "fail", "near_goal", "side", "side_clear", "side_arc", "side_open"),
                    help="lowclear = branch only where the bottom third of the GT depth has an obstacle closer than --lowclear-m; "
                         "fail = before collision / during stuck, plus one low-clearance and one random state; "
                         "near_goal = spawn on a ring around the goal, fly past arrival, branch states within --near-pick-m; "
                         "side = on the depth-network image, one side is within 4 m, the other side is beyond 8 m, "
                         "and the inner center is beyond 8 m; kept only when the current ensemble still "
                         "scores the forward branch at or above --surv-min, the forward branch contacts, "
                         "and a yaw branch does not; "
                         "side_clear = the same picture and the same still-high score, kept only when the "
                         "forward branch does not contact; "
                         "side_arc = the same picture and the same still-high score, but the rolled command "
                         "is the one the flight scores: forward 0.6 m with yaw at most 0.10, held for the "
                         "horizon. Contacts and non-contacts are both kept, --n-states of each; "
                         "side_open = the same side picture, including frames whose forward score "
                         "is already below --surv-min. Kept only when a 0.6 m yaw toward the open "
                         "side contacts and an in-place yaw on that same picture does not")
    ap.add_argument("--exclude-routes", default="od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21",
                    help="train route ids skipped so they stay available for validation")
    ap.add_argument("--near-r", default="3,12", help="near_goal: spawn ring radius range around the goal (m)")
    ap.add_argument("--near-pick-m", type=float, default=8.0, help="near_goal: branch only states this close to the goal")
    ap.add_argument("--near-per-flight", type=int, default=3)
    ap.add_argument("--near-tight-m", type=float, default=5.0,
                    help="near_goal: a goal is tight when GT depth at the goal is below this in any of 4 headings")
    ap.add_argument("--exclude-goals", default="",
                    help="near_goal: comma list of annotations whose goal surroundings are held out")
    ap.add_argument("--exclude-goal-m", type=float, default=30.0)
    ap.add_argument("--water-raster", default="", help="near_goal: measured water npz for the inland check")
    ap.add_argument("--min-inland-m", type=float, default=100.0)
    ap.add_argument("--stuck-s", type=float, default=0.0,
                    help="end the flight as stuck after this many seconds without 1 m gain (0 = off)")
    ap.add_argument("--lowclear-m", type=float, default=2.0)
    ap.add_argument("--lowclear-per-flight", type=int, default=3)
    ap.add_argument("--fail-lowclear-n", type=int, default=1, help="low-clearance states per flight in fail mode")
    ap.add_argument("--lowclear-region", default="bottom", choices=("bottom", "any"),
                    help="bottom = bottom third of the GT depth; any = whole frame (side walls, roof edges)")
    ap.add_argument("--alt-fix", action="store_true",
                    help="same flight rule as mpc_consequence_eval --alt-fix")
    ap.add_argument("--alt-band", action="store_true", help="same flight rule as mpc_consequence_eval --alt-band")
    ap.add_argument("--branch-extra-families", default="",
                    help="comma list from EXTRA_MACROS added to the branched candidates (labels only)")
    ap.add_argument("--choose-extra-families", default="",
                    help="comma list from EXTRA_MACROS added to the flying policy's candidates")
    ap.add_argument("--alt-band-return", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    import yaml

    from experiments.aerial.rl.collect_consequence_dataset import (
        CONES, EXTRA_MACROS, encode_jpeg, goal_bearing_body, pool_min, sample_macro, wrap,
    )
    from experiments.aerial.rl.consequence_model import (
        altitude_allowed, altitude_band_allowed, depth_input, goal_features, load_checkpoint,
        select_candidate, survival,
    )
    from experiments.aerial.rl.depth_geometry import cone_clearances
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.mpc_consequence_eval import smooth_turn
    from experiments.aerial.rl.probe_counterfactual_airsim import FAMILIES
    from experiments.aerial.rl.train_rl import _build_env

    def extra(spec: str) -> tuple:
        names = tuple(s for s in spec.split(",") if s)
        bad = [s for s in names if s not in EXTRA_MACROS]
        if bad:
            raise SystemExit(f"unknown extra families {bad}; choose from {EXTRA_MACROS}")
        return names

    choose_fams = tuple(FAMILIES) + extra(args.choose_extra_families)
    branch_fams = tuple(FAMILIES) + extra(args.branch_extra_families)
    dev = torch.device(args.device)
    models = [load_checkpoint(root / p, device=args.device) for p in args.ckpts.split(",") if p]
    for m in models:
        m.eval()
    H = models[0].horizon
    pred = DepthMinPredictor.from_checkpoint(root / args.depth_ckpt, device=args.device)
    offsets = [int(x) for x in args.before_collision.split(",") if x]

    routes = json.loads((root / args.annotation).read_text(encoding="utf-8"))["episodes"]
    if any(r.get("complexity_meta", {}).get("split") != "train" for r in routes):
        raise SystemExit("refusing: on-policy branching uses train routes only")
    excluded = {s for s in args.exclude_routes.split(",") if s}
    routes = [r for r in routes if r.get("route_id") not in excluded]
    cfg = yaml.safe_load((root / args.config).read_text()) or {}
    env_cfg = dict(cfg.get("env") or {})
    env_cfg.update({"backend": "airsim", "step_hz": 5.0, "grab_depth": True})
    env = _build_env(env_cfg)

    out = root / args.out_dir
    (out / "probes").mkdir(parents=True, exist_ok=True)
    review_path = out / "AUTO_REVIEW.json"
    review: Dict[str, Any] = (json.loads(review_path.read_text()) if review_path.is_file()
                              else {"probes": [], "flights": [], "attempts": 0, "families": list(branch_fams),
                                    "args": vars(args)})
    rng = np.random.default_rng(args.seed + review["attempts"])

    def da3_side_corner(depth56: np.ndarray) -> bool:
        """One side is near, the other is far, and the inner center stays far.

        The minimum of the whole center third is pulled down by a few corner pixels,
        so the inner half of that third is what must stay beyond 8 m.
        """
        d = np.asarray(depth56, np.float32)
        hh, ww = d.shape
        rows = d[hh // 4: hh - hh // 4]

        def _near(block: np.ndarray) -> float:
            vals = block[np.isfinite(block) & (block > 0.05)]
            return float(vals.min()) if vals.size else 100.0

        left = _near(rows[:, : ww // 3])
        right = _near(rows[:, ww - ww // 3:])
        c0, c1 = ww // 3, ww - ww // 3
        span = c1 - c0
        inner = _near(rows[:, c0 + span // 4: c1 - span // 4])
        one_side = (left <= 4.0 and right >= 8.0) or (right <= 4.0 and left >= 8.0)
        return bool(one_side and inner >= 8.0)

    def side_measures(depth56: np.ndarray) -> tuple:
        """Left, whole-center, right, and inner-center minima on the middle rows."""
        d = np.asarray(depth56, np.float32)
        hh, ww = d.shape
        rows = d[hh // 4: hh - hh // 4]

        def _near(block: np.ndarray) -> float:
            vals = block[np.isfinite(block) & (block > 0.05)]
            return float(vals.min()) if vals.size else 100.0

        left = _near(rows[:, : ww // 3])
        right = _near(rows[:, ww - ww // 3:])
        c0, c1 = ww // 3, ww - ww // 3
        span = c1 - c0
        inner = _near(rows[:, c0 + span // 4: c1 - span // 4])
        center = _near(rows[:, c0:c1])
        return left, center, right, inner

    def open_yaw_sign(depth56: np.ndarray) -> float | None:
        """Positive yaw turns toward image right. +1 means the open side is to the right.

        The wall also has to reach the center third. Step 13 was 2 m there, while a
        center of 8 m is a different picture.
        """
        left, center, right, inner = side_measures(depth56)
        if center > 4.0 or inner < 8.0:
            return None
        if left <= 4.0 and right >= 8.0:
            return 1.0
        if right <= 4.0 and left >= 8.0:
            return -1.0
        return None

    def fwd_ensemble(depth56: np.ndarray, state: np.ndarray, goal: np.ndarray, prev: np.ndarray,
                     action: np.ndarray | None = None) -> float:
        """Mean forward-macro survival of the loaded ensemble."""
        if action is None:
            action = np.zeros((H, 4), np.float32)
            action[:, 0] = 0.8
        else:
            action = np.asarray(action, np.float32)[:H]
        d = torch.as_tensor(depth_input(np.asarray(depth56, np.float32))[None], device=dev)
        g = torch.as_tensor(goal_features(np.asarray(state, np.float64), np.asarray(goal, np.float64))[None], device=dev)
        pa = torch.as_tensor(np.asarray(prev, np.float32)[None], device=dev)
        a = torch.as_tensor(action[None], device=dev)
        scores = []
        with torch.no_grad():
            for m in models:
                out = m(d, g, pa, a)
                scores.append(float(survival(out["hazard_logit"])[..., -1].detach().cpu()))
        return float(np.mean(scores))

    if args.branch_mode == "side":
        from experiments.aerial.rl.env.obs import Observation
        from experiments.aerial.rl.precompute_da3_consequence import decode_frames

        yaw_names = {"yaw_fwd", "goal_turn_fwd", "yaw_only"}
        for rec in review["probes"]:
            if rec.get("side_rule") == "da3_still_high":
                continue
            fams = rec.get("families") or review.get("families") or []
            collided = rec.get("collided") or []
            fwd_hit = bool(collided and collided[0])
            yaw_safe = any(i < len(collided) and fams[i] in yaw_names and not collided[i] for i in range(len(fams)))
            z = np.load(out / "probes" / rec["file"])
            if rec.get("side_rule") != "da3_inner":
                frames = decode_frames({"jpeg": z["hist_jpeg"], "jpeg_offsets": z["hist_jpeg_offsets"]})
                pred.reset()
                seen = pred.predict_depth(Observation(
                    rgb=np.ascontiguousarray(frames[-1]), state=np.zeros(7, np.float32), collided=False))
                geom = seen is not None and da3_side_corner(pool_min(np.asarray(seen, np.float32)))
                rec["da3_side"] = bool(geom)
            else:
                geom = bool(rec.get("da3_side"))
            surv = 0.0
            if geom and fwd_hit and yaw_safe:
                hist = np.asarray(z["hist_action"], np.float32)
                prev = hist[-2] if len(hist) >= 2 else np.zeros(4, np.float32)
                frames = decode_frames({"jpeg": z["hist_jpeg"], "jpeg_offsets": z["hist_jpeg_offsets"]})
                pred.reset()
                seen = pred.predict_depth(Observation(
                    rgb=np.ascontiguousarray(frames[-1]), state=np.zeros(7, np.float32), collided=False))
                if seen is not None:
                    surv = fwd_ensemble(pool_min(np.asarray(seen, np.float32)), z["state"], z["goal"], prev,
                                        np.asarray(z["candidates"][0]))
            rec["fwd_hit"] = fwd_hit
            rec["fwd_surv"] = round(surv, 4)
            rec["train_keep"] = bool(geom and fwd_hit and yaw_safe and surv >= float(args.surv_min))
            rec["side_rule"] = "da3_still_high"
        review_path.write_text(json.dumps(review, indent=1), encoding="utf-8")
        logger.info("side keep %d / %d probes on disk",
                    sum(1 for p in review["probes"] if p.get("train_keep")), len(review["probes"]))

    def arc_counts() -> tuple:
        hit = clear = 0
        for p in review["probes"]:
            if not p.get("train_keep"):
                continue
            if p.get("arc_hit"):
                hit += 1
            else:
                clear += 1
        return hit, clear

    def n_kept() -> int:
        if args.branch_mode == "side_arc":
            hit, clear = arc_counts()
            return min(hit, clear)
        if args.branch_mode in ("side", "side_clear", "side_open"):
            return sum(1 for p in review["probes"] if p.get("train_keep"))
        return len(review["probes"])

    band = [0.0, 0.0]

    def fresh_contact(ts0) -> str:
        ci = getattr(env, "collision_info", lambda: {})()
        return str(ci.get("object_name", "")) if ci.get("has_collided") and ci.get("time_stamp") != ts0 else ""

    def stamp():
        return getattr(env, "collision_info", lambda: {})().get("time_stamp")

    near = args.branch_mode == "near_goal"
    if near:
        from experiments.aerial.phase3_unified.region_geometry import classify_spawn_xy, load_regions, path_inland_metrics

        regions = load_regions()
        shore = None
        if args.water_raster:
            sys.path.insert(0, str(root / "experiments" / "aerial" / "scripts" / "consequence"))
            from water_audit import Shore
            shore = Shore(str(root / args.water_raster))
        held = [np.asarray(e["pos"][-1][:2], np.float64) for a in args.exclude_goals.split(",") if a
                for e in json.loads((root / a).read_text(encoding="utf-8"))["episodes"]]
        r_lo, r_hi = (float(x) for x in args.near_r.split(","))

        def inland_ok(x: float, y: float) -> bool:
            s = classify_spawn_xy(float(x), float(y), regions)
            return bool(s["in_urban"] and s["in_interior"] and not s["in_water"] and s["spawn_inland_m"] >= 100.0
                        and (shore is None or shore(float(x), float(y)) >= float(args.min_inland_m)))

        def goal_check(g: np.ndarray) -> Dict[str, Any]:
            mins = []
            for k in range(4):
                yaw = wrap(k * math.pi / 2)
                ts0 = stamp()
                try:
                    o = env.reset({"pos": [g.tolist(), g.tolist()], "yaw": [yaw, yaw]})
                    o, _ = env.step(np.zeros(4))
                except RuntimeError as exc:
                    return {"ok": False, "reason": f"reset_failed {exc}"}
                hit = fresh_contact(ts0)
                if hit or bool(o.collided):
                    return {"ok": False, "reason": f"goal_contact {hit}"}
                if float(np.linalg.norm(np.asarray(o.position) - g)) > 1.0:
                    return {"ok": False, "reason": "goal_nudged"}
                mins.append(float(np.asarray(o.depth, np.float32).min()))
            return {"ok": True, "class": "tight" if min(mins) < float(args.near_tight_m) else "open",
                    "depth_min_m": [round(m, 2) for m in mins]}

        goals = review.setdefault("goals", {})
        for r in routes:
            rid = r["route_id"]
            if rid in goals:
                continue
            g = np.asarray(r["pos"][-1], np.float64)
            if any(float(np.linalg.norm(g[:2] - h)) < float(args.exclude_goal_m) for h in held):
                goals[rid] = {"ok": False, "reason": "near_held_out_goal"}
            elif not inland_ok(g[0], g[1]):
                goals[rid] = {"ok": False, "reason": "goal_geography"}
            else:
                goals[rid] = goal_check(g)
            logger.info("goal %s %s", rid, goals[rid])
            review_path.write_text(json.dumps(review, indent=1), encoding="utf-8")
        usable = [r for r in routes if goals[r["route_id"]].get("ok")]
        if not usable:
            raise SystemExit("no usable near-goal routes")

    def low_clear(gt56: np.ndarray) -> float:
        return float(gt56[38:].min() if args.lowclear_region == "bottom" else gt56.min())

    def choose(depth, state, goal, prev):
        bearing = goal_bearing_body(state[:3], float(state[6]), goal)
        cands, names = [], []
        for fam in choose_fams:
            name, seq = sample_macro(rng, H, bearing, name=fam)
            if fam == "fwd":
                seq[:, 0] = 0.8
            cands.append(seq)
            names.append(name)
        if args.alt_fix and state[2] > goal[2] + 1.0:
            name, seq = sample_macro(rng, H, bearing, name="descend")
            cands.append(seq)
            names.append(name)
        cands = np.stack(cands)
        M = len(cands)
        allowed = altitude_allowed(cands, state, goal)
        if args.alt_fix:
            allowed &= altitude_allowed(cands[:, :int(args.replan_every)], state, goal)
        if args.alt_band:
            banded = allowed & altitude_band_allowed(
                cands, state[2], band[0], band[1], int(args.replan_every), args.alt_band_return)
            allowed = banded if banded.any() else allowed
        d = torch.as_tensor(np.repeat(depth_input(pool_min(np.asarray(depth, np.float32)))[None], M, 0), device=dev)
        g = torch.as_tensor(np.repeat(goal_features(state, goal)[None], M, 0), device=dev)
        pa = torch.as_tensor(np.repeat(prev[None].astype(np.float32), M, 0), device=dev)
        a = torch.as_tensor(cands.astype(np.float32), device=dev)
        s_all, p_all = [], []
        with torch.no_grad():
            for m in models:
                o = m(d, g, pa, a)
                s_all.append(survival(o["hazard_logit"])[..., -1].cpu().numpy())
                p_all.append((-o["dgoal"][..., -1]).cpu().numpy())
        c = select_candidate(np.mean(s_all, 0), np.mean(p_all, 0), allowed, args.surv_min)
        return cands[c], names[c]

    def branch(rec_t: Dict[str, Any], goal: np.ndarray) -> Dict[str, Any] | None:
        s0 = rec_t["state"]
        pose = {"pos": [s0[:3].tolist(), goal.tolist()], "yaw": [float(s0[6]), float(s0[6])]}
        bearing = goal_bearing_body(s0[:3], float(s0[6]), goal)
        cands, names = [], []
        if args.branch_mode == "side_arc":
            # The flight rewrites a yaw-then-forward macro so the executed step
            # already moves 0.6 m while yawing at most 0.10, then scores that
            # command held for the horizon. Label that command, not the macro.
            for _ in range(4):
                _, seq = sample_macro(rng, H, bearing, name="yaw_fwd")
                seq = smooth_turn(seq, 0.6, 0.10)
                held = np.repeat(seq[:1], H, axis=0)
                if float(held[0, 0]) < 0.6 - 1e-6 or abs(float(held[0, 3])) > 0.10 + 1e-6:
                    raise RuntimeError(f"arc command left the flight box: {held[0].tolist()}")
                cands.append(held)
                names.append("arc_hold")
        elif args.branch_mode == "side_open":
            # Label the command that still hits: 0.6 m toward the open side.
            # An in-place yaw on the same picture has to stay clear, so the
            # pair distinguishes that yaw from a turn into the near wall.
            sign = float(rec_t.get("open_sign") or 0.0)
            if sign == 0.0:
                return {"skip": "no_open_side"}
            for _ in range(3):
                # The flown step was about 0.07, not the 0.10 clip.
                held = np.zeros((H, 4), np.float64)
                held[:, 0] = 0.6
                held[:, 3] = float(rng.uniform(0.04, 0.09)) * sign
                if float(held[0, 3]) * sign <= 0.0:
                    return {"skip": "open_yaw_sign"}
                cands.append(held)
                names.append("open_hold")
            for _ in range(2):
                _, seq = sample_macro(rng, H, bearing, name="yaw_only")
                held = np.repeat(seq[:1], H, axis=0)
                if float(held[0, 0]) >= 0.6:
                    return {"skip": "yaw_only_moved"}
                cands.append(held)
                names.append("yaw_only")
        else:
            for fam in branch_fams:
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
        coll_obj = [""] * M
        for m in range(M):
            try:
                o = env.reset(pose)
            except RuntimeError as exc:
                logger.warning("re-place failed: %s", exc)
                return None
            replace_err[m] = float(np.linalg.norm(np.asarray(o.position) - s0[:3]))
            if bool(o.collided):
                return None
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
                    coll_obj[m] = getattr(env, "collision_info", lambda: {})().get("object_name", "")
                    break
            if args.branch_mode == "side" and m == 0 and coll_step[0] < 0:
                return {"skip": "fwd_clear"}
            if args.branch_mode == "side_clear" and m == 0 and coll_step[0] >= 0:
                return {"skip": "fwd_hit"}
        if args.branch_mode == "side":
            yaw_names = {"yaw_fwd", "goal_turn_fwd", "yaw_only"}
            yaw_safe = any(names[i] in yaw_names and coll_step[i] < 0 for i in range(M))
            if not yaw_safe:
                return {"skip": "no_safe_yaw"}
        if args.branch_mode == "side_open":
            open_hit = [i for i, name in enumerate(names) if name == "open_hold" and coll_step[i] >= 0]
            yaw_safe = [i for i, name in enumerate(names) if name == "yaw_only" and coll_step[i] < 0]
            if not open_hit:
                return {"skip": "open_clear"}
            if not yaw_safe:
                return {"skip": "yaw_hit"}
            keep = open_hit + yaw_safe
            cands = cands[keep]
            names = [names[i] for i in keep]
            coll_step = coll_step[keep]
            dist = dist[keep]
            cones = cones[keep]
            replace_err = replace_err[keep]
            step_depth = step_depth[keep]
            coll_obj = [coll_obj[i] for i in keep]
        return {"cands": cands, "names": names, "coll_step": coll_step, "dist": dist, "cones": cones,
                "replace_err": replace_err, "step_depth": step_depth, "coll_obj": coll_obj}

    stop_file = Path("/tmp/arc06_one_usable")

    def one_usable_stop() -> bool:
        return args.branch_mode == "side_arc" and stop_file.is_file()

    while n_kept() < int(args.n_states) and not one_usable_stop():
        review["attempts"] += 1
        if near:
            have = {"tight": 0, "open": 0}
            for p in review["probes"]:
                if p.get("goal_class") in have:
                    have[p["goal_class"]] += 1
            want = min(have, key=have.get)
            pool = [r for r in usable if goals[r["route_id"]]["class"] == want] or usable
            r = pool[int(rng.integers(0, len(pool)))]
            goal = np.asarray(r["pos"][-1], dtype=np.float64)
            ang = float(rng.uniform(-math.pi, math.pi))
            rad = float(rng.uniform(r_lo, r_hi))
            start = goal + np.array([rad * math.cos(ang), rad * math.sin(ang), 0.0])
            if not inland_ok(start[0], start[1]):
                review["flights"].append({"attempt": review["attempts"], "route_id": r["route_id"], "end": "spawn_geography"})
                continue
            yaw0 = wrap(ang + math.pi + float(rng.uniform(-math.pi / 4, math.pi / 4)))
            ts0 = stamp()
            try:
                obs = env.reset({"pos": [start.tolist(), goal.tolist()], "yaw": [yaw0, yaw0]})
            except RuntimeError as exc:
                logger.warning("spawn reset failed: %s", exc)
                continue
            if bool(obs.collided) or fresh_contact(ts0) or float(np.linalg.norm(np.asarray(obs.position) - start)) > 1.0:
                review["flights"].append({"attempt": review["attempts"], "route_id": r["route_id"], "end": "spawn_contact"})
                continue
        else:
            r = routes[int(rng.integers(0, len(routes)))]
            pos = np.asarray(r["pos"], dtype=np.float64)
            goal = pos[-1].copy()
            i0 = int(rng.integers(0, max(1, len(pos) // 2)))
            yaw0 = wrap(float(r["yaw"][0]) + float(rng.uniform(-math.pi / 4, math.pi / 4)))
            try:
                obs = env.reset({"pos": [pos[i0].tolist(), goal.tolist()], "yaw": [yaw0, yaw0]})
            except RuntimeError as exc:
                logger.warning("spawn reset failed: %s", exc)
                continue
            if bool(obs.collided):
                continue
        z_ref = float(np.asarray(obs.position)[2])
        band[:] = [min(z_ref, float(goal[2])) - 1.0, max(z_ref, float(goal[2])) + 1.0]
        pred.reset()
        depth = pred.predict_depth(obs)
        prev = np.zeros(4, np.float32)
        plan: List[np.ndarray] = []
        def _da356(depth_map: Any) -> np.ndarray | None:
            if depth_map is None:
                return None
            return pool_min(np.asarray(depth_map, np.float32))

        log: List[Dict[str, Any]] = [{"jpeg": encode_jpeg(np.asarray(obs.rgb)),
                                      "gt56": pool_min(np.asarray(obs.depth, np.float32)),
                                      "da356": _da356(depth),
                                      "state": np.asarray(obs.state, np.float64), "action": None}]
        end = "max_steps"
        flight_obj = ""
        best_d, best_step = float(np.linalg.norm(np.asarray(obs.position) - goal)), 0
        try:
            for step in range(int(args.max_steps)):
                if not plan or step % int(args.replan_every) == 0:
                    seq, _ = choose(depth, np.asarray(obs.state, np.float64), goal, prev)
                    plan = [np.asarray(a, np.float64) for a in seq]
                a = plan.pop(0)
                log[-1]["action"] = a.astype(np.float32)
                obs, _ = env.step(a)
                prev = a.astype(np.float32)
                depth = pred.predict_depth(obs)
                log.append({"jpeg": encode_jpeg(np.asarray(obs.rgb)),
                            "gt56": pool_min(np.asarray(obs.depth, np.float32)),
                            "da356": _da356(depth),
                            "state": np.asarray(obs.state, np.float64), "action": None})
                if bool(obs.collided):
                    end = "collision"
                    flight_obj = getattr(env, "collision_info", lambda: {})().get("object_name", "")
                    break
                d_now = float(np.linalg.norm(np.asarray(obs.position) - goal))
                if d_now <= 3.0 and not near:
                    end = "arrived"
                    break
                if d_now < best_d - 1.0:
                    best_d, best_step = d_now, step
                elif args.stuck_s > 0 and (step - best_step) / 5.0 >= float(args.stuck_s):
                    end = "stuck"
                    break
        except Exception as exc:  # AirSim occasionally returns an empty frame
            logger.warning("flight aborted on sim error: %s", exc)
            review["flights"].append({"attempt": review["attempts"], "route_id": r["route_id"], "end": "sim_error"})
            review_path.write_text(json.dumps(review, indent=1), encoding="utf-8")
            continue
        T = len(log) - 1
        picks: List[tuple] = []
        high_surv: Dict[int, float] = {}
        d_t = [float(np.linalg.norm(x["state"][:3] - goal)) for x in log]
        geo = ""
        if near:
            xyz = [x["state"][:3].tolist() for x in log]
            if (path_inland_metrics(xyz, regions)["path_min_inland_m"] < 80.0
                    or (shore is not None and min(shore(p[0], p[1]) for p in xyz) < 80.0)):
                geo = "path_geography"
        if near:
            pick_m = float(args.near_pick_m)
            if end == "collision" and not geo:
                picks += [(T - k, f"near_before_collision_{k}") for k in offsets if T - k >= 1 and d_t[T - k] <= pick_m]
            close = [] if geo else [t for t in range(1, T) if d_t[t] <= pick_m]
            for t in rng.permutation(close).tolist():
                if len(picks) >= int(args.near_per_flight):
                    break
                if all(abs(t - p[0]) >= 3 for p in picks):
                    picks.append((int(t), "near"))
            picks.sort()
        elif args.branch_mode in ("side", "side_clear", "side_arc"):
            # Depth-network image: one side near, the other far, inner center far.
            side = []
            for t in range(1, T):
                seen = log[t].get("da356")
                if seen is not None and da3_side_corner(seen):
                    side.append(t)
            logger.info("side frames %d / %d on %s", len(side), max(T - 1, 0), r["route_id"])
            # Roll every frame the ensemble still calls safe. A frame within 3 steps of one
            # already chosen is the same moment, so it is not rolled again. The per-flight cap
            # does not apply in this mode.
            high: List[int] = []
            for t in side:
                prev_a = log[t - 1].get("action")
                prev_a = np.zeros(4, np.float32) if prev_a is None else np.asarray(prev_a, np.float32)
                seen = log[t].get("da356")
                surv = fwd_ensemble(seen, log[t]["state"], goal, prev_a) if seen is not None else 0.0
                if surv < float(args.surv_min):
                    review.setdefault("dropped", []).append({"attempt": review["attempts"], "route_id": r["route_id"],
                                                             "t": t, "reason": "already_low",
                                                             "fwd_surv": round(surv, 4)})
                    logger.info("drop %s t=%d already_low surv=%.3f keep=%d", r["route_id"], t, surv, n_kept())
                    continue
                high.append(t)
                high_surv[t] = surv
            logger.info("side still_high %d / %d on %s", len(high), len(side), r["route_id"])
            for t in rng.permutation(high).tolist():
                if all(abs(t - p[0]) >= 3 for p in picks):
                    picks.append((int(t), "side"))
            picks.sort()
        elif args.branch_mode == "side_open":
            # Step 13: the wall has entered the center, and straight flight is still
            # near 0.94. A center of 8 m, or a straight score of 0.99 or 0.33, is not it.
            side = []
            for t in range(1, T):
                seen = log[t].get("da356")
                sign = open_yaw_sign(seen) if seen is not None else None
                if sign is None:
                    continue
                prev_a = log[t - 1].get("action")
                prev_a = np.zeros(4, np.float32) if prev_a is None else np.asarray(prev_a, np.float32)
                surv = fwd_ensemble(seen, log[t]["state"], goal, prev_a)
                if not (0.85 <= surv <= 0.97):
                    review.setdefault("dropped", []).append({"attempt": review["attempts"], "route_id": r["route_id"],
                                                             "t": t, "reason": "fwd_outside",
                                                             "fwd_surv": round(surv, 4)})
                    logger.info("drop %s t=%d fwd_outside surv=%.3f keep=%d", r["route_id"], t, surv, n_kept())
                    continue
                log[t]["open_sign"] = sign
                high_surv[t] = surv
                side.append(t)
            logger.info("step13-like frames %d / %d on %s", len(side), max(T - 1, 0), r["route_id"])
            for t in rng.permutation(side).tolist():
                if all(abs(t - p[0]) >= 3 for p in picks):
                    picks.append((int(t), "side_open"))
            picks.sort()
        elif args.branch_mode == "lowclear":
            low = [t for t in range(1, T) if low_clear(log[t]["gt56"]) < float(args.lowclear_m)]
            for t in rng.permutation(low).tolist():
                if len(picks) >= int(args.lowclear_per_flight):
                    break
                if all(abs(t - p[0]) >= 3 for p in picks):
                    why = f"lowclear_before_collision_{T - t}" if end == "collision" and T - t <= 15 else "lowclear"
                    picks.append((int(t), why))
            picks.sort()
        elif args.branch_mode == "fail":
            if end == "collision":
                picks += [(T - k, f"before_collision_{k}") for k in offsets if T - k >= 1]
            elif end in ("stuck", "max_steps"):
                picks += [(T - k, f"{end}_{k}") for k in (10, 40, 80) if T - k >= 1]
            taken = {p[0] for p in picks}
            low = [t for t in range(1, T) if low_clear(log[t]["gt56"]) < float(args.lowclear_m)
                   and all(abs(t - u) >= 3 for u in taken)]
            for t in rng.permutation(low).tolist():
                if sum(p[1] == "lowclear" for p in picks) >= int(args.fail_lowclear_n):
                    break
                if all(abs(t - p[0]) >= 3 for p in picks):
                    picks.append((int(t), "lowclear"))
            if T >= 2:
                picks.append((int(rng.integers(1, T)), "random"))
            picks.sort()
        else:
            if end == "collision":
                picks += [(T - k, f"before_collision_{k}") for k in offsets if T - k >= 1]
            if T >= 2:
                picks.append((int(rng.integers(1, T)), "random"))
        review["flights"].append({"attempt": review["attempts"], "route_id": r["route_id"], "end": end, "steps": T,
                                  "branch_steps": [p[0] for p in picks], "collision_object": flight_obj,
                                  "d_min_m": round(min(d_t), 2), "drop": geo})
        for t, why in picks:
            if n_kept() >= int(args.n_states) or one_usable_stop():
                break
            kept_surv = None
            if args.branch_mode in ("side", "side_clear", "side_arc"):
                if t in high_surv:
                    kept_surv = high_surv[t]
                else:
                    prev_a = log[t - 1].get("action")
                    prev_a = np.zeros(4, np.float32) if prev_a is None else np.asarray(prev_a, np.float32)
                    seen = log[t].get("da356")
                    kept_surv = fwd_ensemble(seen, log[t]["state"], goal, prev_a) if seen is not None else 0.0
                if kept_surv < float(args.surv_min):
                    review.setdefault("dropped", []).append({"attempt": review["attempts"], "route_id": r["route_id"],
                                                             "t": t, "reason": "already_low",
                                                             "fwd_surv": round(kept_surv, 4)})
                    logger.info("drop %s t=%d already_low surv=%.3f keep=%d", r["route_id"], t, kept_surv, n_kept())
                    continue
            try:
                b = branch(log[t], goal)
            except Exception as exc:
                logger.warning("branch dropped on sim error: %s", exc)
                b = None
            if b is None:
                review.setdefault("dropped", []).append({"attempt": review["attempts"], "route_id": r["route_id"],
                                                         "t": t, "reason": "replace_failed_or_collided"})
                continue
            if b.get("skip"):
                review.setdefault("dropped", []).append({"attempt": review["attempts"], "route_id": r["route_id"],
                                                         "t": t, "reason": b["skip"]})
                logger.info("drop %s t=%d %s keep=%d", r["route_id"], t, b["skip"], n_kept())
                continue
            h0 = max(0, t - int(args.hist_frames) + 1)
            hist = log[h0:t + 1]
            jpgs = [x["jpeg"] for x in hist]
            hist_action = np.stack([x["action"] if (i < len(hist) - 1) else np.zeros(4, np.float32)
                                    for i, x in enumerate(hist)]).astype(np.float32)
            idx = len(review["probes"])
            fname = f"probe_{idx:04d}.npz"
            np.savez(
                out / "probes" / fname,
                hist_jpeg=np.frombuffer(b"".join(jpgs), dtype=np.uint8),
                hist_jpeg_offsets=np.cumsum([0] + [len(j) for j in jpgs]).astype(np.int64),
                hist_action=hist_action,
                gt_depth56=log[t]["gt56"],
                state=log[t]["state"].astype(np.float32),
                goal=goal.astype(np.float32),
                candidates=b["cands"],
                coll_step=b["coll_step"],
                dist=b["dist"],
                cones=b["cones"],
                step_depth56=b["step_depth"],
                replace_err=b["replace_err"],
            )
            rec = {
                "probe": idx, "file": fname, "route_id": r["route_id"], "source": "onpolicy", "why": why,
                "flight_end": end, "families": b["names"], "train_keep": True,
                "fwd_hit": bool(b["coll_step"][0] >= 0),
                "bottom_min_m": round(float(log[t]["gt56"][38:].min()), 2),
                "collided": [int(c >= 0) for c in b["coll_step"]],
                "collision_object": b["coll_obj"],
                "progress_m": [None if b["coll_step"][m] >= 0 else round(float(b["dist"][m, 0] - b["dist"][m, H]), 2)
                               for m in range(len(b["names"]))],
                "max_replace_err_m": round(float(b["replace_err"].max()), 2),
                "goal_dist_m": round(d_t[t], 2),
                "goal_class": goals[r["route_id"]]["class"] if near else None,
            }
            if args.branch_mode in ("side", "side_clear", "side_arc", "side_open"):
                rec["da3_side"] = True
                rec["fwd_surv"] = None if kept_surv is None else round(float(kept_surv), 4)
            if args.branch_mode == "side_open":
                seen = log[t].get("da356")
                if kept_surv is None and seen is not None:
                    prev_a = log[t - 1].get("action")
                    prev_a = np.zeros(4, np.float32) if prev_a is None else np.asarray(prev_a, np.float32)
                    kept_surv = fwd_ensemble(seen, log[t]["state"], goal, prev_a)
                    rec["fwd_surv"] = round(float(kept_surv), 4)
                rec["side_rule"] = "da3_open_hit"
                rec["open_sign"] = float(log[t].get("open_sign") or 0.0)
                rec["open_hit"] = True
                logger.info("state %d %s %s flight=%s collided=%s fwd_surv=%s keep=%d/%d",
                            idx, r["route_id"], why, end, rec["collided"], rec["fwd_surv"],
                            n_kept() + 1, int(args.n_states))
            if args.branch_mode == "side_arc":
                rec["side_rule"] = "da3_arc06"
                rec["arc_hit"] = bool(np.any(np.asarray(b["coll_step"]) >= 0))
                hit_n, clear_n = arc_counts()
                # arc_counts reads probes already stored; this record is not in yet.
                hit_n += int(rec["arc_hit"])
                clear_n += int(not rec["arc_hit"])
                logger.info("state %d %s %s flight=%s collided=%s arc_hit=%d arc_clear=%d",
                            idx, r["route_id"], why, end, rec["collided"], hit_n, clear_n)
            elif args.branch_mode == "side_clear":
                rec["side_rule"] = "da3_still_clear"
            elif args.branch_mode == "side":
                rec["side_rule"] = "da3_still_high"
            review["probes"].append(rec)
            review_path.write_text(json.dumps(review, indent=1), encoding="utf-8")
            if args.branch_mode != "side_arc":
                logger.info("state %d %s %s flight=%s collided=%s keep=%d/%d",
                            idx, r["route_id"], why, end, rec["collided"], n_kept(), int(args.n_states))
        review_path.write_text(json.dumps(review, indent=1), encoding="utf-8")
    if args.branch_mode == "side_arc":
        hit_n, clear_n = arc_counts()
        if one_usable_stop():
            logger.info("stop file %s", stop_file)
        logger.info("DONE states=%d arc_hit=%d arc_clear=%d flights=%d",
                    len(review["probes"]), hit_n, clear_n, len(review["flights"]))
    elif args.branch_mode == "side_open":
        logger.info("DONE states=%d open_pairs=%d flights=%d",
                    len(review["probes"]), n_kept(), len(review["flights"]))
    else:
        logger.info("DONE states=%d keep=%d flights=%d", len(review["probes"]), n_kept(), len(review["flights"]))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
