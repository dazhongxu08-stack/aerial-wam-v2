#!/usr/bin/env python3
"""Stage 3: shield-off MPC flight with the consequence-model ensemble.

Every ``--replan-every`` steps: sample one candidate 10-step sequence per family
(same families and macro sampler as the Stage 2 probes), keep those that pass
the hold-altitude rule, score them with the ensemble (mean survival, mean
predicted goal progress) and pick with ``select_candidate`` (the rule fixed on
the training-region held-out set). Execute the first steps of the pick, replan.
No shield, no hand-written turns: the only control logic is the model's choice.

Per route: arrived (within --success-dist of the goal), severe collision (any
collision ends the episode), progress ratio = max(0, d0 - d_final) / route
length, stuck (no 1 m gain on the best distance for --stuck-s seconds).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("mpc_consequence")


def candidate_side(cands: np.ndarray) -> np.ndarray:
    """+1 / -1 for candidates ending turned (or displaced) to one side, 0 for straight or in-place.

    Yaw and lateral velocity share a sign convention (mirror_batch negates both)."""
    c = np.asarray(cands, np.float64)
    turn, lat = c[..., 3].sum(-1), c[..., 1].sum(-1)
    return np.where(np.abs(turn) > 0.15, np.sign(turn), np.where(np.abs(lat) > 0.2, np.sign(lat), 0.0))


def goal_offset(cands: np.ndarray, bearing: float, dt: float):
    """Angle of each candidate's net horizontal displacement from the goal direction (+ = the +yaw side), and its length.

    Body frame at plan time; +vy points to the +yaw side, the same sense as ``goal_bearing_body``."""
    c = np.asarray(cands, np.float64)
    th = np.cumsum(c[..., 3], -1) * dt
    dx = (c[..., 0] * np.cos(th) - c[..., 1] * np.sin(th)).sum(-1) * dt
    dy = (c[..., 0] * np.sin(th) + c[..., 1] * np.cos(th)).sum(-1) * dt
    return (np.arctan2(dy, dx) - bearing + np.pi) % (2 * np.pi) - np.pi, np.hypot(dx, dy)


def follow_fan(k: int, yaw_limit: float) -> np.ndarray:
    """Fixed yaw-then-forward fan both ways plus full-rate turns in place (yaw_fwd / yaw_only shapes)."""
    seqs = [np.tile([0.6, 0.0, 0.0, 0.0], (k, 1))]
    for sgn in (-1.0, 1.0):
        for frac in (0.5, 1.0):
            for m in (1, 2, 3, 4):
                a = np.zeros((k, 4))
                a[:m, 3], a[:m, 0], a[m:, 0] = sgn * frac * yaw_limit, 0.1, 0.6
                seqs.append(a)
        a = np.zeros((k, 4))
        a[:, 3] = sgn * yaw_limit
        seqs.append(a)
    return np.stack(seqs)


def follow_pick(alpha: np.ndarray, length: np.ndarray, safe: np.ndarray, sigma: float,
                min_len: float = 0.5, cross_tol: float = math.radians(15.0)) -> int:
    """Boundary following: the safe moving candidate heading closest to the goal, crossing to the other side by at
    most ``cross_tol``.

    -1 when no safe candidate moves at least ``min_len``."""
    a = sigma * np.asarray(alpha, np.float64)
    ok = np.asarray(safe, bool) & (np.asarray(length) >= min_len) & (a >= -cross_tol)
    return int(np.argmin(np.where(ok, np.abs(a), np.inf))) if ok.any() else -1


def near_pick(cands: np.ndarray, bearing: float, safe: np.ndarray, prog: np.ndarray, dt: float = 0.2,
              max_deg: float = 30.0, min_len: float = 0.3) -> int:
    """Most progressing safe candidate whose net displacement heads within ``max_deg`` of the goal; -1 if none."""
    alpha, length = goal_offset(cands, bearing, dt)
    ok = np.asarray(safe, bool) & (length >= min_len) & (np.abs(alpha) <= math.radians(max_deg))
    return int(np.argmax(np.where(ok, prog, -np.inf))) if ok.any() else -1


def nopass_pick(cands: np.ndarray, bearing: float, surv: np.ndarray, prog: np.ndarray, allowed: np.ndarray,
                margin: float, floor: float, min_prog: float, dt: float = 0.2, min_len: float = 0.5) -> int:
    """Most progressing moving candidate nearly as safe as the safest allowed one; -1 if none."""
    _, length = goal_offset(cands, bearing, dt)
    allowed = np.asarray(allowed, bool)
    s = np.asarray(surv, np.float64)
    bar = max(float(floor), float(s[allowed].max()) - float(margin))
    ok = allowed & (s >= bar) & (length >= min_len) & (np.asarray(prog) >= min_prog)
    return int(np.argmax(np.where(ok, prog, -np.inf))) if ok.any() else -1


def candidate_waypoints(cands: np.ndarray, state: np.ndarray, k: int) -> np.ndarray:
    """World (x, y, z-up) after each of the first ``k`` per-step body displacements of every candidate, (M, k, 3)."""
    c = np.asarray(cands, np.float64)[:, :int(k)]
    th = float(state[6]) + np.concatenate([np.zeros((len(c), 1)), np.cumsum(c[:, :-1, 3], -1)], -1)
    dx = c[..., 0] * np.cos(th) - c[..., 1] * np.sin(th)
    dy = c[..., 0] * np.sin(th) + c[..., 1] * np.cos(th)
    return (np.asarray(state[:3], np.float64)
            + np.stack([np.cumsum(dx, -1), np.cumsum(dy, -1), np.cumsum(c[..., 2], -1)], -1))


def depth_points(depth: np.ndarray, state: np.ndarray, max_range: float, cam_dz: float = -1.0,
                 hfov_deg: float = 90.0, vfov_deg: float = 58.7, voxel: float = 0.2) -> np.ndarray:
    """World points (x, y, z-up) of a pooled planar depth map closer than ``max_range``, voxel-deduplicated.

    The pooled map squeezes the 16:9 frame to a square, so the two axes get their own focal length."""
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    fx, fy = (w / 2) / math.tan(math.radians(hfov_deg / 2)), (h / 2) / math.tan(math.radians(vfov_deg / 2))
    v, u = np.mgrid[0:h, 0:w]
    ok = np.isfinite(d) & (d > 0.05) & (d < max_range)
    fwd = d[ok]
    right = (u[ok] + 0.5 - w / 2) / fx * fwd
    down = (v[ok] + 0.5 - h / 2) / fy * fwd
    yaw = float(state[6])
    pts = np.stack([state[0] + fwd * math.cos(yaw) - right * math.sin(yaw),
                    state[1] + fwd * math.sin(yaw) + right * math.cos(yaw),
                    state[2] + cam_dz - down], -1)
    return np.unique(np.round(pts / voxel), axis=0) * voxel if len(pts) else pts.reshape(0, 3)


def memory_clear(wps: np.ndarray, here: np.ndarray, pts: np.ndarray, clear_m: float, dz_m: float,
                 slack: float = 0.1) -> np.ndarray:
    """True for candidates that keep ``clear_m`` horizontally from remembered points within ``dz_m``, or, when the
    drone is already closer than that, do not get more than ``slack`` closer than it is now."""
    flat = wps.reshape(-1, 3)
    allp = np.vstack([flat, np.asarray(here, np.float64)[None, :3]])
    lo, hi = allp.min(0) - clear_m, allp.max(0) + clear_m
    pts = pts[np.all((pts[:, :2] >= lo[:2]) & (pts[:, :2] <= hi[:2]), -1)]
    if not len(pts):
        return np.ones(len(wps), bool)

    def clearance(q: np.ndarray) -> np.ndarray:
        dxy = np.linalg.norm(q[:, None, :2] - pts[None, :, :2], axis=-1)
        return np.where(np.abs(q[:, None, 2] - pts[None, :, 2]) < dz_m, dxy, np.inf).min(-1)

    bar = min(clear_m, float(clearance(allp[-1:])[0]) - slack)
    return clearance(flat).reshape(wps.shape[:2]).min(-1) >= bar


def brake_action(state: np.ndarray, step_hz: float, limits: np.ndarray, hold_z: Optional[float] = None,
                 hold_gain: float = 0.5, hold_max: float = 0.2) -> np.ndarray:
    """Per-step body displacement that commands the opposite of the current horizontal velocity, no turn.

    With ``hold_z``, also steer the height toward it: a hard stop pitches the drone back and lifts it ~0.15 m."""
    vx, vy = float(state[3]), float(state[4])
    yaw = float(state[6])
    a = np.zeros(4)
    a[0] = -(vx * math.cos(yaw) + vy * math.sin(yaw)) / step_hz
    a[1] = -(-vx * math.sin(yaw) + vy * math.cos(yaw)) / step_hz
    if hold_z is not None:
        a[2] = float(np.clip(hold_gain * (hold_z - float(state[2])), -hold_max, hold_max))
    return np.clip(a, -limits, limits)


def open_turn_sign(depth: np.ndarray, cap: float = 10.0) -> float:
    """+1 (yaw rate > 0 turns toward the image right) when the right half of the depth image is more open, else -1."""
    col = np.minimum(np.asarray(depth, np.float64).min(0), cap)
    h = col.shape[0] // 2
    return 1.0 if col[h:].mean() > col[:h].mean() else -1.0


def follow_side_choice(alpha: np.ndarray, length: np.ndarray, safe: np.ndarray, clear: np.ndarray, bearing: float) -> float:
    """Side whose strictly-on-side best candidate heads closer to the goal; ties go to the larger predicted clearance."""
    best = {}
    for sd in (-1.0, 1.0):
        i = follow_pick(alpha, length, safe, sd, cross_tol=0.0)
        if i >= 0:
            best[sd] = (abs(float(alpha[i])), -float(clear[i]))
    if not best:
        return 1.0 if bearing >= 0 else -1.0
    return min(best, key=best.get)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation", default="experiments/aerial/phase3_unified/annotations/outdoor_complex_od_holdout.json")
    ap.add_argument("--routes", default="", help="comma list of route_id; empty = all")
    ap.add_argument("--ckpts", required=True, help="comma list of consequence checkpoints (ensemble)")
    ap.add_argument("--depth-ckpt", default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt")
    ap.add_argument("--config", default="configs/_tmp_feasible_side_h125.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--surv-min", type=float, default=0.7)
    ap.add_argument("--replan-every", type=int, default=3)
    ap.add_argument("--success-dist", type=float, default=3.0)
    ap.add_argument("--max-steps-per-m", type=float, default=4.0)
    ap.add_argument("--budget-ref", choices=("chord", "astar"), default="chord",
                    help="length the step budget scales with: annotation polyline, or complexity_meta astar_1p5_m")
    ap.add_argument("--stuck-s", type=float, default=60.0)
    ap.add_argument("--min-z-m", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="model", choices=("model", "goal_line"),
                    help="goal_line = baseline that always takes the goal_turn_fwd candidate (model ignored)")
    ap.add_argument("--input-depth", default="da3", choices=("da3", "gt"),
                    help="gt = simulator depth as model input; diagnostic only, must match the checkpoints")
    ap.add_argument("--alt-fix", action="store_true",
                    help="hold-altitude rule also on the executed prefix; add a descend candidate when >1 m above the goal")
    ap.add_argument("--alt-band", action="store_true",
                    help="executed prefix must keep altitude within 1 m of the [start, goal] altitude band")
    ap.add_argument("--fwd-dx", type=float, default=0.8, help="forward command of the fwd family")
    ap.add_argument("--dx-scale", type=float, default=1.0,
                    help="scale forward commands of the sampled candidates, capped at 1.0 (the training range)")
    ap.add_argument("--goal-surv-min", type=float, default=0.0,
                    help="if >0, survival threshold used for the pick within --goal-surv-dist of the goal")
    ap.add_argument("--goal-surv-dist", type=float, default=8.0)
    ap.add_argument("--extra-families", default="",
                    help="comma list from EXTRA_MACROS (hover, spin, back) added to the candidates")
    ap.add_argument("--fov-max-deg", type=float, default=0.0,
                    help="if >0, executed horizontal motion must be within this angle of the nose (camera sees +-45)")
    ap.add_argument("--escape-after-s", type=float, default=0.0,
                    help="if >0, after this long without 1 m of goal progress, spend --escape-s choosing the survivor "
                         "with the largest predicted forward clearance instead of the largest goal progress")
    ap.add_argument("--escape-s", type=float, default=4.0)
    ap.add_argument("--commit-after-s", type=float, default=0.0,
                    help="if >0, within --commit-dist of the goal and this long without 1 m of goal progress, lock one "
                         "side (the one whose surviving candidates have the larger predicted forward clearance) for "
                         "--commit-s; picks stay goal-progress among that side's candidates")
    ap.add_argument("--commit-s", type=float, default=4.0)
    ap.add_argument("--commit-dist", type=float, default=15.0)
    ap.add_argument("--follow-after-s", type=float, default=0.0,
                    help="if >0, within --follow-dist of the goal and this long without 1 m of goal progress, follow the "
                         "obstacle boundary on one side of the goal direction (follow_pick) until --follow-leave-m closer "
                         "than where it started with the goal direction clear; one side switch after --follow-max-s, a "
                         "270 deg turn or getting --follow-away-m farther than where that side started; the leave "
                         "distance is kept across the switch; after both sides, back to the default rule")
    ap.add_argument("--follow-dist", type=float, default=15.0)
    ap.add_argument("--follow-leave-m", type=float, default=1.5)
    ap.add_argument("--follow-leave-deg", type=float, default=20.0)
    ap.add_argument("--follow-max-s", type=float, default=8.0)
    ap.add_argument("--follow-away-m", type=float, default=4.0)
    ap.add_argument("--follow-mline-m", type=float, default=0.0,
                    help="if >0 (Bug2), leaving the boundary also requires being within this distance of the "
                         "start-goal line")
    ap.add_argument("--progress-ckpts", default="",
                    help="comma list of checkpoints whose dgoal head gives the goal progress (e.g. geodesic-label "
                         "models); survival and clearance still come from --ckpts")
    ap.add_argument("--nopass-margin", type=float, default=0.0,
                    help="if >0, when no allowed candidate reaches the survival threshold, pick the most progressing "
                         "moving candidate (>= 0.5 m net, predicted progress >= --nopass-min-prog) whose survival is "
                         "within this margin of the safest allowed one and >= --nopass-floor, instead of the safest")
    ap.add_argument("--park-before-reset", type=float, default=0.0,
                    help="if >0, hover this many seconds --park-alt-m above the start before the reset teleport; the "
                         "teleport keeps the velocity the previous episode ended with")
    ap.add_argument("--park-alt-m", type=float, default=0.0,
                    help="parking 20-40 m up made the eval drone sink ~3 m on its first step after the teleport down")
    ap.add_argument("--settle-after-reset", type=float, default=0.0,
                    help="if >0, hover after reset until the speed drops below --settle-speed or this many seconds "
                         "pass, then observe again; reset does not zero the velocity")
    ap.add_argument("--settle-speed", type=float, default=0.1)
    ap.add_argument("--brake-speed", type=float, default=0.0,
                    help="if >0, when no allowed candidate reaches the survival threshold (and no nopass pick) and the "
                         "horizontal speed exceeds this (m/s), command the opposite velocity for one step instead of "
                         "the safest candidate; zero-velocity picks only coast to a stop")
    ap.add_argument("--brake-hold-alt", action="store_true",
                    help="brake steps also steer the height toward the middle of the start/goal altitude band; braking "
                         "alone climbs ~0.15 m per stop and lifted od_hard_02 above SM_NYCB_579's rendered roof, whose "
                         "collision extends ~7 m higher")
    ap.add_argument("--spin-nopass", action="store_true",
                    help="when no allowed candidate reaches the survival threshold (and no nopass pick or brake), turn "
                         "in place at full yaw rate toward the more open image half, keeping that side until one passes; "
                         "the safest candidate otherwise still creeps forward (every family has dx >= 0)")
    ap.add_argument("--mem-clear-m", type=float, default=0.0,
                    help="if >0, disallow candidates whose first --mem-steps waypoints come within this horizontal "
                         "distance of points seen in the last --mem-frames model-input depth frames (walls that left "
                         "the view), or closer than the drone already is; ignored when it would disallow every "
                         "candidate")
    ap.add_argument("--mem-steps", type=int, default=3)
    ap.add_argument("--mem-frames", type=int, default=25)
    ap.add_argument("--mem-range", type=float, default=6.0)
    ap.add_argument("--mem-dz", type=float, default=1.0)
    ap.add_argument("--nopass-floor", type=float, default=0.85)
    ap.add_argument("--nopass-min-prog", type=float, default=-1.0)
    ap.add_argument("--nopass-stall-prog", type=float, default=0.0,
                    help="if >0 (with --nopass-margin), when the best passing candidate predicts less progress than "
                         "this and the pick barely moves, take the most progressing passing candidate that moves")
    ap.add_argument("--near-goal-dist", type=float, default=0.0,
                    help="if >0, within this distance of the goal and --near-goal-after-s without 1 m of goal "
                         "progress, add slow goal-turn candidates (--near-goal-dx) and pick the most progressing "
                         "survivor heading within --near-goal-deg of the goal; survival threshold unchanged")
    ap.add_argument("--near-goal-after-s", type=float, default=3.0)
    ap.add_argument("--near-goal-dx", default="0.3,0.5")
    ap.add_argument("--near-goal-deg", type=float, default=30.0)
    ap.add_argument("--keep-frames", type=int, default=15,
                    help="frames kept in last_frames_*.npz; 0 keeps the whole flight")
    ap.add_argument("--use-spawn-z", action="store_true",
                    help="start and goal at the validated spawn_z_ok (validate_od_routes_airsim) instead of the annotation z")
    ap.add_argument("--hover-after-reset", action="store_true",
                    help="hold a hover from reset until the first action; the first plan takes ~2 s and an "
                         "uncommanded drone can sink several metres in that gap")
    ap.add_argument("--alt-band-return", type=float, default=0.0,
                    help="outside the band, required commanded return toward it per executed step (m)")
    ap.add_argument("--tune-on-train", action="store_true",
                    help="fly train routes instead, for choosing decision settings without touching held-out routes")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.follow_after_s > 0 and (args.commit_after_s > 0 or args.escape_after_s > 0):
        raise SystemExit("--follow-after-s excludes --commit-after-s and --escape-after-s")

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    import yaml

    from experiments.aerial.rl.collect_consequence_dataset import (
        EXTRA_MACROS, LIMITS, encode_jpeg, goal_bearing_body, pool_min, sample_macro,
    )
    from experiments.aerial.rl.consequence_model import (
        altitude_allowed, altitude_band_allowed, history_inputs, goal_features, load_checkpoint, select_candidate, survival,
        visible_motion_allowed,
    )
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.precompute_env_consequence import env_pair
    from experiments.aerial.rl.probe_counterfactual_airsim import FAMILIES
    from experiments.aerial.rl.train_rl import _build_env

    extra = tuple(s for s in args.extra_families.split(",") if s)
    if any(s not in EXTRA_MACROS for s in extra):
        raise SystemExit(f"unknown extra families {extra}; choose from {EXTRA_MACROS}")
    fams = tuple(FAMILIES) + extra
    dev = torch.device(args.device)
    models = [load_checkpoint(root / p, device=args.device) for p in args.ckpts.split(",") if p]
    prog_models = [load_checkpoint(root / p, device=args.device) for p in args.progress_ckpts.split(",") if p]
    for m in models + prog_models:
        m.eval()
    hist_len = max([8] + [max(m.frame_offsets) + 1 for m in models + prog_models]
                   + [m.hist_actions + 1 for m in models + prog_models])
    H = models[0].horizon
    fan = follow_fan(H, float(LIMITS[3]))
    near_dx = [float(x) for x in args.near_goal_dx.split(",") if x]
    pred = DepthMinPredictor.from_checkpoint(root / args.depth_ckpt, device=args.device)

    routes = json.loads((root / args.annotation).read_text(encoding="utf-8"))["episodes"]
    want = "train" if args.tune_on_train else "holdout"
    if any(r.get("complexity_meta", {}).get("split") != want for r in routes):
        raise SystemExit(f"refusing: this mode flies {want} routes only")
    only = [r for r in args.routes.split(",") if r]
    if only:
        routes = [r for r in routes if r["route_id"] in only]

    cfg = yaml.safe_load((root / args.config).read_text()) or {}
    env_cfg = dict(cfg.get("env") or {})
    env_cfg.update({"backend": "airsim", "step_hz": 5.0, "grab_depth": True})
    env = _build_env(env_cfg)
    step_hz = 5.0

    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "MPC_EVAL.json"
    results: Dict[str, Any] = (json.loads(res_path.read_text()) if res_path.is_file()
                               else {"args": vars(args), "episodes": []})
    done_ids = {e["route_id"] for e in results["episodes"]}
    rng = np.random.default_rng(args.seed)

    def score(dseq: List[np.ndarray], aseq: List[np.ndarray], state: np.ndarray, goal: np.ndarray,
              prev: np.ndarray, cands: np.ndarray, rgb: np.ndarray):
        M = len(cands)
        g = torch.as_tensor(np.repeat(goal_features(state, goal)[None], M, 0), device=dev)
        pa = torch.as_tensor(np.repeat(prev[None].astype(np.float32), M, 0), device=dev)
        a = torch.as_tensor(cands.astype(np.float32), device=dev)
        env = None
        if any(getattr(m, "env_dim", 0) for m in models + prog_models):
            e, _ = env_pair(pred._model, [rgb])
            env = torch.as_tensor(np.repeat(e[:1].astype(np.float32), M, 0), device=dev)
        s_all, p_all, f_all, q_all = [], [], [], []
        with torch.no_grad():
            for i, m in enumerate(models + prog_models):
                d_k, a_k = history_inputs(dseq, aseq, m.frame_offsets, m.hist_actions)
                d_k = d_k[0] if len(m.frame_offsets) == 1 else d_k
                d = torch.as_tensor(np.repeat(d_k[None], M, 0), device=dev)
                ha = torch.as_tensor(np.repeat(a_k[None], M, 0), device=dev) if m.hist_actions else None
                o = m(d, g, pa, a, ha, env=env) if getattr(m, "env_dim", 0) else m(d, g, pa, a, ha)
                if i >= len(models):
                    q_all.append((-o["dgoal"][..., -1]).cpu().numpy())
                    continue
                s_all.append(survival(o["hazard_logit"])[..., -1].cpu().numpy())
                p_all.append((-o["dgoal"][..., -1]).cpu().numpy())
                f_all.append(o["log_cones"][..., -1, 0].cpu().numpy())
        return np.mean(s_all, 0), np.mean(q_all or p_all, 0), np.mean(f_all, 0)

    for r in routes:
        if r["route_id"] in done_ids:
            continue
        pos = np.asarray(r["pos"], dtype=np.float64)
        start, goal = pos[0].copy(), pos[-1].copy()
        if args.use_spawn_z and r.get("complexity_meta", {}).get("spawn_z_ok") is not None:
            start[2] = goal[2] = float(r["complexity_meta"]["spawn_z_ok"])
        ref_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

        def mline_dist(xy: np.ndarray, a: np.ndarray = start[:2].copy(), b: np.ndarray = goal[:2].copy()) -> float:
            ab = b - a
            t = float(np.clip(np.dot(xy - a, ab) / max(1e-9, float(np.dot(ab, ab))), 0.0, 1.0))
            return float(np.linalg.norm(xy - (a + t * ab)))

        yaw0 = float(r["yaw"][0])
        budget_len = ref_len
        if args.budget_ref == "astar":
            astar = r.get("complexity_meta", {}).get("astar_1p5_m")
            if astar is None:
                raise SystemExit(f"{r['route_id']}: --budget-ref astar needs complexity_meta.astar_1p5_m")
            budget_len = float(astar)
        max_steps = int(args.max_steps_per_m * budget_len)
        if args.park_before_reset > 0:
            c, airsim_ = env._connect(), env._airsim
            c.enableApiControl(True, **env._vk)
            c.armDisarm(True, **env._vk)
            c.simSetVehiclePose(airsim_.Pose(airsim_.Vector3r(float(start[0]), float(start[1]),
                                                              -(float(start[2]) + args.park_alt_m)),
                                             airsim_.to_quaternion(0.0, 0.0, 0.0)), True, **env._vk)
            c.hoverAsync(**env._vk)
            time.sleep(args.park_before_reset)
        try:
            obs = env.reset({"pos": [start.tolist(), goal.tolist()], "yaw": [yaw0, yaw0]})
        except RuntimeError as exc:
            logger.warning("%s reset failed: %s", r["route_id"], exc)
            results["episodes"].append({"route_id": r["route_id"], "end_reason": "reset_failed"})
            res_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
            continue
        if args.hover_after_reset or args.settle_after_reset > 0:
            env._client.hoverAsync(**env._vk)
        settle_s = 0.0
        if args.settle_after_reset > 0:
            t_settle = time.time()
            while time.time() - t_settle < args.settle_after_reset:
                v = env.observe_state()[3:6]
                if float(np.linalg.norm(v)) < args.settle_speed:
                    break
                time.sleep(0.1)
            settle_s = time.time() - t_settle
            obs = env.observe()
        pred.reset()
        get_contact = getattr(env, "collision_info", lambda: {})
        reset_contact = get_contact()
        z_ref = float(np.asarray(obs.position)[2])
        z_lo, z_hi = min(z_ref, float(goal[2])) - 1.0, max(z_ref, float(goal[2])) + 1.0
        d0 = float(np.linalg.norm(np.asarray(obs.position) - goal))
        best_d, best_step = d0, 0
        escape_until, escape_end = -1, 0
        commit_until, commit_end, commit_side = -1, 0, 0.0
        follow_side, follow_start, follow_end, follow_d, follow_turn, follow_switched = 0.0, 0, 0, 0.0, 0.0, False
        follow_side_d = 0.0
        spin_side = 0.0
        prev = np.zeros(4, np.float32)
        plan: List[np.ndarray] = []
        steps: List[Dict[str, Any]] = []
        traj = [np.asarray(obs.position, dtype=np.float64).tolist()]
        end = "max_steps"
        t0 = time.time()
        depth = pred.predict_depth(obs)
        recent: deque = deque(maxlen=int(args.keep_frames) or None)

        def remember(o: Any, d_hat: np.ndarray) -> None:
            recent.append((encode_jpeg(np.asarray(o.rgb)), pool_min(np.asarray(d_hat, np.float32)),
                           pool_min(np.asarray(o.depth, np.float32))))

        remember(obs, depth)
        dhist: deque = deque(maxlen=hist_len)
        ahist: deque = deque(maxlen=hist_len)
        dhist.append(pool_min(np.asarray(obs.depth if args.input_depth == "gt" else depth, np.float32)))
        mem: deque = deque(maxlen=max(1, int(args.mem_frames)))
        if args.mem_clear_m > 0:
            mem.append(depth_points(dhist[-1], np.asarray(obs.state, np.float64), args.mem_range))
        for step in range(max_steps):
            if not plan or step % int(args.replan_every) == 0:
                state = np.asarray(obs.state, dtype=np.float64)
                bearing = goal_bearing_body(state[:3], float(state[6]), goal)
                cands, names = [], []
                for fam in fams:
                    name, seq = sample_macro(rng, H, bearing, name=fam)
                    if fam == "fwd":
                        seq[:, 0] = args.fwd_dx
                    if args.dx_scale != 1.0:
                        seq[:, 0] = np.where(seq[:, 0] > 0, np.minimum(seq[:, 0] * args.dx_scale, 1.0), seq[:, 0])
                    cands.append(seq)
                    names.append(name)
                if args.alt_fix and state[2] > goal[2] + 1.0:
                    name, seq = sample_macro(rng, H, bearing, name="descend")
                    cands.append(seq)
                    names.append(name)
                d_here = float(np.linalg.norm(state[:3] - goal))
                near_due = (args.near_goal_dist > 0 and d_here <= args.near_goal_dist
                            and (step - best_step) / step_hz >= args.near_goal_after_s)
                if near_due:
                    for dx in near_dx:
                        _, seq = sample_macro(rng, H, bearing, name="goal_turn_fwd")
                        seq[seq[:, 3] == 0, 0] = dx
                        cands.append(seq)
                        names.append("near_creep")
                cands = np.stack(cands)
                follow_due = (args.follow_after_s > 0 and not follow_side and d_here <= args.follow_dist
                              and (step - max(best_step, follow_end)) / step_hz >= args.follow_after_s)
                if follow_side or follow_due:
                    cands = np.concatenate([cands, fan])
                    names += ["follow_fan"] * len(fan)
                allowed = altitude_allowed(cands, state, goal)
                if args.alt_fix:
                    allowed &= altitude_allowed(cands[:, :int(args.replan_every)], state, goal)
                if args.alt_band:
                    banded = allowed & altitude_band_allowed(
                        cands, state[2], z_lo, z_hi, int(args.replan_every), args.alt_band_return)
                    allowed = banded if banded.any() else allowed
                if args.fov_max_deg > 0:
                    seen = allowed & visible_motion_allowed(cands, int(args.replan_every), args.fov_max_deg)
                    allowed = seen if seen.any() else allowed
                mem_blocked = 0
                if args.mem_clear_m > 0:
                    clear = memory_clear(candidate_waypoints(cands, state, args.mem_steps), state[:3],
                                         np.concatenate(mem), args.mem_clear_m, args.mem_dz)
                    mem_blocked = int((allowed & ~clear).sum())
                    allowed = allowed & clear if (allowed & clear).any() else allowed
                s, p, fwd_clear = score(list(dhist), list(ahist), state, goal, prev, cands, np.asarray(obs.rgb))
                if (args.escape_after_s > 0 and step >= escape_until
                        and (step - max(best_step, escape_end)) / step_hz >= args.escape_after_s):
                    escape_until = step + int(round(args.escape_s * step_hz))
                    escape_end = escape_until
                escaping = step < escape_until
                pick_allowed = allowed
                if args.commit_after_s > 0:
                    side = candidate_side(cands)
                    safe = allowed & (s >= args.surv_min)
                    if (step >= commit_until and np.linalg.norm(state[:3] - goal) <= args.commit_dist
                            and (step - max(best_step, commit_end)) / step_hz >= args.commit_after_s):
                        best = {sd: float(fwd_clear[safe & (side == sd)].max()) for sd in (-1.0, 1.0)
                                if (safe & (side == sd)).any()}
                        if best:
                            commit_side = max(best, key=best.get)
                            commit_until = step + int(round(args.commit_s * step_hz))
                            commit_end = commit_until
                    if step < commit_until and (safe & (side == commit_side)).any():
                        pick_allowed = allowed & (side == commit_side)
                committing = pick_allowed is not allowed
                follow_c, alpha, nopass_used, braking, spinning = -1, None, False, False, False
                if follow_side or follow_due:
                    alpha, dlen = goal_offset(cands, bearing, 1.0 / step_hz)
                    safe = allowed & (s >= args.surv_min)
                    if follow_due:
                        follow_side = follow_side_choice(alpha, dlen, safe, fwd_clear, bearing)
                        follow_start, follow_d, follow_turn, follow_switched = step, d_here, 0.0, False
                        follow_side_d = d_here
                    elif (d_here <= follow_d - args.follow_leave_m
                          and (args.follow_mline_m <= 0 or mline_dist(state[:2]) <= args.follow_mline_m)
                          and (safe & (dlen >= 0.5) & (np.abs(alpha) <= math.radians(args.follow_leave_deg))).any()):
                        follow_side, follow_end = 0.0, step
                    elif ((step - follow_start) / step_hz >= args.follow_max_s or abs(follow_turn) >= 1.5 * math.pi
                          or d_here > follow_side_d + args.follow_away_m):
                        if follow_switched:
                            follow_side, follow_end = 0.0, step
                        else:
                            follow_side, follow_start, follow_turn, follow_side_d = -follow_side, step, 0.0, d_here
                            follow_switched = True
                    if follow_side:
                        follow_c = follow_pick(alpha, dlen, safe, follow_side)
                        if follow_c < 0:
                            spin = [i for i in range(len(cands)) if names[i] == "follow_fan" and allowed[i]
                                    and dlen[i] < 0.5 and np.sign(cands[i, 0, 3]) == follow_side]
                            follow_c = spin[0] if spin else -1
                if args.policy == "goal_line":
                    c = names.index("goal_turn_fwd")
                elif follow_c >= 0:
                    c = follow_c
                elif near_due and (near_c := near_pick(cands, bearing, allowed & (s >= args.surv_min), p,
                                                       1.0 / step_hz, args.near_goal_deg)) >= 0:
                    c = near_c
                else:
                    surv_pick = (args.goal_surv_min if args.goal_surv_min > 0 and d_here <= args.goal_surv_dist
                                 else args.surv_min)
                    c = select_candidate(s, fwd_clear if escaping else p, pick_allowed, surv_pick)
                    passing = pick_allowed & (s >= surv_pick)
                    c_np = -1
                    if args.nopass_margin > 0 and not escaping and pick_allowed.any():
                        if not passing.any():
                            c_np = nopass_pick(cands, bearing, s, p, pick_allowed, args.nopass_margin,
                                               args.nopass_floor, args.nopass_min_prog, 1.0 / step_hz)
                        elif (args.nopass_stall_prog > 0 and float(p[passing].max()) < args.nopass_stall_prog
                              and goal_offset(cands[c:c + 1], bearing, 1.0 / step_hz)[1][0] < 0.5):
                            c_np = nopass_pick(cands, bearing, s, p, passing, 1.0, surv_pick,
                                               args.nopass_min_prog, 1.0 / step_hz)
                    if c_np >= 0:
                        c, nopass_used = c_np, True
                    elif (args.brake_speed > 0 and not passing.any()
                          and math.hypot(float(state[3]), float(state[4])) > args.brake_speed):
                        braking = True
                    elif args.spin_nopass and not passing.any():
                        spinning = True
                        spin_side = spin_side or open_turn_sign(dhist[-1])
                    if passing.any():
                        spin_side = 0.0
                plan = [np.asarray(a, np.float64) for a in cands[c]]
                if braking:
                    hold_z = 0.5 * (z_lo + z_hi) if args.brake_hold_alt else None
                    plan = [brake_action(state, step_hz, np.asarray(LIMITS, np.float64), hold_z)] + [np.zeros(4)] * (H - 1)
                elif spinning:
                    plan = [np.array([0.0, 0.0, 0.0, spin_side * float(LIMITS[3])])] + [np.zeros(4)] * (H - 1)
                family = "brake" if braking else "spin_nopass" if spinning else names[c]
                steps.append({"step": step, "family": family, "surv": round(float(s[c]), 3),
                              "pred_prog": round(float(p[c]), 2), "escape": bool(escaping),
                              "commit": int(commit_side) if committing else 0,
                              "follow": int(follow_side), "near": bool(near_due), "nopass": bool(nopass_used),
                              "alpha_deg": None if alpha is None else round(math.degrees(float(alpha[c])), 1),
                              "n_pass": int((allowed & (s >= args.surv_min)).sum()),
                              "d_goal": round(float(np.linalg.norm(state[:3] - goal)), 1),
                              "yaw": round(float(state[6]), 4),
                              "z_off": round(float(state[2] - goal[2]), 2),
                              "speed": round(math.hypot(float(state[3]), float(state[4])), 2),
                              "mem_blocked": mem_blocked,
                              "cand_family": names, "cand_surv": np.round(s, 3).tolist(),
                              "cand_pred_prog": np.round(p, 2).tolist(), "cand_allowed": allowed.tolist()})
            a = plan.pop(0)
            if follow_side:
                follow_turn += float(a[3]) / step_hz
            obs, _ = env.step(a)
            prev = a.astype(np.float32)
            depth = pred.predict_depth(obs)
            remember(obs, depth)
            ahist.append(prev)
            dhist.append(pool_min(np.asarray(obs.depth if args.input_depth == "gt" else depth, np.float32)))
            if args.mem_clear_m > 0:
                mem.append(depth_points(dhist[-1], np.asarray(obs.state, np.float64), args.mem_range))
            p_now = np.asarray(obs.position, dtype=np.float64)
            traj.append(p_now.tolist())
            d_now = float(np.linalg.norm(p_now - goal))
            if bool(obs.collided):
                end = "collision"
                break
            if d_now <= float(args.success_dist):
                end = "arrived"
                break
            if p_now[2] < float(args.min_z_m):
                end = "altitude_low"
                break
            if d_now <= best_d - 1.0:
                best_d, best_step = d_now, step
            elif (step - best_step) / step_hz >= float(args.stuck_s):
                end = "stuck"
                break
        p_end = np.asarray(traj[-1])
        d_final = float(np.linalg.norm(p_end - goal))
        d_min = float(min(np.linalg.norm(np.asarray(traj) - goal, axis=1)))
        fam_count: Dict[str, int] = {}
        for s_ in steps:
            fam_count[s_["family"]] = fam_count.get(s_["family"], 0) + 1
        ep = {
            "route_id": r["route_id"],
            "route_len_m": round(ref_len, 1),
            "end_reason": end,
            "arrived": end == "arrived",
            "severe_collision": end == "collision",
            "stuck": end == "stuck",
            "progress_ratio": round(float(np.clip(max(0.0, d0 - d_final) / max(1e-3, ref_len), 0, 1)), 3),
            "collided_at_step": len(traj) - 1 if end == "collision" else None,
        "d_start_m": round(d0, 1),
            "d_final_m": round(d_final, 1),
            "d_min_m": round(d_min, 1),
            "steps": len(traj) - 1,
            "wall_s": round(time.time() - t0, 1),
            "picked_families": fam_count,
            "z_range": [round(float(np.min(np.asarray(traj)[:, 2])), 1), round(float(np.max(np.asarray(traj)[:, 2])), 1)],
            "reset_contact": reset_contact,
            "settle_s": round(settle_s, 2),
            "start_offset_m": round(float(np.linalg.norm(np.asarray(traj[0]) - start)), 2),
            "collision_contact": get_contact() if end == "collision" else None,
        }
        np.savez_compressed(out / f"traj_{r['route_id']}.npz", traj=np.asarray(traj, np.float32), goal=goal)
        jpgs = [x[0] for x in recent]
        np.savez_compressed(
            out / f"last_frames_{r['route_id']}.npz",
            jpeg=np.frombuffer(b"".join(jpgs), dtype=np.uint8),
            jpeg_offsets=np.cumsum([0] + [len(b) for b in jpgs]).astype(np.int64),
            da3_depth56=np.stack([x[1] for x in recent]).astype(np.float16),
            gt_depth56=np.stack([x[2] for x in recent]).astype(np.float16),
        )
        (out / f"steps_{r['route_id']}.json").write_text(json.dumps(steps), encoding="utf-8")
        results["episodes"].append(ep)
        eps = [e for e in results["episodes"] if "arrived" in e]
        results["summary"] = {
            "n": len(eps),
            "arrived": sum(e["arrived"] for e in eps),
            "severe_collision": sum(e["severe_collision"] for e in eps),
            "stuck": sum(e["stuck"] for e in eps),
            "mean_progress_ratio": round(float(np.mean([e["progress_ratio"] for e in eps])), 3),
        }
        res_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        logger.info("%s end=%s prog=%.2f d_final=%.1f steps=%d fam=%s", r["route_id"], end,
                    ep["progress_ratio"], d_final, ep["steps"], fam_count)
    logger.info("SUMMARY %s", results.get("summary"))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
