"""Collect depth-scene expert rollouts (auto OA demos, not PathExpert).

Expert = SceneIntent yaw-fan + D̂ cones + forward body step + yaw-to-subgoal.
Keeps only human-like arrivals: arrived ∧ path_inflate ≤ max ∧ min length.

Urban-interior gate (always):
  in_urban ∧ in_interior ∧ ¬in_water ∧ spawn/path inland thresholds.
Auto-review after each keep; loop until --min-kept quality gate passes.

    # mock smoke:
    python -m experiments.aerial.rl.collect_depth_scene_expert_dataset \\
      --backend mock --episodes 2 --max-steps 80 \\
      --annotation experiments/aerial/tests/fixtures/mini_openfly/seen_mini.json \\
      --out /tmp/dataset_depth_scene_mock --min-kept 1 --until-gate

    # 125 AirSim (interior inland until gate):
    source experiments/aerial/scripts/env_4090.sh
    $AERIAL_PY -m experiments.aerial.rl.collect_depth_scene_expert_dataset \\
      --backend airsim \\
      --annotation experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json \\
      --out experiments/aerial/rl/artifacts/dataset_urban_depth_scene_expert_$(date +%Y%m%d) \\
      --min-spawn-z 38 --max-inflate 1.8 --require-arrived \\
      --require-interior --min-spawn-inland-m 100 --min-path-inland-m 80 \\
      --min-kept 12 --until-gate --max-rounds 40
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from experiments.aerial.phase3_unified.region_geometry import (
    DEFAULT_REGIONS_PATH,
    classify_spawn_xy,
    load_regions,
    path_inland_metrics,
)
from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.collector import RolloutCollector
from experiments.aerial.rl.depth_scene_expert import DepthSceneExpertPolicy
from experiments.aerial.rl.reward import DEFAULT_ONLINE_SUCCESS_DIST_M, RewardConfig
from experiments.aerial.rl.safety import NullSafetyShield
from experiments.aerial.rl.spawn_utils import collect_episode_with_spawn_retries, lift_episode_z
from experiments.aerial.rl.train_rl import _build_env, _build_safety, _load_episodes

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_routes(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _episode_stats(transitions: list, success_dist_m: float) -> Dict[str, Any]:
    if not transitions:
        return {"arrived": False, "path_length_m": 0.0, "d0": 0.0, "inflate": 0.0, "n_interventions": 0}
    goal = None
    for t in transitions:
        for bag in (t.info, getattr(t.obs, "info", {}) or {}):
            if isinstance(bag, dict) and bag.get("goal") is not None:
                goal = np.asarray(bag["goal"], dtype=np.float64).reshape(3)
                break
        if goal is not None:
            break
    positions = []
    n_int = 0
    for t in transitions:
        positions.append(np.asarray(t.obs.position, dtype=np.float64).reshape(3))
        if isinstance(t.info, dict) and t.info.get("intervention"):
            n_int += 1
    last = transitions[-1]
    end = np.asarray(
        last.next_obs.position if last.next_obs is not None else last.obs.position,
        dtype=np.float64,
    ).reshape(3)
    positions.append(end)
    path_len = 0.0
    for a, b in zip(positions[:-1], positions[1:]):
        path_len += float(np.linalg.norm(b - a))
    start = positions[0]
    d0 = float(np.linalg.norm(goal - start)) if goal is not None else path_len
    collided = bool(
        (last.next_obs.collided if last.next_obs is not None else False) or last.obs.collided
    )
    arrived = (
        goal is not None
        and (not collided)
        and float(np.linalg.norm(end - goal)) < float(success_dist_m)
    )
    inflate = (path_len / max(d0, 1e-3)) if d0 > 1e-3 else 0.0

    # Honest failure diagnostics (cheap, no keep-failures/npz needed): where did
    # a not-arrived episode actually go? Distinguishes "never got close" (real
    # blocking obstacle / wrong direction) from "got close then diverged"
    # (overshoot/oscillation) from "crept in a straight line but ran out of
    # budget" (chronic braking, not a wrong turn).
    final_dist_to_goal_m = float(np.linalg.norm(end - goal)) if goal is not None else None
    min_dist_to_goal_m = None
    max_lateral_offset_m = None
    if goal is not None and d0 > 1e-3:
        pts_arr = np.asarray(positions, dtype=np.float64)
        dists = np.linalg.norm(pts_arr - goal.reshape(1, 3), axis=1)
        min_dist_to_goal_m = float(dists.min())
        u = (goal - start)[:2] / d0
        perp = np.array([-u[1], u[0]])
        offs = (pts_arr[:, :2] - start[:2]) @ perp
        max_lateral_offset_m = float(offs[np.argmax(np.abs(offs))])

    return {
        "arrived": bool(arrived),
        "path_length_m": float(path_len),
        "d0": float(d0),
        "inflate": float(inflate),
        "n_interventions": int(n_int),
        "steps": len(transitions),
        "collided": bool(collided),
        "final_dist_to_goal_m": final_dist_to_goal_m,
        "min_dist_to_goal_m": min_dist_to_goal_m,
        "max_lateral_offset_m": max_lateral_offset_m,
        "traj_positions": positions,
    }


def _geo_ok(
    ep: Dict[str, Any],
    regions: Any,
    *,
    require_interior: bool,
    min_spawn_inland_m: float,
    min_path_inland_m: float,
) -> Tuple[bool, Dict[str, Any]]:
    pts = np.asarray(ep.get("pos", []), dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return False, {"reason": "empty_polyline"}
    spawn_meta = classify_spawn_xy(float(pts[0, 0]), float(pts[0, 1]), regions)
    inland = path_inland_metrics(pts, regions)
    reasons: List[str] = []
    if spawn_meta.get("in_water"):
        reasons.append("spawn_in_water")
    if not spawn_meta.get("in_urban"):
        reasons.append("not_urban")
    if require_interior and not spawn_meta.get("in_interior"):
        reasons.append("not_interior")
    if float(spawn_meta.get("spawn_inland_m", 0.0)) < float(min_spawn_inland_m):
        reasons.append(
            f"spawn_inland={spawn_meta.get('spawn_inland_m')}<{min_spawn_inland_m}"
        )
    if float(inland.get("path_min_inland_m", 0.0)) < float(min_path_inland_m):
        reasons.append(
            f"path_inland={inland.get('path_min_inland_m')}<{min_path_inland_m}"
        )
    meta = {
        "spawn_inland_m": float(spawn_meta.get("spawn_inland_m", 0.0)),
        "path_min_inland_m": float(inland.get("path_min_inland_m", 0.0)),
        "in_urban": bool(spawn_meta.get("in_urban")),
        "in_interior": bool(spawn_meta.get("in_interior")),
        "in_water": bool(spawn_meta.get("in_water")),
        "geo_reasons": reasons,
    }
    return (len(reasons) == 0), meta


def _traj_geo_ok(
    positions: List[np.ndarray],
    regions: Any,
    *,
    require_interior: bool,
    min_path_inland_m: float,
) -> Tuple[bool, Dict[str, Any]]:
    if not positions:
        return False, {"reason": "empty_traj"}
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    inland = path_inland_metrics(pts, regions)
    sample = pts[:: max(1, len(pts) // 12)]
    n_out = 0
    for pt in sample:
        cls = classify_spawn_xy(float(pt[0]), float(pt[1]), regions)
        if cls.get("in_water") or not cls.get("in_urban"):
            n_out += 1
        elif require_interior and not cls.get("in_interior"):
            n_out += 1
    frac_bad = float(n_out) / float(max(len(sample), 1))
    ok = float(inland.get("path_min_inland_m", 0.0)) >= float(min_path_inland_m) and frac_bad <= 0.15
    return ok, {
        "traj_path_min_inland_m": float(inland.get("path_min_inland_m", 0.0)),
        "traj_frac_out_interior": round(frac_bad, 3),
    }


def _write_npz(out_dir: Path, index: int, transitions: list, meta: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{index:05d}.npz"
    arrays = ds.episode_arrays(transitions)
    for k, v in meta.items():
        if k == "traj_positions":
            continue
        arrays[k] = np.asarray(v)
    np.savez_compressed(path, **arrays)
    return path


def _dump_review(out_dir: Path, reviews: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "AUTO_REVIEW.json").write_text(
        json.dumps({"summary": summary, "episodes": reviews}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="mock", choices=("mock", "airsim"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=41451)
    p.add_argument("--step-hz", type=float, default=5.0)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--episodes", type=int, default=20, help="routes per round (cap)")
    p.add_argument("--routes", default=None, help="comma route indices into annotation")
    p.add_argument(
        "--annotation",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--r-m", type=float, default=100.0)
    p.add_argument("--cruise-speed", type=float, default=10.0)
    p.add_argument("--success-dist", type=float, default=DEFAULT_ONLINE_SUCCESS_DIST_M)
    p.add_argument("--min-spawn-z", type=float, default=38.0)
    p.add_argument("--spawn-z-retry-m", type=float, default=14.0)
    p.add_argument("--spawn-z-max-retries", type=int, default=3)
    p.add_argument("--max-inflate", type=float, default=1.8)
    p.add_argument("--min-path-m", type=float, default=20.0)
    p.add_argument(
        "--require-arrived",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="only keep arrived episodes that also pass inflate (default: true)",
    )
    p.add_argument("--keep-failures", action="store_true", help="also write non-passing eps")
    p.add_argument("--no-shield", action="store_true", help="null shield (pure expert)")
    p.add_argument(
        "--shield-fwd-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--tti-coeff", type=float, default=2.5)
    p.add_argument(
        "--depth-ckpt",
        default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--config", default="configs/aerial_rl_urban_complex_p2c.yaml")
    p.add_argument("--require-interior", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-spawn-inland-m", type=float, default=100.0)
    p.add_argument("--min-path-inland-m", type=float, default=80.0)
    p.add_argument(
        "--min-kept",
        type=int,
        default=12,
        help="quality gate: number of kept_quality episodes required",
    )
    p.add_argument(
        "--min-kept-per-route",
        type=int,
        default=0,
        help="if >0, each --routes index must keep at least this many (hard-route coverage)",
    )
    p.add_argument(
        "--until-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep collecting rounds until min-kept quality gate passes (default: true)",
    )
    p.add_argument("--max-rounds", type=int, default=40, help="safety cap on until-gate rounds")
    p.add_argument("--append", action="store_true", help="append to existing out dir")
    p.add_argument("--d-clear", type=float, default=40.0, help="scene-intent soft clearance (m)")
    p.add_argument("--d-danger", type=float, default=3.0, help="scene-intent hard danger (m)")
    p.add_argument("--max-dyaw", type=float, default=0.314, help="expert max yaw step (rad)")
    p.add_argument(
        "--descent-radius-m",
        type=float,
        default=0.0,
        help="hold current altitude (clear rooftops) until within this horizontal "
        "distance of goal; 0 disables (legacy blend-toward-goal-z every step)",
    )
    p.add_argument(
        "--replan-period-s",
        type=float,
        default=2.0,
        help="scene-intent hold period between replans (s); larger = more "
        "human-like commitment to a chosen lateral direction, less oscillation",
    )
    p.add_argument(
        "--w-jump",
        type=float,
        default=0.05,
        help="scene-intent penalty for switching away from the previous "
        "candidate; larger = stickier / less back-and-forth",
    )
    p.add_argument(
        "--stuck-escape-after-s",
        type=float,
        default=12.0,
        help="scene-intent: after this many seconds with no real net progress "
        "toward goal, widen the yaw fan (near-reverse headings included) and "
        "pick purely by clearance, ignoring goal-progress scoring; 0 disables",
    )
    p.add_argument(
        "--stuck-escape-hold-s",
        type=float,
        default=5.0,
        help="scene-intent: how long to commit to a stuck-escape heading once chosen",
    )
    p.add_argument(
        "--retreat-max-dyaw-rad",
        type=float,
        default=0.35,
        help="ThreeZoneSpeedShield: turn toward the clearer side while the "
        "emergency/exclusion retreat is engaged (rad/step); 0 = legacy "
        "zero-yaw pure -x retreat (freezes heading against a wall)",
    )
    p.add_argument(
        "--gt-depth-diagnostic",
        action="store_true",
        help="DIAGNOSTIC ONLY (never deploy/train): feed AirSim ground-truth "
        "DepthPlanar to the shield+expert instead of D̂, to isolate whether a "
        "chronic-intervention failure is a D̂ perception bug or a real obstacle. "
        "Forces per-step depth grab (slower).",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[depth-scene-expert] %(message)s")
    root = _repo_root()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    cfg: Dict[str, Any] = {}
    cfg_path = root / args.config
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    env_cfg = dict(cfg.get("env") or {})
    env_cfg.update(
        {
            "backend": args.backend,
            "host": args.host,
            "port": int(args.port),
            "step_hz": float(args.step_hz),
            "grab_depth": bool(args.gt_depth_diagnostic),
        }
    )
    cfg["env"] = env_cfg
    cfg["annotation"] = str(args.annotation)
    cfg["max_episodes"] = 0

    all_episodes = _load_episodes(cfg) or []
    route_idxs = _parse_routes(args.routes)
    if route_idxs is not None:
        pool: List[Tuple[int, Dict[str, Any]]] = [
            (i, all_episodes[i]) for i in route_idxs if 0 <= i < len(all_episodes)
        ]
    else:
        pool = [(i, ep) for i, ep in enumerate(all_episodes[: int(args.episodes)])]
    if not pool:
        logger.error("no episodes")
        return 1

    regions = load_regions(DEFAULT_REGIONS_PATH)
    filtered: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for ri, ep in pool:
        ok, geo = _geo_ok(
            ep,
            regions,
            require_interior=bool(args.require_interior),
            min_spawn_inland_m=float(args.min_spawn_inland_m),
            min_path_inland_m=float(args.min_path_inland_m),
        )
        if not ok:
            logger.warning("route %d PRE-FILTER drop: %s", ri, geo.get("geo_reasons"))
            continue
        filtered.append((ri, lift_episode_z(ep, min_spawn_z=float(args.min_spawn_z)), geo))
    if not filtered:
        logger.error("no urban-interior inland routes after geography filter")
        return 1
    logger.info(
        "geography filter kept %d/%d routes (interior=%s inland>=%.0f/%.0f)",
        len(filtered),
        len(pool),
        args.require_interior,
        args.min_spawn_inland_m,
        args.min_path_inland_m,
    )

    env = _build_env(env_cfg)
    depth_pred = None
    if args.backend == "airsim":
        if args.gt_depth_diagnostic:
            from experiments.aerial.rl.depth_predictor import GTDepthAdapter

            depth_pred = GTDepthAdapter()
            logger.warning(
                "GT-DEPTH DIAGNOSTIC MODE — feeding AirSim ground-truth DepthPlanar "
                "to shield+expert instead of D̂. NOT a deploy/training run."
            )
        else:
            from experiments.aerial.rl.depth_predictor import DepthMinPredictor

            depth_path = root / args.depth_ckpt
            if depth_path.is_file():
                depth_pred = DepthMinPredictor.from_checkpoint(str(depth_path), device=str(args.device))
                logger.info("depth ckpt %s", depth_path)
            else:
                logger.warning("depth ckpt missing %s — scene fan falls back to d_fwd=None", depth_path)

    safety: Any = NullSafetyShield()
    if not args.no_shield and args.backend == "airsim":
        sf = dict(cfg.get("safety") or {})
        sf["tti_coeff"] = float(args.tti_coeff)
        sf["exclusion_forward_only"] = bool(args.shield_fwd_only)
        sf["v_cruise_m_s"] = float(args.cruise_speed)
        safety = _build_safety(sf)
        if hasattr(safety, "retreat_max_dyaw_rad"):
            safety.retreat_max_dyaw_rad = float(args.retreat_max_dyaw_rad)

    policy = DepthSceneExpertPolicy(
        goal_getter=lambda: getattr(env, "goal", None),
        r_m=float(args.r_m),
        cruise_speed=float(args.cruise_speed),
        step_m=float(args.cruise_speed) / float(args.step_hz),
        max_dyaw=float(args.max_dyaw),
        d_clear=float(args.d_clear),
        d_danger=float(args.d_danger),
        descent_radius_m=float(args.descent_radius_m),
        replan_period_s=float(args.replan_period_s),
        w_jump=float(args.w_jump),
        stuck_escape_after_s=float(args.stuck_escape_after_s),
        stuck_escape_hold_s=float(args.stuck_escape_hold_s),
    )
    buf = ReplayBuffer(capacity_episodes=max(8, len(filtered) * 4), seed=0)
    reward_cfg = RewardConfig(
        w_progress=1.0,
        w_collision=1.0,
        w_maneuver=0.01,
        w_eff_strafe=0.05,
        w_eff_heading=0.05,
        w_eff_idle=0.02,
        success_dist_m=float(args.success_dist),
    )
    collector = RolloutCollector(
        env,
        policy,
        buf,
        reward_cfg=reward_cfg,
        safety=safety,
        max_steps=int(args.max_steps),
        target_hz=float(args.step_hz),
        min_spawn_z=float(args.min_spawn_z),
        spawn_z_retry_m=float(args.spawn_z_retry_m),
        spawn_z_max_retries=int(args.spawn_z_max_retries),
        depth_predictor=depth_pred,
        skip_reset_collision=True,
    )

    # Honest-retry grid: DepthSceneExpertPolicy + AirSim are deterministic, so
    # replaying the same route with identical params on every --until-gate
    # round reproduces the identical failing trajectory forever. Cycle a small,
    # logged, deterministic set of parameter variants across rounds so repeated
    # attempts genuinely explore different solutions instead of spinning.
    base_max_steps = int(args.max_steps)
    base_r_m = float(args.r_m)
    base_cruise = float(args.cruise_speed)
    base_d_clear = float(args.d_clear)
    base_d_danger = float(args.d_danger)
    base_descent_radius = float(args.descent_radius_m)
    base_replan_period = float(args.replan_period_s)
    base_w_jump = float(args.w_jump)
    base_yaw_offsets = tuple(policy.intent.yaw_offsets_deg)
    wide_yaw_offsets = tuple(
        sorted(set(base_yaw_offsets) | {-90.0, 90.0, -105.0, 105.0})
    )
    # ROOT CAUSE (hard134, 2026-09-18, confirmed via GT-depth A/B — see
    # test_emergency_retreat_turns_toward_clearer_side): the shield's
    # emergency/exclusion retreat used to command pure -x with dyaw=0, which
    # FREEZES heading while the vehicle is pinned against a wall. Even a
    # perfect (ground-truth-depth) expert then oscillates in place forever
    # (96-99% intervention, near-zero lateral offset, identical camera frame
    # for 800+ steps) because it can never turn while the shield is engaged —
    # not a perception bug, not "needs more time/altitude/commitment". Fixed
    # in safety.py (ThreeZoneSpeedShield: turn toward the clearer cone side
    # while retreating). Post-fix evidence: route 1 and route 3 (previously
    # chronic not_arrived) ARRIVED on the very next `baseline` round.
    # `human_like_commit` (long replan hold + high w_jump) REGRESSED route 1
    # back to not_arrived on the round right after — long hold means fewer
    # chances to re-plan into the now-available escape, so it is deprioritised
    # below the faster-replan / stronger-turn variants that actually help.
    RETRY_GRID: List[Dict[str, Any]] = [
        {"label": "baseline"},
        {
            "label": "fast_replan_escape",
            "replan_period_s": 1.0,
            "w_jump": 0.05,
        },
        {
            "label": "strong_retreat_turn",
            "retreat_max_dyaw_rad": 0.6,
        },
        {
            "label": "fast_replan_strong_turn",
            "replan_period_s": 1.0,
            "retreat_max_dyaw_rad": 0.6,
        },
        # Evidence (hard134 route 0, 2026-09-18): even after the retreat-turn
        # fix, a corner/recess can settle into a STABLE few-metre-off-line
        # equilibrium — progress-weighted scoring keeps pulling it back before
        # it clears the obstacle, regardless of replan cadence or retreat-turn
        # strength (identical ~lat_off across 4+ variants above). Shorten the
        # no-progress trigger (default 12s is long relative to a ~90m route)
        # so the wide-fan / clearance-only escape kicks in well before the
        # step budget runs out.
        {
            "label": "quick_stuck_escape",
            "stuck_escape_after_s": 6.0,
            "stuck_escape_hold_s": 6.0,
        },
        {
            "label": "fast_replan_quick_escape",
            "replan_period_s": 1.0,
            "retreat_max_dyaw_rad": 0.6,
            "stuck_escape_after_s": 6.0,
            "stuck_escape_hold_s": 6.0,
        },
        {
            "label": "wide_fan_more_lookahead",
            "max_steps_mult": 1.3,
            "r_m": base_r_m * 1.3,
            "yaw_offsets_deg": wide_yaw_offsets,
        },
        {
            "label": "more_time_looser_clear",
            "max_steps_mult": 1.6,
            "d_clear": max(20.0, base_d_clear * 0.75),
            "d_danger": max(2.0, base_d_danger * 0.8),
        },
        {
            "label": "slower_cruise_more_time",
            "max_steps_mult": 1.4,
            "cruise_speed": max(4.0, base_cruise * 0.6),
        },
        {
            "label": "human_like_commit",
            "replan_period_s": 5.0,
            "w_jump": 0.35,
            "max_steps_mult": 1.2,
        },
        {
            "label": "human_like_commit_wide_fan",
            "replan_period_s": 5.0,
            "w_jump": 0.35,
            "r_m": base_r_m * 1.3,
            "yaw_offsets_deg": wide_yaw_offsets,
            "max_steps_mult": 1.4,
        },
        {
            "label": "hold_altitude_40m",
            "descent_radius_m": 25.0,
            "max_steps_mult": 1.3,
        },
    ]

    base_retreat_dyaw = float(args.retreat_max_dyaw_rad)
    base_stuck_after = float(args.stuck_escape_after_s)
    base_stuck_hold = float(args.stuck_escape_hold_s)

    def _apply_retry_variant(round_i: int) -> Dict[str, Any]:
        variant = RETRY_GRID[round_i % len(RETRY_GRID)]
        if hasattr(safety, "retreat_max_dyaw_rad"):
            safety.retreat_max_dyaw_rad = float(
                variant.get("retreat_max_dyaw_rad", base_retreat_dyaw)
            )
        policy.intent.stuck_escape_after_s = float(
            variant.get("stuck_escape_after_s", base_stuck_after)
        )
        policy.intent.stuck_escape_hold_s = float(
            variant.get("stuck_escape_hold_s", base_stuck_hold)
        )
        policy.intent.r_m = float(variant.get("r_m", base_r_m))
        policy.intent.d_clear = float(variant.get("d_clear", base_d_clear))
        policy.intent.d_danger = float(variant.get("d_danger", base_d_danger))
        policy.intent.yaw_offsets_deg = variant.get("yaw_offsets_deg", base_yaw_offsets)
        policy.intent.descent_radius_m = float(
            variant.get("descent_radius_m", base_descent_radius)
        )
        policy.intent.replan_period_s = float(
            variant.get("replan_period_s", base_replan_period)
        )
        policy.intent.w_jump = float(variant.get("w_jump", base_w_jump))
        cruise = float(variant.get("cruise_speed", base_cruise))
        policy.intent.cruise_speed = cruise
        policy.step_m = cruise / float(args.step_hz)
        collector.max_steps = int(round(base_max_steps * float(variant.get("max_steps_mult", 1.0))))
        logger.info(
            "round %d retry-variant=%s r_m=%.1f d_clear=%.1f d_danger=%.1f cruise=%.1f "
            "descent_radius=%.1f replan_period_s=%.1f w_jump=%.2f retreat_dyaw=%.2f "
            "stuck_after_s=%.1f stuck_hold_s=%.1f max_steps=%d",
            round_i,
            variant.get("label", "?"),
            policy.intent.r_m,
            policy.intent.d_clear,
            policy.intent.d_danger,
            policy.intent.cruise_speed,
            policy.intent.descent_radius_m,
            policy.intent.replan_period_s,
            policy.intent.w_jump,
            getattr(safety, "retreat_max_dyaw_rad", -1.0),
            policy.intent.stuck_escape_after_s,
            policy.intent.stuck_escape_hold_s,
            collector.max_steps,
        )
        return variant

    manifest: List[Dict[str, Any]] = []
    reviews: List[Dict[str, Any]] = []
    kept = 0
    n_arrived = 0
    quality_kept = 0
    if args.append and (out_dir / "manifest.json").is_file():
        try:
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            if isinstance(manifest, dict):
                manifest = list(manifest.get("episodes") or [])
            reviews = json.loads((out_dir / "AUTO_REVIEW.json").read_text(encoding="utf-8")).get(
                "episodes", []
            )
            kept = len(manifest)
            quality_kept = sum(1 for m in reviews if m.get("keep"))
            n_arrived = sum(1 for m in manifest if m.get("arrived"))
            logger.info("append: existing kept=%d quality=%d", kept, quality_kept)
        except Exception as exc:
            logger.warning("append load failed (%s) — starting fresh", exc)
            manifest, reviews, kept, quality_kept, n_arrived = [], [], 0, 0, 0

    round_i = 0
    max_rounds = int(args.max_rounds) if args.until_gate else 1
    gate = int(args.min_kept)
    min_per = int(args.min_kept_per_route)
    required_routes = [ri for ri, _, _ in filtered]

    def _keep_by_route() -> Dict[int, int]:
        counts: Dict[int, int] = {ri: 0 for ri in required_routes}
        for row in reviews:
            if row.get("keep"):
                r = int(row.get("route_idx", -1))
                if r in counts:
                    counts[r] += 1
        return counts

    def _coverage_ok() -> bool:
        if min_per <= 0:
            return True
        counts = _keep_by_route()
        return all(counts.get(ri, 0) >= min_per for ri in required_routes)

    def _gate_ok() -> bool:
        return quality_kept >= gate and _coverage_ok()

    try:
        while (not _gate_ok()) and round_i < max_rounds:
            round_i += 1
            variant = _apply_retry_variant(round_i - 1)
            cov = _keep_by_route()
            logger.info(
                "=== collect round %d/%d quality_kept=%d/%d per_route=%s (need>=%d) variant=%s ===",
                round_i,
                max_rounds,
                quality_kept,
                gate,
                cov,
                min_per,
                variant.get("label", "?"),
            )
            for local_i, (ri, ep, geo_pre) in enumerate(filtered):
                if _gate_ok() and args.until_gate:
                    break
                # Prefer routes still under per-route quota when coverage is required.
                if min_per > 0 and _keep_by_route().get(ri, 0) >= min_per and not _coverage_ok():
                    short = [r for r, c in _keep_by_route().items() if c < min_per]
                    if short and ri not in short:
                        continue
                policy.reset()
                ep_used, transitions, stats = collect_episode_with_spawn_retries(
                    collector,
                    ep,
                    min_spawn_z=float(args.min_spawn_z),
                    spawn_z_retry_m=float(args.spawn_z_retry_m),
                    spawn_z_max_retries=int(args.spawn_z_max_retries),
                )
                del ep_used
                if stats is None or stats.skipped or not transitions:
                    logger.info("route %d skipped (spawn)", ri)
                    reviews.append(
                        {
                            "route_idx": ri,
                            "round": round_i,
                            "keep": False,
                            "drop_reason": "spawn_skip",
                            **geo_pre,
                        }
                    )
                    _dump_review(
                        out_dir,
                        reviews,
                        {"quality_kept": quality_kept, "gate": gate, "round": round_i},
                    )
                    continue
                q = _episode_stats(transitions, float(args.success_dist))
                traj_ok, traj_geo = _traj_geo_ok(
                    list(q.pop("traj_positions")),
                    regions,
                    require_interior=bool(args.require_interior),
                    min_path_inland_m=float(args.min_path_inland_m),
                )
                if q["arrived"]:
                    n_arrived += 1
                pass_q = True
                drop_reasons: List[str] = []
                if args.require_arrived and not args.keep_failures and not q["arrived"]:
                    pass_q = False
                    drop_reasons.append("not_arrived")
                if pass_q and q["inflate"] > float(args.max_inflate):
                    pass_q = False
                    drop_reasons.append(f"inflate={q['inflate']:.2f}>{args.max_inflate}")
                if pass_q and q["path_length_m"] < float(args.min_path_m) and q["arrived"]:
                    pass_q = False
                    drop_reasons.append("path_too_short")
                if not traj_ok:
                    pass_q = False
                    drop_reasons.append(
                        f"traj_geo frac_out={traj_geo.get('traj_frac_out_interior')} "
                        f"inland={traj_geo.get('traj_path_min_inland_m')}"
                    )

                review_row = {
                    "route_idx": ri,
                    "local_idx": local_i,
                    "round": round_i,
                    "retry_variant": variant.get("label", "?"),
                    "arrived": q["arrived"],
                    "inflate": q["inflate"],
                    "path_length_m": q["path_length_m"],
                    "d0": q["d0"],
                    "steps": q["steps"],
                    "final_dist_to_goal_m": q.get("final_dist_to_goal_m"),
                    "min_dist_to_goal_m": q.get("min_dist_to_goal_m"),
                    "max_lateral_offset_m": q.get("max_lateral_offset_m"),
                    "n_interventions": q.get("n_interventions"),
                    "intervention_frac": (
                        float(q["n_interventions"]) / max(1, int(q["steps"]))
                        if q.get("n_interventions") is not None
                        else None
                    ),
                    "keep": bool(pass_q),
                    "drop_reason": ",".join(drop_reasons) if drop_reasons else "",
                    **geo_pre,
                    **traj_geo,
                }
                reviews.append(review_row)
                logger.info(
                    "AUTO-REVIEW route %d arr=%s inflate=%.2f L=%.1f final_d=%.1f "
                    "min_d=%.1f lat_off=%.1f interventions=%d/%d(%.0f%%) keep=%s %s",
                    ri,
                    q["arrived"],
                    q["inflate"],
                    q["path_length_m"],
                    q.get("final_dist_to_goal_m") or -1.0,
                    q.get("min_dist_to_goal_m") or -1.0,
                    q.get("max_lateral_offset_m") or 0.0,
                    q.get("n_interventions") or 0,
                    q["steps"],
                    100.0 * (review_row.get("intervention_frac") or 0.0),
                    pass_q,
                    review_row["drop_reason"] or "ok",
                )

                if pass_q or args.keep_failures:
                    meta = {
                        "arrived": q["arrived"],
                        "path_inflate": q["inflate"],
                        "path_length_m": q["path_length_m"],
                        "d0_m": q["d0"],
                        "expert": "depth_scene",
                        "kept_quality": bool(pass_q),
                        "route_idx": int(ri),
                        "spawn_inland_m": geo_pre.get("spawn_inland_m"),
                        "path_min_inland_m": geo_pre.get("path_min_inland_m"),
                        "in_interior": geo_pre.get("in_interior"),
                    }
                    path = _write_npz(out_dir, kept, transitions, meta)
                    manifest.append(
                        {
                            "file": path.name,
                            "route_idx": int(ri),
                            "route_local_idx": local_i,
                            "round": round_i,
                            **{k: v for k, v in q.items() if k != "traj_positions"},
                            "kept_quality": bool(pass_q),
                            **{k: geo_pre[k] for k in ("spawn_inland_m", "path_min_inland_m", "in_interior")},
                        }
                    )
                    kept += 1
                    if pass_q:
                        quality_kept += 1

                _dump_review(
                    out_dir,
                    reviews,
                    {
                        "quality_kept": quality_kept,
                        "gate": gate,
                        "round": round_i,
                        "n_written": kept,
                        "n_arrived": n_arrived,
                    },
                )
                (out_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )

            if not args.until_gate:
                break
            if not _gate_ok():
                logger.warning(
                    "gate NOT met kept=%d/%d per_route=%s after round %d — continuing",
                    quality_kept,
                    gate,
                    _keep_by_route(),
                    round_i,
                )
    finally:
        try:
            env.close()
        except Exception:
            pass

    gate_ok = _gate_ok()
    summary = {
        "kind": "depth_scene_expert",
        "n_requested_pool": len(filtered),
        "n_written": kept,
        "n_arrived": n_arrived,
        "quality_kept": quality_kept,
        "min_kept_gate": gate,
        "min_kept_per_route": min_per,
        "keep_by_route": _keep_by_route(),
        "gate_passed": gate_ok,
        "rounds": round_i,
        "max_inflate": float(args.max_inflate),
        "require_arrived": bool(args.require_arrived),
        "require_interior": bool(args.require_interior),
        "min_spawn_inland_m": float(args.min_spawn_inland_m),
        "min_path_inland_m": float(args.min_path_inland_m),
        "annotation": str(args.annotation),
        "routes": route_idxs,
        "shield": "null" if args.no_shield else "three_zone_fwd",
        "note": (
            "urban-interior inland only; auto-review each ep; "
            "until-gate keeps collecting until min_kept quality passes"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "path_expert_meta.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "QUALITY_SUMMARY.json").write_text(
        json.dumps(
            {
                "episodes": kept,
                "arrived": n_arrived,
                "quality_kept": quality_kept,
                "gate": gate,
                "gate_passed": gate_ok,
                "mean_inflate": (
                    float(np.mean([m["inflate"] for m in manifest if m.get("kept_quality")]))
                    if any(m.get("kept_quality") for m in manifest)
                    else 0.0
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _dump_review(out_dir, reviews, summary)
    logger.info(
        "wrote %d eps quality_kept=%d/%d gate_passed=%s -> %s",
        kept,
        quality_kept,
        gate,
        gate_ok,
        out_dir,
    )
    if args.until_gate and not gate_ok:
        logger.error("STOPPED at max_rounds without passing gate — resume with --append")
        return 3
    return 0 if quality_kept > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
