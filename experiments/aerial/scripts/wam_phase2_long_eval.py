#!/usr/bin/env python3
"""Phase 2 long-horizon evaluator (goal+scene E0/E1).

Stack:
  subgoal_source ∈ {polyline, toward_g, direct_g, scene}
    → LatentActorDeployPolicy → optional ImaginationPlanner
    → ThreeZoneSpeedShield → env.step

Main product path: toward_g / scene (no GT polyline carrot).
polyline / --rolling-global remain waterline对照 (default polyline until E0 DECLARE).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("wam_phase2_eval")


def _goal_dist(pos: np.ndarray, goal: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.asarray(goal, dtype=np.float64).reshape(3)
            - np.asarray(pos, dtype=np.float64).reshape(3)
        )
    )


def _goal_closure(d_start_m: float, d_min_m: float) -> float:
    """Honest Euclidean closure toward G: 1 - d_min/d_start, clipped to [0, 1]."""
    d0 = float(max(1e-3, d_start_m))
    return float(np.clip(1.0 - float(d_min_m) / d0, 0.0, 1.0))


def _monotone_inflate(progress_ratio: float, d_min_m: float, *, prog_min: float = 0.9, d_min_floor_m: float = 30.0) -> bool:
    """True when arc-s Prog looks near-done but Euclidean d_min stays far from G."""
    return bool(float(progress_ratio) >= float(prog_min) and float(d_min_m) >= float(d_min_floor_m))


def _episode_better(candidate: dict, incumbent: dict) -> bool:
    """Prefer arrived episodes; else lower d_min_m."""
    if candidate.get("arrived") and not incumbent.get("arrived"):
        return True
    if incumbent.get("arrived") and not candidate.get("arrived"):
        return False
    return float(candidate["d_min_m"]) < float(incumbent["d_min_m"])


def _segment_min_dist(p0: np.ndarray, p1: np.ndarray, goal: np.ndarray) -> float:
    p0_arr = np.asarray(p0, dtype=np.float64).reshape(3)
    p1_arr = np.asarray(p1, dtype=np.float64).reshape(3)
    g = np.asarray(goal, dtype=np.float64).reshape(3)
    v = p1_arr - p0_arr
    v_sq = float(np.sum(v**2))
    if v_sq < 1e-8:
        return float(np.linalg.norm(p0_arr - g))
    t = float(np.clip(np.dot(g - p0_arr, v) / v_sq, 0.0, 1.0))
    return float(np.linalg.norm(p0_arr + t * v - g))


# L0 PASS gates: Euclidean arrival (SR) + safety only.
# SPL is diagnostic: toward_g in dense forest without a map physically cannot
# match hand-flown reference paths (typical L_act ≈ 3× L_ref → SPL ≈ 0.33).
# IR is diagnostic: any shield change (hard brake OR soft TTI governor).
# Prefer hard_brake_rate for "takeover"; governor_cap_rate is speed-rule only.
# Rates above 0.25 are expected in cluttered urban / dense forest at cs=10
# with tti_coeff=2.5.
PASS_THRESHOLDS: Dict[str, float] = {
    "arrival_rate_min": 0.80,
    "spl_min": 0.30,                       # diagnostic floor (not a hard gate)
    "severe_collision_rate_max": 0.10,
    "mean_intervention_rate_max": 0.50,    # diagnostic; 0.25 is too tight for forest
    "mean_progress_ratio_diagnostic_only": 0.90,
}


def aggregate_metrics(
    results: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], str]:
    """Episode dicts → (scored, spawn_fails, metrics, verdict).

    Shared with the split-run merger (`merge_phase2_split_eval.py`) so a row
    merged from two boxes can never drift from a single-box row.
    progress_ratio stays diagnostic; it must not cosplay as near-success.
    """
    scored = [r for r in results if not r.get("spawn_fail")]
    spawn_fails = [r for r in results if r.get("spawn_fail")]

    def _mean(key: str) -> float:
        return float(np.mean([r[key] for r in scored])) if scored else 0.0

    sr = _mean("arrived")
    scr = _mean("severe_collision")
    mean_spl = _mean("spl")
    n_replans = int(sum(r.get("n_intent_replans", 0) for r in scored))
    n_offaxis = int(sum(r.get("n_intent_offaxis", 0) for r in scored))
    metrics: Dict[str, Any] = {
        "arrival_rate": round(sr, 4),
        "spl": round(mean_spl, 4),
        "severe_collision_rate": round(scr, 4),
        "mean_goal_closure": round(_mean("goal_closure"), 4),
        "n_monotone_inflate": int(sum(1 for r in scored if r.get("monotone_inflate"))),
        "mean_progress_ratio": round(_mean("progress_ratio"), 4),
        "mean_intervention_rate": round(_mean("intervention_rate"), 4),
        "mean_hard_brake_rate": round(_mean("hard_brake_rate"), 4),
        "mean_governor_cap_rate": round(_mean("governor_cap_rate"), 4),
        "mean_global_replans": (
            round(float(np.mean([r.get("n_global_replans", 0) for r in scored])), 2)
            if scored
            else 0.0
        ),
        # E1 diagnostics only — never gate on these.
        "n_intent_replans": n_replans,
        "n_intent_offaxis": n_offaxis,
        "intent_offaxis_frac": round(n_offaxis / n_replans, 4) if n_replans else 0.0,
        "max_intent_dev_deg": round(
            max((r.get("max_intent_dev_deg", 0.0) for r in scored), default=0.0), 2
        ),
    }
    verdict = (
        "PASS"
        if (
            sr >= PASS_THRESHOLDS["arrival_rate_min"]
            and scr <= PASS_THRESHOLDS["severe_collision_rate_max"]
        )
        else "FAIL"
    )
    return scored, spawn_fails, metrics, verdict


def select_route_indices(n_available: int, episodes: int, routes_arg: str | None) -> List[int]:
    """Which annotation rows this box owns. `--routes` overrides `--episodes`."""
    if not routes_arg:
        return list(range(min(int(episodes), int(n_available))))
    idxs = [int(t) for t in str(routes_arg).split(",") if t.strip()]
    out_of_range = [i for i in idxs if not 0 <= i < int(n_available)]
    if out_of_range:
        raise SystemExit(
            f"refuse: --routes {out_of_range} out of range (annotation has {n_available})"
        )
    if len(set(idxs)) != len(idxs):
        raise SystemExit(f"refuse: --routes has duplicates: {idxs}")
    return idxs


def format_summary_line(summary: Dict[str, Any], *, prefix: str) -> str:
    m = summary["metrics"]
    line = (
        f"{prefix} Verdict={summary['verdict']} | "
        f"SR={m['arrival_rate']*100:.1f}% SPL={m['spl']*100:.1f}% "
        f"SCR={m['severe_collision_rate']*100:.1f}% "
        f"closure={m['mean_goal_closure']*100:.1f}% "
        f"inflate={m['n_monotone_inflate']}/{summary['n_scored']} "
        f"Prog={m['mean_progress_ratio']*100:.1f}% (diag) "
        f"IR={m['mean_intervention_rate']*100:.1f}%"
        f"(HB={m.get('mean_hard_brake_rate', 0.0)*100:.1f}%"
        f"/GC={m.get('mean_governor_cap_rate', 0.0)*100:.1f}%)"
    )
    if m.get("n_intent_replans"):
        line += (
            f" | replan={m['n_intent_replans']} offaxis={m['n_intent_offaxis']}"
            f" ({m['intent_offaxis_frac']*100:.1f}%)"
            f" maxdev={m['max_intent_dev_deg']:.1f}deg"
        )
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 mainline long-horizon eval")
    parser.add_argument("--config", default="configs/aerial_rl.yaml")
    parser.add_argument(
        "--wm-ckpt",
        default="experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt",
    )
    parser.add_argument(
        "--actor-ckpt",
        default="experiments/aerial/rl/artifacts/v4_ac_ckpt_step_e_20260828/v4_ac_latest.pt",
    )
    parser.add_argument("--annotation", default="artifacts/seen_airsim16_long_routes.json")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument(
        "--routes",
        type=str,
        default=None,
        help=(
            "Comma-separated 0-based route indices into the annotation, e.g. '0,1'. "
            "Overrides --episodes so two boxes can split one arm; merge the JSONs "
            "with merge_phase2_split_eval.py before filling a DECLARE row."
        ),
    )
    parser.add_argument("--step-hz", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--cruise-speed", type=float, default=25.0)
    parser.add_argument(
        "--a-max",
        type=float,
        default=None,
        help="Override ThreeZoneShield a_max_m_s2 (default: use config/2.5). "
             "E.g. --a-max 5 halves engage_outer at cs=25 (134→72 m).",
    )
    parser.add_argument(
        "--tti-coeff",
        type=float,
        default=None,
        help="Override ThreeZoneSpeedShield tti_coeff (default: 4.0). "
             "Trigger distance = tti_coeff × v_ref. Lower = shield fires later.",
    )
    parser.add_argument(
        "--shield-tti-hysteresis",
        type=float,
        default=0.0,
        help="Forward TTI cap release hysteresis frac (0=OFF). Hold cap until "
             "d_fwd >= trigger × (1 + frac).",
    )
    parser.add_argument(
        "--shield-tti-relax-on-corridor",
        action="store_true",
        default=False,
        help="On straight corridor (low CTE), scale tti_coeff down for fewer brakes.",
    )
    parser.add_argument(
        "--shield-tti-relax-scale",
        type=float,
        default=0.82,
        help="Multiply base tti_coeff when --shield-tti-relax-on-corridor (lower=fewer caps).",
    )
    parser.add_argument(
        "--shield-tti-relax-cte-m",
        type=float,
        default=1.5,
        help="CTE gate for corridor tti relax (m).",
    )
    parser.add_argument(
        "--shield-tti-relax-rem-m",
        type=float,
        default=25.0,
        help="Only relax tti when rem_dist exceeds this (m).",
    )
    parser.add_argument(
        "--no-shield",
        action="store_true",
        help="Disable safety shield (NullSafetyShield). For ablation vs three_zone.",
    )
    parser.add_argument(
        "--shield-exclusion-forward-only",
        action="store_true",
        help="Exclusion zone uses forward cone only, not full-FOV depth_min_pred.",
    )
    parser.add_argument("--success-dist", type=float, default=3.0)
    parser.add_argument(
        "--save-dataset",
        type=str,
        default=None,
        help="If set, write per-route expert rollouts to this directory (episode_XXXXX.npz)",
    )
    parser.add_argument(
        "--save-dataset-append",
        action="store_true",
        help="Do not wipe existing episode_*.npz; write at next free index",
    )
    parser.add_argument(
        "--save-arrived-only",
        action="store_true",
        help="Only write npz when episode arrived (teacher collect)",
    )
    parser.add_argument(
        "--stuck-abort-after-s",
        type=float,
        default=0.0,
        help="Abort episode after this many seconds with <stuck-abort-progress-m "
             "improvement in rem-band (0=off). Speeds failed R0 teacher attempts.",
    )
    parser.add_argument(
        "--stuck-abort-progress-m",
        type=float,
        default=1.0,
        help="Min d_to_goal improvement to reset stuck-abort timer",
    )
    parser.add_argument(
        "--stuck-abort-rem-lo",
        type=float,
        default=25.0,
        help="Only apply stuck-abort when d_to_goal >= this (m)",
    )
    parser.add_argument(
        "--stuck-abort-rem-hi",
        type=float,
        default=45.0,
        help="Only apply stuck-abort when d_to_goal <= this (m)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--planner", action="store_true")
    parser.add_argument("--planner-horizon", type=int, default=5)
    parser.add_argument(
        "--planner-rollout",
        choices=("open_loop", "closed_loop"),
        default="open_loop",
        help="open_loop repeats the candidate for H steps (hand-rule path, frozen). "
             "closed_loop uses the candidate only at step 0, then the actor; "
             "score is WM return only.",
    )
    parser.add_argument(
        "--planner-mock",
        choices=("pass", "rules", "wm_bare", "wm_escape"),
        default=None,
        help="Planner ablations: pass=return π unchanged; "
             "rules=same candidates+hand biases, geometric progress (no WM); "
             "wm_bare=same candidates, WM return only (no hand biases); "
             "wm_escape=WM return + open-side escape/stuck bias only.",
    )
    parser.add_argument(
        "--goal-feat-mode",
        choices=("meter", "g_norm"),
        default="meter",
        help="Actor goal conditioning: metre goal_rel (Step E) or F9 g_norm",
    )
    parser.add_argument(
        "--depth-ckpt",
        default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt",
    )
    parser.add_argument(
        "--use-gt-depth",
        action="store_true",
        help="E3 upper bound: ThreeZone/scene depth from AirSim GT DepthPlanar "
        "(GTDepthAdapter) instead of D̂ checkpoint. Never for deploy claims.",
    )
    parser.add_argument(
        "--tau-ckpt",
        default="experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--out", default="artifacts/wam_phase2_accept_result.json")
    parser.add_argument(
        "--spawn-tol-m",
        type=float,
        default=12.0,
        help="F1: max ||p_reset - start_pos||; beyond → spawn_fail skip (no SCR inflate)",
    )
    parser.add_argument(
        "--min-spawn-z",
        type=float,
        default=24.0,
        help="Uniform z-lift on annotated polyline before reset (urban flyability)",
    )
    parser.add_argument(
        "--spawn-z-retry-m",
        type=float,
        default=5.0,
        help="On spawn fail/collision, raise polyline z by this per retry",
    )
    parser.add_argument(
        "--spawn-z-max-retries",
        type=int,
        default=4,
        help="Spawn retries after min-spawn-z lift",
    )
    parser.add_argument(
        "--min-spawn-clear-m",
        type=float,
        default=5.0,
        help="Reject spawn if GT forward clearance < this (0 disables); retry z/xy",
    )
    parser.add_argument(
        "--heading-assist",
        action="store_true",
        default=False,
        help="F7 fuse: path-tangent dyaw (OFF by default; mainline SR must not rely on this)",
    )
    parser.add_argument(
        "--heading-assist-cte-max-m",
        type=float,
        default=8.0,
        help="Heading assist only when CTE <= this (m)",
    )
    parser.add_argument(
        "--heading-assist-cos-thr",
        type=float,
        default=0.7,
        help="Heading assist when cos(heading,tangent) < this",
    )
    parser.add_argument(
        "--heading-assist-lateral-scale",
        type=float,
        default=0.25,
        help="Scale body-lateral dy when cos(heading,tangent) < 0.3 during assist",
    )
    parser.add_argument(
        "--cte-reentry-m",
        type=float,
        default=None,
        help="AdaptiveSubgoal cte_reentry_m (default 2.0)",
    )
    parser.add_argument(
        "--cte-lock-freeze-m",
        type=float,
        default=None,
        help="AdaptiveSubgoal cte_lock_freeze_m (default 5.0)",
    )
    parser.add_argument(
        "--heading-reentry-cos",
        type=float,
        default=None,
        help="AdaptiveSubgoal heading_reentry_cos peel threshold (default 0.7)",
    )
    parser.add_argument(
        "--lookahead-feedback",
        action="store_true",
        default=False,
        help="L1 DECLARE: no-progress / CTE feedback on carrot (OFF by default)",
    )
    parser.add_argument(
        "--rolling-global",
        action="store_true",
        default=False,
        help="P0 receding GlobalRefPlanner: carrot on short P_ref (OFF by default)",
    )
    parser.add_argument(
        "--global-horizon-m",
        type=float,
        default=60.0,
        help="GlobalRefPlanner forward horizon (m)",
    )
    parser.add_argument(
        "--global-replan-period-s",
        type=float,
        default=1.0,
        help="GlobalRefPlanner replan period (s); 0 = every step",
    )
    parser.add_argument(
        "--subgoal-source",
        type=str,
        default="polyline",
        choices=("polyline", "toward_g", "direct_g", "scene"),
        help=(
            "polyline=waterline; toward_g=main (clip toward G, policy handles obstacles); "
            "scene=alias for toward_g (fan intent removed; see E1r2 analysis); "
            "direct_g=ablation A"
        ),
    )
    parser.add_argument(
        "--r-m-intent",
        type=float,
        default=100.0,
        help="TowardGoalIntent / SceneIntentPlanner clip radius (m); default 100",
    )
    parser.add_argument(
        "--traj-out",
        default=None,
        help="If set, write per-step JSONL trace (pos, intent_target, d_fwd, "
             "chosen_idx, dev_deg, yaw, replan) for trajectory forensics.",
    )
    parser.add_argument(
        "--terminal-pin-rem-m",
        type=float,
        default=20.0,
        help="Pin subgoal to route goal when rem_dist <= this (m); 0=OFF",
    )
    parser.add_argument(
        "--terminal-creep-rem-m",
        type=float,
        default=None,
        help="Bleed speed when rem_dist <= this (m); default=subgoal generator (8)",
    )
    parser.add_argument(
        "--near-miss-retry-m",
        type=float,
        default=0.0,
        help="If d_min <= this but not arrived, retry episode (0=OFF)",
    )
    parser.add_argument(
        "--near-miss-retries",
        type=int,
        default=1,
        help="Max near-miss retries per route (each retry is a full reset)",
    )
    parser.add_argument(
        "--recover-on-near-miss-retry",
        action="store_true",
        default=False,
        help="Run recover_renderer before each near-miss retry (AirSim drift).",
    )
    parser.add_argument(
        "--recover-script",
        default=None,
        help="recover_renderer.sh path for --recover-on-near-miss-retry",
    )
    parser.add_argument(
        "--terminal-direct-rem-m",
        type=float,
        default=0.0,
        help="When rem_dist<=this and d_to_goal<=terminal-direct-d-m, subgoal=goal (0=OFF)",
    )
    parser.add_argument(
        "--terminal-direct-d-m",
        type=float,
        default=20.0,
        help="Euclidean gate paired with --terminal-direct-rem-m",
    )
    parser.add_argument(
        "--terminal-shield-tti-rem-m",
        type=float,
        default=0.0,
        help="When rem_dist<=this and CTE low, scale up tti_coeff (fewer terminal brakes).",
    )
    parser.add_argument(
        "--terminal-shield-tti-cte-m",
        type=float,
        default=4.0,
        help="CTE gate for --terminal-shield-tti-rem-m",
    )
    parser.add_argument(
        "--terminal-shield-tti-scale",
        type=float,
        default=1.25,
        help="Multiply tti_coeff under terminal shield relax",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch
    from experiments.aerial.rl.actor_critic import (
        LatentActorCritic,
        LatentActorDeployPolicy,
    )
    from experiments.aerial.rl.env.action import body_delta_limits, clip_body_delta
    from experiments.aerial.rl.path_heading_assist import apply_path_heading_assist
    from experiments.aerial.rl.goal_features import body_vel_from_obs
    from experiments.aerial.rl.planner import ImaginationPlanner
    from experiments.aerial.rl.reward import RewardConfig
    from experiments.aerial.rl.global_ref_planner import GlobalRefConfig, GlobalRefPlanner
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor, GTDepthAdapter
    from experiments.aerial.rl.tau_predictor import make_tau_predictor
    from experiments.aerial.rl.scene_intent import SceneIntentPlanner, TowardGoalIntent
    from experiments.aerial.rl.subgoal_generator import (
        AdaptiveSubgoalGenerator,
        nearest_on_polyline,
    )
    from experiments.aerial.rl.train_rl import (
        _build_env,
        _build_safety,
        load_torch_dynamics,
    )

    cfg_file = (root / args.config).resolve()
    cfg = yaml.safe_load(cfg_file.read_text()) if cfg_file.is_file() else {}

    device_str = "cpu" if (args.mock or not torch.cuda.is_available()) else args.device
    device = torch.device(device_str)
    logger.info(f"Using device: {device} (mock={args.mock})")

    anno_path = (
        (root / args.annotation).resolve()
        if not Path(args.annotation).is_absolute()
        else Path(args.annotation)
    )
    with open(anno_path, "r", encoding="utf-8") as f:
        anno_data = json.load(f)
    if isinstance(anno_data, dict):
        routes = anno_data.get("routes") or anno_data.get("episodes")
        if routes is None:
            raise SystemExit(f"annotation {anno_path} missing routes/episodes list")
    else:
        routes = anno_data
    route_idxs = select_route_indices(len(routes), args.episodes, args.routes)
    n_routes = len(route_idxs)

    env_cfg = dict(cfg.get("env") or {})
    env_cfg["backend"] = "mock" if args.mock else "airsim"
    env_cfg["step_hz"] = float(args.step_hz)
    env_cfg["grab_depth"] = True
    env = _build_env(env_cfg)

    wm_cfg = cfg.get("world_model") or {}
    wm_path = (
        (root / args.wm_ckpt).resolve()
        if not Path(args.wm_ckpt).is_absolute()
        else Path(args.wm_ckpt)
    )
    dynamics, _ = load_torch_dynamics(
        wm_cfg, str(wm_path), device=device_str, success_dist_m=float(args.success_dist)
    )

    actor_path = (
        (root / args.actor_ckpt).resolve()
        if not Path(args.actor_ckpt).is_absolute()
        else Path(args.actor_ckpt)
    )
    if not args.mock and actor_path.exists():
        actor_ac = LatentActorCritic.load_from_checkpoint(actor_path, device=device_str)
        # Re-anchor: Step E expects metre features; CLI overrides ckpt default.
        actor_ac.config.goal_feat_mode = str(args.goal_feat_mode)
        logger.info(
            "Loaded actor-critic from %s (goal_feat_mode=%s)",
            actor_path,
            actor_ac.config.goal_feat_mode,
        )
    else:
        actor_ac = LatentActorCritic.from_config(
            {"latent_dim": dynamics.latent_dim, "device": device_str}
        )

    depth_path = (
        (root / args.depth_ckpt).resolve()
        if not Path(args.depth_ckpt).is_absolute()
        else Path(args.depth_ckpt)
    )
    if bool(getattr(args, "use_gt_depth", False)):
        depth_pred = GTDepthAdapter()
        logger.warning("E3: using GTDepthAdapter (AirSim GT depth) — not a deploy claim")
        depth_path = Path("GTDepthAdapter")
    else:
        depth_pred = (
            DepthMinPredictor.from_checkpoint(depth_path, device=device_str)
            if (not args.mock and depth_path.is_file())
            else None
        )
        if depth_pred is None and not args.mock:
            raise SystemExit(
                f"ABORT: depth checkpoint not found at {depth_path}\n"
                "ThreeZoneShield forward cap requires depth. "
                "Pass --depth-ckpt <path>, --use-gt-depth, or --mock."
            )

    tau_path = (
        (root / args.tau_ckpt).resolve()
        if not Path(args.tau_ckpt).is_absolute()
        else Path(args.tau_ckpt)
    )
    tau_pred = make_tau_predictor(
        kind="foe_calibrated",
        ckpt=tau_path if (not args.mock and tau_path.is_file()) else None,
        device=device_str,
    )
    if not args.mock and not tau_path.is_file():
        logger.warning(
            "tau checkpoint not found at %s — running uncalibrated FOE tau (τ latch active but less accurate)",
            tau_path,
        )

    phys_limits = body_delta_limits(1.0 / float(args.step_hz))
    vx_max_step = float(
        min(float(args.cruise_speed) / float(args.step_hz), float(phys_limits[0]))
    )
    action_limits = np.array(
        [
            vx_max_step,
            float(phys_limits[1]),
            float(phys_limits[2]),
            float(phys_limits[3]),
        ],
        dtype=np.float64,
    )

    reward_cfg = RewardConfig(**(cfg.get("reward") or {}))
    reward_cfg.success_dist_m = float(args.success_dist)

    policy = LatentActorDeployPolicy(
        dynamics, actor_ac, deterministic=True, stream_latent=True
    )

    planner = None
    if args.planner:
        planner = ImaginationPlanner(
            dynamics=dynamics,
            horizon=int(args.planner_horizon),
            reward_cfg=reward_cfg,
            action_limits=action_limits,
            mock_mode=args.planner_mock,
            rollout_mode=str(args.planner_rollout),
            tail_policy=policy if str(args.planner_rollout) == "closed_loop" else None,
        )
        if args.planner_mock:
            logger.info("planner mock_mode=%s", args.planner_mock)
        logger.info(
            "planner rollout=%s horizon=%d hand_bias=%s",
            args.planner_rollout,
            int(args.planner_horizon),
            "off" if str(args.planner_rollout) == "closed_loop" else "on",
        )

    safety_cfg = dict(cfg.get("safety") or {})
    if bool(args.no_shield):
        safety_cfg["kind"] = "null"
        logger.warning("safety disabled (--no-shield)")
    elif str(safety_cfg.get("kind", "null")) in ("null", "none", "None"):
        safety_cfg["kind"] = "three_zone"
        logger.warning("safety.kind was null — forcing three_zone for mainline Phase 2")
    if bool(args.shield_exclusion_forward_only):
        safety_cfg["exclusion_forward_only"] = True
        logger.info("shield exclusion: forward cone only (ignore full-FOV min)")
    # Keep three-zone cruise assumption aligned with --cruise-speed (engage scales as v²).
    safety_cfg["v_cruise_m_s"] = float(args.cruise_speed)
    if args.a_max is not None:
        safety_cfg["a_max_m_s2"] = float(args.a_max)
    if args.tti_coeff is not None:
        safety_cfg["tti_coeff"] = float(args.tti_coeff)
    if float(args.shield_tti_hysteresis) > 0.0:
        safety_cfg["tti_hysteresis_release_frac"] = float(args.shield_tti_hysteresis)
    safety_cfg.pop("schedule_margin_l1_m", None)
    safety_cfg.pop("schedule_margin_l2_m", None)
    safety_cfg.pop("disc_lag_steps", None)
    shield = _build_safety(safety_cfg)
    if hasattr(shield, "zone"):
        logger.info(
            "three_zone v_cruise=%.1f engage_outer=%.1fm margins L1/L2=%.2f/%.2f tti_coeff=%.1f",
            float(shield.zone.v_cruise_m_s),
            float(shield.zone.engage_outer_m),
            float(shield.zone.schedule_margin_l1_m),
            float(shield.zone.schedule_margin_l2_m),
            float(shield.tti_coeff),
        )

    # Local carrot for step_e π (H1 + P1 sweep 2026-09-01): r_base=25 /
    # cte_reentry=2 passed R01 ds>=25 & cte_end<=15; long routes still slide.
    sg_kw: Dict[str, Any] = {
        "r_base": 25.0 if args.cruise_speed >= 8.0 else 20.0,
        "r_min": 15.0 if args.cruise_speed >= 8.0 else 12.0,
        "d_clear": 22.0 if args.cruise_speed >= 8.0 else 12.0,
        "d_danger": 3.0,
        "cruise_speed": args.cruise_speed,
        "cte_reentry_m": 2.0,
        "lookahead_feedback": bool(args.lookahead_feedback),
    }
    if args.cte_reentry_m is not None:
        sg_kw["cte_reentry_m"] = float(args.cte_reentry_m)
    if args.cte_lock_freeze_m is not None:
        sg_kw["cte_lock_freeze_m"] = float(args.cte_lock_freeze_m)
    if args.heading_reentry_cos is not None:
        sg_kw["heading_reentry_cos"] = float(args.heading_reentry_cos)
    if float(args.terminal_pin_rem_m) > 0.0:
        sg_kw["terminal_pin_rem_m"] = float(args.terminal_pin_rem_m)
    if args.terminal_creep_rem_m is not None:
        sg_kw["terminal_creep_rem_m"] = float(args.terminal_creep_rem_m)
    subgoal_gen = AdaptiveSubgoalGenerator(**sg_kw)
    if float(args.terminal_pin_rem_m) > 0.0:
        logger.info("terminal_pin_rem_m=%.1f", float(args.terminal_pin_rem_m))
    if args.lookahead_feedback:
        logger.info("L1 lookahead_feedback=ON (opt-in; mainline default remains OFF)")

    subgoal_source = str(args.subgoal_source)
    intent: Any = None
    r_intent = float(args.r_m_intent)
    if subgoal_source in ("toward_g", "direct_g"):
        intent = TowardGoalIntent(
            r_m=r_intent,
            mode=subgoal_source,
            cruise_speed=float(args.cruise_speed),
        )
    elif subgoal_source == "scene":
        # Set d_clear proportional to the ThreeZoneShield engage threshold so E1
        # starts routing before the shield fires (not after).  At cs=25 the shield
        # engages at ~134 m; with d_clear=40 m (old default) E1 only routes at
        # d_fwd<35 m — well into the braking zone where the drone is nearly stopped.
        _cs = float(args.cruise_speed)
        _engage_m = 10.0 + (_cs ** 2 - 4.0) / 5.0  # l1_sched + need(cs,v1=2,a=2.5)
        _d_clear_intent = float(max(40.0, 0.75 * _engage_m))
        intent = SceneIntentPlanner(
            r_m=r_intent,
            cruise_speed=_cs,
            d_clear=_d_clear_intent,
        )
    if intent is not None and bool(args.rolling_global):
        raise SystemExit(
            "refuse: --rolling-global incompatible with non-polyline --subgoal-source"
        )
    logger.info("subgoal_source=%s", subgoal_source)

    global_planner: Any = None
    if bool(args.rolling_global):
        global_planner = GlobalRefPlanner(
            GlobalRefConfig(
                horizon_m=float(args.global_horizon_m),
                replan_period_s=float(args.global_replan_period_s),
                step_hz=float(args.step_hz),
            )
        )
        logger.info(
            "P0 rolling_global=ON horizon=%.1fm replan=%.2fs (default remains OFF)",
            float(args.global_horizon_m),
            float(args.global_replan_period_s),
        )

    logger.info(
        f"Starting Phase 2 mainline eval on {n_routes} native routes "
        f"(idx={route_idxs}, cruise={args.cruise_speed} m/s, limits={action_limits.tolist()})"
    )

    results: List[Dict[str, Any]] = []
    save_dir: Path | None = None
    next_save_slot = 0
    if args.save_dataset:
        from experiments.aerial.rl import dataset as ds_mod

        save_dir = (
            Path(args.save_dataset).resolve()
            if Path(args.save_dataset).is_absolute()
            else (root / args.save_dataset).resolve()
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        if args.save_dataset_append:
            existing = []
            for p in save_dir.glob("episode_*.npz"):
                try:
                    existing.append(int(p.stem.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            next_save_slot = (max(existing) + 1) if existing else 0
            logger.info("save-dataset append from slot %d: %s", next_save_slot, save_dir)
        else:
            for old in save_dir.glob("episode_*.npz"):
                old.unlink()
            logger.info("save-dataset: %s", save_dir)

    for slot, ep_idx in enumerate(route_idxs):
        r_info = routes[ep_idx]
        pts = np.array(r_info.get("pos", r_info.get("positions")), dtype=np.float64)
        goal_pos = pts[-1].copy()
        start_pos = pts[0].copy()
        yaws = np.array(r_info.get("yaw", [0.0] * len(pts)), dtype=np.float64)
        start_yaw = float(yaws[0]) if len(yaws) else 0.0
        ref_len = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))

        subgoal_gen.reset()
        if intent is not None:
            intent.reset()
        # NOTE: global_planner.reset(pts, goal=goal_pos) is deferred until after
        # reset_with_spawn_retries() below, once pts/goal_pos have been
        # recomputed from the actually-lifted+retried episode. Calling it here
        # with the pre-lift `pts` would seed the planner with a goal/corridor
        # altitude that disagrees with where the vehicle actually spawns
        # whenever --min-spawn-z lifts the polyline (see fix below).
        if shield is not None:
            shield.reset()
        if depth_pred is not None:
            depth_pred.reset()
        tau_pred.reset()
        policy.reset()
        if planner is not None:
            planner.reset()

        ep_dict = {
            # Full polyline so reset uses true start; goal = last waypoint
            "pos": pts.tolist(),
            "yaw": yaws.tolist() if len(yaws) == len(pts) else [start_yaw] * len(pts),
            "gpt_instruction": r_info.get("gpt_instruction", ""),
        }
        from experiments.aerial.rl.spawn_utils import reset_with_spawn_retries

        ep_dict, obs, spawn_err, spawn_failed = reset_with_spawn_retries(
            env,
            ep_dict,
            min_spawn_z=float(args.min_spawn_z),
            spawn_z_retry_m=float(args.spawn_z_retry_m),
            spawn_z_max_retries=int(args.spawn_z_max_retries),
            spawn_tol_m=float(args.spawn_tol_m),
            mock=bool(args.mock),
            min_spawn_clear_m=float(args.min_spawn_clear_m),
        )
        # BUGFIX (2026-09-20): pts/goal_pos/start_pos above were computed from
        # the RAW annotation (pre-lift). reset_with_spawn_retries() internally
        # applies lift_episode_z() (uniform +z shift so spawn >= min_spawn_z)
        # and possibly extra retry nudges, but returned that in `ep_dict` — the
        # local pts/goal_pos/start_pos were never updated to match. Every
        # downstream consumer (intent.compute goal=, nearest_on_polyline,
        # terminal-arrival d_to_goal, carrot/subgoal target_world, global
        # planner corridor) was therefore scored against a goal/corridor
        # sitting `min_spawn_z - orig_z0` metres BELOW where the vehicle
        # actually spawns and flies (33m for hard134's z=12→45 lift). This
        # silently turned every flat-corridor route into a forced descent the
        # policy was never trained on (training's collector.py/
        # collect_depth_scene_expert_dataset.py lift the WHOLE episode,
        # including goal, before it ever reaches goal-conditioning — so BC/AC
        # training is self-consistent; only this eval script's scoring path
        # was not). Recompute from the actually-used (lifted+retried) episode.
        pts = np.asarray(ep_dict["pos"], dtype=np.float64)
        goal_pos = pts[-1].copy()
        start_pos = pts[0].copy()
        if global_planner is not None:
            global_planner.reset(pts, goal=goal_pos)
        p_curr = np.array(obs.position, dtype=np.float64)
        curr_yaw = float(obs.yaw) if hasattr(obs, "yaw") else 0.0

        if spawn_failed:
            logger.error(
                "Route %02d F1 spawn_fail err=%.1fm — skip episode (not counted as SCR)",
                ep_idx + 1,
                spawn_err,
            )
            results.append(
                {
                    "route_idx": ep_idx,
                    "base_route_idx": r_info.get("base_route_idx", ep_idx),
                    "L_ref": ref_len,
                    "L_act": 0.0,
                    "d0": float("nan"),
                    "min_d": float("nan"),
                    "arrived": False,
                    "collided": False,
                    "severe_collision": False,
                    "progress_ratio": 0.0,
                    "spl": 0.0,
                    "intervention_rate": 0.0,
                    "spawn_fail": True,
                    "spawn_err_m": spawn_err,
                    "fail_tag": "F1",
                }
            )
            continue

        max_attempts = 1
        if float(args.near_miss_retry_m) > 0.0 and int(args.near_miss_retries) > 0:
            max_attempts = 1 + int(args.near_miss_retries)
        best_ep: dict | None = None

        for near_attempt in range(1, max_attempts + 1):
            if near_attempt > 1:
                logger.warning(
                    "Route %02d near-miss retry %d/%d (prev d_min=%.2fm)",
                    ep_idx + 1,
                    near_attempt - 1,
                    int(args.near_miss_retries),
                    float(prev_min_d),
                )
                if bool(args.recover_on_near_miss_retry):
                    recover = args.recover_script
                    if recover is None:
                        recover = str(
                            Path.home() / "aerial_airsim_persistent/recover_renderer.sh"
                        )
                    recover_path = Path(recover)
                    if recover_path.is_file():
                        logger.info("recover_renderer before near-miss retry: %s", recover_path)
                        subprocess.run(["bash", str(recover_path)], check=False)
                        time.sleep(5.0)
                    else:
                        logger.warning("recover script missing: %s", recover_path)
                subgoal_gen.reset()
                if intent is not None:
                    intent.reset()
                if global_planner is not None:
                    global_planner.reset(pts, goal=goal_pos)
                if shield is not None:
                    shield.reset()
                if depth_pred is not None:
                    depth_pred.reset()
                tau_pred.reset()
                policy.reset()
                if planner is not None:
                    planner.reset()
                obs = env.reset(ep_dict)
                p_curr = np.array(obs.position, dtype=np.float64)
                curr_yaw = float(obs.yaw) if hasattr(obs, "yaw") else start_yaw

            d0 = _goal_dist(p_curr, goal_pos)
            min_d = d0
            d_final = d0
            traj = [p_curr.copy()]
            traj_writer = None
            if args.traj_out:
                _traj_path = Path(args.traj_out).with_suffix("") / f"route{ep_idx:02d}.jsonl"
                _traj_path.parent.mkdir(parents=True, exist_ok=True)
                traj_writer = _traj_path.open("w")
                logger.info("traj-out: %s", _traj_path)
            arrived = False
            collided = False
            severe_coll = False
            fail_tag: str | None = None
            interventions = 0
            hard_brakes = 0
            governor_caps = 0
            intervened_steps: set[int] = set()
            hard_brake_steps: set[int] = set()
            governor_cap_steps: set[int] = set()
            s_prog = 0.0
            last_true_s: float | None = None
            dev_degs: List[float] = []
            transitions: List[Any] = []
            oa_n_steps = 0
            oa_n_offer_escape = 0
            oa_n_chose_climb = 0
            oa_n_climb_beats = 0
            oa_n_oc_probes = 0
            oa_sum_planner_delta = 0.0
            nav_reward = None
            goal_stamp = np.asarray(goal_pos, dtype=np.float32).reshape(3)
            if save_dir is not None:
                from experiments.aerial.rl.buffer import Transition
                from experiments.aerial.rl.reward import NavigationReward

                nav_reward = NavigationReward(goal_pos, reward_cfg)
                nav_reward.reset(goal_pos, p_curr)
            prev_obs = obs
            stuck_abort_s = float(getattr(args, "stuck_abort_after_s", 0.0) or 0.0)
            stuck_best_d = float(d0)
            stuck_best_step = 0
            stuck_aborted = False

            for step in range(args.max_steps):
                d_fwd = None
                obs.info.pop("depth_min_pred", None)
                obs.info.pop("depth_cones_pred", None)
                obs.info.pop("tau_pred", None)
                if depth_pred is not None and obs.rgb is not None:
                    pred_both = getattr(depth_pred, "predict_min_and_cones", None)
                    if callable(pred_both):
                        d_min, cones = pred_both(obs)
                        if step < 3:
                            import logging as _log
                            _log.getLogger(__name__).info("DIAG step=%d d_min=%s cones_type=%s cones=%s", step, d_min, type(cones).__name__, cones)
                        if d_min is not None:
                            obs.info["depth_min_pred"] = float(d_min)
                            d_fwd = float(d_min)
                        if isinstance(cones, dict):
                            obs.info["depth_cones_pred"] = {
                                k: (float(v) if v is not None else None)
                                for k, v in cones.items()
                            }
                            cf = cones.get("forward")
                            if cf is not None and np.isfinite(float(cf)):
                                d_fwd = float(cf)
                    else:
                        d_fwd = depth_pred.predict_min(obs)
                        if d_fwd is not None:
                            obs.info["depth_min_pred"] = float(d_fwd)
                tau_v = tau_pred.predict_tau(obs)
                if tau_v is not None:
                    obs.info["tau_pred"] = float(tau_v)
                path_for_carrot = pts
                rem_full = None
                true_s_full = None
                if intent is not None:
                    g_rel_body, s_info = intent.compute(
                        curr_pos=p_curr,
                        curr_yaw=curr_yaw,
                        goal=goal_pos,
                        d_fwd_hat=d_fwd,
                        depth_cones=obs.info.get("depth_cones_pred") if subgoal_source == "scene" else None,
                    )
                    target_world = np.array(s_info["target_world"], dtype=np.float64)
                    rem_dist = float(s_info["rem_dist"])
                    s_prog = 0.0
                    safe_v = float(s_info.get("safe_speed_limit", args.cruise_speed))
                    if s_info.get("dev_deg") is not None:
                        dev_degs.append(float(s_info["dev_deg"]))
                else:
                    if global_planner is not None:
                        true_proj, _seg, true_s_full, rem_full = nearest_on_polyline(
                            p_curr, pts
                        )
                        cte_full = float(np.linalg.norm(p_curr - true_proj))
                        if last_true_s is None:
                            progressed = float("inf")
                        else:
                            progressed = float(true_s_full) - float(last_true_s)
                        last_true_s = float(true_s_full)
                        path_for_carrot = global_planner.step(
                            p_curr,
                            curr_yaw,
                            cte_m=cte_full,
                            progressed_m=progressed,
                        )

                    g_rel_body, s_info = subgoal_gen.compute_subgoal(
                        curr_pos=p_curr,
                        curr_yaw=curr_yaw,
                        global_path=path_for_carrot,
                        d_fwd_hat=d_fwd,
                    )
                    target_world = np.array(s_info["target_world"], dtype=np.float64)
                    if (
                        global_planner is not None
                        and true_s_full is not None
                        and rem_full is not None
                    ):
                        # Metrics / stop use full corridor + Euclidean G, not short P_ref rem.
                        s_prog = float(true_s_full)
                        rem_dist = float(rem_full)
                    else:
                        s_prog = float(s_info["s_progress"])
                        rem_dist = float(s_info["rem_dist"])
                    safe_v = float(s_info.get("safe_speed_limit", args.cruise_speed))

                # Align step box with physics body_delta_limits and v_safe (F3/planner)
                phys = body_delta_limits(1.0 / float(args.step_hz))
                vx_step_limit = float(min(safe_v / float(args.step_hz), float(phys[0])))
                cur_limits = np.array(
                    [vx_step_limit, float(phys[1]), float(phys[2]), float(phys[3])],
                    dtype=np.float64,
                )
                if planner is not None:
                    planner.action_limits = cur_limits

                d_to_goal = _goal_dist(p_curr, goal_pos)
                d_final = float(d_to_goal)
                if d_to_goal < min_d:
                    min_d = d_to_goal
                if intent is not None:
                    s_prog = float(max(0.0, d0 - d_to_goal))

                if (
                    float(args.terminal_direct_rem_m) > 0.0
                    and float(rem_dist) <= float(args.terminal_direct_rem_m)
                    and float(d_to_goal) <= float(args.terminal_direct_d_m)
                ):
                    target_world = goal_pos.copy()
                    delta_w = target_world - p_curr
                    cos_y, sin_y = math.cos(curr_yaw), math.sin(curr_yaw)
                    dx_b = cos_y * delta_w[0] + sin_y * delta_w[1]
                    dy_b = -sin_y * delta_w[0] + cos_y * delta_w[1]
                    dz_b = delta_w[2]
                    dist = float(np.linalg.norm(delta_w))
                    g_rel_body = np.array([dx_b, dy_b, dz_b, dist], dtype=np.float32)

                # Terminal arrival: Euclidean G for intent / rolling; else rem∧euclid.
                euclid_only = intent is not None or bool(args.rolling_global)
                if euclid_only:
                    if d_to_goal <= float(args.success_dist):
                        arrived = True
                        break
                elif rem_dist <= float(args.success_dist) and d_to_goal <= float(args.success_dist):
                    arrived = True
                    break

                # In-distribution local goal for Phase 1 policy / planner
                obs.info["goal"] = target_world.tolist()
                obs.info["goal_rel"] = g_rel_body.tolist()
                # Body yaw-to-carrot for F15 heading cost / diagnostics.
                if float(np.hypot(g_rel_body[0], g_rel_body[1])) > 1e-6:
                    obs.info["yaw_err_rad"] = float(
                        np.arctan2(float(g_rel_body[1]), float(g_rel_body[0]))
                    )
                elif s_info.get("yaw_err_rad") is not None:
                    obs.info["yaw_err_rad"] = float(s_info["yaw_err_rad"])
                # Shield terminal soft-exclusion needs rem + alignment.
                obs.info["d_to_g"] = float(d_to_goal)
                obs.info["rem_dist"] = float(rem_dist)
                if planner is not None:
                    planner.set_goal(target_world)

                action = policy.act(obs)
                action_actor = np.asarray(action, dtype=np.float64).reshape(4).copy()
                oa_step: dict = {}
                if planner is not None:
                    action = planner.plan(obs, action, latent=policy._latent)
                    plan_meta = getattr(planner, "last_plan_meta", None) or {}
                    if isinstance(plan_meta, dict):
                        oa_step = {
                            "offer_escape": bool(plan_meta.get("offer_escape")),
                            "chose_climb": bool(plan_meta.get("chose_climb")),
                            "force_peel_exec": bool(plan_meta.get("force_peel_exec")),
                            "chosen_idx": plan_meta.get("chosen_idx"),
                            "n_candidates": plan_meta.get("n_candidates"),
                            "d_fwd_plan": plan_meta.get("d_fwd"),
                        }
                    # Head ranking diagnostic: oc(climb) vs oc(wall-fwd).
                    feat = getattr(policy, "_latent", None)
                    if (
                        feat is not None
                        and bool(getattr(dynamics, "obstacle_cost_trained", False))
                        and hasattr(dynamics, "predict_obstacle_cost")
                    ):
                        lim = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                        a_fwd = np.clip(
                            np.array([0.7, 0.0, 0.0, 0.0], dtype=np.float64), -lim, lim
                        )
                        a_climb = np.clip(
                            np.array([0.2, 0.0, 0.7, 0.0], dtype=np.float64), -lim, lim
                        )
                        try:
                            oc_fwd = float(dynamics.predict_obstacle_cost(feat, a_fwd))
                            oc_climb = float(
                                dynamics.predict_obstacle_cost(feat, a_climb)
                            )
                            oa_step["oc_fwd"] = oc_fwd
                            oa_step["oc_climb"] = oc_climb
                            oa_step["oc_climb_beats_fwd"] = bool(
                                oc_climb + 1e-4 < oc_fwd
                            )
                        except Exception:
                            pass
                    act_delta = float(
                        np.linalg.norm(
                            np.asarray(action, dtype=np.float64).reshape(4) - action_actor
                        )
                    )
                    oa_step["planner_delta"] = act_delta
                    oa_n_steps += 1
                    oa_n_offer_escape += int(bool(oa_step.get("offer_escape")))
                    oa_n_chose_climb += int(bool(oa_step.get("chose_climb")))
                    oa_sum_planner_delta += act_delta
                    if "oc_climb_beats_fwd" in oa_step:
                        oa_n_oc_probes += 1
                        oa_n_climb_beats += int(bool(oa_step["oc_climb_beats_fwd"]))

                action = clip_body_delta(
                    action,
                    cur_limits,
                    forbid_backward=bool(
                        getattr(reward_cfg, "forbid_backward_motion", False)
                    ),
                )
                if bool(args.heading_assist) and intent is None:
                    # Heading assist must use the full corridor polyline + projection
                    # on ``pts``.  With --rolling-global, ``s_info['seg_idx']`` indexes
                    # the short P_ref window, not ``pts`` — mixing them injects wrong
                    # tangents and causes late-phase zigzag in open terrain.
                    _ha_proj, _ha_seg, _, _ = nearest_on_polyline(p_curr, pts)
                    _ha_cte = float(np.linalg.norm(p_curr - _ha_proj))
                    action, _ha, _ = apply_path_heading_assist(
                        action,
                        yaw=curr_yaw,
                        path=pts,
                        seg_idx=int(_ha_seg),
                        cte_m=_ha_cte,
                        limits=cur_limits,
                        cte_max_m=float(args.heading_assist_cte_max_m),
                        cos_thr=float(args.heading_assist_cos_thr),
                        lateral_scale_when_misaligned=float(
                            args.heading_assist_lateral_scale
                        ),
                    )
                    if _ha:
                        action = clip_body_delta(
                            action,
                            cur_limits,
                            forbid_backward=bool(
                                getattr(reward_cfg, "forbid_backward_motion", False)
                            ),
                        )

                # Probe p_coll for shield without consuming deploy latent stream
                wm_out = None
                if policy._latent is not None and hasattr(dynamics, "step"):
                    try:
                        wm_out = dynamics.step(
                            policy._latent,
                            action,
                            goal_rel=g_rel_body,
                            body_vel=body_vel_from_obs(obs),
                        )
                    except Exception:
                        wm_out = None

                if shield is not None:
                    base_tti = float(
                        args.tti_coeff
                        if args.tti_coeff is not None
                        else getattr(shield, "tti_coeff", 4.0)
                    )
                    eff_tti = base_tti
                    _cte_raw = s_info.get("cte_m")
                    if _cte_raw is None:
                        _cte_raw = s_info.get("cte")
                    cte_for_shield = float(_cte_raw if _cte_raw is not None else float("inf"))
                    if bool(args.shield_tti_relax_on_corridor) and intent is None:
                        if (
                            float(cte_for_shield) <= float(args.shield_tti_relax_cte_m)
                            and float(rem_dist) > float(args.shield_tti_relax_rem_m)
                        ):
                            eff_tti = base_tti * float(args.shield_tti_relax_scale)
                    if (
                        float(args.terminal_shield_tti_rem_m) > 0.0
                        and float(rem_dist) <= float(args.terminal_shield_tti_rem_m)
                        and float(cte_for_shield) <= float(args.terminal_shield_tti_cte_m)
                    ):
                        eff_tti = base_tti * float(args.terminal_shield_tti_scale)
                    obs.info["shield_tti_coeff"] = float(eff_tti)
                    act_safe, overridden = shield.apply_action(
                        action, obs, wm_out=wm_out, limits=cur_limits
                    )
                    if overridden:
                        interventions += 1
                        intervened_steps.add(step)
                    if bool(obs.info.get("shield_hard_brake")):
                        hard_brakes += 1
                        hard_brake_steps.add(step)
                    if bool(obs.info.get("shield_governor_cap")):
                        governor_caps += 1
                        governor_cap_steps.add(step)
                    action = act_safe

                if bool(getattr(reward_cfg, "forbid_backward_motion", False)):
                    from experiments.aerial.rl.env.action import forbid_backward_dx

                    action = forbid_backward_dx(action)

                # Snapshot cones before step — post-step obs has no depth_cones_pred.
                _cones_pre = dict(obs.info.get("depth_cones_pred") or {})

                step_out = env.step(action)
                if len(step_out) == 4:
                    obs, _rew, done, step_info = step_out
                else:
                    obs, step_info = step_out
                    done = bool(getattr(obs, "collided", False))

                p_prev = p_curr.copy()
                p_curr = np.array(obs.position, dtype=np.float64)
                curr_yaw = float(obs.yaw) if hasattr(obs, "yaw") else curr_yaw

                # F-tele: detect AirSim teleportation (position jump > 20 m in one
                # step is physically impossible; flag episode so it is not reported
                # as a navigation result).
                _step_jump = float(np.linalg.norm(p_curr - p_prev))
                if _step_jump > 20.0:
                    logger.error(
                        "Route %02d F-tele teleportation at step %d: jump=%.1fm — episode invalidated",
                        ep_idx + 1, step, _step_jump,
                    )
                    fail_tag = "F-tele"  # teleport: episode invalid, not a real collision
                    break

                traj.append(p_curr.copy())

                if traj_writer is not None:
                    _srec = s_info if intent is not None else {}
                    _cones = _cones_pre
                    traj_writer.write(json.dumps({
                        "step": step,
                        "pos": p_curr.tolist(),
                        "yaw_deg": round(float(np.degrees(curr_yaw)), 2),
                        "d_to_g": round(float(np.linalg.norm(goal_pos - p_curr)), 2),
                        "d_fwd": round(float(d_fwd), 3) if d_fwd is not None else None,
                        "d_left": round(float(_cones["left"]), 2) if _cones.get("left") is not None else None,
                        "d_right": round(float(_cones["right"]), 2) if _cones.get("right") is not None else None,
                        "intent_target": _srec.get("target_world"),
                        "chosen_idx": _srec.get("chosen_idx"),
                        "dev_deg": round(float(_srec["dev_deg"]), 2) if _srec.get("dev_deg") is not None else None,
                        "replan": _srec.get("replan"),
                        "n_feasible": _srec.get("n_feasible"),
                        "intervened": bool(step in intervened_steps),
                        "hard_brake": bool(step in hard_brake_steps),
                        "governor_cap": bool(step in governor_cap_steps),
                        "offer_escape": oa_step.get("offer_escape") if oa_step else None,
                        "chose_climb": oa_step.get("chose_climb") if oa_step else None,
                        "force_peel_exec": (
                            oa_step.get("force_peel_exec") if oa_step else None
                        ),
                        "oc_fwd": oa_step.get("oc_fwd") if oa_step else None,
                        "oc_climb": oa_step.get("oc_climb") if oa_step else None,
                        "oc_climb_beats_fwd": (
                            oa_step.get("oc_climb_beats_fwd") if oa_step else None
                        ),
                    }) + "\n")

                seg_d = _segment_min_dist(p_prev, p_curr, goal_pos)
                if euclid_only:
                    if seg_d <= float(args.success_dist):
                        arrived = True
                        min_d = min(min_d, seg_d)
                        d_final = float(seg_d)
                        break
                elif rem_dist <= float(args.success_dist) and seg_d <= float(args.success_dist):
                    arrived = True
                    min_d = min(min_d, seg_d)
                    d_final = float(seg_d)
                    break

                if save_dir is not None and nav_reward is not None:
                    r_step, done_r, terms = nav_reward.step(obs, action)
                    ep_info = dict(step_info or {})
                    ep_info.update(terms)
                    ep_info["goal"] = goal_stamp.copy()
                    ep_info["scene"] = "outdoor_long"
                    ep_info["route_idx"] = int(ep_idx)
                    transitions.append(
                        Transition(
                            obs=prev_obs,
                            action=np.asarray(action, dtype=np.float32),
                            reward=float(r_step),
                            done=bool(done_r or done),
                            next_obs=obs,
                            info=ep_info,
                        )
                    )
                prev_obs = obs

                if done:
                    collided = bool(
                        getattr(obs, "collided", False) or step_info.get("collided", False)
                    )
                    if step_info.get("severe_collision", False) or collided:
                        severe_coll = True
                    break

                # Mid-course stuck abort (teacher collect throughput): R0 choke ~32m.
                if stuck_abort_s > 0.0:
                    d_now = float(np.linalg.norm(goal_pos - p_curr))
                    rem_lo = float(args.stuck_abort_rem_lo)
                    rem_hi = float(args.stuck_abort_rem_hi)
                    prog_need = float(args.stuck_abort_progress_m)
                    if rem_lo <= d_now <= rem_hi:
                        if d_now <= stuck_best_d - prog_need:
                            stuck_best_d = d_now
                            stuck_best_step = step
                        else:
                            elapsed_s = (step - stuck_best_step) / max(
                                1e-6, float(args.step_hz)
                            )
                            if elapsed_s >= stuck_abort_s:
                                stuck_aborted = True
                                fail_tag = "stuck_abort"
                                d_final = d_now
                                logger.info(
                                    "Route %02d stuck-abort at step %d d_to=%.1fm "
                                    "(no ≥%.1fm progress for %.1fs in rem[%.0f,%.0f])",
                                    ep_idx + 1,
                                    step,
                                    d_now,
                                    prog_need,
                                    elapsed_s,
                                    rem_lo,
                                    rem_hi,
                                )
                                break
                    else:
                        # Outside band: keep best tracker warm but don't abort.
                        if d_now < stuck_best_d:
                            stuck_best_d = d_now
                            stuck_best_step = step

            save_this = bool(
                save_dir is not None
                and transitions
                and (arrived or not bool(getattr(args, "save_arrived_only", False)))
            )
            if save_this:
                write_slot = int(next_save_slot) if args.save_dataset_append else int(slot)
                ds_mod.write_episode(save_dir, write_slot, transitions)
                if args.save_dataset_append:
                    next_save_slot = write_slot + 1
                logger.info(
                    "save-dataset: route %02d -> episode_%05d.npz (%d steps) arrived=%s",
                    ep_idx + 1,
                    write_slot,
                    len(transitions),
                    arrived,
                )
            elif save_dir is not None and transitions and args.save_arrived_only:
                logger.info(
                    "save-dataset: skip non-arrived route %02d (%d steps, tag=%s)",
                    ep_idx + 1,
                    len(transitions),
                    fail_tag,
                )

            actual_len = (
                float(np.sum(np.linalg.norm(np.diff(np.array(traj), axis=0), axis=1)))
                if len(traj) > 1
                else 0.0
            )
            prog_ratio = float(np.clip(s_prog / max(1e-3, ref_len), 0.0, 1.0))
            # Design SPL; shortcuts (L_act < L_ref) do not inflate above 1.0
            ep_spl = (ref_len / max(ref_len, actual_len)) if arrived else 0.0
            goal_closure = _goal_closure(d0, min_d)
            inflate = _monotone_inflate(prog_ratio, min_d)

            ep_result = {
                "route_idx": ep_idx,
                "base_route_idx": r_info.get("base_route_idx"),
                "nominal_length_m": round(ref_len, 2),
                "actual_length_m": round(actual_len, 2),
                "steps": len(traj),
                "d_start_m": round(d0, 2),
                "d_min_m": round(min_d, 2),
                "d_final_m": round(float(d_final), 2),
                "goal_closure": round(goal_closure, 4),
                "monotone_inflate": bool(inflate),
                "arrived": arrived,
                "collided": collided,
                "severe_collision": severe_coll,
                "progress_ratio": round(prog_ratio, 4),
                "spl": round(ep_spl, 4),
                "intervention_rate": round(interventions / max(1, len(traj)), 4),
                "hard_brake_rate": round(hard_brakes / max(1, len(traj)), 4),
                "governor_cap_rate": round(governor_caps / max(1, len(traj)), 4),
                "n_global_replans": (
                    int(global_planner.replan_count) if global_planner is not None else 0
                ),
                "subgoal_source": subgoal_source,
                # E1: replans / how often the fan left the direct-to-G ray. offaxis 0
                # means `scene` reduced to `toward_g` on this route.
                "n_intent_replans": int(getattr(intent, "replan_count", 0)),
                "n_intent_offaxis": int(getattr(intent, "offaxis_count", 0)),
                "mean_intent_dev_deg": round(float(np.mean(dev_degs)), 2) if dev_degs else 0.0,
                "max_intent_dev_deg": round(float(np.max(dev_degs)), 2) if dev_degs else 0.0,
                "fail_tag": fail_tag,
                # Directional OA online diagnostics (planner + obstacle head).
                "oa_n_steps": int(oa_n_steps),
                "oa_offer_escape_rate": round(
                    oa_n_offer_escape / max(1, oa_n_steps), 4
                ),
                "oa_chose_climb_rate": round(
                    oa_n_chose_climb / max(1, oa_n_steps), 4
                ),
                "oa_oc_climb_beats_fwd_rate": round(
                    oa_n_climb_beats / max(1, oa_n_oc_probes), 4
                ),
                "oa_mean_planner_delta": round(
                    oa_sum_planner_delta / max(1, oa_n_steps), 4
                ),
            }
            if traj_writer is not None:
                traj_writer.close()
            ep_result["near_miss_attempt"] = int(near_attempt)
            if best_ep is None or _episode_better(ep_result, best_ep):
                best_ep = ep_result
            if (
                arrived
                or float(args.near_miss_retry_m) <= 0.0
                or near_attempt >= max_attempts
                or float(min_d) > float(args.near_miss_retry_m)
            ):
                break
            prev_min_d = float(min_d)

        results.append(best_ep)
        ep_result = best_ep
        intent_tail = (
            f" | replan={ep_result['n_intent_replans']}"
            f" offaxis={ep_result['n_intent_offaxis']}"
            f" dev={ep_result['mean_intent_dev_deg']:.1f}/{ep_result['max_intent_dev_deg']:.1f}deg"
            if ep_result["n_intent_replans"]
            else ""
        )
        _retry_tag = (
            f" | near_attempt={ep_result.get('near_miss_attempt', 1)}"
            if int(ep_result.get("near_miss_attempt", 1)) > 1
            else ""
        )
        logger.info(
            f"Route {ep_idx+1:02d} ({slot+1}/{n_routes}) | L_ref={ref_len:.1f}m | "
            f"L_act={ep_result['actual_length_m']:.1f}m | "
            f"min_d={ep_result['d_min_m']:.2f}m | d_final={ep_result['d_final_m']:.2f}m | "
            f"closure={ep_result['goal_closure']:.2f} | "
            f"arrived={ep_result['arrived']} | prog={ep_result['progress_ratio']*100:.1f}% | "
            f"inflate={ep_result['monotone_inflate']} | "
            f"spl={ep_result['spl']:.3f} | IR={ep_result['intervention_rate']:.3f}"
            f"{intent_tail}{_retry_tag}"
            + (
                f" | oa_esc={ep_result.get('oa_offer_escape_rate', 0):.2f}"
                f" climb={ep_result.get('oa_chose_climb_rate', 0):.2f}"
                f" oc↑={ep_result.get('oa_oc_climb_beats_fwd_rate', 0):.2f}"
                if int(ep_result.get("oa_n_steps") or 0) > 0
                else ""
            )
        )

    scored, spawn_fails, metrics, verdict = aggregate_metrics(results)
    summary = {
        "protocol_version": "wam_phase2_goal_scene_e0_20260903",
        "goal_feat_mode": str(args.goal_feat_mode),
        "actor_ckpt": str(actor_path),
        "depth_ckpt": str(depth_path),
        "tau_ckpt": str(tau_path),
        "r_m_intent": r_intent,
        "cruise_speed_m_s": args.cruise_speed,
        "max_steps": args.max_steps,
        "spawn_tol_m": args.spawn_tol_m,
        "subgoal_source": subgoal_source,
        "rolling_global": bool(args.rolling_global),
        "global_horizon_m": float(args.global_horizon_m),
        "global_replan_period_s": float(args.global_replan_period_s),
        "route_indices": list(route_idxs),
        "n_scored": len(scored),
        "n_spawn_fail_f1": len(spawn_fails),
        "metrics": metrics,
        "thresholds": dict(PASS_THRESHOLDS),
        "verdict": verdict,
        "episodes": results,
    }

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(format_summary_line(summary, prefix="Phase 2 mainline complete."))
    if len(route_idxs) < len(routes):
        logger.warning(
            "PARTIAL RUN: routes %s of %d — merge with the other box via "
            "merge_phase2_split_eval.py before filling any DECLARE row",
            route_idxs,
            len(routes),
        )
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
