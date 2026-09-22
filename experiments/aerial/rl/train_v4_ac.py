#!/usr/bin/env python3
"""V4 pure-imagination AC short train entrypoint (H100 or local CPU smoke).

Does **not** modify ``configs/aerial_rl.yaml`` — pass overrides on CLI or edit
the in-memory cfg dict only.

    python -m experiments.aerial.rl.train_v4_ac --iters 5 --device cuda

On H100 after bundle/pull, run with ``dynamics.kind=torch`` and a loaded WM ckpt.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from experiments.aerial.rl.collect_dataset import (
    _mock_goal_episode,
    approach_bias_episodes,
)
from experiments.aerial.rl.train_rl import (
    bind_loaded_dynamics,
    build_from_config,
    load_torch_dynamics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overlay into a shallow copy of base (dicts only)."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_cfg(repo: Path, config_rel: str = "configs/aerial_rl.yaml") -> Dict[str, Any]:
    cfg_path = repo / config_rel
    return yaml.safe_load(cfg_path.read_text())


def main() -> int:
    p = argparse.ArgumentParser(description="V4 imagination AC short train")
    p.add_argument(
        "--config",
        default="configs/aerial_rl.yaml",
        help="yaml config (use configs/aerial_rl_phase3_unified.yaml for Phase-3)",
    )
    p.add_argument(
        "--config-overlay",
        default="",
        help="optional yaml deep-merged on top of --config "
        "(e.g. configs/aerial_rl_urban_complex_directional_oa.yaml)",
    )
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--episodes-per-iter", type=int, default=2)
    p.add_argument("--imagine-batch", type=int, default=16)
    p.add_argument("--imagine-horizon", type=int, default=15)
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    p.add_argument("--ckpt-dir", default=None, help="write actor ckpt dir")
    p.add_argument("--backend", default="mock", choices=("mock", "airsim"))
    p.add_argument(
        "--dynamics",
        default=None,
        choices=("stub", "torch"),
        help="dynamics.kind override (default: stub for mock backend, else yaml)",
    )
    p.add_argument(
        "--wm-ckpt",
        default=None,
        help="WM checkpoint path when --dynamics torch (required for serious train)",
    )
    p.add_argument(
        "--annotation",
        default=None,
        help="OpenFly annotation JSON for start→goal episodes (real or mock collect)",
    )
    p.add_argument(
        "--approach-bias",
        action="store_true",
        help="rewrite goals to start+dist along start yaw (matches collect_dataset)",
    )
    p.add_argument(
        "--approach-dist-m",
        type=float,
        default=25.0,
        help="goal distance along start heading when --approach-bias",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="preload real RGB replay episodes for z0 encode (offline RGB align)",
    )
    p.add_argument(
        "--skip-collect",
        action="store_true",
        help="skip env collect each iter (use with --dataset for offline z0 AC)",
    )
    p.add_argument(
        "--w-collision",
        type=float,
        default=None,
        help="override reward.w_collision for imagination AC (default: yaml)",
    )
    p.add_argument(
        "--w-eff-strafe",
        type=float,
        default=None,
        help="F15 override reward.w_eff_strafe (default: yaml / 0)",
    )
    p.add_argument(
        "--w-eff-heading",
        type=float,
        default=None,
        help="F15 override reward.w_eff_heading (default: yaml / 0)",
    )
    p.add_argument(
        "--w-eff-idle",
        type=float,
        default=None,
        help="F15 override reward.w_eff_idle (default: yaml / 0)",
    )
    p.add_argument(
        "--w-intervention",
        type=float,
        default=None,
        help=(
            "override reward.w_intervention (default: yaml / 0 = no-op). "
            "Real path: charged only on shield_hard_brake (exclusion / "
            "τ·p_coll emergency), not soft TTI governor caps. Imagination: "
            "binary charge when d_fwd_hat ≤ hard_brake_depth_m (= exclusion_m). "
            "Soft clearance still goes through w_collision × clear_risk."
        ),
    )
    p.add_argument(
        "--init-actor-ckpt",
        default=None,
        help="warm-start actor/critic from an existing v4_ac_*.pt (F15 short FT)",
    )
    p.add_argument(
        "--phase2",
        action="store_true",
        help=(
            "Phase-2 training: replace HeuristicPolicy with learned AC + toward_g "
            "outer loop for data collection; enable online policy update "
            "(WM update stays gated by corrector.enable_wm_update / forced off "
            "in this entrypoint)."
        ),
    )
    p.add_argument(
        "--r-m",
        type=float,
        default=100.0,
        help="Phase-2 toward_g clip radius (m); default 100 matches eval",
    )
    p.add_argument(
        "--near-goal-frac",
        type=float,
        default=0.0,
        help=(
            "Fraction of training episodes that spawn near goal (0=disabled). "
            "E.g. 0.4 → 40%% of episodes start within --near-goal-dist of goal."
        ),
    )
    p.add_argument("--near-goal-dist-min", type=float, default=5.0,
                   help="Min distance from goal for near-goal spawns (m).")
    p.add_argument("--near-goal-dist-max", type=float, default=30.0,
                   help="Max distance from goal for near-goal spawns (m).")
    p.add_argument(
        "--enable-bc",
        action="store_true",
        help=(
            "Behavior-clone actor toward demo/planner actions each corrector "
            "iter (after imagination AC). With --dataset uses expert demos; "
            "with --planner and no dataset, harvests online planner≠actor steps."
        ),
    )
    p.add_argument(
        "--bc-only",
        action="store_true",
        help=(
            "Offline expert BC only: no imagination AC and no env collect. "
            "Implies --enable-bc and --skip-collect. Requires --dataset."
        ),
    )
    p.add_argument("--bc-batch", type=int, default=64,
                   help="Expert transition batch size per BC update.")
    p.add_argument("--bc-updates-per-iter", type=int, default=4,
                   help="Number of BC gradient steps per corrector iteration.")
    p.add_argument("--bc-loss-scale", type=float, default=1.0,
                   help="Multiplier on BC MSE loss.")
    p.add_argument(
        "--min-steps-for-best",
        type=int,
        default=None,
        help="Ignore collects shorter than this when updating v4_ac_best (default 20)",
    )
    p.add_argument(
        "--cs-values",
        type=str,
        default="",
        help=(
            "Comma-separated cruise speeds for variable-cs training "
            "(e.g. '3,5,7,10'). Empty = fixed cs from config (default)."
        ),
    )
    p.add_argument(
        "--start-iter",
        type=int,
        default=0,
        help="skip corrector iters [0, start_iter) when resuming online collect",
    )
    p.add_argument(
        "--save-every-iter",
        action="store_true",
        help="write v4_ac_latest.pt after each corrector iteration (125 long runs)",
    )
    p.add_argument(
        "--renderer-restart-every",
        type=int,
        default=None,
        help="restart AirSim renderer every N corrector iters (omit to use config; 0=disabled)",
    )
    p.add_argument(
        "--renderer-restart-script",
        type=str,
        default="",
        help="recover_renderer_scene.sh path (required when --renderer-restart-every > 0)",
    )
    p.add_argument(
        "--renderer-restart-scene",
        type=str,
        default="outdoor",
        help="scene arg passed to renderer restart script (outdoor/building99/...)",
    )
    p.add_argument(
        "--min-spawn-z",
        type=float,
        default=None,
        help="Lift episode z uniformly so spawn z >= this (urban flyability; default: yaml corrector)",
    )
    p.add_argument(
        "--spawn-z-retry-m",
        type=float,
        default=None,
        help="On spawn collision, raise polyline z by this per retry (default: yaml corrector)",
    )
    p.add_argument(
        "--spawn-z-max-retries",
        type=int,
        default=None,
        help="Spawn-collision retries after min-spawn-z lift (default: yaml corrector)",
    )
    p.add_argument(
        "--min-spawn-clear-m",
        type=float,
        default=None,
        help="Reject train-collect spawn if GT forward clearance < this (0 disables)",
    )
    p.add_argument(
        "--planner",
        action="store_true",
        help="enable imagination planner during collect (overrides yaml planner.enable)",
    )
    p.add_argument(
        "--no-planner",
        action="store_true",
        help="disable planner during collect",
    )
    p.add_argument(
        "--planner-horizon",
        type=int,
        default=None,
        help="planner imagination horizon (eval mainline uses 1)",
    )
    p.add_argument(
        "--planner-rollout",
        type=str,
        default=None,
        choices=("open_loop", "closed_loop"),
        help="collect-time planner rollout (match eval closed_loop to close train/eval gap)",
    )
    p.add_argument(
        "--tti-coeff",
        type=float,
        default=None,
        help="ThreeZoneSpeedShield tti_coeff override (eval mainline: 2.5)",
    )
    p.add_argument(
        "--shield-exclusion-forward-only",
        action="store_true",
        help="shield uses forward cone only for exclusion depth (match eval)",
    )
    p.add_argument(
        "--no-shield-exclusion-forward-only",
        action="store_true",
        help="disable forward-only shield exclusion (full-FOV min depth)",
    )
    p.add_argument(
        "--no-shield",
        action="store_true",
        help="null safety shield for collect (directional-OA train; deploy re-enables)",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[3]
    cfg = _load_cfg(repo, str(args.config))
    if args.config_overlay:
        overlay = _load_cfg(repo, str(args.config_overlay))
        cfg = _deep_merge(cfg, overlay)
        logger.info("config: %s + overlay %s", args.config, args.config_overlay)
    else:
        logger.info("config: %s", args.config)
    cfg.setdefault("corrector", {})
    cfg.setdefault("env", {})
    cfg.setdefault("imagination", {})
    cfg.setdefault("v4", {})
    cfg.setdefault("dynamics", {})
    cfg.setdefault("tau_predictor", {})
    cfg.setdefault("reward", {})
    cfg.setdefault("planner", {})
    cfg.setdefault("safety", {})
    cfg.setdefault("world_model", {})
    if args.w_collision is not None:
        cfg["reward"]["w_collision"] = float(args.w_collision)
    if args.w_eff_strafe is not None:
        cfg["reward"]["w_eff_strafe"] = float(args.w_eff_strafe)
    if args.w_eff_heading is not None:
        cfg["reward"]["w_eff_heading"] = float(args.w_eff_heading)
    if args.w_eff_idle is not None:
        cfg["reward"]["w_eff_idle"] = float(args.w_eff_idle)
    if args.w_intervention is not None:
        cfg["reward"]["w_intervention"] = float(args.w_intervention)
    logger.info(
        "F15 reward weights: w_eff_strafe=%s w_eff_heading=%s w_eff_idle=%s w_collision=%s "
        "w_intervention=%s",
        cfg["reward"].get("w_eff_strafe", 0.0),
        cfg["reward"].get("w_eff_heading", 0.0),
        cfg["reward"].get("w_eff_idle", 0.0),
        cfg["reward"].get("w_collision"),
        cfg["reward"].get("w_intervention", 0.0),
    )
    cfg["corrector"]["iterations"] = int(args.iters)
    cfg["corrector"]["episodes_per_iter"] = int(args.episodes_per_iter)
    if args.min_spawn_z is not None:
        cfg["corrector"]["min_spawn_z"] = float(args.min_spawn_z)
    if args.spawn_z_retry_m is not None:
        cfg["corrector"]["spawn_z_retry_m"] = float(args.spawn_z_retry_m)
    if args.spawn_z_max_retries is not None:
        cfg["corrector"]["spawn_z_max_retries"] = int(args.spawn_z_max_retries)
    if getattr(args, "min_spawn_clear_m", None) is not None:
        cfg["corrector"]["min_spawn_clear_m"] = float(args.min_spawn_clear_m)
    cc_spawn = cfg["corrector"]
    if (
        cc_spawn.get("min_spawn_z")
        or cc_spawn.get("spawn_z_max_retries")
        or cc_spawn.get("min_spawn_clear_m")
    ):
        logger.info(
            "spawn: min_z=%s retry_m=%s max_retries=%s min_clear=%s",
            cc_spawn.get("min_spawn_z", 0),
            cc_spawn.get("spawn_z_retry_m", 0),
            cc_spawn.get("spawn_z_max_retries", 0),
            cc_spawn.get("min_spawn_clear_m", 0),
        )
    if bool(getattr(args, "bc_only", False)):
        args.enable_bc = True
        args.skip_collect = True
        cfg["corrector"]["enable_policy_update"] = False
        logger.info("BC-only: imagination AC off, env collect off")
    else:
        cfg["corrector"]["enable_policy_update"] = True
    # Phase-2 Direction A: enable joint WM+AC update. freeze=False in load_torch_dynamics
    # keeps WM params trainable so dynamics.update() can backprop after ckpt load.
    cfg["corrector"]["enable_wm_update"] = False  # WM is env-valid; AC-only FT here
    cfg["imagination"]["horizon"] = int(args.imagine_horizon)
    cfg["imagination"]["batch"] = int(args.imagine_batch)
    cfg["env"]["backend"] = str(args.backend)
    if args.annotation:
        cfg["annotation"] = str(args.annotation)
    if args.skip_collect:
        cfg["corrector"]["episodes_per_iter"] = 0
    if args.dynamics is not None:
        dyn_kind = str(args.dynamics)
    elif args.backend == "mock":
        dyn_kind = "stub"
    else:
        dyn_kind = str(cfg["dynamics"].get("kind", "stub"))
    cfg["dynamics"]["kind"] = dyn_kind
    cfg["tau_predictor"]["enable"] = False if args.backend == "mock" else cfg["tau_predictor"].get("enable", False)
    if args.backend != "mock" and cfg["tau_predictor"].get("enable"):
        cfg["tau_predictor"]["device"] = str(args.device)
    if args.backend == "mock":
        cfg.setdefault("world_model", {}).setdefault("depth_head", {})["enable"] = False
    if args.no_planner:
        cfg["planner"]["enable"] = False
    elif args.planner:
        cfg["planner"]["enable"] = True
    if args.planner_horizon is not None:
        cfg["planner"]["horizon"] = int(args.planner_horizon)
    if args.planner_rollout is not None:
        cfg["planner"]["rollout_mode"] = str(args.planner_rollout)
        # closed_loop collect implies planner on (deploy-align).
        if str(args.planner_rollout) == "closed_loop" and not args.no_planner:
            cfg["planner"]["enable"] = True
    if args.tti_coeff is not None:
        cfg["safety"]["tti_coeff"] = float(args.tti_coeff)
    if args.shield_exclusion_forward_only:
        cfg["safety"]["exclusion_forward_only"] = True
    elif args.no_shield_exclusion_forward_only:
        cfg["safety"]["exclusion_forward_only"] = False
    if args.no_shield:
        cfg["safety"]["kind"] = "null"
        # No shield → intervention term must stay zero (plan task 6).
        cfg.setdefault("reward", {})["w_intervention"] = 0.0
    cfg["v4"]["device"] = str(args.device)

    wm_ckpt_path = None
    if dyn_kind == "torch":
        wm_ckpt_path = args.wm_ckpt
        if not wm_ckpt_path:
            wm_dir = cfg.get("world_model", {}).get("checkpoint_dir")
            if wm_dir:
                cand = repo / wm_dir / "wm_step_5000.pt"
                if cand.is_file():
                    wm_ckpt_path = str(cand)
        if not wm_ckpt_path:
            logger.error("--wm-ckpt required when --dynamics torch")
            return 1
        cfg.setdefault("world_model", {})["device"] = str(args.device)

    loop = build_from_config(cfg)
    # Mock dry-run with no annotation: inject a goal so imagined progress / RH
    # aux are non-trivial (collect_dataset does the same for mock collect).
    if args.backend == "mock" and loop.episodes is None:
        loop.episodes = [_mock_goal_episode()]
        logger.info("mock backend: injected synthetic start→goal episode")
    if args.approach_bias and loop.episodes is not None:
        loop.episodes = approach_bias_episodes(
            loop.episodes, dist_m=float(args.approach_dist_m),
        )
        logger.info(
            "approach-bias ON: goals -> start + %.1f m along start yaw (%d eps)",
            float(args.approach_dist_m), len(loop.episodes),
        )
    if args.dataset:
        from experiments.aerial.rl import dataset as ds
        from experiments.aerial.rl.goal_features import attach_goal, resolve_episode_goal

        ds_path = Path(args.dataset)
        if not ds_path.is_dir():
            logger.error("--dataset %s is not a directory", ds_path)
            return 1
        loaded = ds.load_dataset(ds_path, skip_quarantined=True)
        stamped = 0
        expert_flat = []
        for ep in loaded:
            goal = resolve_episode_goal(ep, allow_end_proxy=True)
            if goal is not None:
                attach_goal(ep, goal)
                stamped += 1
            loop.buffer.add_episode(ep)
            expert_flat.extend(ep)
        loop.expert_transitions = expert_flat
        logger.info(
            "preloaded %d episodes (%d with goals, %d transitions) from %s for real-RGB z0",
            len(loaded), stamped, len(expert_flat), ds_path,
        )
    if bool(getattr(args, "enable_bc", False)):
        online_ok = bool(getattr(args, "planner", False)) and not bool(
            getattr(args, "bc_only", False)
        )
        if not loop.expert_transitions and not online_ok:
            logger.error(
                "--enable-bc requires --dataset (expert) or --planner (online distill)"
            )
            return 1
        loop.config.enable_bc_update = True
        loop.config.bc_batch = int(args.bc_batch)
        loop.config.bc_updates_per_iter = int(args.bc_updates_per_iter)
        loop.config.bc_loss_scale = float(args.bc_loss_scale)
        if online_ok and not loop.expert_transitions:
            loop.config.enable_online_planner_bc = True
            logger.info(
                "BC ON (online planner distill): batch=%d updates/iter=%d "
                "loss_scale=%.3g (pool grows from CL collect)",
                loop.config.bc_batch,
                loop.config.bc_updates_per_iter,
                loop.config.bc_loss_scale,
            )
        else:
            if online_ok:
                loop.config.enable_online_planner_bc = True
            logger.info(
                "BC ON: batch=%d updates/iter=%d loss_scale=%.3g n_expert=%d "
                "online_harvest=%s",
                loop.config.bc_batch,
                loop.config.bc_updates_per_iter,
                loop.config.bc_loss_scale,
                len(loop.expert_transitions),
                bool(loop.config.enable_online_planner_bc),
            )
    if getattr(args, "min_steps_for_best", None) is not None:
        loop.config.min_steps_for_best = int(args.min_steps_for_best)
        logger.info("min_steps_for_best=%d", loop.config.min_steps_for_best)
    if dyn_kind == "torch" and wm_ckpt_path:
        wm_cfg = cfg.get("world_model", {})
        success_dist_m = float(cfg.get("reward", {}).get("success_dist_m", 3.0))
        dynamics, wm_payload = load_torch_dynamics(
            wm_cfg,
            wm_ckpt_path,
            device=str(args.device),
            success_dist_m=success_dist_m,
            freeze=not args.phase2,
        )
        bind_loaded_dynamics(loop, dynamics)
        if loop.actor_critic is not None:
            ac_dim = int(loop.actor_critic.config.latent_dim)
            if ac_dim != int(dynamics.latent_dim):
                logger.error(
                    "actor latent_dim=%d != WM latent_dim=%d — rebuild with matching config",
                    ac_dim,
                    int(dynamics.latent_dim),
                )
                return 1
        logger.info(
            "loaded WM ckpt %s step=%s latent_dim=%d depth_decoder_trained=%s "
            "(collector.dynamics is loop.dynamics=%s)",
            wm_ckpt_path,
            wm_payload.get("step"),
            int(dynamics.latent_dim),
            bool(wm_payload.get("depth_decoder_trained", False)),
            loop.collector.dynamics is dynamics if getattr(loop, "collector", None) else False,
        )
        w_int = float(cfg.get("reward", {}).get("w_intervention", 0.0) or 0.0)
        if w_int > 0.0 and not bool(wm_payload.get("depth_decoder_trained", False)):
            logger.warning(
                "reward.w_intervention=%.3g but WM ckpt has NO depth_decoder weights "
                "— imagination clearance_risk / soft-zone charge will be inert "
                "(d_fwd_hat gated). Use wm_ckpt_depth_aux_* for OA FT.",
                w_int,
            )
    # Directional OA: refuse silent zero-obstacle training.
    use_learned = bool(cfg.get("reward", {}).get("use_learned_obstacle_cost", False))
    if use_learned:
        dyn_live = getattr(loop, "dynamics", None)
        obs_trained = bool(getattr(dyn_live, "obstacle_cost_trained", False))
        if not obs_trained:
            logger.error(
                "reward.use_learned_obstacle_cost=true but obstacle_cost_trained=False "
                "— refuse start (would train with zero obstacle cost). "
                "Run train_obstacle_cost_labels train and load that WM ckpt."
            )
            return 1
        w_coll = float(cfg.get("reward", {}).get("w_collision", 10.0) or 10.0)
        # Calibrated 2026-09-22: w_collision=2 (gate); warn only if still on legacy F15-scale.
        if w_coll >= 5.0:
            logger.warning(
                "directional OA expects w_collision≈2 (got %.3g); "
                "use --config-overlay configs/aerial_rl_urban_complex_directional_oa.yaml",
                w_coll,
            )
        logger.info(
            "directional OA: use_learned_obstacle_cost=true "
            "obstacle_cost_trained=true w_collision=%.3g w_straight=%.3g",
            w_coll,
            float(cfg.get("reward", {}).get("w_straight", 0.0) or 0.0),
        )
    if loop.actor_critic is None:
        logger.error("actor_critic not built — install torch")
        return 1
    if args.init_actor_ckpt:
        from experiments.aerial.rl.actor_critic import ImaginationActorPolicy, LatentActorCritic

        init_path = Path(args.init_actor_ckpt)
        if not init_path.is_file():
            logger.error("--init-actor-ckpt missing: %s", init_path)
            return 1
        warmed = LatentActorCritic.load_from_checkpoint(
            init_path, device=str(args.device),
        )
        if not warmed.bounded:
            logger.error(
                "refusing warm-start from unbounded policy_class=%s",
                warmed.config.policy_class,
            )
            return 1
        loop.actor_critic = warmed
        loop.imagination_policy = ImaginationActorPolicy(warmed)
        logger.info(
            "warm-started actor from %s (goal_feat_mode=%s condition_on_goal=%s)",
            init_path,
            warmed.config.goal_feat_mode,
            warmed.config.condition_on_goal,
        )
    ac_cfg = loop.actor_critic.config
    if not loop.actor_critic.bounded:
        # Red line (C2, 2026-08-18): the pre-C2 unbounded policy class is
        # invalidated; it may be replayed for audit but never trained.
        logger.error(
            "refusing to train policy_class=%s — retrain from scratch (proposal §4.1)",
            ac_cfg.policy_class,
        )
        return 1
    logger.info(
        "policy: class=%s action_limits=%s (step_hz=%.3f) action_scale=%.3f",
        ac_cfg.policy_class, ac_cfg.action_limits, ac_cfg.step_hz, ac_cfg.action_scale,
    )

    # Deploy-align: closed_loop collect needs actor as imagination tail.
    pl = getattr(getattr(loop, "collector", None), "planner", None)
    if pl is not None and str(getattr(pl, "rollout_mode", "open_loop")) == "closed_loop":
        from experiments.aerial.rl.actor_critic import LatentActorDeployPolicy

        if loop.dynamics is None or not hasattr(loop.dynamics, "encode"):
            logger.error("closed_loop collect requires torch WM dynamics")
            return 1
        pl.tail_policy = LatentActorDeployPolicy(
            loop.dynamics, loop.actor_critic, stream_latent=True
        )
        logger.info(
            "closed_loop collect: planner H=%s tail=LatentActorDeployPolicy",
            getattr(pl, "horizon", None),
        )

    if args.phase2:
        from experiments.aerial.rl.train_rl import Phase2CollectionPolicy
        from experiments.aerial.rl.actor_critic import LatentActorDeployPolicy

        if loop.dynamics is None or not hasattr(loop.dynamics, "encode"):
            logger.error("--phase2 requires --dynamics torch with a loaded WM checkpoint")
            return 1
        inner_policy = LatentActorDeployPolicy(
            loop.dynamics, loop.actor_critic, stream_latent=True
        )
        env_ref = loop.collector.env
        loop.collector.policy = Phase2CollectionPolicy(
            inner_policy,
            goal_getter=lambda: getattr(env_ref, "goal", None),
            r_m=float(args.r_m),
            cruise_speed=float(
                (cfg.get("safety") or {}).get("v_cruise_m_s")
                or (cfg.get("scene_profiles") or {})
                .get("outdoor_complex", {})
                .get("safety", {})
                .get("v_cruise_m_s", 10.0)
            ),
        )
        pl_cfg = cfg.get("planner", {})
        sf_cfg = cfg.get("safety", {})
        dh_cfg = (cfg.get("world_model") or {}).get("depth_head") or {}
        logger.info(
            "Phase-2: collection policy = step_e AC + toward_g r_m=%.0f; "
            "wm_update=%s policy_update=%s",
            args.r_m,
            bool(cfg.get("corrector", {}).get("enable_wm_update")),
            bool(cfg.get("corrector", {}).get("enable_policy_update")),
        )
        logger.info(
            "deploy-align collect: planner=%s H=%s tti_coeff=%s fwd_excl=%s "
            "depth_head=%s tau=%s max_steps=%s cs=%s",
            bool(pl_cfg.get("enable")),
            pl_cfg.get("horizon"),
            sf_cfg.get("tti_coeff"),
            sf_cfg.get("exclusion_forward_only"),
            bool(dh_cfg.get("enable")),
            bool(cfg.get("tau_predictor", {}).get("enable")),
            cfg.get("corrector", {}).get("max_steps"),
            sf_cfg.get("v_cruise_m_s"),
        )

    if args.near_goal_frac > 0.0 and loop.episodes is not None:
        from experiments.aerial.rl.train_rl import augment_near_goal_episodes
        loop.episodes = augment_near_goal_episodes(
            loop.episodes,
            near_frac=args.near_goal_frac,
            dist_min_m=args.near_goal_dist_min,
            dist_max_m=args.near_goal_dist_max,
        )

    if args.cs_values and loop.episodes is not None:
        from experiments.aerial.rl.train_rl import assign_variable_cruise_speed
        cs_list = [float(x.strip()) for x in args.cs_values.split(",") if x.strip()]
        if cs_list:
            loop.episodes = assign_variable_cruise_speed(loop.episodes, cs_list)

    loop.config.start_iter = int(args.start_iter)
    if args.ckpt_dir:
        loop.config.ckpt_dir = str(args.ckpt_dir)
    loop.config.save_every_iter = bool(args.save_every_iter)
    if args.renderer_restart_every is not None:
        loop.config.renderer_restart_every = int(args.renderer_restart_every)
    if args.renderer_restart_script:
        loop.config.renderer_restart_script = str(args.renderer_restart_script)
    if args.renderer_restart_scene:
        loop.config.renderer_restart_scene = str(args.renderer_restart_scene)

    reports = loop.run()
    losses = []
    diag_goal_rel: list[float] = []
    diag_progress: list[float] = []
    diag_return: list[float] = []
    for i, r in enumerate(reports):
        rl = r.rl
        logger.info("iter %d rl=%s", i, rl)
        if rl.get("status") == "updated":
            losses.append(float(rl.get("actor_loss", float("nan"))))
            if "mean_abs_goal_rel" in rl:
                diag_goal_rel.append(float(rl["mean_abs_goal_rel"]))
            if "mean_progress" in rl:
                diag_progress.append(float(rl["mean_progress"]))
            if "mean_return" in rl:
                diag_return.append(float(rl["mean_return"]))

    meta = {
        "iters": len(reports),
        "losses": losses,
        "mean_actor_loss": float(sum(losses) / len(losses)) if losses else None,
        "mean_abs_goal_rel": float(sum(diag_goal_rel) / len(diag_goal_rel)) if diag_goal_rel else None,
        "mean_progress": float(sum(diag_progress) / len(diag_progress)) if diag_progress else None,
        "mean_return": float(sum(diag_return) / len(diag_return)) if diag_return else None,
        "policy_class": str(ac_cfg.policy_class),
        "action_limits": [float(x) for x in ac_cfg.action_limits],
        "action_scale": float(ac_cfg.action_scale),
        "step_hz": float(ac_cfg.step_hz),
        "device": str(args.device),
        "dynamics_kind": dyn_kind,
        "wm_ckpt": wm_ckpt_path,
        "latent_dim": int(getattr(loop.dynamics, "latent_dim", 8)),
        "dataset": args.dataset,
        "skip_collect": bool(args.skip_collect),
        "mock_goal_injected": bool(args.backend == "mock" and args.annotation is None),
    }
    print(json.dumps(meta, indent=2))

    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            import torch

            ac = loop.actor_critic
            torch.save(
                {
                    "actor": ac._actor.state_dict(),
                    "critic": ac._critic.state_dict(),
                    "log_std": ac._log_std.detach().cpu(),
                    "config": ac.config.__dict__,
                },
                ckpt_dir / "v4_ac_latest.pt",
            )
            logger.info("wrote %s", ckpt_dir / "v4_ac_latest.pt")
        except Exception as exc:
            logger.warning("ckpt save skipped: %s", exc)

    ok = any(r.rl.get("status") == "updated" for r in reports)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
