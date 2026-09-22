"""Entrypoint wiring the serial-corrector loop from ``configs/aerial_rl.yaml``.

    python -m experiments.aerial.rl.train_rl                     # mock, V0 loop
    python -m experiments.aerial.rl.train_rl corrector.smoke=true env.backend=airsim

``build_from_config`` assembles env / buffer / dynamics / policy / collector /
corrector from a plain (Omega)Conf-like mapping and is directly unit-testable
without Hydra. ``main`` is the Hydra wrapper (config dir = repo ``configs/``).

The default policy is a lightweight goal-seeking heuristic (``HeuristicPolicy``)
so the V0 collection loop runs end-to-end with no checkpoint. Swap in
``FastWAMAerialPolicy`` via ``build_policy`` once a checkpoint is available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.collector import RolloutCollector
from experiments.aerial.rl.corrector import CorrectorConfig, SerialCorrectorLoop
from experiments.aerial.rl.dynamics import StubLatentDynamics
from experiments.aerial.rl.env.action import DEFAULT_STEP_HZ, body_delta_limits
from experiments.aerial.rl.env.obs import PolicyObservation
from experiments.aerial.rl.reward import DEFAULT_ONLINE_SUCCESS_DIST_M, RewardConfig
from experiments.aerial.rl.safety import (
    DepthTauShield,
    NullSafetyShield,
    ThreeZoneSpeedShield,
    ThresholdSafetyShield,
)
from experiments.aerial.rl.three_zone import ThreeZoneSpec

logger = logging.getLogger(__name__)


class HeuristicPolicy:
    """PRIVILEGED goal-seeking stand-in — NOT an RGB policy.

    Steers toward the (externally supplied) goal using only proprio (x, y, z,
    yaw), so it respects the RGB-only input boundary — it receives a
    ``PolicyObservation`` and never touches depth/IMU. But it reaches the goal
    via a straight-line oracle rather than perception, so it exists only to
    exercise the V0 collection path end-to-end with no checkpoint. Swap in a
    learned ``act(view)`` policy once one exists. With no goal it idles.
    """

    def __init__(self, goal_getter, step_m: float = 3.0) -> None:
        self._goal_getter = goal_getter
        self.step_m = float(step_m)

    def reset(self) -> None:
        return None

    def act(self, obs: PolicyObservation) -> np.ndarray:
        goal = self._goal_getter()
        if goal is None:
            return np.zeros(4, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64).reshape(3)
        d_world = goal - obs.position
        yaw = obs.yaw
        c, s = np.cos(yaw), np.sin(yaw)
        dx = c * d_world[0] + s * d_world[1]      # world -> body
        dy = -s * d_world[0] + c * d_world[1]
        dz = d_world[2]
        vec = np.array([dx, dy, dz], dtype=np.float64)
        n = np.linalg.norm(vec)
        if n > self.step_m:
            vec = vec / n * self.step_m
        # Return the step_m-scaled body delta RAW — do NOT clip to the default
        # 30 Hz cap here. Doing so silently locked every step to
        # ``body_delta_limits(1/30) ≈ [0.167, .067, .067] m`` regardless of
        # ``step_m`` or the env's actual ``step_hz``, so a 5 Hz rollout crawled
        # ~0.167 m/step instead of the physical 1.0 m cap (5 m/s ÷ 5 Hz) and the
        # probe/eval could never reach an obstacle. The collector's ``act_delta``
        # re-clips with the rate-correct ``body_delta_limits(1/step_hz)``, so this
        # command is bounded downstream; ``step_m`` bounds ‖vec‖ here.
        return np.array([vec[0], vec[1], vec[2], 0.0], dtype=np.float64)


class Phase2CollectionPolicy:
    """Phase-2 data collection: learned AC + toward_g outer loop.

    Collector must call ``stamp_local_goal(obs)`` on the **live** Observation
    *before* ``act_delta`` (same order as ``wam_phase2_long_eval``). That writes
    the clipped subgoal to ``obs.info["goal"]`` so planner / WM ``goal_rel`` /
    buffer stamps / imagination all see toward_g — not only the frozen
    ``PolicyObservation`` inside ``act()``.

    Uses ``TowardGoalIntent`` (depth-adaptive ``r`` + ``safe_speed_limit``) so
    train collect matches gate eval's outer loop.
    """

    def __init__(
        self,
        inner: Any,
        goal_getter: Any,
        r_m: float = 100.0,
        *,
        cruise_speed: float = 10.0,
        d_danger: float = 3.0,
        d_clear: float = 22.0,
        min_creep_speed: float = 1.0,
    ) -> None:
        from experiments.aerial.rl.scene_intent import TowardGoalIntent

        self._inner = inner
        self._goal_getter = goal_getter
        self._r_m = float(r_m)
        self._intent = TowardGoalIntent(
            r_m=float(r_m),
            mode="toward_g",
            cruise_speed=float(cruise_speed),
            d_danger=float(d_danger),
            d_clear=float(d_clear),
            min_creep_speed=float(min_creep_speed),
        )
        self._last_target: Optional[np.ndarray] = None
        self._last_safe_speed: float = float(cruise_speed)

    def reset(self) -> None:
        self._last_target = None
        self._last_safe_speed = float(self._intent.cruise_speed)
        self._intent.reset()
        if hasattr(self._inner, "reset"):
            self._inner.reset()

    def _d_fwd_hat(self, obs: Any) -> Optional[float]:
        info = getattr(obs, "info", None)
        if not isinstance(info, dict):
            return None
        cones = info.get("depth_cones_pred")
        if isinstance(cones, dict) and cones.get("forward") is not None:
            try:
                v = float(cones["forward"])
                return v if np.isfinite(v) else None
            except (TypeError, ValueError):
                return None
        raw = info.get("depth_min_pred")
        if raw is None:
            return None
        try:
            v = float(raw)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    def stamp_local_goal(self, obs: Any) -> Optional[np.ndarray]:
        """Depth-adaptive toward_g carrot + safe_speed → live ``obs.info``."""
        goal_G = self._goal_getter()
        if goal_G is None:
            self._last_target = None
            return None
        pos = np.asarray(obs.position, dtype=np.float64)
        yaw = float(getattr(obs, "yaw", 0.0))
        _g_rel, info = self._intent.compute(
            pos,
            yaw,
            np.asarray(goal_G, dtype=np.float64),
            d_fwd_hat=self._d_fwd_hat(obs),
        )
        target = np.asarray(info["target_world"], dtype=np.float64).reshape(3)
        safe_v = float(info.get("safe_speed_limit", self._intent.cruise_speed))
        obs_info = getattr(obs, "info", None)
        if isinstance(obs_info, dict):
            obs_info["goal"] = target.tolist()
            obs_info["safe_speed_limit"] = safe_v
            obs_info["r_lookahead"] = info.get("r_lookahead")
            if info.get("yaw_err_rad") is not None:
                obs_info["yaw_err_rad"] = float(info["yaw_err_rad"])
        self._last_target = target.copy()
        self._last_safe_speed = safe_v
        return self._last_target

    def act(self, obs: Any) -> np.ndarray:
        # Prefer collector-stamped target; fall back to intent if stamp was skipped.
        target = self._last_target
        if target is None:
            self.stamp_local_goal(obs)
            target = self._last_target
        if target is not None:
            target = np.asarray(target, dtype=np.float64).reshape(3)
            try:
                object.__setattr__(obs, "goal", target)
            except (AttributeError, TypeError):
                info = getattr(obs, "info", None)
                if isinstance(info, dict):
                    info["goal"] = target.tolist()
        return self._inner.act(obs)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _build_env(env_cfg: Any) -> Any:
    backend = str(_get(env_cfg, "backend", "mock"))
    if backend == "mock":
        from experiments.aerial.rl.env.mock_env import MockAirSimDroneEnv, MockEnvConfig

        return MockAirSimDroneEnv(MockEnvConfig(
            width=int(_get(env_cfg, "width", 224)),
            height=int(_get(env_cfg, "height", 224)),
            step_hz=float(_get(env_cfg, "step_hz", 30.0)),
            seed=int(_get(env_cfg, "seed", 0)),
        ))
    if backend == "airsim":
        from experiments.aerial.rl.env.airsim_env import AirSimDroneEnv, AirSimEnvConfig

        return AirSimDroneEnv(AirSimEnvConfig(
            host=str(_get(env_cfg, "host", "127.0.0.1")),
            port=int(_get(env_cfg, "port", 41451)),
            camera=str(_get(env_cfg, "camera", "front_custom")),
            vehicle=str(_get(env_cfg, "vehicle", "drone_1")),
            width=int(_get(env_cfg, "width", 224)),
            height=int(_get(env_cfg, "height", 224)),
            step_hz=float(_get(env_cfg, "step_hz", 30.0)),
            health_check=bool(_get(env_cfg, "health_check", True)),
            grab_depth=bool(_get(env_cfg, "grab_depth", True)),
            fanout_rgb=bool(_get(env_cfg, "fanout_rgb", False)),
            wam_encode_size=int(_get(env_cfg, "wam_encode_size", 224)),
        ))
    raise ValueError(f"unknown env backend {backend!r} (expected mock|airsim)")


def _build_dynamics(dyn_cfg: Any, *, success_dist_m: float, wm_cfg: Any = None) -> Any:
    """Dispatch on ``dynamics.kind``. ``wan`` is an offline distillation source
    (spec §4.4/§11) — it must NOT drive the online corrector, so selecting it
    here is a hard error rather than a silent fall-back to the stub. ``torch``
    is the real DreamerV3 RSSM WM (V1) and needs torch (H100); it reads the
    ``world_model:`` block, imported lazily so the stub/mock path stays torch-free."""
    kind = str(_get(dyn_cfg, "kind", "stub"))
    if kind == "stub":
        return StubLatentDynamics(
            goal=None,  # set per-episode by the corrector before imagination
            latent_dim=int(_get(dyn_cfg, "latent_dim", 8)),
            collide_radius_m=float(_get(dyn_cfg, "collide_radius_m", 2.0)),
            success_dist_m=float(success_dist_m),
        )
    if kind == "wan":
        raise ValueError(
            "dynamics.kind='wan' (WanImaginationDynamics) is an OFFLINE "
            "distillation source only (spec §4.4/§11) — stepping the Wan2.2 "
            "pixel model in the online RL loop is a non-goal. Use kind='stub' "
            "for V0; the real fast latent WM (V1) drops into the stub slot."
        )
    if kind == "torch":
        try:
            from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics
        except ImportError as exc:  # torch absent (dev host) -> clear H100 pointer
            raise RuntimeError(
                "dynamics.kind='torch' needs the torch DreamerV3 RSSM WM, which "
                "runs on the H100 (torch 2.7.1+cu128) — it is not importable here. "
                f"Use kind='stub' on the GPU-less host. (import error: {exc})"
            ) from exc
        return TorchRSSMDynamics.from_config(wm_cfg or {})
    raise ValueError(f"unknown dynamics kind {kind!r} (expected stub|wan|torch)")


def bind_loaded_dynamics(loop: Any, dynamics: Any) -> None:
    """Point loop + collector (+ planner) at the same loaded WM instance.

    ``build_from_config`` may construct a randomly-init torch WM; after
    ``load_torch_dynamics`` the collect / shield / imagination path must share
    the loaded weights (2026-09-20 dual-WM audit).
    """
    loop.dynamics = dynamics
    collector = getattr(loop, "collector", None)
    if collector is not None:
        collector.dynamics = dynamics
        planner = getattr(collector, "planner", None)
        if planner is not None and hasattr(planner, "dynamics"):
            planner.dynamics = dynamics


def load_torch_dynamics(
    wm_cfg: Any,
    ckpt_path: Any,
    *,
    device: str = "cuda",
    success_dist_m: float = 3.0,
    freeze: bool = True,
) -> tuple[Any, Dict[str, Any]]:
    """Build ``TorchRSSMDynamics`` and load WM weights (shared train + deploy path).

    ``freeze=True`` (default): freeze all WM params for inference-only use.
    ``freeze=False``: keep params trainable for joint WM+AC update (Phase-2 Direction A).
    """
    from pathlib import Path

    ckpt = Path(ckpt_path).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"WM checkpoint not found: {ckpt}")
    cfg = dict(wm_cfg or {})
    cfg["device"] = str(device)
    dynamics = _build_dynamics(
        {"kind": "torch"},
        success_dist_m=float(success_dist_m),
        wm_cfg=cfg,
    )
    payload = dynamics.load_checkpoint(str(ckpt))
    if freeze:
        dynamics.eval()
        for param in dynamics.parameters():
            param.requires_grad_(False)
    else:
        dynamics.train()
    return dynamics, payload


def _build_safety(safety_cfg: Any) -> Any:
    kind = str(_get(safety_cfg, "kind", "null"))
    if kind in ("null", "none", "None"):
        return NullSafetyShield()
    if kind == "threshold":
        # min_depth_m is the reaction STANDOFF, not the ④a near-collision metric
        # (frozen at 1.5). Default 3.0 m gives the shield room to intervene before
        # the band (frozen-spec ④a re-freeze 2026-08-11).
        return ThresholdSafetyShield(
            min_depth_m=float(_get(safety_cfg, "min_depth_m", 3.0)),
            min_tau_s=float(_get(safety_cfg, "min_tau_s", 1.0)),
            max_p_coll=float(_get(safety_cfg, "max_p_coll", 0.5)),
        )
    if kind == "depth_tau":
        return DepthTauShield(
            min_depth_m=float(_get(safety_cfg, "min_depth_m", 3.0)),
            min_tau_s=float(_get(safety_cfg, "min_tau_s", 1.0)),
            max_p_coll=float(_get(safety_cfg, "max_p_coll", 0.5)),
        )
    if kind in ("three_zone", "three_zone_speed"):
        zone = ThreeZoneSpec.from_mapping(safety_cfg)
        return ThreeZoneSpeedShield(
            zone=zone,
            min_tau_s=float(_get(safety_cfg, "min_tau_s", 1.0)),
            max_p_coll=float(_get(safety_cfg, "max_p_coll", 0.5)),
            retreat_step_m=float(_get(safety_cfg, "retreat_step_m", 3.0)),
            retreat_max_dyaw_rad=float(_get(safety_cfg, "retreat_max_dyaw_rad", 0.0)),
            retreat_max_cum_yaw_rad=float(
                _get(safety_cfg, "retreat_max_cum_yaw_rad", 0.70)
            ),
            retreat_max_head_vs_goal_rad=float(
                _get(safety_cfg, "retreat_max_head_vs_goal_rad", 1.047)
            ),
            tti_coeff=float(_get(safety_cfg, "tti_coeff", 4.0)),
            tti_hysteresis_release_frac=float(
                _get(safety_cfg, "tti_hysteresis_release_frac", 0.0)
            ),
            exclusion_forward_only=bool(
                _get(safety_cfg, "exclusion_forward_only", False)
            ),
        )
    raise ValueError(f"unknown safety kind {kind!r}")


def _build_depth_predictor(wm_cfg: Any) -> Optional[Any]:
    dh = _get(wm_cfg, "depth_head", {}) if wm_cfg else {}
    if not bool(_get(dh, "enable", False)):
        return None
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor

    ckpt_path = _get(dh, "checkpoint_path", None)
    if ckpt_path is None:
        ckpt_dir = _get(dh, "checkpoint_dir", None)
        if ckpt_dir:
            candidates = sorted(Path(str(ckpt_dir)).glob("depth_step_*.pt"))
            if candidates:
                ckpt_path = str(candidates[-1])
    if ckpt_path:
        device = str(_get(wm_cfg, "device", "cpu"))
        return DepthMinPredictor.from_checkpoint(ckpt_path, device=device)
    return DepthMinPredictor(n_frames=int(_get(dh, "n_frames", 4)))


def _build_tau_predictor(tau_cfg: Any) -> Optional[Any]:
    if not bool(_get(tau_cfg, "enable", False)):
        return None
    from experiments.aerial.rl.tau_predictor import make_tau_predictor

    kind = str(_get(tau_cfg, "kind", "gt_proxy"))
    ckpt = _get(tau_cfg, "ckpt", None)
    return make_tau_predictor(
        kind=kind,
        center_frac=float(_get(tau_cfg, "center_frac", 0.5)),
        min_closing_m_s=float(_get(tau_cfg, "min_closing_m_s", 0.05)),
        max_tau_s=float(_get(tau_cfg, "max_tau_s", 60.0)),
        use_gt_depth=bool(_get(tau_cfg, "use_gt_depth", True)),
        dt_s=float(_get(tau_cfg, "dt_s", 0.1)),
        ckpt=ckpt,
        device=str(_get(tau_cfg, "device", "cpu")),
    )


def _build_planner(cfg: Any, dynamics: Any, reward_cfg: RewardConfig) -> Optional[Any]:
    pc = _get(cfg, "planner", {})
    if not bool(_get(pc, "enable", False)):
        return None
    from experiments.aerial.rl.planner import ImaginationPlanner

    step_hz = float(_get(_get(cfg, "env", {}), "step_hz", DEFAULT_STEP_HZ))
    limits = body_delta_limits(1.0 / step_hz)
    rollout_mode = str(_get(pc, "rollout_mode", "open_loop") or "open_loop")
    return ImaginationPlanner(
        dynamics,
        horizon=int(_get(pc, "horizon", 5)),
        reward_cfg=reward_cfg,
        action_limits=limits,
        rollout_mode=rollout_mode,
        # closed_loop tail_policy wired after actor warm-start in train_v4_ac.
        tail_policy=None,
    )


def _load_episodes(cfg: Any) -> Optional[List[Dict[str, Any]]]:
    ann = _get(cfg, "annotation", None)
    if not ann:
        return None
    from experiments.aerial.eval.run_closed_loop import load_annotation

    episodes = load_annotation(Path(str(ann)))
    max_eps = int(_get(cfg, "max_episodes", 20))
    if max_eps <= 0:
        return episodes
    return episodes[:max_eps]


def augment_near_goal_episodes(
    episodes: List[Dict[str, Any]],
    near_frac: float,
    dist_min_m: float = 5.0,
    dist_max_m: float = 30.0,
    rng: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Append near-goal spawn variants to the episode list.

    For each route, samples waypoints within [dist_min_m, dist_max_m] of the
    goal as alternative start positions.  These teach the policy to complete
    the final approach — a scenario underrepresented when every episode starts
    100-500 m from goal.

    The augmented episodes are appended so the collector cycles through them
    alongside the originals.  Mix ratio = near_frac / (1 - near_frac).
    """
    if near_frac <= 0.0:
        return episodes
    _rng = np.random.default_rng(rng)
    near_eps: List[Dict[str, Any]] = []
    for ep in episodes:
        pts = np.array(ep.get("pos", []), dtype=np.float64)
        if len(pts) < 2:
            continue
        goal = pts[-1]
        # distances of each waypoint (except goal itself) from goal
        dists = np.linalg.norm(pts[:-1] - goal, axis=1)
        idxs = np.where((dists >= dist_min_m) & (dists <= dist_max_m))[0]
        if len(idxs) == 0:
            continue
        for idx in idxs:
            pt = pts[idx]
            d_vec = goal[:2] - pt[:2]
            spawn_yaw = float(np.arctan2(d_vec[1], d_vec[0]))
            near_ep: Dict[str, Any] = {
                "pos": [pt.tolist(), goal.tolist()],
                "yaw": [spawn_yaw, spawn_yaw],
                "gpt_instruction": ep.get("gpt_instruction", ""),
                "_near_goal_spawn": True,
            }
            if ep.get("scene"):
                near_ep["scene"] = ep["scene"]
            if ep.get("pose_source"):
                near_ep["pose_source"] = ep["pose_source"]
            near_eps.append(near_ep)
    if not near_eps:
        logger.warning("augment_near_goal_episodes: no waypoints in [%.0f, %.0f]m of goal", dist_min_m, dist_max_m)
        return episodes
    # target: near_frac fraction of total
    n_orig = len(episodes)
    n_near_target = max(1, int(round(n_orig * near_frac / max(1.0 - near_frac, 1e-6))))
    # sample with replacement if needed
    chosen = [near_eps[i % len(near_eps)] for i in _rng.permutation(n_near_target)]
    logger.info(
        "near-goal augment: %d original + %d near-goal episodes "
        "(frac=%.0f%%, d=[%.0f, %.0f]m)",
        n_orig, len(chosen), 100.0 * near_frac, dist_min_m, dist_max_m,
    )
    combined = list(episodes) + chosen
    _rng.shuffle(combined)
    return combined


def assign_variable_cruise_speed(
    episodes: List[Dict[str, Any]],
    cs_values: List[float],
    rng: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Annotate each episode with a randomly chosen cruise_speed from cs_values.

    The collector reads ``episode["cruise_speed"]`` and caps ``limits[0]`` +
    updates ``shield.zone.v_cruise_m_s`` for that episode, so the policy
    trains under varied speed regimes rather than a fixed cs=10.
    """
    if not cs_values:
        return episodes
    _rng = np.random.default_rng(rng)
    cs_arr = np.array([float(c) for c in cs_values])
    result = []
    for ep in episodes:
        chosen = float(_rng.choice(cs_arr))
        result.append({**ep, "cruise_speed": chosen})
    logger.info(
        "variable-cs: %d episodes annotated with cs in %s",
        len(result), sorted(float(c) for c in cs_values),
    )
    return result


def _build_actor_critic(cfg: Any, latent_dim: int) -> Optional[Any]:
    """Build V4 actor-critic when torch is available (lazy import)."""
    v4 = _get(cfg, "v4", {})
    if not v4:
        return None
    try:
        from experiments.aerial.rl.actor_critic import ActorCriticConfig, LatentActorCritic
    except RuntimeError:
        return None
    # C2 (2026-08-18): the policy's action box comes from the SAME ``step_hz``
    # the env is built with, so the bounded policy distribution and the
    # deployment-side clip (``collector.py:167``) agree by construction.
    step_hz = float(_get(_get(cfg, "env", {}), "step_hz", 30.0))
    ac_cfg = ActorCriticConfig(
        latent_dim=int(latent_dim),
        lambda_gae=float(_get(v4, "lambda_gae", 0.95)),
        gamma=float(_get(v4, "gamma", 0.997)),
        entropy_scale=float(_get(v4, "entropy_scale", 3.0e-4)),
        actor_lr=float(_get(v4, "actor_lr", 1.0e-4)),
        critic_lr=float(_get(v4, "critic_lr", 1.0e-4)),
        action_scale=float(_get(v4, "action_scale", 1.0)),
        step_hz=step_hz,
        device=str(_get(v4, "device", "cpu")),
    )
    ac = LatentActorCritic(config=ac_cfg)
    logger.info(
        "actor-critic: policy_class=%s step_hz=%.3f action_limits=%s action_scale=%.3f",
        ac_cfg.policy_class, step_hz, ac_cfg.action_limits, ac_cfg.action_scale,
    )
    return ac


def build_from_config(cfg: Any) -> SerialCorrectorLoop:
    env = _build_env(_get(cfg, "env", {}))
    buf_cfg = _get(cfg, "buffer", {})
    buffer = ReplayBuffer(
        capacity_episodes=int(_get(buf_cfg, "capacity_episodes", 1000)),
        seed=int(_get(buf_cfg, "seed", 0)),
    )

    rc = _get(cfg, "reward", {})
    w_maneuver = float(_get(rc, "w_maneuver", 0.01))
    reward_cfg = RewardConfig(
        w_progress=float(_get(rc, "w_progress", 1.0)),
        w_collision=float(_get(rc, "w_collision", 10.0)),
        w_maneuver=w_maneuver,
        # Shield-intervention cost (default 0 = no-op). Must be read here —
        # train_v4_ac --w-intervention only writes cfg["reward"]; dropping it
        # from RewardConfig construction silently zeroed every e5/clearrisk run
        # (diagnosed 2026-09-20 full-path audit).
        w_intervention=float(_get(rc, "w_intervention", 0.0)),
        # Imag hard-brake band: prefer reward override, else safety.exclusion_m.
        hard_brake_depth_m=float(
            _get(
                rc,
                "hard_brake_depth_m",
                _get(_get(cfg, "safety", {}), "exclusion_m", 3.0),
            )
        ),
        # F15 efficiency (default 0 = no-op until DECLARE short-train overrides).
        w_eff_strafe=float(_get(rc, "w_eff_strafe", 0.0)),
        w_eff_heading=float(_get(rc, "w_eff_heading", 0.0)),
        w_eff_idle=float(_get(rc, "w_eff_idle", 0.0)),
        eff_strafe_thr=float(_get(rc, "eff_strafe_thr", 0.5)),
        eff_idle_ds_thr_m=float(_get(rc, "eff_idle_ds_thr_m", 0.05)),
        # Directional OA + path shaping (2026-09-21 plan); default 0 / False = no-op.
        w_straight=float(_get(rc, "w_straight", 0.0)),
        w_idle_body=float(_get(rc, "w_idle_body", 0.0)),
        w_away=float(_get(rc, "w_away", 0.0)),
        w_level_flight=float(_get(rc, "w_level_flight", 0.0)),
        w_backward=float(_get(rc, "w_backward", 0.0)),
        forbid_backward_motion=bool(_get(rc, "forbid_backward_motion", False)),
        idle_body_trans_thr_m=float(_get(rc, "idle_body_trans_thr_m", 0.05)),
        yaw_align_cos_thr=float(_get(rc, "yaw_align_cos_thr", 0.5)),
        goal_ahead_cos_thr=float(_get(rc, "goal_ahead_cos_thr", 0.0)),
        gate_progress_by_heading=bool(_get(rc, "gate_progress_by_heading", False)),
        level_window=int(_get(rc, "level_window", 5)),
        level_dz_thr_m=float(_get(rc, "level_dz_thr_m", 0.05)),
        level_dyaw_thr_rad=float(_get(rc, "level_dyaw_thr_rad", 0.05)),
        use_learned_obstacle_cost=bool(_get(rc, "use_learned_obstacle_cost", False)),
        obstacle_cost_ceiling=float(_get(rc, "obstacle_cost_ceiling", 1.0)),
        blend_clearance_risk=bool(_get(rc, "blend_clearance_risk", False)),
        near_miss_d_fwd_m=float(_get(rc, "near_miss_d_fwd_m", 3.0)),
        near_miss_d_clear_m=float(_get(rc, "near_miss_d_clear_m", 12.0)),
        # Online arrival/termination radius — tighter than the eval SR metric
        # (EVAL_SUCCESS_DIST_M=20 m); falls back to the tight online default.
        success_dist_m=float(_get(rc, "success_dist_m", DEFAULT_ONLINE_SUCCESS_DIST_M)),
        success_bonus=float(_get(rc, "success_bonus", 10.0)),
        # Maneuver-penalty curriculum (§2.4); defaults leave it a no-op.
        w_maneuver_final=float(_get(rc, "w_maneuver_final", w_maneuver)),
        maneuver_curriculum_threshold=float(_get(rc, "maneuver_curriculum_threshold", 0.0)),
        maneuver_curriculum_ramp=float(_get(rc, "maneuver_curriculum_ramp", 1.0)),
    )

    # Imagined dynamics shares the reward's arrival radius so imagined and real
    # returns agree on when the goal is reached. The world_model block feeds the
    # torch DreamerV3 WM (kind=torch); ignored by stub/wan.
    dynamics = _build_dynamics(
        _get(cfg, "dynamics", {}),
        success_dist_m=reward_cfg.success_dist_m,
        wm_cfg=_get(cfg, "world_model", {}),
    )

    cc = _get(cfg, "corrector", {})
    ic = _get(cfg, "imagination", {})
    env_cfg = _get(cfg, "env", {})
    corrector_cfg = CorrectorConfig(
        iterations=int(_get(cc, "iterations", 10)),
        episodes_per_iter=int(_get(cc, "episodes_per_iter", 1)),
        enable_wm_update=bool(_get(cc, "enable_wm_update", False)),
        enable_policy_update=bool(_get(cc, "enable_policy_update", False)),
        strict_gates=bool(_get(cc, "strict_gates", False)),
        wm_batch=int(_get(cc, "wm_batch", 32)),
        wm_window=int(_get(cc, "wm_window", 8)),
        imagine_batch=int(_get(ic, "batch", 64)),
        imagine_horizon=int(_get(ic, "horizon", 10)),
        smoke=bool(_get(cc, "smoke", False)),
        renderer_restart_every=int(_get(cc, "renderer_restart_every", 0)),
        renderer_restart_script=_get(cc, "renderer_restart_script", None),
        renderer_restart_scene=str(_get(cc, "renderer_restart_scene", "outdoor")),
        renderer_restart_wait_s=float(_get(cc, "renderer_restart_wait_s", 30.0)),
        renderer_host=str(_get(cc, "renderer_host", _get(env_cfg, "host", "127.0.0.1"))),
        renderer_port=int(_get(cc, "renderer_port", _get(env_cfg, "port", 41451))),
    )

    policy = HeuristicPolicy(goal_getter=lambda: getattr(env, "goal", None))
    planner = _build_planner(cfg, dynamics, reward_cfg)
    scene_profiles = None
    scene_profiles_raw = _get(cfg, "scene_profiles", None)
    if scene_profiles_raw:
        from experiments.aerial.rl.scene_profile import load_scene_profiles_from_mapping

        scene_profiles = load_scene_profiles_from_mapping(scene_profiles_raw)

    collector = RolloutCollector(
        env, policy, buffer,
        reward_cfg=reward_cfg,
        safety=_build_safety(_get(cfg, "safety", {})),
        max_steps=int(_get(cc, "max_steps", 200)),
        target_hz=float(_get(_get(cfg, "env", {}), "step_hz", 30.0)),
        min_spawn_z=float(_get(cc, "min_spawn_z", 0.0)),
        spawn_z_retry_m=float(_get(cc, "spawn_z_retry_m", 0.0)),
        spawn_z_max_retries=int(_get(cc, "spawn_z_max_retries", 0)),
        depth_predictor=_build_depth_predictor(_get(cfg, "world_model", {})),
        tau_predictor=_build_tau_predictor(_get(cfg, "tau_predictor", {})),
        planner=planner,
        dynamics=dynamics,
        scene_profiles=scene_profiles,
    )
    episodes = _load_episodes(cfg)
    latent_dim = int(getattr(dynamics, "latent_dim", 8))
    actor_critic = _build_actor_critic(cfg, latent_dim)
    imagination_policy = None
    if actor_critic is not None:
        from experiments.aerial.rl.actor_critic import ImaginationActorPolicy

        imagination_policy = ImaginationActorPolicy(actor_critic)
    return SerialCorrectorLoop(
        collector, buffer, dynamics,
        imagination_policy=imagination_policy,
        actor_critic=actor_critic,
        config=corrector_cfg, episodes=episodes,
    )


def main() -> None:  # pragma: no cover - Hydra wrapper
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(version_base="1.3", config_path="../../../configs", config_name="aerial_rl")
    def _run(cfg: "DictConfig") -> None:
        logging.basicConfig(level=logging.INFO)
        loop = build_from_config(OmegaConf.to_container(cfg, resolve=True))
        reports = loop.run()
        total_steps = sum(r.collect.steps for r in reports)
        logger.info("done: %d iters, %d env steps", len(reports), total_steps)

    _run()


if __name__ == "__main__":  # pragma: no cover
    main()
