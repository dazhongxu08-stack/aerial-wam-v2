"""Composite navigation reward (spec §4.5) + Phase-2 F15 efficiency terms.

Base shape:

    reward = w_prog * progress
           - w_coll * collision_risk
           - w_man  * maneuver_cost
           - w_eff  * efficiency_cost   # F15; weights default 0 = no-op

  * ``progress`` / ``collision_risk`` / ``maneuver_cost`` — as before.
  * ``efficiency_cost`` (F15) — invalid corridor motion: lateral chase, heading
    misalignment while strafing, along-track idle. Soft ``L_act/L_ref`` is
    episode-level (not per-step here).

Product goal is obstacle-aware corridor progress — not only ``Δd_goal − crash``.
Raising ``w_eff_*`` is a training-contract change: declare before use.

``NavigationReward`` is stateful; ``reward_terms`` / ``efficiency_cost`` are pure.
Arrival radius defaults to ``DEFAULT_ONLINE_SUCCESS_DIST_M`` (3 m online), not the
loose 20 m eval SR radius.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from experiments.aerial.eval.metrics import OPENFLY_SUCCESS_DIST_M
from experiments.aerial.rl.env.obs import Observation

# Online arrival / termination radius (m). Tighter than the eval SR metric so a
# bare NavigationReward()/RewardConfig() defaults to THIS, not the loose eval
# radius — code paths that skip the YAML must not silently terminate at 20 m.
DEFAULT_ONLINE_SUCCESS_DIST_M = 3.0
# The loose eval success-rate radius (metrics.OPENFLY_SUCCESS_DIST_M = 20 m),
# re-exported for reference. Intentionally NOT the per-step termination gate.
EVAL_SUCCESS_DIST_M = float(OPENFLY_SUCCESS_DIST_M)


@dataclass
class RewardConfig:
    w_progress: float = 1.0
    w_collision: float = 10.0
    w_maneuver: float = 0.01               # curriculum START weight
    w_curiosity: float = 0.0              # Near-obstacle lateral clearance curiosity
    # Shield-intervention penalty (2026-09-21 contract v2):
    # Real path: charged only when ``obs.info["shield_hard_brake"]`` (exclusion /
    # τ·p_coll emergency). Soft TTI governor may set intervened=True but does
    # not consume this weight. Imagination: binary charge when
    # d_fwd ≤ hard_brake_depth_m (hard-brake analogue), not the soft TTI zone.
    # Default 0 = no-op (declare-before-use).
    w_intervention: float = 0.0
    #: Imagination hard-brake band (m). Keep equal to shield ``exclusion_m`` /
    #: TowardGoalIntent ``d_danger`` (wired from yaml in ``build_from_config``).
    hard_brake_depth_m: float = 3.0
    # F15 efficiency (default 0 = no-op until declared + retrain)
    w_eff_strafe: float = 0.0             # |dy|/max(|dx|,eps) excess above thr
    w_eff_heading: float = 0.0            # |yaw_err| while strafing
    w_eff_idle: float = 0.0               # step with along-track |Δs| ≈ 0
    eff_strafe_thr: float = 0.5
    eff_idle_ds_thr_m: float = 0.05
    # Directional OA + path shaping (2026-09-21 plan). Defaults 0 = no-op until
    # task-3 coefficients are written; then raise and lock via gate sorts.
    w_straight: float = 0.0               # nose-forward along-goal (goal ahead only)
    w_idle_body: float = 0.0              # body translation ≈ 0 (hover / spin)
    w_away: float = 0.0                   # extra cost when progress < 0
    w_level_flight: float = 0.0           # sustained level + non-decreasing dist
    #: Cost for any body-backward ``dx<0`` (stops reverse-SR / nose-away back-in).
    #: Applied even when the carrot is behind — yaw-align first, then nose-forward.
    w_backward: float = 0.0
    #: Hard clamp: never execute/imagine ``dx<0`` (no rear depth sensor).
    forbid_backward_motion: bool = False
    idle_body_trans_thr_m: float = 0.05
    yaw_align_cos_thr: float = 0.5        # goal-aligned pure yaw exemption
    #: Min body-x / |goal_xy| to treat carrot as ahead (cos yaw-err).
    goal_ahead_cos_thr: float = 0.0       # 0 ⇒ forward hemisphere; raise to tighten
    level_window: int = 5
    level_dz_thr_m: float = 0.05
    level_dyaw_thr_rad: float = 0.05
    #: When True, prefer dynamics ``obstacle_cost`` over legacy clearance×10.
    use_learned_obstacle_cost: bool = False
    obstacle_cost_ceiling: float = 1.0    # collided / near-wall upper bound
    #: When learned OA is on, also take max with **action-aligned** cone clearance
    #: so head under-reporting near walls cannot zero the collision term.
    #: Forward clutter must NOT tax left/right/climb escapes (plan failure mode).
    blend_clearance_risk: bool = False
    #: Extra near-miss band (m) for clearance blend; matches shield exclusion.
    near_miss_d_fwd_m: float = 3.0
    near_miss_d_clear_m: float = 12.0
    #: If True, positive Δdist only counts when carrot is ahead of the nose.
    gate_progress_by_heading: bool = False
    curiosity_fwd_thresh_m: float = 3.5   # Trigger exploration when forward depth <= thresh
    curiosity_max_bonus: float = 2.0      # Max curiosity bonus per step
    success_dist_m: float = DEFAULT_ONLINE_SUCCESS_DIST_M
    success_bonus: float = 10.0
    # Maneuver-penalty curriculum (design doc §2.4): keep the aggressive-maneuver
    # penalty small early (exploration matters more than smoothness), then ramp it
    # up as competence rises. ``w_maneuver`` is the start; the effective weight
    # ramps linearly toward ``w_maneuver_final`` over the competence band
    # ``[threshold, threshold + ramp]``. Defaults make the curriculum a NO-OP
    # (final == start), so unconfigured runs behave exactly as before.
    # NOTE (§1.5): the threshold is a project-tuned placeholder for OUR 4-D
    # kinematic SEARCH regime — it is deliberately NOT DreamerV3's reward-50.0.
    w_maneuver_final: float = 0.01
    maneuver_curriculum_threshold: float = 0.0
    maneuver_curriculum_ramp: float = 1.0


# Task-3 locked coefficients (plan 2026-09-21). Empty forward cost = 0;
# near-wall cost = OBSTACLE_COST_NEAR (label / ceiling). w_collision=1 so the
# scalar head (not ×10) sets the metre-scale. Straight loses to near-wall on a
# 1 m head-on step: progress+straight − near ≈ 1+0.5−1 = 0.5 < empty 0.8 m.
OBSTACLE_COST_EMPTY: float = 0.0
OBSTACLE_COST_NEAR: float = 1.0
STRAIGHT_WEIGHT: float = 0.5
IDLE_BODY_WEIGHT: float = 0.5
AWAY_WEIGHT: float = 1.0
LEVEL_FLIGHT_WEIGHT: float = 0.3
BACKWARD_WEIGHT: float = 5.0  # soft backup; hard forbid is primary (no rear sensor)
LEVEL_WINDOW_STEPS: int = 5
LEVEL_DZ_THR_M: float = 0.05
LEVEL_DYAW_THR_RAD: float = 0.05


def directional_oa_reward_cfg(**overrides) -> RewardConfig:
    """RewardConfig for learned directional OA + path shaping (task 3 lock)."""
    base = dict(
        w_progress=1.0,
        # Empirically w_collision=1 leaves wall-fwd ≥ climb-around on real
        # fwd_near features (oc≈1 vs 0.45); 2× restores plan intent that
        # near-wall obstacle beats straight bonus.
        w_collision=2.0,
        w_maneuver=0.01,
        w_straight=STRAIGHT_WEIGHT,
        w_idle_body=IDLE_BODY_WEIGHT,
        w_away=AWAY_WEIGHT,
        w_level_flight=LEVEL_FLIGHT_WEIGHT,
        w_backward=BACKWARD_WEIGHT,
        forbid_backward_motion=True,
        level_window=LEVEL_WINDOW_STEPS,
        level_dz_thr_m=LEVEL_DZ_THR_M,
        level_dyaw_thr_rad=LEVEL_DYAW_THR_RAD,
        goal_ahead_cos_thr=0.0,
        gate_progress_by_heading=True,
        use_learned_obstacle_cost=True,
        obstacle_cost_ceiling=OBSTACLE_COST_NEAR,
        blend_clearance_risk=True,
        near_miss_d_fwd_m=3.0,
        near_miss_d_clear_m=12.0,
        w_intervention=0.0,
    )
    base.update(overrides)
    return RewardConfig(**base)


def goal_ahead_cos(goal_rel: np.ndarray) -> float:
    """``cos(yaw_err)`` = body-x / |goal_xy|; +1 nose-on, ≤0 goal behind/abeam."""
    g = np.asarray(goal_rel, dtype=np.float64).reshape(-1)
    if g.size < 2:
        return 0.0
    gx, gy = float(g[0]), float(g[1])
    n = float(np.hypot(gx, gy))
    if n < 1e-8:
        return 0.0
    return gx / n


def nose_aligned_progress(
    progress: float,
    goal_rel: np.ndarray,
    *,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """Drop positive progress credit when the carrot is behind the nose.

    Closing distance while facing away (reverse / crab-to-goal) previously
    earned full ``w_progress`` and produced reverse-SR arrivals. Negative
    progress (away) is unchanged so ``w_away`` still fires.
    """
    p = float(progress)
    if not bool(getattr(cfg, "gate_progress_by_heading", False)):
        return p
    if p <= 0.0:
        return p
    cos_err = goal_ahead_cos(goal_rel)
    if cos_err < float(cfg.goal_ahead_cos_thr):
        return 0.0
    return p


def maneuver_weight_at(metric: float, cfg: RewardConfig, w_start: Optional[float] = None) -> float:
    """Effective ``w_maneuver`` for a competence ``metric`` (e.g. mean episode return).

    Linearly ramps from the START weight (``w_start`` if given, else
    ``cfg.w_maneuver``) toward ``cfg.w_maneuver_final`` across the band
    ``[threshold, threshold + ramp]``; flat before the threshold. Pass ``w_start``
    explicitly (a snapshot of the base weight) when the caller mutates
    ``cfg.w_maneuver`` between iterations, so the schedule never feeds its own
    output back in as the start. Pure function of scalars — no side effects.
    """
    start = float(cfg.w_maneuver if w_start is None else w_start)
    final = float(cfg.w_maneuver_final)
    threshold = float(cfg.maneuver_curriculum_threshold)
    ramp = float(cfg.maneuver_curriculum_ramp)
    if final == start or metric < threshold:
        return start
    if ramp <= 0.0:
        return final                       # step at the threshold
    frac = min(1.0, max(0.0, (float(metric) - threshold) / ramp))
    return start + frac * (final - start)


def efficiency_cost(
    action: np.ndarray,
    *,
    yaw_err_rad: float = 0.0,
    ds_true_m: float = 1.0,
    cfg: RewardConfig = RewardConfig(),
) -> Dict[str, float]:
    """F15 per-step invalid-motion cost (pure). Soft L_act/L_ref is episode-level.

    With default ``w_eff_*=0``, scalar ``efficiency_cost`` is 0 (no behavior change).
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    dx = float(a[0]) if a.size > 0 else 0.0
    dy = float(a[1]) if a.size > 1 else 0.0
    eps = 1e-3
    strafe_ratio = abs(dy) / max(abs(dx), eps)
    strafe = max(0.0, strafe_ratio - float(cfg.eff_strafe_thr))
    # DECLARE: |yaw_err| × 1{planar maneuver OR pure yaw} — peel must cost, and
    # a pure turn must also see heading cost (else H=1 imag never pays for
    # facing while strafing still earns progress).
    dyaw = float(a[3]) if a.size > 3 else 0.0
    maneuvering = (abs(dx) + abs(dy) + abs(dyaw)) > eps
    heading = abs(float(yaw_err_rad)) if maneuvering else 0.0
    idle = 1.0 if abs(float(ds_true_m)) < float(cfg.eff_idle_ds_thr_m) else 0.0
    cost = (
        float(cfg.w_eff_strafe) * strafe
        + float(cfg.w_eff_heading) * heading
        + float(cfg.w_eff_idle) * idle
    )
    return {
        "efficiency_cost": float(cost),
        "strafe_ratio": float(strafe_ratio),
        "strafe_excess": float(strafe),
        "heading_term": float(heading),
        "idle": float(idle),
    }


def clearance_risk_from_depth(
    d_fwd_m: Optional[float],
    *,
    d_danger: float = 3.0,
    d_clear: float = 22.0,
) -> float:
    """Map predicted forward clearance (m) → [0, 1] risk for imagination RL.

    1.0 at/under ``d_danger``, 0.0 at/above ``d_clear``, linear in between.
    Matches the SceneIntent soft-penalty band so imagined avoidance pressure
    lives on the same metre scale the shield/depth cones already use. Pure.
    """
    if d_fwd_m is None:
        return 0.0
    d = float(d_fwd_m)
    if not np.isfinite(d):
        return 0.0
    lo = float(d_danger)
    hi = float(d_clear)
    if hi <= lo:
        return 1.0 if d <= lo else 0.0
    if d <= lo:
        return 1.0
    if d >= hi:
        return 0.0
    return float(np.clip((hi - d) / (hi - lo), 0.0, 1.0))


def clearance_m_along_action(
    action: np.ndarray,
    cones: Optional[Dict[str, Any]],
    d_fwd_fallback: Optional[float] = None,
) -> Optional[float]:
    """Pick a depth-cone clearance aligned with the body action (not always forward).

    Dominant axis of ``(dx, dy, dz)`` selects forward / left / right / up / down.
    Backward-dominant actions return ``None`` (no blend — avoid taxing peel with
    a forward cliff). Missing cone keys also return ``None``.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size < 3:
        a = np.pad(a, (0, 3 - a.size))
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    cones_d = cones if isinstance(cones, dict) else {}

    def _get(key: str, fallback: Optional[float] = None) -> Optional[float]:
        v = cones_d.get(key)
        if v is not None and np.isfinite(float(v)):
            return float(v)
        if fallback is not None and np.isfinite(float(fallback)):
            return float(fallback)
        return None

    if abs(ax) >= abs(ay) and abs(ax) >= abs(az):
        if ax >= 0.0:
            return _get("forward", d_fwd_fallback)
        return None
    if abs(ay) >= abs(az):
        return _get("left" if ay > 0.0 else "right")
    return _get("up" if az > 0.0 else "down")


def straight_to_goal_bonus(
    action: np.ndarray,
    goal_rel: np.ndarray,
    *,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """Bonus for nose-forward flight into an ahead carrot.

    Requires (1) carrot ahead of the nose (``goal_ahead_cos ≥ thr``) and
    (2) positive body ``dx``. Backing toward a behind-carrot previously scored
    as "straight" because ``dot(disp, goal_rel)>0`` with negative ``dx``.
    """
    w = float(cfg.w_straight)
    if w <= 0.0:
        return 0.0
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    g = np.asarray(goal_rel, dtype=np.float64).reshape(-1)
    dx = float(a[0]) if a.size > 0 else 0.0
    if dx <= 0.0:
        return 0.0
    if goal_ahead_cos(g) < float(cfg.goal_ahead_cos_thr):
        return 0.0
    disp = a[:3] if a.size >= 3 else np.zeros(3)
    gvec = g[:3] if g.size >= 3 else np.zeros(3)
    dn = float(np.linalg.norm(disp))
    gn = float(np.linalg.norm(gvec))
    if dn < 1e-8 or gn < 1e-8:
        return 0.0
    along = float(np.dot(disp, gvec) / gn)
    if along <= 0.0:
        return 0.0
    lat = float(np.linalg.norm(disp - (along / gn) * gvec))
    purity = along / max(along + lat, 1e-6)
    return w * along * purity


def backward_flight_cost(
    action: np.ndarray,
    goal_rel: np.ndarray,
    *,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """Cost for body-backward ``dx<0`` regardless of carrot bearing.

    Holefill reverse collapse: when the carrot is behind, ``gate_progress`` zeros
    credit and the old ahead-only exemption left reverse free — so ``dx<0`` toward
    a behind-carrot became a local optimum (FOV obstacle looks clear, nose away).
    Always tax reverse; pure yaw (``dx≈0``) stays free so face-then-go still works.
    ``goal_rel`` is kept for API symmetry with other path-shaping terms.
    """
    w = float(getattr(cfg, "w_backward", 0.0) or 0.0)
    if w <= 0.0:
        return 0.0
    _ = goal_rel  # unused; signature stable for callers / tests
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    dx = float(a[0]) if a.size > 0 else 0.0
    if dx >= 0.0:
        return 0.0
    return w * float(-dx)


def idle_body_cost(
    action: np.ndarray,
    goal_rel: np.ndarray,
    *,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """Cost when planar/vertical translation ≈ 0, unless goal-aligned yaw.

    Goal-aligned pure yaw (reduces |atan2(left, fwd)|) is exempt so corridor
    face-then-go remains learnable.
    """
    w = float(cfg.w_idle_body)
    if w <= 0.0:
        return 0.0
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    g = np.asarray(goal_rel, dtype=np.float64).reshape(-1)
    dx = float(a[0]) if a.size > 0 else 0.0
    dy = float(a[1]) if a.size > 1 else 0.0
    dz = float(a[2]) if a.size > 2 else 0.0
    dyaw = float(a[3]) if a.size > 3 else 0.0
    trans = abs(dx) + abs(dy) + abs(dz)
    if trans > float(cfg.idle_body_trans_thr_m):
        return 0.0
    # Exemption: yaw that clearly reduces body-frame heading error to goal.
    if g.size >= 2 and abs(dyaw) > 1e-6:
        yaw_err0 = float(np.arctan2(float(g[1]), float(g[0]))) if (
            abs(float(g[0])) + abs(float(g[1])) > 1e-6
        ) else 0.0
        # Already facing goal well enough → pure yaw is idle spin, not align.
        if float(np.cos(yaw_err0)) >= float(cfg.yaw_align_cos_thr):
            return w
        yaw_err1 = yaw_err0 - dyaw
        # wrap to [-pi, pi]
        yaw_err1 = float((yaw_err1 + np.pi) % (2 * np.pi) - np.pi)
        if abs(yaw_err1) + 1e-6 < abs(yaw_err0):
            return 0.0
    return w


def away_from_goal_cost(
    progress: float,
    *,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """Extra cost when the step increases distance to the (carrot) goal."""
    w = float(cfg.w_away)
    if w <= 0.0:
        return 0.0
    if float(progress) >= 0.0:
        return 0.0
    return w * float(-progress)


def level_flight_cost(
    action: np.ndarray,
    progress: float,
    *,
    hist_level: bool,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """Cost for sustained level flight with non-decreasing goal distance.

    ``hist_level`` is True when the last ``level_window`` steps were all
    |dz|<thr, |dyaw|<thr, and progress≤0. Caller maintains the window.
    A single goal-closing horizontal step does not pay.
    """
    w = float(cfg.w_level_flight)
    if w <= 0.0 or not hist_level:
        return 0.0
    if float(progress) > 0.0:
        return 0.0
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    dz = abs(float(a[2])) if a.size > 2 else 0.0
    dyaw = abs(float(a[3])) if a.size > 3 else 0.0
    if dz > float(cfg.level_dz_thr_m) or dyaw > float(cfg.level_dyaw_thr_rad):
        return 0.0
    return w


def path_shaping_terms(
    action: np.ndarray,
    goal_rel: np.ndarray,
    progress: float,
    *,
    hist_level: bool = False,
    cfg: RewardConfig = RewardConfig(),
) -> Dict[str, float]:
    """Straight − idle − away − level − backward. Pure; real and imagined."""
    prog = nose_aligned_progress(progress, goal_rel, cfg=cfg)
    straight = straight_to_goal_bonus(action, goal_rel, cfg=cfg)
    idle = idle_body_cost(action, goal_rel, cfg=cfg)
    away = away_from_goal_cost(prog, cfg=cfg)
    level = level_flight_cost(action, prog, hist_level=hist_level, cfg=cfg)
    backward = backward_flight_cost(action, goal_rel, cfg=cfg)
    return {
        "straight_bonus": float(straight),
        "idle_body_cost": float(idle),
        "away_cost": float(away),
        "level_flight_cost": float(level),
        "backward_cost": float(backward),
        "progress_eff": float(prog),
        "path_shaping": float(straight - idle - away - level - backward),
    }


def reward_terms(
    progress: float,
    collision_risk: float,
    maneuver_cost: float,
    cfg: RewardConfig = RewardConfig(),
    curiosity_gain: float = 0.0,
    efficiency_cost_val: float = 0.0,
    path_shaping_val: float = 0.0,
) -> Dict[str, float]:
    """Pure term breakdown + scalar reward. Used by both real and imagined paths."""
    r = (
        cfg.w_progress * float(progress)
        - cfg.w_collision * float(collision_risk)
        - cfg.w_maneuver * float(maneuver_cost)
        + cfg.w_curiosity * float(np.clip(curiosity_gain, 0.0, cfg.curiosity_max_bonus))
        - float(efficiency_cost_val)
        + float(path_shaping_val)
    )
    return {
        "reward": float(r),
        "progress": float(progress),
        "collision_risk": float(collision_risk),
        "maneuver_cost": float(maneuver_cost),
        "curiosity_gain": float(curiosity_gain),
        "efficiency_cost": float(efficiency_cost_val),
        "path_shaping": float(path_shaping_val),
    }


class NavigationReward:
    """Stateful per-episode reward: progress toward ``goal`` − risk − maneuver + curiosity."""

    def __init__(self, goal: Optional[np.ndarray], cfg: Optional[RewardConfig] = None) -> None:
        self.cfg = cfg or RewardConfig()
        self._goal = None if goal is None else np.asarray(goal, dtype=np.float64).reshape(3)
        self._prev_dist: Optional[float] = None
        self._cum_curiosity: float = 0.0
        self._level_flags: list = []

    def reset(self, goal: Optional[np.ndarray], start_pos: np.ndarray) -> None:
        self._goal = None if goal is None else np.asarray(goal, dtype=np.float64).reshape(3)
        self._prev_dist = self._dist(np.asarray(start_pos, dtype=np.float64).reshape(3))
        self._cum_curiosity = 0.0
        self._level_flags = []

    def _dist(self, pos: np.ndarray) -> Optional[float]:
        if self._goal is None:
            return None
        return float(np.linalg.norm(pos - self._goal))

    def step(
        self,
        obs: Observation,
        action: np.ndarray,
        p_coll: Optional[float] = None,
        intervened: bool = False,
    ) -> tuple[float, bool, Dict[str, float]]:
        """Return ``(reward, done, terms)`` for one real env transition.

        ``intervened`` — shield ``apply_action`` return (hard brake OR soft TTI
        governor). Logged as ``terms["intervened"]``; ``w_intervention`` follows
        ``shield_hard_brake`` only. Imagination has no ``safety.py``; there
        ``w_intervention`` is a binary hard-brake analogue
        (d_fwd ≤ ``hard_brake_depth_m``).
        """
        pos = obs.position
        dist = self._dist(pos)
        info = obs.info if isinstance(obs.info, dict) else {}
        a_arr = np.asarray(action, dtype=np.float64).reshape(-1)

        # Carrot / goal_rel first (toward_g stack) so real path matches imagination.
        goal_rel = info.get("goal_rel")
        if goal_rel is None and self._goal is not None:
            from experiments.aerial.rl.goal_features import goal_rel_body

            goal_rel = goal_rel_body(obs.position, float(obs.yaw), self._goal)
        if goal_rel is None:
            goal_rel = np.zeros(4, dtype=np.float64)
        else:
            goal_rel = np.asarray(goal_rel, dtype=np.float64).reshape(-1)

        progress = 0.0
        use_carrot_progress = bool(
            getattr(self.cfg, "use_learned_obstacle_cost", False)
            or info.get("goal_rel") is not None
        )
        if use_carrot_progress and float(np.linalg.norm(goal_rel[:3])) > 1e-8:
            from experiments.aerial.rl.goal_features import advance_goal_rel_body

            g0 = float(goal_rel[3]) if goal_rel.size > 3 else float(
                np.linalg.norm(goal_rel[:3])
            )
            g1 = float(advance_goal_rel_body(goal_rel, a_arr)[3])
            progress = g0 - g1
        elif dist is not None and self._prev_dist is not None:
            progress = self._prev_dist - dist
        self._prev_dist = dist

        # Near-obstacle lateral clearance curiosity (spec 20260828):
        curiosity_gain = 0.0
        if self.cfg.w_curiosity > 0:
            cones = info.get("depth_cones_pred")
            fwd_d = None
            left_d = None
            right_d = None
            if isinstance(cones, dict):
                fwd_d = cones.get("forward")
                left_d = cones.get("left")
                right_d = cones.get("right")
            if fwd_d is None:
                fwd_d = info.get("depth_min_pred")

            if fwd_d is not None and np.isfinite(float(fwd_d)):
                if float(fwd_d) <= float(self.cfg.curiosity_fwd_thresh_m) and progress <= 0.1:
                    l_val = float(left_d) if (left_d is not None and np.isfinite(float(left_d))) else 5.0
                    r_val = float(right_d) if (right_d is not None and np.isfinite(float(right_d))) else 5.0
                    lat_adv = max(0.0, max(l_val, r_val) - float(fwd_d))
                    dyaw = abs(float(action[3])) if len(action) > 3 else 0.0
                    raw_gain = lat_adv * dyaw
                    avail = max(0.0, 3.0 - self._cum_curiosity)
                    curiosity_gain = min(raw_gain, avail)
                    self._cum_curiosity += curiosity_gain

        clear_risk = 0.0
        d_fwd = info.get("depth_min_pred")
        cones = info.get("depth_cones_pred")
        if isinstance(cones, dict) and cones.get("forward") is not None:
            d_fwd = cones.get("forward")
        if bool(getattr(self.cfg, "use_learned_obstacle_cost", False)):
            # Directional head is primary; optional *action-aligned* clearance
            # blend plugs under-report without taxing sideways escapes with a
            # forward cliff (left-wall / open-fwd failure mode).
            collision_risk = 0.0
            oc = info.get("obstacle_cost")
            if oc is not None and np.isfinite(float(oc)):
                collision_risk = float(oc)
            if bool(getattr(self.cfg, "blend_clearance_risk", False)):
                d_along = clearance_m_along_action(a_arr, cones, d_fwd)
                if d_along is not None:
                    clear_risk = float(
                        clearance_risk_from_depth(
                            d_along,
                            d_danger=float(getattr(self.cfg, "near_miss_d_fwd_m", 3.0)),
                            d_clear=float(getattr(self.cfg, "near_miss_d_clear_m", 12.0)),
                        )
                    )
                    collision_risk = float(max(collision_risk, clear_risk))
            if obs.collided:
                collision_risk = max(
                    float(collision_risk), float(self.cfg.obstacle_cost_ceiling)
                )
        else:
            collision_risk = 1.0 if obs.collided else float(p_coll or 0.0)
            if not obs.collided:
                # Legacy path: shield/depth-head forward clearance.
                clear_risk = float(clearance_risk_from_depth(d_fwd))
                collision_risk = float(max(collision_risk, clear_risk))
        maneuver_cost = float(np.linalg.norm(a_arr))
        # F15: optional geometry from caller via obs.info (default weights 0 → no-op)
        yaw_err = float(info.get("yaw_err_rad", 0.0) or 0.0)
        ds_true = float(info.get("ds_true_m", 1.0) if "ds_true_m" in info else 1.0)
        eff = efficiency_cost(
            action, yaw_err_rad=yaw_err, ds_true_m=ds_true, cfg=self.cfg
        )
        level_step = (
            abs(float(a_arr[2]) if a_arr.size > 2 else 0.0) <= float(self.cfg.level_dz_thr_m)
            and abs(float(a_arr[3]) if a_arr.size > 3 else 0.0) <= float(self.cfg.level_dyaw_thr_rad)
            and float(progress) <= 0.0
        )
        self._level_flags.append(bool(level_step))
        wlen = max(1, int(self.cfg.level_window))
        if len(self._level_flags) > wlen:
            self._level_flags = self._level_flags[-wlen:]
        hist_level = len(self._level_flags) >= wlen and all(self._level_flags)
        shaping = path_shaping_terms(
            a_arr, goal_rel, float(progress), hist_level=hist_level, cfg=self.cfg
        )
        prog_eff = float(shaping.get("progress_eff", progress))
        terms = reward_terms(
            prog_eff,
            collision_risk,
            maneuver_cost,
            self.cfg,
            curiosity_gain=curiosity_gain,
            efficiency_cost_val=float(eff["efficiency_cost"]),
            path_shaping_val=float(shaping["path_shaping"]),
        )
        terms.update({k: eff[k] for k in ("strafe_ratio", "strafe_excess", "heading_term", "idle")})
        terms.update(shaping)
        terms["clearance_risk"] = float(clear_risk)
        terms["progress_raw"] = float(progress)

        intervention_cost = 0.0
        hard_brake = False
        governor_cap = False
        if isinstance(obs.info, dict):
            hard_brake = bool(obs.info.get("shield_hard_brake"))
            governor_cap = bool(obs.info.get("shield_governor_cap"))
        # Charge w_intervention only on hard brake (exclusion/emergency), not
        # mild TTI governor caps — those are speed rules, not pilot takeover.
        if hard_brake and float(self.cfg.w_intervention) > 0.0:
            intervention_cost = float(self.cfg.w_intervention)
            terms["reward"] -= intervention_cost
        terms["intervention_cost"] = float(intervention_cost)
        terms["intervened"] = float(bool(intervened))
        terms["hard_brake"] = float(hard_brake)
        terms["governor_cap"] = float(governor_cap)

        arrived = dist is not None and dist < self.cfg.success_dist_m
        if arrived:
            terms["reward"] += self.cfg.success_bonus
        done = bool(obs.collided or arrived)
        terms["arrived"] = float(arrived)
        return terms["reward"], done, terms
