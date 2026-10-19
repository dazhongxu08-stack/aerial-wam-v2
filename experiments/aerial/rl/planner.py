"""Short-horizon imagination planner (V1b, frozen spec §7).

Scores a small set of candidate body deltas by rolling them forward through a
``LatentDynamics`` model for ``horizon`` steps (≤ ``MAX_IMAGINATION_HORIZON``),
then returns the first action of the highest-return sequence. This is the
test-time imagination scoring path — distinct from V4 actor-critic training.

2026-09-21 face-goal / anti-crab (three clearance zones) + R01 escape:
- Open (d_fwd ≥ 8 m): drop pure-strafe off-axis + strong face bias.
- Mid (3–8 m): face bias + strafe score penalty; keep lateral ability.
  Tight mid (<5 m): also offer escape peel.
- Blocked (< 3 m): ban pure crab; escape toward clearer side cone (or goal /
  both-side fallback). Re-offer escape after goal-turn filter so open-side ≠
  carrot still has a leave-wall atom (R01).
- Extreme |yaw_err| in tight nose: suppress face-into-wall; score-boost escape;
  tax pure-forward / zero when nose is jammed.
- Open-lane / standoff disabled (v7–v9 regressed SR).
- Unknown depth: mid-like soft face + escape available.
"""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiments.aerial.rl.dynamics import LatentDynamics
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.goal_features import body_vel_from_obs, goal_rel_from_obs
from experiments.aerial.rl.imagination import MAX_IMAGINATION_HORIZON, imagine
from experiments.aerial.rl.reward import RewardConfig

# Body-frame fwd above this → subgoal is ahead; drop pure-backward planner atom.
_SUBGOAL_AHEAD_FWD_M = 0.05
#: |yaw_err| above this → drop pure lateral candidates in *open* space (≈35°).
_YAW_OFF_AXIS_THR_RAD = float(np.deg2rad(35.0))
#: Open: hard anti-crab (drop + strong face). Matches prior facegoal open gate.
_FACE_OPEN_FWD_M = 8.0
#: Below this ≈ shield hard-brake band — no face-into-wall; ban pure crab.
_FACE_BLOCK_FWD_M = 3.0
#: Face residual weight in open space (beats ~0.4 m strafe progress).
_FACE_YAW_SCORE_W = 2.0
#: Mid / unknown: still prefer facing, but weaker than open.
_FACE_YAW_SCORE_W_MID = 1.2
_FACE_YAW_SCORE_W_UNKNOWN = 1.0
#: Mid / unknown: extra score cost on |dy| when yaw is off-axis (discourage crab
#: without removing the candidate — ability preserved for tight gaps).
_STRAFE_SCORE_W_MID = 1.0
#: Min side-cone advantage (m) to pick an escape side in the block zone.
_ESCAPE_CONE_MARGIN_M = 1.0
#: Also offer escape candidates in this tight-mid band (urban wall approach).
_ESCAPE_TIGHT_MID_M = 5.0
#: When |yaw_err| exceeds this in a tight nose, suppress face-into-wall bias.
_ESCAPE_YAW_OVERRIDE_RAD = float(np.deg2rad(60.0))
#: Soft peel bias only in hard-block (v10's 2.0 + mid-band tax killed R1/R4).
_ESCAPE_SCORE_W = 0.6
#: Mild tax on freeze/dive only when nose is hard-jammed (< block thresh).
_ESCAPE_STUCK_TAX = 0.5
#: Disable *mid-open* peel only in the last metres. Hard-block (<3 m) and
#: tight-mid (<5 m) always offer escape even near goal (urban crash ~25 m rem).
_ESCAPE_DISABLE_NEAR_GOAL_M = 8.0
#: Final approach band where open-nose gating may starve escape atoms
#: (CL-FT stall ~15–25 m rem). Modes: see ``resolve_approach_escape_mode``.
_ESCAPE_FORCE_APPROACH_M = 25.0
_APPROACH_ESCAPE_MODES = ("off", "side_climb", "side_only")


def resolve_approach_escape_mode(
    explicit: Optional[str] = None,
) -> str:
    """``off`` | ``side_climb`` | ``side_only`` (env ``OA_APPROACH_ESCAPE``)."""
    raw = explicit if explicit is not None else os.environ.get("OA_APPROACH_ESCAPE", "off")
    mode = str(raw or "off").strip().lower()
    if mode not in _APPROACH_ESCAPE_MODES:
        raise ValueError(
            f"approach_escape_mode={mode!r}; expected one of {_APPROACH_ESCAPE_MODES}"
        )
    return mode


def oa_offer_escape(
    d_fwd: Optional[float],
    *,
    near_goal: bool,
    goal_rem_xy: Optional[float] = None,
    approach_mode: str = "off",
) -> bool:
    """When to add climb/side escape atoms under directional OA.

    ``approach_mode`` only affects the open-nose / starved case inside
    ``_ESCAPE_FORCE_APPROACH_M``; jam-band natural offers are unchanged.
    """
    natural = _oa_offer_escape_natural(d_fwd, near_goal=near_goal)
    if natural:
        return True
    mode = resolve_approach_escape_mode(approach_mode)
    if mode == "off":
        return False
    if goal_rem_xy is not None and float(goal_rem_xy) < float(_ESCAPE_FORCE_APPROACH_M):
        return True
    return False


def _oa_offer_escape_natural(
    d_fwd: Optional[float],
    *,
    near_goal: bool,
) -> bool:
    """Pre-approach-force gating (tight / mid-open / open nose)."""
    if d_fwd is None:
        return not near_goal
    d = float(d_fwd)
    if d < float(_FACE_BLOCK_FWD_M):
        return True
    if d < float(_ESCAPE_TIGHT_MID_M):
        # Always peel/climb in the jam band — including last 8 m.
        return True
    if d < float(_FACE_OPEN_FWD_M):
        return not near_goal
    return False



class ConstantLatentPolicy:
    """Imagination policy that repeats one body delta every step."""

    def __init__(self, action: np.ndarray) -> None:
        self._action = np.asarray(action, dtype=np.float64).reshape(4)

    def act_latent(self, z: np.ndarray, goal_rel: Optional[np.ndarray] = None) -> np.ndarray:
        del z, goal_rel
        return self._action.copy()


class FirstActionThenActor:
    """Closed-loop imagination: commit ``first`` once, then call ``tail`` each step.

    ``imagine()`` is batch-1 in the planner (one candidate at a time). The step
    counter is reset by constructing a new instance per candidate.
    """

    def __init__(self, first: np.ndarray, tail: Any) -> None:
        self._first = np.asarray(first, dtype=np.float64).reshape(4)
        self._tail = tail
        self._t = 0

    def act_latent(self, z: np.ndarray, goal_rel: Optional[np.ndarray] = None) -> np.ndarray:
        t = self._t
        self._t += 1
        if t == 0:
            return self._first.copy()
        return np.asarray(
            self._tail.act_latent(z, goal_rel=goal_rel), dtype=np.float64
        ).reshape(4)


def drop_backward_if_subgoal_ahead(
    candidates: Sequence[np.ndarray],
    goal_rel: np.ndarray,
) -> List[np.ndarray]:
    """Remove the hardcoded pure-backward candidate when carrot is ahead in body frame."""
    if float(goal_rel[0]) <= _SUBGOAL_AHEAD_FWD_M:
        return list(candidates)
    kept: List[np.ndarray] = []
    for cand in candidates:
        c = np.asarray(cand, dtype=np.float64).reshape(4)
        pure_back = c[0] < -0.01 and np.all(np.abs(c[1:]) < 1e-6)
        if pure_back:
            continue
        kept.append(c)
    return kept if kept else list(candidates)


def drop_backward_motion(
    candidates: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Drop any candidate with body ``dx<0`` (no rear sensor → reverse illegal)."""
    kept: List[np.ndarray] = []
    for cand in candidates:
        c = np.asarray(cand, dtype=np.float64).reshape(4)
        if float(c[0]) < -1e-6:
            continue
        kept.append(c)
    return kept if kept else list(candidates)


def drop_strafe_if_yaw_off_axis(
    candidates: Sequence[np.ndarray],
    goal_rel: np.ndarray,
    *,
    yaw_thr_rad: float = _YAW_OFF_AXIS_THR_RAD,
    dyaw_min_rad: float = 0.05,
) -> List[np.ndarray]:
    """Drop pure lateral candidates when the carrot is far off the nose.

    Progress-only H=1 imagination prefers ``+dy`` at ~90° (distance shrinks
    without facing). Call only in open forward space — blocked nose must keep
    side-step as a legal detour.
    """
    g = np.asarray(goal_rel, dtype=np.float64).reshape(-1)
    if g.size < 2 or float(np.hypot(g[0], g[1])) <= 1e-6:
        return list(candidates)
    yaw_err = float(np.arctan2(float(g[1]), float(g[0])))
    if abs(yaw_err) <= float(yaw_thr_rad):
        return list(candidates)
    turn_sign = float(np.sign(yaw_err))  # +err = carrot left → need +dyaw
    kept: List[np.ndarray] = []
    for cand in candidates:
        c = np.asarray(cand, dtype=np.float64).reshape(4)
        dx, dy, dyaw = float(c[0]), float(c[1]), float(c[3])
        pure_strafe = abs(dy) > max(abs(dx), 0.05) and abs(dyaw) < float(dyaw_min_rad)
        wrong_turn_strafe = (
            abs(dy) > max(abs(dx), 0.05)
            and abs(dyaw) >= float(dyaw_min_rad)
            and float(np.sign(dyaw)) != turn_sign
        )
        if pure_strafe or wrong_turn_strafe:
            continue
        kept.append(c)
    return kept if kept else list(candidates)


def drop_pure_lateral_strafe(
    candidates: Sequence[np.ndarray],
    *,
    dyaw_min_rad: float = 0.05,
) -> List[np.ndarray]:
    """Always drop pure |dy|-dominant atoms with negligible dyaw (block-zone crab)."""
    kept: List[np.ndarray] = []
    for cand in candidates:
        c = np.asarray(cand, dtype=np.float64).reshape(4)
        dx, dy, dyaw = float(c[0]), float(c[1]), float(c[3])
        pure = abs(dy) > max(abs(dx), 0.05) and abs(dyaw) < float(dyaw_min_rad)
        if pure:
            continue
        kept.append(c)
    return kept if kept else list(candidates)


def escape_clearer_cone_candidates(
    obs: Observation,
    limits: Optional[np.ndarray] = None,
    goal_rel: Optional[np.ndarray] = None,
    *,
    include_climb: bool = True,
) -> List[np.ndarray]:
    """Tight-nose escape: strafe + turn toward clearer side (or goal-side fallback).

    Keeps the ability to leave a tight nose without pure wall-slide crabbing.
    Body +y / +dyaw = left. When left/right cones are missing (common on some
    deploy paths), fall back to carrot side / both sides so escape still exists.

    ``include_climb=False`` keeps side peel only (approach ablation C).
    """
    side = escape_side(obs, goal_rel=goal_rel)
    lim_dy = 0.4
    lim_dyaw = 0.3141592653589793
    lim_dx = 1.0
    if limits is not None:
        lim = np.abs(np.asarray(limits, dtype=np.float64).reshape(4))
        lim_dx = float(lim[0])
        lim_dy = float(lim[1])
        lim_dyaw = float(lim[3])
    creep = 0.15 * lim_dx
    dy = 0.85 * lim_dy
    turn = lim_dyaw

    lim_dz = 0.4
    if limits is not None:
        lim_dz = float(np.abs(np.asarray(limits, dtype=np.float64).reshape(4)[2]))
    climb = 0.85 * lim_dz

    def _atom(s: float) -> np.ndarray:
        return np.array([creep, s * dy, 0.0, s * turn], dtype=np.float64)

    # Climb escapes: optional — jam-band natural offers keep them; approach
    # ``side_only`` mode omits them so we do not over-rely on climb.
    climbs: List[np.ndarray] = []
    if include_climb:
        climbs = [
            np.array([creep, 0.0, climb, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, climb, 0.0], dtype=np.float64),
            np.array([creep, 0.0, climb, turn], dtype=np.float64),
            np.array([creep, 0.0, climb, -turn], dtype=np.float64),
        ]

    if side is None or side == 0.0:
        return [
            _atom(1.0),
            _atom(-1.0),
            np.array([0.0, dy, 0.0, turn], dtype=np.float64),
            np.array([0.0, -dy, 0.0, -turn], dtype=np.float64),
            *climbs,
        ]
    s = float(side)
    out = [
        _atom(s),
        np.array([0.0, s * dy, 0.0, s * turn], dtype=np.float64),
        _atom(-s),  # backup opposite
        *climbs,
    ]
    if include_climb:
        out.append(np.array([creep, s * dy, climb, s * turn], dtype=np.float64))
    return out


def _cone_lr(obs: Observation) -> tuple[Optional[float], Optional[float]]:
    info = getattr(obs, "info", None)
    cones = info.get("depth_cones_pred") if isinstance(info, dict) else None
    if not isinstance(cones, dict):
        return None, None

    def _f(key: str) -> Optional[float]:
        raw = cones.get(key)
        if raw is None:
            return None
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    return _f("left"), _f("right")


_YAW_SIDE_CLEAR_M = 2.0
_YAW_BURST_STEPS = 3
_FORWARD_AFTER_YAW_STEPS = 4
# Above this, the road ahead is already open. A side cone a few metres longer
# is the flanking building, not a reason to yaw off the street.
_YAW_DETOUR_MAX_FWD_M = 12.0
# One control step of yaw while the road centre is still short. Re-measure
# every step and stop when the centre opens, instead of a three-step full-rate burst.
_YAW_STEP_RAD = 0.10
_YAW_BUDGET_RAD = 1.2
# Road centre still open, but the subgoal has swung into a side street.
# One small yaw tracks that bearing; a goal still down the street does not.
_CORNER_YAW_ERR_RAD = 0.40


def yaw_detour_sign(
    d_fwd: Optional[float],
    left: Optional[float],
    right: Optional[float],
) -> Optional[float]:
    """+1 left / -1 right when one side is the open detour.

    A street whose nose is already open flies forward. A detour exists only
    when the nose is the short direction and one side is clearly longer.
    ``None`` means fly forward toward the goal.
    """
    if d_fwd is None or left is None or right is None:
        return None
    d, l, r = float(d_fwd), float(left), float(right)
    if not all(np.isfinite(v) for v in (d, l, r)):
        return None
    if d >= float(_YAW_DETOUR_MAX_FWD_M):
        return None
    if d >= l and d >= r:
        return None
    # A side only a couple of metres longer than the nose is the flanking
    # building. Yawing into it walks the vehicle off the street.
    if l >= r + _YAW_SIDE_CLEAR_M and l >= d + 4.0:
        return 1.0
    if r >= l + _YAW_SIDE_CLEAR_M and r >= d + 4.0:
        return -1.0
    return None


def yaw_only_candidates(
    sign: float, limits: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """A small yaw toward the latched open side. No strafe and no climb.

    The step stays well under the body yaw limit so the next frame can see
    whether the road centre has opened.
    """
    lim_dyaw = 0.3141592653589793
    if limits is not None:
        lim_dyaw = float(np.abs(np.asarray(limits, dtype=np.float64).reshape(4)[3]))
    turn = float(np.sign(sign)) * min(float(_YAW_STEP_RAD), lim_dyaw)
    return [
        np.array([0.0, 0.0, 0.0, turn], dtype=np.float64),
        np.array([0.15, 0.0, 0.0, turn], dtype=np.float64),
    ]


def escape_side(
    obs: Observation,
    goal_rel: Optional[np.ndarray] = None,
) -> Optional[float]:
    """+1 left / -1 right / 0 offer-both / None if no cue at all."""
    left, right = _cone_lr(obs)
    if left is None and right is None:
        if goal_rel is not None:
            g = np.asarray(goal_rel, dtype=np.float64).reshape(-1)
            if g.size >= 2 and abs(float(g[1])) > 0.1:
                return float(np.sign(g[1]))
        return 0.0  # both sides
    if left is None:
        return -1.0
    if right is None:
        return 1.0
    if float(left) >= float(right) + float(_ESCAPE_CONE_MARGIN_M):
        return 1.0
    if float(right) >= float(left) + float(_ESCAPE_CONE_MARGIN_M):
        return -1.0
    return 0.0


def forward_clearance_m(obs: Observation) -> Optional[float]:
    """Forward cone / depth_min if present, else None (unknown ≠ open)."""
    info = getattr(obs, "info", None)
    if not isinstance(info, dict):
        return None
    cones = info.get("depth_cones_pred")
    raw = None
    if isinstance(cones, dict):
        raw = cones.get("forward")
    if raw is None:
        raw = info.get("depth_min_pred")
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def face_goal_candidates(
    goal_rel: np.ndarray, limits: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """Creep + turn toward body-frame carrot (and pure-turn fallback)."""
    g = np.asarray(goal_rel, dtype=np.float64).reshape(-1)
    if g.size < 2 or float(np.hypot(g[0], g[1])) <= 1e-6:
        return []
    yaw_err = float(np.arctan2(float(g[1]), float(g[0])))
    if abs(yaw_err) < 1e-3:
        return []
    lim_dyaw = 0.3141592653589793
    lim_dx = 1.0
    if limits is not None:
        lim = np.abs(np.asarray(limits, dtype=np.float64).reshape(4))
        lim_dx = float(lim[0])
        lim_dyaw = float(lim[3])
    turn = float(np.sign(yaw_err)) * min(abs(yaw_err), lim_dyaw)
    creep = 0.35 * lim_dx
    return [
        np.array([creep, 0.0, 0.0, turn], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, turn], dtype=np.float64),
        np.array([creep, 0.0, 0.0, 0.5 * turn], dtype=np.float64),
    ]


def default_candidates(base_action: np.ndarray) -> List[np.ndarray]:
    """Small discrete set around the policy proposal."""
    base = np.asarray(base_action, dtype=np.float64).reshape(4)
    dx, dy, dz, dyaw = base
    return [
        base,
        np.array([dx * 0.5, dy, dz, dyaw], dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        np.array([-max(abs(dx), 0.5), 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([dx, dy * 0.5, dz, dyaw], dtype=np.float64),
        np.array([max(abs(dx), 1.0), 0.0, 0.0, 0.0], dtype=np.float64),
    ]


@dataclass
class ImaginationPlanner:
    """Pick the best first action via batched short imagined rollouts."""

    dynamics: LatentDynamics
    horizon: int = 5
    reward_cfg: Optional[RewardConfig] = None
    candidate_fn: Any = field(default=default_candidates)
    #: Optional deployed action box. ``None`` (default) keeps the V1-merged
    #: behaviour: candidates are scored in the UNCLIPPED space and only clipped
    #: later at ``collector.py:167`` — a same-origin inconsistency with §A.4,
    #: logged 2026-08-18 but deliberately NOT changed by default, since flipping
    #: it would alter the V1 deployed path and require a V1 re-gate.
    action_limits: Optional[np.ndarray] = None
    #: ``pass``: return ``base_action`` unchanged (skip all candidate logic).
    #: ``rules``: same candidates + hand biases; score = 1-step geometric
    #: progress (no WM imagine) — ablation: "are rules enough?".
    #: ``wm_bare``: same candidates; score = WM return only (no face/escape/
    #: strafe/stuck biases) — ablation: "can WM decide alone?".
    #: ``wm_escape``: WM return + open-side escape/stuck score only (no face /
    #: strafe tax). Overnight R1 forensics: L≫R yet climb wins under wm_bare
    #: because escape_score_w was stripped with the full hand pack.
    #: Use ``horizon=1`` for one-step WM scoring without adding a separate mode.
    mock_mode: Optional[str] = None
    #: Extra score weight on residual |yaw_err - dyaw| when forward is open.
    face_yaw_score_w: float = _FACE_YAW_SCORE_W
    #: Mid-clearance face weight (urban corridor default).
    face_yaw_score_w_mid: float = _FACE_YAW_SCORE_W_MID
    #: Weaker weight when forward depth is unknown (depth head miss).
    face_yaw_score_w_unknown: float = _FACE_YAW_SCORE_W_UNKNOWN
    #: Mid/unknown |dy| score penalty when yaw is off-axis.
    strafe_score_w_mid: float = _STRAFE_SCORE_W_MID
    #: R01 escape: boost peel toward clearer/fallback side when escape is active.
    #: Hand-rule biases are frozen (v5 is the sealed baseline). New runs use
    #: ``rollout_mode="closed_loop"`` and do not apply these terms.
    escape_score_w: float = _ESCAPE_SCORE_W
    escape_stuck_tax: float = _ESCAPE_STUCK_TAX
    #: ``open_loop``: repeat the candidate for H steps (legacy).
    #: ``closed_loop``: candidate is only step 0; steps 1..H-1 call ``tail_policy``
    #: (the deployed actor). Score is WM return only — no face/escape/strafe bias.
    rollout_mode: str = "open_loop"
    #: Deploy actor used as imagination tail when ``rollout_mode=closed_loop``.
    tail_policy: Any = None
    #: Approach-band escape injection: ``off`` | ``side_climb`` | ``side_only``.
    #: Default from env ``OA_APPROACH_ESCAPE`` (else ``off``). Natural jam-band
    #: offers always keep climb atoms; ``side_only`` only strips climb on the
    #: *forced* open-nose approach inject.
    approach_escape_mode: Optional[str] = None
    #: Filled by the last ``plan()`` call (eval / distill diagnostics).
    last_plan_meta: Dict[str, Any] = field(default_factory=dict)
    #: Hold open-side escape commitment for this many steps (0=off). Stops L/R
    #: flip-flop when WM scores chatter (R0 yaw churn forensics 2026-09-23).
    escape_commit_steps: int = 10
    #: Cap |Δdyaw| vs previous planned action (rad). 0 disables.
    max_ddyaw: float = 0.12
    _esc_commit_side: float = field(default=0.0, init=False, repr=False)
    _esc_commit_left: int = field(default=0, init=False, repr=False)
    _yaw_latch: float = field(default=0.0, init=False, repr=False)
    _yaw_burst_left: int = field(default=0, init=False, repr=False)
    _yaw_turned: float = field(default=0.0, init=False, repr=False)
    _forward_commit_left: int = field(default=0, init=False, repr=False)
    _prev_planned: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mock_mode is not None and self.mock_mode not in (
            "pass",
            "rules",
            "wm_bare",
            "wm_escape",
        ):
            raise ValueError(f"unsupported mock_mode={self.mock_mode!r}")
        if self.rollout_mode not in ("open_loop", "closed_loop"):
            raise ValueError(f"unsupported rollout_mode={self.rollout_mode!r}")
        if self.rollout_mode == "closed_loop" and self.mock_mode in ("pass", "rules"):
            raise ValueError("closed_loop requires WM scoring")
        self.approach_escape_mode = resolve_approach_escape_mode(
            self.approach_escape_mode
        )
        self.horizon = int(self.horizon)
        if self.action_limits is not None:
            lim = np.abs(np.asarray(self.action_limits, dtype=np.float64).reshape(-1))
            if lim.shape != (4,) or not np.all(lim > 0):
                raise ValueError(
                    f"action_limits must be 4 positive values, got {self.action_limits!r}"
                )
            self.action_limits = lim
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.horizon > MAX_IMAGINATION_HORIZON:
            raise ValueError(
                f"planner horizon {self.horizon} exceeds cap {MAX_IMAGINATION_HORIZON}"
            )

    def reset(self) -> None:
        """Per-episode hook — clear escape/yaw commitment state."""
        self._esc_commit_side = 0.0
        self._esc_commit_left = 0
        self._yaw_latch = 0.0
        self._yaw_burst_left = 0
        self._yaw_turned = 0.0
        self._forward_commit_left = 0
        self._prev_planned = None
        self.last_plan_meta = {}

    def set_goal(self, goal: Optional[np.ndarray]) -> None:
        set_goal = getattr(self.dynamics, "set_goal", None)
        if callable(set_goal):
            set_goal(goal)

    def plan(
        self,
        obs: Observation,
        base_action: np.ndarray,
        *,
        latent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the candidate first action with highest imagined return.

        When ``latent`` is provided (deploy streaming posterior), imagination
        scores from that state instead of resetting via ``encode(obs)``.
        """
        base = np.asarray(base_action, dtype=np.float64).reshape(4)
        if self.mock_mode == "pass":
            return base.copy()
        if latent is not None:
            z0 = np.asarray(latent, dtype=np.float64).reshape(-1)
        else:
            z0 = np.asarray(self.dynamics.encode(obs), dtype=np.float64)
        goal_rel = goal_rel_from_obs(obs)
        body_vel = body_vel_from_obs(obs)
        d_fwd = forward_clearance_m(obs)
        closed = self.rollout_mode == "closed_loop"
        hold_altitude = False
        if closed and self.tail_policy is None:
            raise RuntimeError("closed_loop planner requires tail_policy")
        if closed:
            # Base: actor-local proposals (v5 seal). Directional OA additionally
            # scores climb/side escapes so the learned obstacle head can beat
            # wall-fwd — actor-local atoms alone never invent climb.
            candidates = list(self.candidate_fn(np.asarray(base_action, dtype=np.float64)))
            offer_escape_cands = False
            natural_escape = False
            w_face = 0.0
            w_strafe = 0.0
            yaw_err0 = 0.0
            if float(np.hypot(goal_rel[0], goal_rel[1])) > 1e-6:
                yaw_err0 = float(np.arctan2(float(goal_rel[1]), float(goal_rel[0])))
            if bool(getattr(self.reward_cfg, "use_learned_obstacle_cost", False)):
                goal_rem_xy = float(np.hypot(float(goal_rel[0]), float(goal_rel[1])))
                near_goal = goal_rem_xy < float(_ESCAPE_DISABLE_NEAR_GOAL_M)
                natural_escape = _oa_offer_escape_natural(d_fwd, near_goal=near_goal)
                offer_escape_cands = oa_offer_escape(
                    d_fwd,
                    near_goal=near_goal,
                    goal_rem_xy=goal_rem_xy,
                    approach_mode=str(self.approach_escape_mode),
                )
                if offer_escape_cands:
                    # Jam-band natural → full climb+side. Forced approach only
                    # → climb iff mode is side_climb (ablation C = side_only).
                    include_climb = bool(natural_escape) or (
                        str(self.approach_escape_mode) == "side_climb"
                    )
                    candidates.extend(
                        escape_clearer_cone_candidates(
                            obs,
                            self.action_limits,
                            goal_rel=goal_rel,
                            include_climb=include_climb,
                        )
                    )
                candidates.extend(face_goal_candidates(goal_rel, self.action_limits))
            else:
                goal_rem_xy = float(np.hypot(float(goal_rel[0]), float(goal_rel[1])))
            candidates = drop_backward_if_subgoal_ahead(candidates, goal_rel)
            candidates = drop_backward_motion(candidates)
            # Stash for eval / distill diagnostics.
            _rem = float(np.hypot(float(goal_rel[0]), float(goal_rel[1])))
            self.last_plan_meta = {
                "n_candidates": int(len(candidates)),
                "offer_escape": bool(offer_escape_cands),
                "d_fwd": float(d_fwd) if d_fwd is not None and np.isfinite(float(d_fwd)) else None,
                "near_goal": bool(_rem < float(_ESCAPE_DISABLE_NEAR_GOAL_M)),
                "approach_escape_mode": str(self.approach_escape_mode),
                "approach_force_escape": bool(
                    (not natural_escape)
                    and _rem < float(_ESCAPE_FORCE_APPROACH_M)
                    and str(self.approach_escape_mode) != "off"
                    and bool(offer_escape_cands)
                ),
                "rollout_mode": "closed_loop",
            }
        else:
            goal_rem_xy = float(np.hypot(float(goal_rel[0]), float(goal_rel[1])))
            near_goal = goal_rem_xy < float(_ESCAPE_DISABLE_NEAR_GOAL_M)
            # Three zones. A yaw detour exists only when one side is clearly
            # longer than the nose. A 5–8 m street with no such side flies
            # forward; it does not keep offering strafe and climb.
            left_c, right_c = _cone_lr(obs)
            hard = d_fwd is not None and float(d_fwd) < float(_FACE_BLOCK_FWD_M)
            road_open = (
                d_fwd is not None and float(d_fwd) >= float(_YAW_DETOUR_MAX_FWD_M)
            )
            raw = None if road_open else yaw_detour_sign(d_fwd, left_c, right_c)
            # Yaw one small step while the road centre is short. Stop as soon
            # as it opens. A lost side cue keeps the latched direction so the
            # turn is not dropped in the same frame the brake holds forward speed.
            hold_forward = False
            if road_open:
                self._forward_commit_left = 0
                self._yaw_latch = 0.0
                self._yaw_burst_left = 0
                self._yaw_turned = 0.0
                detour = None
            elif int(self._forward_commit_left) > 0:
                self._forward_commit_left -= 1
                detour = None
                self._yaw_latch = 0.0
                self._yaw_burst_left = 0
                self._yaw_turned = 0.0
                hold_forward = True
            else:
                sign = raw
                if (
                    sign is None
                    and abs(float(self._yaw_latch)) > 1e-6
                    and float(self._yaw_turned) < float(_YAW_BUDGET_RAD)
                ):
                    sign = float(self._yaw_latch)
                if sign is None and hard:
                    side = escape_side(obs, goal_rel=goal_rel)
                    g = np.asarray(goal_rel, dtype=np.float64).reshape(-1)
                    if side is None or abs(float(side)) < 1e-6:
                        sign = (
                            float(np.sign(g[1]))
                            if g.size >= 2 and abs(float(g[1])) > 0.1
                            else 1.0
                        )
                    else:
                        sign = float(np.sign(side))
                if sign is None:
                    detour = None
                    self._yaw_latch = 0.0
                    self._yaw_burst_left = 0
                    self._yaw_turned = 0.0
                elif float(self._yaw_turned) + float(_YAW_STEP_RAD) >= float(_YAW_BUDGET_RAD) - 1e-6:
                    self._forward_commit_left = int(_FORWARD_AFTER_YAW_STEPS)
                    self._yaw_latch = 0.0
                    self._yaw_burst_left = 0
                    self._yaw_turned = 0.0
                    detour = None
                    hold_forward = True
                else:
                    if (
                        abs(float(self._yaw_latch)) < 1e-6
                        or float(np.sign(self._yaw_latch)) != float(np.sign(sign))
                    ):
                        self._yaw_latch = float(np.sign(sign))
                        self._yaw_turned = 0.0
                    detour = float(self._yaw_latch)
                    self._yaw_turned = float(self._yaw_turned) + float(_YAW_STEP_RAD)
            yaw_only = detour is not None and not hold_forward
            drop_open_strafe = False
            drop_block_pure_strafe = False
            offer_escape_cands = False
            w_strafe = 0.0
            if d_fwd is None:
                w_face = float(self.face_yaw_score_w_unknown)
                w_strafe = float(self.strafe_score_w_mid)
                offer_face_cands = True
                # Unknown nose near goal: prefer face, not peel-away.
                offer_escape_cands = not near_goal
            elif float(d_fwd) >= float(_FACE_OPEN_FWD_M):
                w_face = float(self.face_yaw_score_w)
                offer_face_cands = True
                drop_open_strafe = True
                if yaw_only:
                    offer_escape_cands = True
                    drop_open_strafe = False
            elif float(d_fwd) >= float(_FACE_BLOCK_FWD_M):
                w_face = float(self.face_yaw_score_w_mid)
                w_strafe = float(self.strafe_score_w_mid)
                offer_face_cands = True
                # Yaw only when one side is clearly open. Unknown sides in the
                # tight band keep the old escape fallback. A measured street
                # does not.
                if yaw_only:
                    offer_escape_cands = True
                elif (
                    float(d_fwd) < float(_ESCAPE_TIGHT_MID_M)
                    and (left_c is None or right_c is None)
                ):
                    offer_escape_cands = True
            else:
                # Hard proximity: a short yaw, then stop spinning.
                w_face = 0.0
                offer_face_cands = False
                if hold_forward:
                    offer_escape_cands = False
                    yaw_only = False
                    drop_block_pure_strafe = False
                else:
                    drop_block_pure_strafe = True
                    offer_escape_cands = True
                    yaw_only = True

            natural_escape = bool(offer_escape_cands)
            # Approach-band force (same modes as closed_loop).
            if (
                str(self.approach_escape_mode) != "off"
                and goal_rem_xy < float(_ESCAPE_FORCE_APPROACH_M)
            ):
                offer_escape_cands = True

            yaw_err0 = 0.0
            if float(np.hypot(goal_rel[0], goal_rel[1])) > 1e-6:
                yaw_err0 = float(np.arctan2(float(goal_rel[1]), float(goal_rel[0])))
            # Road centre, carrot still down the street: a longer side is the
            # flanking building. Stay on the centreline.
            if (
                bool(obs.info.get("hold_street"))
                and abs(float(yaw_err0)) <= 0.35
                and d_fwd is not None
                and float(d_fwd) >= float(_FACE_BLOCK_FWD_M)
            ):
                yaw_only = False
                offer_escape_cands = False
                detour = None
            # Extreme off-axis + measured tight nose: don't face into the obstacle.
            # Skip when d_fwd is unknown (soft face still wanted there).
            if (
                offer_escape_cands
                and abs(yaw_err0) >= float(_ESCAPE_YAW_OVERRIDE_RAD)
                and d_fwd is not None
                and float(d_fwd) < float(_ESCAPE_TIGHT_MID_M)
            ):
                w_face = 0.0
                offer_face_cands = False
                drop_block_pure_strafe = True

            candidates = list(self.candidate_fn(np.asarray(base_action, dtype=np.float64)))
            if offer_face_cands:
                candidates.extend(face_goal_candidates(goal_rel, self.action_limits))
            if offer_escape_cands and yaw_only:
                candidates = yaw_only_candidates(float(detour), self.action_limits)
            elif offer_escape_cands:
                include_climb = bool(natural_escape) or (
                    str(self.approach_escape_mode) == "side_climb"
                )
                candidates.extend(
                    escape_clearer_cone_candidates(
                        obs,
                        self.action_limits,
                        goal_rel=goal_rel,
                        include_climb=include_climb,
                    )
                )
            candidates = drop_backward_if_subgoal_ahead(candidates, goal_rel)
            candidates = drop_backward_motion(candidates)
            if drop_open_strafe:
                candidates = drop_strafe_if_yaw_off_axis(candidates, goal_rel)
            if drop_block_pure_strafe:
                candidates = drop_pure_lateral_strafe(candidates)
                candidates = drop_strafe_if_yaw_off_axis(candidates, goal_rel)
                # Re-offer escape ONLY if filters wiped every leave-wall atom
                # (v10 always-reoffer bypassed yaw filter → R1 spin-into-wall).
                if offer_escape_cands:
                    has_peel = any(
                        abs(float(np.asarray(c, dtype=np.float64).reshape(4)[3])) > 0.05
                        and abs(float(np.asarray(c, dtype=np.float64).reshape(4)[1])) > 0.05
                        for c in candidates
                    )
                    if not has_peel:
                        include_climb = bool(natural_escape) or (
                            str(self.approach_escape_mode) == "side_climb"
                        )
                        if yaw_only:
                            candidates = yaw_only_candidates(
                                float(detour), self.action_limits
                            )
                        else:
                            candidates.extend(
                                escape_clearer_cone_candidates(
                                    obs,
                                    self.action_limits,
                                    goal_rel=goal_rel,
                                    include_climb=include_climb,
                                )
                            )
            elif (
                not yaw_only
                and d_fwd is not None
                and float(d_fwd) >= float(_FACE_BLOCK_FWD_M)
                and left_c is not None
                and right_c is not None
            ):
                # Nose is not a one-sided wall: go forward. Drop climb and strafe.
                # A side-street subgoal still gets one small yaw, so the corner
                # is taken while the road centre stays open.
                turn_to_goal = 0.0
                err_abs = abs(float(yaw_err0))
                # A wide bearing with a still-long nose is an early carrot, not the
                # intersection. Stay on the street. A moderate bearing gets one step.
                if road_open and float(_CORNER_YAW_ERR_RAD) < err_abs <= 0.60:
                    turn_to_goal = float(
                        np.clip(
                            float(yaw_err0),
                            -float(_YAW_STEP_RAD),
                            float(_YAW_STEP_RAD),
                        )
                    )
                ahead = [
                    c
                    for c in candidates
                    if float(np.asarray(c, dtype=np.float64).reshape(4)[0]) > 0.2
                    and abs(float(np.asarray(c, dtype=np.float64).reshape(4)[1])) < 0.15
                    and abs(float(np.asarray(c, dtype=np.float64).reshape(4)[3]) - turn_to_goal)
                    < 0.05
                    and float(np.asarray(c, dtype=np.float64).reshape(4)[2]) >= -1e-6
                    and float(np.asarray(c, dtype=np.float64).reshape(4)[2]) < 0.05
                ]
                hold_altitude = True
                if ahead:
                    candidates = ahead
                else:
                    candidates = [
                        np.array([0.7, 0.0, 0.0, turn_to_goal], dtype=np.float64)
                    ]
            elif hold_forward and hard:
                # Yaw burst is over. Do not climb out of the street.
                candidates = [np.array([0.35, 0.0, 0.0, 0.0], dtype=np.float64)]
            # Keep the nose on the carrot. A wide bearing slows down in the
            # intersection. A moderate bearing still yaws toward the carrot, so a
            # side detour cannot reverse the turn once the exit is in front.
            if (
                d_fwd is not None
                and float(d_fwd) >= float(_FACE_BLOCK_FWD_M)
                and abs(float(yaw_err0)) > 0.35
            ):
                err_abs = abs(float(yaw_err0))
                if err_abs > 0.60:
                    step = min(0.22, err_abs)
                    fwd_cmd = 0.15
                else:
                    step = float(_YAW_STEP_RAD)
                    fwd_cmd = 0.45
                turn_to_goal = float(np.clip(float(yaw_err0), -step, step))
                candidates = [
                    np.array([fwd_cmd, 0.0, 0.0, turn_to_goal], dtype=np.float64)
                ]
                hold_altitude = True
        if not candidates:
            return np.asarray(base_action, dtype=np.float64).reshape(4)
        if self.action_limits is not None:
            lim = self.action_limits
            candidates = [np.clip(c, -lim, lim) for c in candidates]

        best_a = candidates[0]
        best_idx = 0
        best_score = -np.inf
        gr0 = np.asarray(goal_rel, dtype=np.float32).reshape(1, -1)
        bv0 = np.asarray(body_vel, dtype=np.float32).reshape(1, -1)
        yaw_off = abs(yaw_err0) > float(_YAW_OFF_AXIS_THR_RAD)
        esc = escape_side(obs, goal_rel=goal_rel) if offer_escape_cands else None
        # Mid-band stick (forensics 2026-09-23): offer_escape+oc↑ on but still pick
        # pure-forward → d_fwd→0.7. Soft bias used to be hard_block-only; extend to
        # tight mid and hard-prefer leave-wall atoms after scoring.
        tight_escape = d_fwd is not None and float(d_fwd) < float(_ESCAPE_TIGHT_MID_M)
        use_wm = self.mock_mode != "rules"
        # Full hand pack (face+strafe+escape) vs escape-only vs bare WM.
        apply_hand = (not closed) and self.mock_mode not in ("wm_bare", "wm_escape")
        apply_escape_bias = (not closed) and (
            apply_hand or self.mock_mode == "wm_escape"
        )
        # wm_escape overnight 0555: tight-mid bias over-peeled R0 into the
        # open-side wall at rem≈14. Escape-only mock: hard-block ALWAYS; also
        # tight-mid when a side is already crushing (<2.5 m) with L/R asymmetry
        # (R1 pocket). Mid-open peel without a near wall stays off.
        escape_zone = tight_escape
        if self.mock_mode == "wm_escape":
            hard_block = d_fwd is not None and float(d_fwd) < float(_FACE_BLOCK_FWD_M)
            left_c, right_c = _cone_lr(obs)
            side_crush = False
            if left_c is not None and right_c is not None:
                side_crush = min(float(left_c), float(right_c)) < 2.5 and abs(
                    float(left_c) - float(right_c)
                ) >= float(_ESCAPE_CONE_MARGIN_M)
            escape_zone = bool(hard_block or (tight_escape and side_crush))
        esc_w = float(self.escape_score_w)
        cand_scores: List[float] = []
        for ci, cand in enumerate(candidates):
            if use_wm:
                if closed:
                    policy = FirstActionThenActor(cand, self.tail_policy)
                else:
                    policy = ConstantLatentPolicy(cand)
                imagine_kw = dict(
                    reward_cfg=self.reward_cfg,
                    goal_rel0=gr0,
                    body_vel0=bv0,
                    propagate_goal_rel=True,
                    action_limits=self.action_limits,
                )
                from experiments.aerial.rl.imagination import imagine as _imagine_impl

                if "profile0" in inspect.signature(_imagine_impl).parameters:
                    info = getattr(obs, "info", None)
                    profile0 = None
                    if isinstance(info, dict) and info.get("depth_profile_pred") is not None:
                        profile0 = np.asarray(
                            info["depth_profile_pred"], dtype=np.float64
                        ).reshape(1, -1)
                    imagine_kw["profile0"] = profile0
                roll = imagine(
                    self.dynamics,
                    policy,
                    z0[None, :],
                    self.horizon,
                    **imagine_kw,
                )
                score = float(roll.returns[0])
            else:
                # 1-step body-frame progress proxy (no latent rollout).
                g = np.asarray(goal_rel, dtype=np.float64).reshape(-1)
                before = float(np.linalg.norm(g[:3])) if g.size >= 3 else 0.0
                after = float(
                    np.linalg.norm(g[:3] - np.asarray(cand, dtype=np.float64)[:3])
                )
                score = before - after
            if apply_hand:
                if w_face > 0.0:
                    resid = abs(yaw_err0 - float(cand[3]))
                    score -= w_face * resid
                if w_strafe > 0.0 and yaw_off:
                    score -= w_strafe * abs(float(cand[1]))
            if apply_escape_bias:
                # Escape bias in hard-block AND tight-mid (was hard-only → mid stick).
                # wm_escape: hard-block only (see escape_zone).
                if (
                    escape_zone
                    and offer_escape_cands
                    and esc is not None
                    and float(esc) != 0.0
                ):
                    s = float(esc)
                    score += float(esc_w) * (
                        s * float(cand[1]) + s * float(cand[3])
                    )
                if escape_zone:
                    # Don't freeze or dive straight into the wall when jammed.
                    if abs(float(cand[1])) < 0.05 and abs(float(cand[3])) < 0.05:
                        score -= float(self.escape_stuck_tax)
                    elif float(cand[0]) > 0.4 and abs(float(cand[3])) < 0.05:
                        score -= 0.75 * float(self.escape_stuck_tax)
                # v2 R1 plateau ~26 m: climb%≈0.54 with mid-open nose. Tax climb
                # only for wm_escape in the final approach band when not jammed.
                if (
                    self.mock_mode == "wm_escape"
                    and float(np.hypot(float(goal_rel[0]), float(goal_rel[1])))
                    < 30.0
                    and d_fwd is not None
                    and float(d_fwd) >= float(_FACE_BLOCK_FWD_M)
                    and float(cand[2]) > 0.15
                ):
                    score -= 0.5
            cand_scores.append(float(score))
            if score > best_score:
                best_score = score
                best_a = cand
                best_idx = int(ci)
        # Escape-side commitment: while holding, re-pick among candidates that
        # peel toward the committed side (dy or dyaw sign matches esc).
        commit_n = int(self.escape_commit_steps)
        if (
            commit_n > 0
            and offer_escape_cands
            and cand_scores
            and self._esc_commit_left > 0
            and abs(float(self._esc_commit_side)) > 0.0
        ):
            side = float(self._esc_commit_side)

            def _peels(c: np.ndarray) -> bool:
                a = np.asarray(c, dtype=np.float64).reshape(4)
                return (side * float(a[1]) > 0.02) or (side * float(a[3]) > 0.02)

            matching = [i for i, c in enumerate(candidates) if _peels(c)]
            if matching:
                new_idx = max(matching, key=lambda i: cand_scores[i])
                best_idx = int(new_idx)
                best_a = candidates[best_idx]
                best_score = float(cand_scores[best_idx])
            self._esc_commit_left = max(0, int(self._esc_commit_left) - 1)
        elif (
            commit_n > 0
            and offer_escape_cands
            and esc is not None
            and abs(float(esc)) > 0.0
        ):
            a = np.asarray(best_a, dtype=np.float64).reshape(4)
            if (float(esc) * float(a[1]) > 0.02) or (float(esc) * float(a[3]) > 0.02):
                self._esc_commit_side = float(esc)
                self._esc_commit_left = int(commit_n)
        force_peel_exec = False
        # Overnight force-restrict/inject hurt more than they helped vs wm_bare
        # (2026-09-23: wm_bare P0=1/2 with R0 arrive; forcepeel stayed 2/2).
        # Gate the whole block behind OA_FORCE_PEEL_RESTRICT (default off).
        _force_restrict = str(
            os.environ.get("OA_FORCE_PEEL_RESTRICT", "0")
        ).lower() in ("1", "true", "yes", "on")
        if (
            _force_restrict
            and offer_escape_cands
            and tight_escape
            and cand_scores
            and self.mock_mode not in ("wm_bare", "wm_escape")
        ):

            def _leave_wall(c: np.ndarray) -> bool:
                a = np.asarray(c, dtype=np.float64).reshape(4)
                return (
                    abs(float(a[1])) > 0.05
                    or abs(float(a[3])) > 0.05
                    or float(a[2]) > 0.15
                )

            peel_idxs = [i for i, c in enumerate(candidates) if _leave_wall(c)]
            if peel_idxs:
                # Restrict to leave-wall set whenever nose is tight (not only when
                # the WM winner is pure-forward). Soft force_peel left R0/R1 at
                # d_fwd≈0.7 with climb% still low (2026-09-23 overnight).
                new_idx = max(peel_idxs, key=lambda i: cand_scores[i])
                if int(new_idx) != int(best_idx) or not _leave_wall(best_a):
                    force_peel_exec = True
                best_idx = int(new_idx)
                best_a = candidates[best_idx]
                best_score = float(cand_scores[best_idx])

            def _esc_strength(c: np.ndarray) -> float:
                a = np.asarray(c, dtype=np.float64).reshape(4)
                return abs(float(a[1])) + abs(float(a[3])) + 1.5 * max(0.0, float(a[2]))

            # v3 inject is opt-in: overnight showed higher climb/force_peel but
            # still final d_fwd≈0.8 (P0 fail). Default off to avoid regressing
            # open corridors; set OA_FORCE_PEEL_INJECT=1 to re-enable.
            if (
                _esc_strength(best_a) < 0.35
                and str(os.environ.get("OA_FORCE_PEEL_INJECT", "0")).lower()
                in ("1", "true", "yes", "on")
            ):
                inj_pool = escape_clearer_cone_candidates(
                    obs,
                    self.action_limits,
                    goal_rel=goal_rel,
                    include_climb=True,
                )
                if inj_pool:
                    best_a = max(inj_pool, key=_esc_strength)
                    best_idx = -1
                    force_peel_exec = True
        out = np.asarray(best_a, dtype=np.float64).reshape(4)
        # Yaw low-pass vs previous plan — cuts high-frequency heading flips.
        max_dd = float(self.max_ddyaw)
        if max_dd > 0.0 and self._prev_planned is not None:
            prev = np.asarray(self._prev_planned, dtype=np.float64).reshape(4)
            dd = float(np.clip(float(out[3]) - float(prev[3]), -max_dd, max_dd))
            out[3] = float(prev[3] + dd)
        if hold_altitude:
            out[2] = max(0.0, float(out[2]))
        if bool(getattr(self.reward_cfg, "forbid_backward_motion", False)):
            from experiments.aerial.rl.env.action import forbid_backward_dx

            out = forbid_backward_dx(out)
        self._prev_planned = np.asarray(out, dtype=np.float64).reshape(4).copy()
        meta = dict(getattr(self, "last_plan_meta", {}) or {})
        _rem_xy = float(np.hypot(float(goal_rel[0]), float(goal_rel[1])))
        meta.update(
            {
                "chosen_idx": int(best_idx),
                "n_candidates": int(len(candidates)),
                "offer_escape": bool(offer_escape_cands),
                "best_score": float(best_score),
                "chose_climb": bool(float(out[2]) > 0.15),
                "d_fwd": (
                    float(d_fwd)
                    if d_fwd is not None and np.isfinite(float(d_fwd))
                    else None
                ),
                "near_goal": bool(_rem_xy < float(_ESCAPE_DISABLE_NEAR_GOAL_M)),
                "approach_escape_mode": str(self.approach_escape_mode),
                "force_peel_exec": bool(force_peel_exec),
                "escape_commit_left": int(self._esc_commit_left),
                "escape_commit_side": float(self._esc_commit_side),
                "approach_force_escape": bool(
                    (not natural_escape)
                    and _rem_xy < float(_ESCAPE_FORCE_APPROACH_M)
                    and str(self.approach_escape_mode) != "off"
                    and bool(offer_escape_cands)
                ),
            }
        )
        self.last_plan_meta = meta
        return out
