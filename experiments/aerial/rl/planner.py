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

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

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
#: Disable tight-mid escape when this close to goal (v11 R4: reached 3.8m then peeled away).
_ESCAPE_DISABLE_NEAR_GOAL_M = 25.0


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
) -> List[np.ndarray]:
    """Tight-nose escape: strafe + turn toward clearer side (or goal-side fallback).

    Keeps the ability to leave a tight nose without pure wall-slide crabbing.
    Body +y / +dyaw = left. When left/right cones are missing (common on some
    deploy paths), fall back to carrot side / both sides so escape still exists.
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

    def _atom(s: float) -> np.ndarray:
        return np.array([creep, s * dy, 0.0, s * turn], dtype=np.float64)

    if side is None or side == 0.0:
        return [
            _atom(1.0),
            _atom(-1.0),
            np.array([0.0, dy, 0.0, turn], dtype=np.float64),
            np.array([0.0, -dy, 0.0, -turn], dtype=np.float64),
        ]
    s = float(side)
    return [
        _atom(s),
        np.array([0.0, s * dy, 0.0, s * turn], dtype=np.float64),
        _atom(-s),  # backup opposite
    ]


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
    tail_policy: Any = None

    def __post_init__(self) -> None:
        if self.mock_mode is not None and self.mock_mode not in (
            "pass",
            "rules",
            "wm_bare",
        ):
            raise ValueError(f"unsupported mock_mode={self.mock_mode!r}")
        if self.rollout_mode not in ("open_loop", "closed_loop"):
            raise ValueError(f"unsupported rollout_mode={self.rollout_mode!r}")
        if self.rollout_mode == "closed_loop" and self.mock_mode in ("pass", "rules"):
            raise ValueError("closed_loop requires WM scoring")
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
        """Per-episode hook (stateless; matches deploy policy interface)."""
        return

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
        if closed and self.tail_policy is None:
            raise RuntimeError("closed_loop planner requires tail_policy")
        if closed:
            # Actor-local proposals only. Face/escape atoms and hand biases stay
            # on the frozen open-loop path (v5 seal); they do not score this run.
            candidates = list(self.candidate_fn(np.asarray(base_action, dtype=np.float64)))
            candidates = drop_backward_if_subgoal_ahead(candidates, goal_rel)
            candidates = drop_backward_motion(candidates)
            offer_escape_cands = False
            w_face = 0.0
            w_strafe = 0.0
            yaw_err0 = 0.0
            if float(np.hypot(goal_rel[0], goal_rel[1])) > 1e-6:
                yaw_err0 = float(np.arctan2(float(goal_rel[1]), float(goal_rel[0])))
        else:
            goal_rem_xy = float(np.hypot(float(goal_rel[0]), float(goal_rel[1])))
            near_goal = goal_rem_xy < float(_ESCAPE_DISABLE_NEAR_GOAL_M)
            # Three zones + tight-mid escape when cones/goal allow a side peel.
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
            elif float(d_fwd) >= float(_FACE_BLOCK_FWD_M):
                w_face = float(self.face_yaw_score_w_mid)
                w_strafe = float(self.strafe_score_w_mid)
                offer_face_cands = True
                # Tight mid: peel only when still far from goal (protect R4 terminal).
                if float(d_fwd) < float(_ESCAPE_TIGHT_MID_M) and not near_goal:
                    offer_escape_cands = True
            else:
                # Hard proximity: no face-into-wall; ban pure crab; escape via cone.
                w_face = 0.0
                offer_face_cands = False
                drop_block_pure_strafe = True
                offer_escape_cands = True

            yaw_err0 = 0.0
            if float(np.hypot(goal_rel[0], goal_rel[1])) > 1e-6:
                yaw_err0 = float(np.arctan2(float(goal_rel[1]), float(goal_rel[0])))
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
            if offer_escape_cands:
                candidates.extend(
                    escape_clearer_cone_candidates(
                        obs, self.action_limits, goal_rel=goal_rel
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
                        candidates.extend(
                            escape_clearer_cone_candidates(
                                obs, self.action_limits, goal_rel=goal_rel
                            )
                        )
        if not candidates:
            return np.asarray(base_action, dtype=np.float64).reshape(4)
        if self.action_limits is not None:
            lim = self.action_limits
            candidates = [np.clip(c, -lim, lim) for c in candidates]

        best_a = candidates[0]
        best_score = -np.inf
        gr0 = np.asarray(goal_rel, dtype=np.float32).reshape(1, -1)
        bv0 = np.asarray(body_vel, dtype=np.float32).reshape(1, -1)
        yaw_off = abs(yaw_err0) > float(_YAW_OFF_AXIS_THR_RAD)
        esc = escape_side(obs, goal_rel=goal_rel) if offer_escape_cands else None
        hard_block = d_fwd is not None and float(d_fwd) < float(_FACE_BLOCK_FWD_M)
        use_wm = self.mock_mode != "rules"
        apply_hand = (not closed) and self.mock_mode != "wm_bare"
        for cand in candidates:
            if use_wm:
                if closed:
                    policy = FirstActionThenActor(cand, self.tail_policy)
                else:
                    policy = ConstantLatentPolicy(cand)
                roll = imagine(
                    self.dynamics,
                    policy,
                    z0[None, :],
                    self.horizon,
                    reward_cfg=self.reward_cfg,
                    goal_rel0=gr0,
                    body_vel0=bv0,
                    propagate_goal_rel=True,
                    action_limits=self.action_limits,
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
                # Soft escape bias only when hard-blocked (not tight-mid), avoid
                # v10 mid-band over-peel that stalled R4 / spun R1 into collision.
                if (
                    hard_block
                    and offer_escape_cands
                    and esc is not None
                    and float(esc) != 0.0
                ):
                    s = float(esc)
                    score += float(self.escape_score_w) * (
                        s * float(cand[1]) + s * float(cand[3])
                    )
                if hard_block:
                    # Don't freeze or dive straight into the wall when jammed.
                    if abs(float(cand[1])) < 0.05 and abs(float(cand[3])) < 0.05:
                        score -= float(self.escape_stuck_tax)
                    elif float(cand[0]) > 0.4 and abs(float(cand[3])) < 0.05:
                        score -= 0.75 * float(self.escape_stuck_tax)
            if score > best_score:
                best_score = score
                best_a = cand
        out = np.asarray(best_a, dtype=np.float64).reshape(4)
        if bool(getattr(self.reward_cfg, "forbid_backward_motion", False)):
            from experiments.aerial.rl.env.action import forbid_backward_dx

            out = forbid_backward_dx(out)
        return out
