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
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("wam_phase2_eval")


class _FixedActionSeq:
    """Open-loop body-delta sequence for a G0 imagination rollout."""

    def __init__(self, actions: Sequence[np.ndarray]) -> None:
        self._actions = [np.asarray(a, dtype=np.float64).reshape(4) for a in actions]
        self._i = 0

    def act_latent(self, z: np.ndarray, goal_rel: Optional[np.ndarray] = None) -> np.ndarray:
        del z, goal_rel
        a = self._actions[min(self._i, len(self._actions) - 1)]
        self._i += 1
        return a


def select_receding_primitive(
    dynamics: Any,
    z: np.ndarray,
    goal_rel: np.ndarray,
    limits: np.ndarray,
    *,
    horizon: int = 5,
    cost_margin: float = 0.25,
    cones: Optional[Dict[str, Any]] = None,
    yaw_err: float = 0.0,
    allow_hits: bool = True,
) -> tuple:
    """One receding step: drop high-cost primitives, then max goal progress.

    Rolls each constant primitive ``horizon`` steps on the frozen dynamics.
    Returns ``(first_action, name, sum_cost, sum_progress)``. Does not train.
    """
    from experiments.aerial.rl.goal_features import advance_goal_rel_body

    lim = np.abs(np.asarray(limits, dtype=np.float64).reshape(4))
    yaw = float(lim[3])
    specs = {
        "fwd": np.array([0.7, 0.0, 0.0, 0.0]),
        "yaw_l": np.array([0.0, 0.0, 0.0, yaw]),
        "yaw_r": np.array([0.0, 0.0, 0.0, -yaw]),
        "peel_l": np.array([0.2, 0.4, 0.0, 0.0]),
        "peel_r": np.array([0.2, -0.4, 0.0, 0.0]),
    }
    z0 = np.asarray(z, dtype=np.float64).reshape(-1).copy()
    g0 = np.asarray(goal_rel, dtype=np.float64).reshape(-1).copy()
    scored = []
    for name, raw in specs.items():
        a = np.clip(np.asarray(raw, dtype=np.float64), -lim, lim)
        a[0] = max(0.0, float(a[0]))
        zz = z0.copy()
        gg = g0.copy()
        cost = 0.0
        prog = 0.0
        hit = False
        for _ in range(int(horizon)):
            out = dynamics.step(zz, a, goal_rel=gg)
            oc = getattr(out, "obstacle_cost", None)
            if oc is not None and np.isfinite(float(oc)):
                cost += float(oc)
                if float(oc) >= 1.0:
                    hit = True
            if float(getattr(out, "p_coll", 0.0) or 0.0) >= 0.5:
                hit = True
            g1 = np.asarray(advance_goal_rel_body(gg, a), dtype=np.float64).reshape(-1)
            if gg.size > 3 and g1.size > 3:
                prog += float(gg[3]) - float(g1[3])
            gg = g1
            zz = np.asarray(out.z_next, dtype=np.float64).reshape(-1)
            if bool(getattr(out, "done", False)):
                break
        scored.append((name, a, cost, prog, hit))

    def _cone_ok(name: str) -> bool:
        if not isinstance(cones, dict):
            return True

        def _num(key: str) -> Optional[float]:
            raw = cones.get(key)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        forward, left, right = _num("forward"), _num("left"), _num("right")
        err = float(yaw_err)
        if abs(err) > 1.0:
            return name == ("yaw_l" if err > 0.0 else "yaw_r")
        if name == "fwd" and forward is not None and forward < 5.0:
            return False
        if name == "fwd" and forward is not None and forward < 8.0 and abs(err) > 0.4:
            return False
        if name == "peel_l" and left is not None and left < 8.0:
            return False
        if name == "peel_r" and right is not None and right < 8.0:
            return False
        # Yaw only toward the goal. Yawing the other way at 5 m spun the nose
        # into the facade and held distance at 39 m.
        if name == "yaw_l" and err <= 0.15:
            return False
        if name == "yaw_r" and err >= -0.15:
            return False
        return True

    scored = [s for s in scored if _cone_ok(s[0])]
    alive = [s for s in scored if not s[4]]
    pool = alive if alive else (scored if allow_hits else [])
    if not pool:
        err = float(yaw_err)
        forward = left = right = None
        if isinstance(cones, dict):
            def _num(key: str) -> Optional[float]:
                raw = cones.get(key)
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    return None
                return value if np.isfinite(value) else None

            forward, left, right = _num("forward"), _num("left"), _num("right")
        err = float(yaw_err)
        goal_side = None
        if left is not None and right is not None:
            goal_side = left if err > 0.0 else right
        goal_open = (
            forward is not None
            and goal_side is not None
            and goal_side >= forward + 0.8
        )
        if goal_open:
            dyaw = math.copysign(min(0.12, float(lim[3])), err if abs(err) > 0.05 else 1.0)
            return np.array([0.0, 0.0, 0.0, dyaw]), "face", float("inf"), 0.0
        if forward is not None and forward < 6.0:
            return np.zeros(4), "hold", float("inf"), 0.0
        if abs(err) <= 0.35 and forward is not None and forward >= 6.0:
            creep = np.array([min(0.35, float(lim[0])), 0.0, 0.0, 0.0])
            return creep, "creep", float("inf"), 0.0
        dyaw = math.copysign(min(0.12, float(lim[3])), err if abs(err) > 0.2 else 1.0)
        stopped = np.array([0.0, 0.0, 0.0, dyaw], dtype=np.float64)
        return stopped, "face", float("inf"), 0.0
    c_min = min(s[2] for s in pool)
    kept = [s for s in pool if s[2] <= c_min + float(cost_margin)]
    if not kept:
        kept = pool
    best = max(kept, key=lambda s: (s[3], -s[2]))
    if isinstance(cones, dict) and str(best[0]).startswith("yaw"):
        def _num(key: str) -> Optional[float]:
            raw = cones.get(key)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        forward, left, right = _num("forward"), _num("left"), _num("right")
        if (
            forward is not None
            and left is not None
            and right is not None
            and forward < 6.0
        ):
            goal_side = left if float(yaw_err) > 0.0 else right
            if goal_side >= forward + 0.8:
                return best[1], best[0], float(best[2]), float(best[3])
            return np.zeros(4), "hold", float(best[2]), float(best[3])
    return best[1], best[0], float(best[2]), float(best[3])


def select_street_action(
    cones: Dict[str, Any],
    yaw_err: float,
    limits: np.ndarray,
    memory: Optional[Dict[str, float]] = None,
) -> tuple:
    """Hold altitude and fly where the nose is clear. Does not train and is not a shield.

    Go straight only while the goal is still in front and the forward cone is
    open. If the goal swings behind, yaw toward it in place. If the nose is
    blocked but the goal is still ahead, shift sideways without yawing or climbing.
    """
    lim = np.abs(np.asarray(limits, dtype=np.float64).reshape(4))
    mem = memory if memory is not None else {}

    def _cone(key: str) -> Optional[float]:
        raw = cones.get(key) if isinstance(cones, dict) else None
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _finish(raw: np.ndarray, name: str) -> tuple:
        action = np.clip(np.asarray(raw, dtype=np.float64), -lim, lim)
        action[0] = max(0.0, float(action[0]))
        return action, name

    forward, left, right = _cone("forward"), _cone("left"), _cone("right")
    if forward is None or left is None or right is None:
        return _finish(np.zeros(4), "nocone")

    yaw_err = float(yaw_err)
    # A clear cone behind the aircraft is not a route. Turn back toward the
    # goal before any forward command. Never climb.
    if abs(yaw_err) > 0.9:
        dyaw = math.copysign(min(0.12, float(lim[3])), yaw_err)
        return _finish(np.array([0.0, 0.0, 0.0, dyaw]), "face")

    if forward >= 8.0 and min(left, right) >= 5.0:
        dx = min(0.55, float(lim[0]))
        return _finish(np.array([dx, 0.0, 0.0, 0.0]), "cruise")

    side = 1.0 if left >= right else -1.0
    dx = min(0.15, float(lim[0])) if forward >= 5.0 else 0.0
    dy = side * min(0.30, float(lim[1]))
    return _finish(np.array([dx, dy, 0.0, 0.0]), "shift")


def g0_leave_returns(
    dynamics: Any,
    z: np.ndarray,
    goal_rel: np.ndarray,
    *,
    reward_cfg: Any,
    limits: np.ndarray,
    dy: float,
    open_sign: float,
    horizon: int = 15,
    yaw_steps: int = 5,
) -> Dict[str, float]:
    """15-step return of strafe, peel, and yaw-then-forward under the existing reward.

    Does not update weights. The open side is fixed from the real cones at this step.
    """
    from experiments.aerial.rl.imagination import imagine

    lim = np.asarray(limits, dtype=np.float64).reshape(4)
    strafe = np.clip(np.array([0.0, dy, 0.0, 0.0], dtype=np.float64), -lim, lim)
    peel = np.clip(np.array([0.2, open_sign * 0.4, 0.0, 0.0], dtype=np.float64), -lim, lim)
    yaw = np.clip(np.array([0.0, 0.0, 0.0, open_sign * lim[3]], dtype=np.float64), -lim, lim)
    fwd = np.clip(np.array([0.7, 0.0, 0.0, 0.0], dtype=np.float64), -lim, lim)
    climb = np.clip(np.array([0.0, 0.0, float(lim[2]), 0.0], dtype=np.float64), -lim, lim)
    n_yaw = max(1, min(int(yaw_steps), horizon - 1))
    seqs = {
        "strafe": [strafe] * horizon,
        "peel": [peel] * horizon,
        "yawgo": [yaw] * n_yaw + [fwd] * (horizon - n_yaw),
        "climb": [climb] * horizon,
    }
    g0 = np.asarray(goal_rel, dtype=np.float64).reshape(1, -1)
    z0 = np.asarray(z, dtype=np.float64).reshape(1, -1)
    out: Dict[str, float] = {}
    for name, actions in seqs.items():
        roll = imagine(
            dynamics,
            _FixedActionSeq(actions),
            z0,
            horizon,
            reward_cfg=reward_cfg,
            goal_rel0=g0,
            action_limits=lim,
        )
        out[name] = float(roll.returns[0])
    return out


def early_yaw_readout(
    dynamics: Any,
    z: np.ndarray,
    goal_rel: np.ndarray,
    *,
    limits: np.ndarray,
    open_sign: float,
    strafe_dy: float,
    max_yaw: int = 10,
    fwd_steps: int = 8,
) -> Dict[str, float]:
    """Model-only score of straight flight versus a latched yaw, then forward.

    The turn sign is fixed from the real cones. Yaw steps stop when the
    predicted forward cone becomes the longest, or after ``max_yaw`` steps.
    Straight and the actor's strafe are rolled for the same number of steps.
    Returns collision probability and predicted goal distance, not reward.
    """
    from experiments.aerial.rl.goal_features import advance_goal_rel_body

    lim = np.asarray(limits, dtype=np.float64).reshape(4)
    yaw_a = np.clip(
        np.array([0.0, 0.0, 0.0, float(open_sign) * float(lim[3])], dtype=np.float64),
        -lim,
        lim,
    )
    fwd_a = np.clip(np.array([0.7, 0.0, 0.0, 0.0], dtype=np.float64), -lim, lim)
    fwd_a[0] = max(0.0, float(fwd_a[0]))
    strafe_a = np.clip(
        np.array([0.0, float(strafe_dy), 0.0, 0.0], dtype=np.float64), -lim, lim
    )
    z0 = np.asarray(z, dtype=np.float64).reshape(-1).copy()
    g0 = np.asarray(goal_rel, dtype=np.float64).reshape(-1).copy()

    def _dist(g: np.ndarray) -> float:
        if g.size > 3 and np.isfinite(float(g[3])):
            return float(g[3])
        return float(np.linalg.norm(g[:3]))

    def _step(zz: np.ndarray, gg: np.ndarray, action: np.ndarray):
        out = dynamics.step(zz, action, goal_rel=gg)
        p = float(getattr(out, "p_coll", 0.0) or 0.0)
        g1 = np.asarray(advance_goal_rel_body(gg, action), dtype=np.float64).reshape(-1)
        z1 = np.asarray(out.z_next, dtype=np.float64).reshape(-1)
        return z1, g1, p

    zz = z0.copy()
    gg = g0.copy()
    yaw_ps: List[float] = []
    yaw_n = 0
    opened = False
    for _ in range(int(max_yaw)):
        zz, gg, p = _step(zz, gg, yaw_a)
        yaw_ps.append(p)
        yaw_n += 1
        hats = None
        cones_fn = getattr(dynamics, "cone_hats", None)
        if callable(cones_fn):
            hats = cones_fn(zz)
        if hats is not None and float(hats[0]) >= max(float(hats[1]), float(hats[2])):
            opened = True
            break
    for _ in range(int(fwd_steps)):
        zz, gg, p = _step(zz, gg, fwd_a)
        yaw_ps.append(p)
    n = len(yaw_ps)
    d0 = _dist(g0)
    yaw_d1 = _dist(gg)

    def _fixed(action: np.ndarray) -> tuple:
        zf = z0.copy()
        gf = g0.copy()
        ps: List[float] = []
        for _ in range(n):
            zf, gf, p = _step(zf, gf, action)
            ps.append(p)
        return ps, _dist(gf)

    straight_ps, straight_d1 = _fixed(fwd_a)
    strafe_ps, strafe_d1 = _fixed(strafe_a)

    def _pmax(ps: List[float]) -> float:
        return float(max(ps)) if ps else 1.0

    yaw_p = _pmax(yaw_ps)
    straight_p = _pmax(straight_ps)
    return {
        "d0": d0,
        "yaw_p": yaw_p,
        "yaw_d1": yaw_d1,
        "yaw_n": float(yaw_n),
        "opened": float(opened),
        "straight_p": straight_p,
        "straight_d1": straight_d1,
        "strafe_p": _pmax(strafe_ps),
        "strafe_d1": strafe_d1,
        "n": float(n),
        "yaw_better": float(
            yaw_p + 0.05 < straight_p and yaw_d1 < d0
        ),
    }


def distribution_yaw_readout(
    dynamics: Any,
    z: np.ndarray,
    goal_rel: np.ndarray,
    *,
    limits: np.ndarray,
    open_sign: float,
    max_yaw: int = 10,
    max_occupied: int = 8,
    near_center_min: float = 0.20,
) -> Dict[str, float]:
    """Score yaw against straight from the predicted forward-view depth.

    The depth decoder's map is split into a horizontal band. Center is the
    middle half of the columns. Collision is compared only while a straight
    step still has at least ``near_center_min`` of that center within 18 m.
    The full-sequence max is recorded and does not decide ``better``.
    """
    from experiments.aerial.rl.depth_geometry import forward_view_mass
    from experiments.aerial.rl.goal_features import advance_goal_rel_body

    depth_fn = getattr(dynamics, "predicted_depth", None)
    empty = {
        "ok": 0.0,
        "near0": 0.0,
        "near_yaw": 0.0,
        "near_straight": 0.0,
        "med0": 0.0,
        "med_open": 0.0,
        "med_yaw": 0.0,
        "yaw_n": 0.0,
        "shifted": 0.0,
        "occ_n": 0.0,
        "straight_p": 1.0,
        "yaw_p": 1.0,
        "seq_straight_p": 1.0,
        "seq_yaw_p": 1.0,
        "better": 0.0,
    }
    if not callable(depth_fn):
        return empty
    lim = np.asarray(limits, dtype=np.float64).reshape(4)
    yaw_a = np.clip(
        np.array([0.0, 0.0, 0.0, float(open_sign) * float(lim[3])], dtype=np.float64),
        -lim,
        lim,
    )
    fwd_a = np.clip(np.array([0.7, 0.0, 0.0, 0.0], dtype=np.float64), -lim, lim)
    fwd_a[0] = max(0.0, float(fwd_a[0]))
    z0 = np.asarray(z, dtype=np.float64).reshape(-1).copy()
    g0 = np.asarray(goal_rel, dtype=np.float64).reshape(-1).copy()

    def _step(zz: np.ndarray, gg: np.ndarray, action: np.ndarray):
        out = dynamics.step(zz, action, goal_rel=gg)
        p = float(getattr(out, "p_coll", 0.0) or 0.0)
        g1 = np.asarray(advance_goal_rel_body(gg, action), dtype=np.float64).reshape(-1)
        z1 = np.asarray(out.z_next, dtype=np.float64).reshape(-1)
        return z1, g1, p

    def _mass(zz: np.ndarray) -> Optional[Dict[str, float]]:
        depth = depth_fn(zz)
        if depth is None:
            return None
        return forward_view_mass(depth)

    mass0 = _mass(z0)
    if mass0 is None:
        return empty
    med_open = float(mass0["med_left"] if float(open_sign) >= 0.0 else mass0["med_right"])

    occ_straight: List[float] = []
    occ_yaw: List[float] = []
    zs = z0.copy()
    gs = g0.copy()
    mass_straight = mass0
    for _ in range(int(max_occupied)):
        mass_s = _mass(zs)
        if mass_s is None or float(mass_s["near_center"]) < float(near_center_min):
            break
        mass_straight = mass_s
        _, _, p_s = _step(zs.copy(), gs.copy(), fwd_a)
        _, _, p_y = _step(zs.copy(), gs.copy(), yaw_a)
        occ_straight.append(p_s)
        occ_yaw.append(p_y)
        zs, gs, _ = _step(zs, gs, fwd_a)

    zy = z0.copy()
    gy = g0.copy()
    yaw_n = 0
    mass_yaw = mass0
    yaw_ps: List[float] = []
    for _ in range(int(max_yaw)):
        zy, gy, p = _step(zy, gy, yaw_a)
        yaw_ps.append(p)
        yaw_n += 1
        mass_y = _mass(zy)
        if mass_y is None:
            break
        mass_yaw = mass_y
        if float(mass_yaw["near_center"]) + 0.10 < float(mass0["near_center"]):
            break
        if float(mass_yaw["near_center"]) < float(near_center_min):
            break
    for _ in range(8):
        zy, gy, p = _step(zy, gy, fwd_a)
        yaw_ps.append(p)
    n_seq = len(yaw_ps)
    zs2 = z0.copy()
    gs2 = g0.copy()
    straight_ps: List[float] = []
    for _ in range(n_seq):
        zs2, gs2, p = _step(zs2, gs2, fwd_a)
        straight_ps.append(p)

    def _pmax(ps: List[float]) -> float:
        return float(max(ps)) if ps else 1.0

    straight_p = _pmax(occ_straight)
    yaw_p = _pmax(occ_yaw)
    shifted = float(
        float(mass_yaw["near_center"]) + 0.10 < float(mass0["near_center"])
    )
    return {
        "ok": 1.0,
        "near0": float(mass0["near_center"]),
        "near_yaw": float(mass_yaw["near_center"]),
        "near_straight": float(mass_straight["near_center"]),
        "med0": float(mass0["med_center"]),
        "med_open": med_open,
        "med_yaw": float(mass_yaw["med_center"]),
        "yaw_n": float(yaw_n),
        "shifted": shifted,
        "occ_n": float(len(occ_straight)),
        "straight_p": straight_p,
        "yaw_p": yaw_p,
        "seq_straight_p": _pmax(straight_ps),
        "seq_yaw_p": _pmax(yaw_ps),
        "better": float(
            len(occ_straight) > 0 and yaw_p + 0.05 < straight_p
        ),
    }


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
        "--encode-current",
        action="store_true",
        help="Each step decides from the current frame only. "
        "Default carries the previous latent and the last action into the next step.",
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
        "--cruise-veto",
        action="store_true",
        help="Score the Phase-2 step with the krs6 consequence ensemble and "
        "replace circles / unsafe cruise steps with a goal-aligned step.",
    )
    parser.add_argument(
        "--consequence-ckpt",
        action="append",
        default=None,
        help="Consequence checkpoint. Repeat for an ensemble. "
        "Default is the three krs6 seeds.",
    )
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
        choices=("polyline", "toward_g", "direct_g", "scene", "road_center"),
        help=(
            "polyline=waterline; toward_g=main (clip toward G, policy handles obstacles); "
            "scene=alias for toward_g (fan intent removed; see E1r2 analysis); "
            "direct_g=ablation A; road_center=follow the open street, turn when it ends"
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
    from experiments.aerial.rl.scene_intent import (
        RoadCenterIntent,
        SceneIntentPlanner,
        TowardGoalIntent,
        annotation_yaw_rad,
    )
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

    cruise_veto = None
    if bool(args.cruise_veto):
        from experiments.aerial.rl.cruise_veto import KRS6_CKPTS, CruiseVeto, ensemble_score_fn

        ckpts = list(args.consequence_ckpt or [])
        if not ckpts:
            ckpts = [str((root / rel).resolve()) for rel in KRS6_CKPTS]
        missing = [p for p in ckpts if not Path(p).is_file()]
        if missing:
            raise SystemExit(f"ABORT: consequence checkpoint missing: {missing}")
        cruise_veto = CruiseVeto(ensemble_score_fn(ckpts, device_str))
        logger.info("cruise veto on, %d consequence ckpts, surv>=%.2f", len(ckpts), cruise_veto.surv_min)

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
        dynamics,
        actor_ac,
        deterministic=True,
        stream_latent=not bool(args.encode_current),
    )
    if bool(args.encode_current):
        logger.warning(
            "encode-current: each step uses the current frame only"
        )
    else:
        logger.info("stream latent: each step carries the previous latent and action")

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
    elif subgoal_source == "road_center":
        intent = RoadCenterIntent(
            r_m=min(r_intent, 25.0),
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

    wall_passage = None
    wall_bits = os.environ.get("WALL_BITS")
    if wall_bits:
        from experiments.aerial.rl.near_wall_turn import WallBitPassage

        wall_passage = WallBitPassage.load(wall_bits)
        logger.warning("wall-bit passage armed (%d bits)", int(wall_passage.bits.size))
    visual_residual = None
    visual_path = os.environ.get("VISUAL_RESIDUAL")
    if visual_path:
        from experiments.aerial.rl.visual_residual import VisualResidual

        visual_residual = VisualResidual.load(visual_path, device=device_str)
        logger.warning("visual residual armed: %s", visual_path)
    clearance_head = None
    clearance_path = os.environ.get("CLEARANCE_HEAD")
    if clearance_path:
        from experiments.aerial.rl.clearance_head import ClearanceHead

        clearance_head = ClearanceHead.load(clearance_path, device=device_str)
        logger.warning("clearance head armed: %s", clearance_path)
    clearance_stick = None
    yaw_prior = os.environ.get("YAW_PRIOR")
    depth_policy = os.environ.get("DEPTH_POLICY")
    depth_stick = os.environ.get("DEPTH_STICK")
    clearance_stick_spec = os.environ.get("CLEARANCE_STICK")
    if yaw_prior:
        from experiments.aerial.rl.wm_yaw_prior import YawPriorDeploy

        clearance_stick = YawPriorDeploy.load(yaw_prior, device_str)
        logger.warning("yaw prior armed: %s", yaw_prior)
    elif depth_policy:
        from experiments.aerial.rl.depth_policy import DepthPolicyDeploy

        clearance_stick = DepthPolicyDeploy.load(depth_policy, device_str)
        logger.warning("depth policy armed: %s", depth_policy)
    elif depth_stick:
        from experiments.aerial.rl.clearance_stick import DepthStickDeploy

        clearance_stick = DepthStickDeploy.load(depth_stick, device_str)
        logger.warning("depth stick armed: %s", depth_stick)
    elif clearance_stick_spec:
        from experiments.aerial.rl.clearance_stick import ClearanceStickDeploy

        clearance_stick = ClearanceStickDeploy.load(clearance_stick_spec, device_str)
        logger.warning("clearance stick armed: %s", clearance_stick_spec)
    action_chunk = None
    chunk_path = os.environ.get("ACTION_CHUNK")
    if chunk_path:
        from experiments.aerial.rl.action_chunk import ActionChunkNet, ActionChunkRunner

        action_chunk = ActionChunkRunner(ActionChunkNet.load(chunk_path, device=device_str))
        logger.warning("action chunk armed: %s", chunk_path)
    passage_player = None
    passage_path = os.environ.get("PASSAGE_CODE")
    if passage_path:
        from experiments.aerial.rl.passage_code import PassagePlayer

        passage_player = PassagePlayer.load(passage_path, device=device_str)
        logger.warning(
            "passage code armed: %s margin=%.3f release=%d",
            passage_path,
            passage_player.margin,
            passage_player.release_idx,
        )
    passage_record = os.environ.get("PASSAGE_RECORD") == "1"
    vision_policy = None
    vision_path = os.environ.get("VISION_POLICY")
    if vision_path:
        from experiments.aerial.rl.visual_residual import VisualResidual

        vision_policy = VisualResidual.load(vision_path, device=device_str)
        logger.warning("vision policy armed: %s", vision_path)
    learned_passage = None
    learned_path = os.environ.get("LEARNED_PASSAGE")
    if learned_path:
        from experiments.aerial.rl.passage_code import LearnedPassage

        learned_passage = LearnedPassage.load(learned_path, device=device_str)
        logger.warning(
            "learned passage armed: %s margin=%.3f", learned_path, learned_passage.margin
        )
    latched_stick = None
    latched_path = os.environ.get("LATCHED_STICK")
    if latched_path and passage_player is not None:
        from experiments.aerial.rl.passage_code import LatchedStick
        from experiments.aerial.rl.visual_residual import VisualResidual

        latched_stick = LatchedStick(
            passage_player, VisualResidual.load(latched_path, device=device_str)
        )
        logger.warning("latched stick armed: %s", latched_path)
        passage_player = None

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
            if isinstance(intent, RoadCenterIntent):
                intent.reset(heading=annotation_yaw_rad(start_yaw))
            else:
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
        if wall_passage is not None:
            wall_passage.reset()
        if action_chunk is not None:
            action_chunk.reset()
        if passage_record:
            from experiments.aerial.rl.gap_passage import reset_close_passage

            reset_close_passage()
        if passage_player is not None:
            passage_player.reset()
        if latched_stick is not None:
            latched_stick.reset()
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
                    if isinstance(intent, RoadCenterIntent):
                        intent.reset(heading=annotation_yaw_rad(start_yaw))
                    else:
                        intent.reset()
                if global_planner is not None:
                    global_planner.reset(pts, goal=goal_pos)
                if shield is not None:
                    shield.reset()
                if depth_pred is not None:
                    depth_pred.reset()
                tau_pred.reset()
                policy.reset()
                if wall_passage is not None:
                    wall_passage.reset()
                if action_chunk is not None:
                    action_chunk.reset()
                if passage_player is not None:
                    passage_player.reset()
                if latched_stick is not None:
                    latched_stick.reset()
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
            veto_counts = {
                "cruise": 0,
                "straight": 0,
                "peel": 0,
                "ahead": 0,
                "stall": 0,
                "hover": 0,
                "nodepth": 0,
            }
            if cruise_veto is not None:
                cruise_veto.reset()
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
            script_peel_left = 0
            script_peel_sign = 0.0
            script_peel_used = False
            script_yaw_phase = None
            script_yaw_sign = 0.0
            script_yaw_n = 0
            script_fwd_n = 0
            script_open_mode = os.environ.get("SCRIPT_OPEN", "").strip()
            script_open_phase = None
            script_open_sign = 0.0
            script_open_left = 0
            script_open_used = False
            yaw_collect = os.environ.get("YAW_COLLECT", "").strip() == "1"
            yaw_collect_phase = None
            yaw_collect_sign = 0.0
            yaw_collect_left = 0
            yaw_collect_used = False
            early_press = os.environ.get("EARLY_PRESS_COLLECT", "").strip() == "1"
            early_press_phase = None
            early_press_sign = 0.0
            early_press_left = 0
            early_press_n = 0
            early_press_hist = []
            street_mem: Dict[str, float] = {}
            around = None
            if os.environ.get("AROUND") == "1":
                from experiments.aerial.rl.around import AroundDetour

                around = AroundDetour()
            corner90 = None
            if os.environ.get("CORNER90") == "1":
                from experiments.aerial.rl.corner90 import Corner90Controller

                corner90 = Corner90Controller.load(
                    os.environ["CORNER90_CKPT"],
                    depth_pred._model,
                    args.device,
                )

            z_hold = float(np.asarray(obs.state, dtype=np.float64).reshape(-1)[2])
            for step in range(args.max_steps):
                d_fwd = None
                obs.info.pop("depth_min_pred", None)
                obs.info.pop("depth_cones_pred", None)
                obs.info.pop("depth_profile_pred", None)
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
                        depth_map = getattr(depth_pred, "last_depth", None)
                        if depth_map is not None:
                            from experiments.aerial.rl.depth_geometry import deploy_cones

                            cones = deploy_cones(np.asarray(depth_map, dtype=np.float64))
                        if isinstance(cones, dict):
                            obs.info["depth_cones_pred"] = {
                                k: (float(v) if v is not None and np.isfinite(float(v)) else None)
                                for k, v in cones.items()
                            }
                            cf = cones.get("forward")
                            if cf is not None and np.isfinite(float(cf)):
                                d_fwd = float(cf)
                                obs.info["depth_min_pred"] = float(cf)
                    else:
                        d_fwd = depth_pred.predict_min(obs)
                        if d_fwd is not None:
                            obs.info["depth_min_pred"] = float(d_fwd)
                    from experiments.aerial.rl.cone_transition import stamp_column_profile

                    stamp_column_profile(obs, depth_pred)
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
                        depth_cones=(
                            obs.info.get("depth_cones_pred")
                            if subgoal_source in ("scene", "road_center")
                            else None
                        ),
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
                obs.info["hold_street"] = bool(
                    s_info.get("subgoal_source") == "road_center"
                    and not s_info.get("road_committed")
                )
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
                if clearance_stick is not None:
                    action = clearance_stick.act(obs)
                    if step % 5 == 0 or float(np.asarray(action).reshape(-1)[0]) < 0.2:
                        logger.info(
                            "CLEARANCE_STICK step=%d act=%s near=%s",
                            step,
                            np.round(np.asarray(action), 3).tolist(),
                            getattr(clearance_stick, "last_logit", None),
                        )
                elif vision_policy is not None:
                    _lim = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                    action = np.clip(vision_policy.residual(obs.rgb), -_lim, _lim)
                    if step < 3 or step % 20 == 0:
                        logger.info(
                            "VISION step=%d act=%s",
                            step,
                            np.round(np.asarray(action), 3).tolist(),
                        )
                elif passage_record:
                    from experiments.aerial.rl.gap_passage import close_then_gap_action, gap_action

                    _close_m = os.environ.get("PASSAGE_CLOSE_M")
                    if _close_m:
                        action, _passage_phase = close_then_gap_action(
                            p_curr,
                            float(curr_yaw),
                            goal_pos,
                            max(float(p_curr[2]), 29.0),
                            getattr(obs, "depth", None),
                            close_m=float(_close_m),
                            press_yaw=float(start_yaw)
                            + float(os.environ.get("PASSAGE_PRESS_OFFSET", "0")),
                        )
                    else:
                        action, _passage_phase = gap_action(
                            p_curr,
                            float(curr_yaw),
                            goal_pos,
                            z_hold=max(float(p_curr[2]), 29.0),
                        )
                    if step < 3 or step % 20 == 0:
                        logger.info(
                            "PASSAGE step=%d phase=%s xy=(%.0f,%.0f) yaw=%.2f",
                            step,
                            _passage_phase,
                            float(p_curr[0]),
                            float(p_curr[1]),
                            float(curr_yaw),
                        )
                elif latched_stick is not None:
                    _latched = latched_stick.maybe(obs.rgb)
                    if _latched is not None:
                        action, _n = _latched
                        if step < 3 or step % 20 == 0:
                            logger.info(
                                "LATCH step=%d n=%s act=%s",
                                step,
                                _n,
                                np.round(np.asarray(action), 3).tolist(),
                            )
                elif learned_passage is not None:
                    _learned = learned_passage.maybe(obs.rgb)
                    if _learned is not None:
                        action, _code = _learned
                        if step < 3 or step % 20 == 0:
                            logger.info(
                                "LEARNED step=%d code=%s act=%s",
                                step,
                                _code,
                                np.round(np.asarray(action), 3).tolist(),
                            )
                elif passage_player is not None:
                    _played = passage_player.maybe(obs.rgb)
                    if _played is not None:
                        action = _played
                        if step < 3 or step % 20 == 0:
                            logger.info(
                                "PASSAGE_CODE step=%d idx=%s act=%s",
                                step,
                                passage_player.idx,
                                np.round(np.asarray(action), 3).tolist(),
                            )
                elif action_chunk is not None:
                    action = action_chunk.next_action(obs.rgb)
                    if step < 3 or step % 20 == 0:
                        logger.info(
                            "CHUNK step=%d q=%d act=%s",
                            step,
                            len(action_chunk.queue),
                            np.round(np.asarray(action), 3).tolist(),
                        )
                elif clearance_head is not None and getattr(obs, "depth", None) is not None:
                    _res = clearance_head.residual(obs)
                    _lim = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                    action = np.clip(np.asarray(action, dtype=np.float64) + _res, -_lim, _lim)
                    if step < 3 or step % 20 == 0:
                        logger.info(
                            "CLEAR step=%d res=%s act=%s",
                            step,
                            np.round(_res, 3).tolist(),
                            np.round(np.asarray(action), 3).tolist(),
                        )
                elif visual_residual is not None:
                    _res = visual_residual.residual(obs.rgb)
                    _lim = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                    action = np.clip(np.asarray(action, dtype=np.float64) + _res, -_lim, _lim)
                    if step < 3 or step % 20 == 0:
                        logger.info(
                            "VRES step=%d res=%s act=%s",
                            step,
                            np.round(_res, 3).tolist(),
                            np.round(np.asarray(action), 3).tolist(),
                        )
                if wall_passage is not None:
                    _z = getattr(policy, "_latent", None)
                    _wall = wall_passage.apply(_z, p_curr, curr_yaw, goal_pos)
                    if _wall is not None:
                        action, _wall_phase = _wall
                        if step < 3 or step % 20 == 0 or _wall_phase != getattr(wall_passage, "_logged", ""):
                            logger.info(
                                "WALL step=%d phase=%s xy=(%.0f,%.0f) yaw=%.2f",
                                step,
                                _wall_phase,
                                float(p_curr[0]),
                                float(p_curr[1]),
                                float(curr_yaw),
                            )
                            wall_passage._logged = _wall_phase
                    elif wall_passage.phase == "release" and not getattr(wall_passage, "_release_logged", False):
                        logger.info(
                            "WALL release step=%d xy=(%.0f,%.0f) yaw=%.2f",
                            step,
                            float(p_curr[0]),
                            float(p_curr[1]),
                            float(curr_yaw),
                        )
                        wall_passage._release_logged = True
                action_actor = np.asarray(action, dtype=np.float64).reshape(4).copy()
                oa_step: dict = {}
                # Cost-head probe, including --no-planner runs: press-forward vs
                # lateral peel toward the more open side. Does not change the action.
                _feat = getattr(policy, "_latent", None)
                if (
                    _feat is not None
                    and bool(getattr(dynamics, "obstacle_cost_trained", False))
                    and hasattr(dynamics, "predict_obstacle_cost")
                ):
                    _lim = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                    _cones_now = obs.info.get("depth_cones_pred") or {}
                    _dl, _dr = _cones_now.get("left"), _cones_now.get("right")
                    _open = 1.0
                    if _dl is not None and _dr is not None:
                        _open = 1.0 if float(_dl) >= float(_dr) else -1.0
                    _a_press = np.clip(
                        np.array([0.7, 0.0, 0.0, 0.0], dtype=np.float64), -_lim, _lim
                    )
                    _a_peel = np.clip(
                        np.array([0.2, _open * 0.7, 0.0, 0.0], dtype=np.float64),
                        -_lim,
                        _lim,
                    )
                    _a_back = np.clip(
                        np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float64), -_lim, _lim
                    )
                    _a_exec = np.clip(
                        np.asarray(action_actor, dtype=np.float64).reshape(4), -_lim, _lim
                    )
                    try:
                        _oc_press = float(dynamics.predict_obstacle_cost(_feat, _a_press))
                        _oc_peel = float(dynamics.predict_obstacle_cost(_feat, _a_peel))
                        _oc_back = float(dynamics.predict_obstacle_cost(_feat, _a_back))
                        _oc_exec = float(dynamics.predict_obstacle_cost(_feat, _a_exec))
                        oa_step["oc_press"] = _oc_press
                        oa_step["oc_peel"] = _oc_peel
                        oa_step["oc_back"] = _oc_back
                        oa_step["oc_exec"] = _oc_exec
                        oa_step["oc_peel_beats_press"] = bool(_oc_peel + 1e-4 < _oc_press)
                        oa_step["oc_peel_beats_back"] = bool(_oc_peel + 1e-4 < _oc_back)
                        oa_step["act_dx"] = float(action_actor[0])
                        oa_step["act_dy"] = float(action_actor[1])
                    except Exception:
                        pass
                if os.environ.get("G0_LEAVE") == "1" and _feat is not None:
                    _cones_g0 = obs.info.get("depth_cones_pred") or {}
                    _df = _cones_g0.get("forward")
                    _dl, _dr = _cones_g0.get("left"), _cones_g0.get("right")
                    if (
                        _df is not None
                        and float(_df) < 8.0
                        and _dl is not None
                        and _dr is not None
                    ):
                        _open = 1.0 if float(_dl) >= float(_dr) else -1.0
                        try:
                            _g0 = g0_leave_returns(
                                dynamics,
                                _feat,
                                g_rel_body,
                                reward_cfg=reward_cfg,
                                limits=cur_limits,
                                dy=float(action_actor[1]),
                                open_sign=_open,
                            )
                            oa_step["g0_strafe"] = _g0["strafe"]
                            oa_step["g0_peel"] = _g0["peel"]
                            oa_step["g0_yawgo"] = _g0["yawgo"]
                            oa_step["g0_climb"] = _g0["climb"]
                            _best = max(_g0, key=_g0.get)
                            oa_step["g0_best"] = _best
                            oa_step["g0_leave_wins"] = bool(
                                max(_g0["peel"], _g0["yawgo"]) > _g0["strafe"] + 0.05
                            )
                            oa_step["g0_climb_wins"] = bool(
                                _g0["climb"] > _g0["strafe"] + 0.05
                            )
                            logger.info(
                                "G0 d_fwd=%.1f strafe=%.3f peel=%.3f yawgo=%.3f climb=%.3f best=%s",
                                float(_df),
                                _g0["strafe"],
                                _g0["peel"],
                                _g0["yawgo"],
                                _g0["climb"],
                                _best,
                            )
                        except Exception:
                            logger.exception("G0 leave rollout failed")
                if os.environ.get("EARLY_YAW_READ") == "1" and _feat is not None:
                    _cones_ey = obs.info.get("depth_cones_pred") or {}
                    _df_ey = _cones_ey.get("forward")
                    _dl_ey, _dr_ey = _cones_ey.get("left"), _cones_ey.get("right")
                    if (
                        _df_ey is not None
                        and _dl_ey is not None
                        and _dr_ey is not None
                        and 8.0 < float(_df_ey) <= 18.0
                    ):
                        from experiments.aerial.rl.depth_geometry import forward_view_mass

                        _gt_near = _gt_med = _enc_near = _enc_med = float("nan")
                        _live_near = _live_med = float("nan")
                        _gt = getattr(obs, "depth", None)
                        if _gt is not None:
                            _gt_a = np.asarray(_gt, dtype=np.float64)
                            if _gt_a.ndim == 3:
                                _gt_a = _gt_a[0]
                            if _gt_a.ndim == 2:
                                _gm = forward_view_mass(_gt_a)
                                _gt_near, _gt_med = float(_gm["near_center"]), float(_gm["med_center"])
                        try:
                            _ed = dynamics.predicted_depth(dynamics.encode(obs))
                            if _ed is not None:
                                _em = forward_view_mass(_ed)
                                _enc_near = float(_em["near_center"])
                                _enc_med = float(_em["med_center"])
                            _ld = dynamics.predicted_depth(_feat)
                            if _ld is not None:
                                _lm = forward_view_mass(_ld)
                                _live_near = float(_lm["near_center"])
                                _live_med = float(_lm["med_center"])
                        except Exception:
                            logger.exception("DEPTH compare failed")
                        logger.info(
                            "DEPTH d_fwd=%.1f gt_near=%.2f gt_med=%.1f enc_near=%.2f enc_med=%.1f live_near=%.2f live_med=%.1f",
                            float(_df_ey),
                            _gt_near,
                            _gt_med,
                            _enc_near,
                            _enc_med,
                            _live_near,
                            _live_med,
                        )
                        _open_m = max(float(_dl_ey), float(_dr_ey))
                        _short_m = min(float(_dl_ey), float(_dr_ey))
                        if _open_m >= float(_df_ey) and _open_m - _short_m >= 2.0:
                            _sign_ey = 1.0 if float(_dl_ey) >= float(_dr_ey) else -1.0
                            try:
                                _ey = distribution_yaw_readout(
                                    dynamics,
                                    _feat,
                                    g_rel_body,
                                    limits=cur_limits,
                                    open_sign=_sign_ey,
                                )
                                logger.info(
                                    "VIEW d_fwd=%.1f sign=%+.0f near0=%.2f near_yaw=%.2f "
                                    "near_straight=%.2f med0=%.1f med_open=%.1f med_yaw=%.1f "
                                    "yaw_n=%d shifted=%s occ_n=%d "
                                    "straight_p=%.3f yaw_p=%.3f "
                                    "seq_straight_p=%.3f seq_yaw_p=%.3f better=%s",
                                    float(_df_ey),
                                    _sign_ey,
                                    _ey["near0"],
                                    _ey["near_yaw"],
                                    _ey["near_straight"],
                                    _ey["med0"],
                                    _ey["med_open"],
                                    _ey["med_yaw"],
                                    int(_ey["yaw_n"]),
                                    bool(_ey["shifted"]),
                                    int(_ey["occ_n"]),
                                    _ey["straight_p"],
                                    _ey["yaw_p"],
                                    _ey["seq_straight_p"],
                                    _ey["seq_yaw_p"],
                                    bool(_ey["better"]),
                                )
                            except Exception:
                                logger.exception("EARLY_YAW readout failed")
                        else:
                            logger.info(
                                "EARLY_YAW skip d_fwd=%.1f L=%.1f R=%.1f side_not_clearer",
                                float(_df_ey),
                                float(_dl_ey),
                                float(_dr_ey),
                            )
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

                if cruise_veto is not None:
                    _depth_map = (
                        getattr(depth_pred, "last_depth", None) if depth_pred is not None else None
                    )
                    action, _vtag = cruise_veto.apply(
                        action, _depth_map, p_curr, float(curr_yaw), goal_pos, cur_limits
                    )
                    veto_counts[_vtag] = veto_counts.get(_vtag, 0) + 1
                    if step < 3 or step % 20 == 0:
                        _vl = cruise_veto.last
                        logger.info(
                            "VETO step=%d tag=%s surv_c=%.2f surv_s=%.2f prog_c=%.2f prog_s=%.2f d=%.1f",
                            step,
                            _vtag,
                            float(_vl.get("surv_c", float("nan"))),
                            float(_vl.get("surv_s", float("nan"))),
                            float(_vl.get("prog_c", float("nan"))),
                            float(_vl.get("prog_s", float("nan"))),
                            float(_vl.get("d_goal", float("nan"))),
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

                if corner90 is not None and obs.rgb is not None:
                    action = corner90.act(obs.rgb, p_curr, curr_yaw, goal_pos)
                    action = np.clip(
                        action,
                        -np.abs(np.asarray(cur_limits, dtype=np.float64)),
                        np.abs(np.asarray(cur_limits, dtype=np.float64)),
                    )
                    oa_step["corner90"] = True
                    if step < 3 or step % 20 == 0:
                        logger.info(
                            "CORNER90 step=%d dx=%.2f dy=%.2f dyaw=%.2f",
                            step,
                            float(action[0]),
                            float(action[1]),
                            float(action[3]),
                        )

                if around is not None:
                    gt_depth = getattr(obs, "depth", None)
                    if gt_depth is not None:
                        gt_arr = np.asarray(gt_depth, dtype=np.float64)
                        if gt_arr.ndim == 3:
                            gt_arr = gt_arr[0]
                        from experiments.aerial.rl.depth_geometry import approach_clearances

                        gf, gl, gr = approach_clearances(gt_arr, center_frac=0.12)
                    else:
                        cones_now = obs.info.get("depth_cones_pred") or {}
                        gf = cones_now.get("forward")
                        gl = cones_now.get("left")
                        gr = cones_now.get("right")
                    if gf is not None and gl is not None and gr is not None:
                        action, around_phase = around.act(
                            p_curr,
                            curr_yaw,
                            goal_pos,
                            float(gf),
                            float(gl),
                            float(gr),
                            cur_limits,
                        )
                        oa_step["around"] = around_phase
                        if step < 5 or step % 15 == 0 or around_phase in (
                            "latch",
                            "relatch",
                            "flip",
                            "recheck",
                        ):
                            logger.info(
                                "AROUND step=%d phase=%s F=%.1f L=%.1f R=%.1f",
                                step,
                                around_phase,
                                float(gf),
                                float(gl),
                                float(gr),
                            )

                if bool(getattr(reward_cfg, "forbid_backward_motion", False)):
                    from experiments.aerial.rl.env.action import forbid_backward_dx

                    action = forbid_backward_dx(action)

                if os.environ.get("ORACLE_STEP") == "1" and policy._latent is not None:
                    try:
                        _oa, _oname, _ocost, _oprog = select_receding_primitive(
                            dynamics,
                            policy._latent,
                            g_rel_body,
                            cur_limits,
                        )
                        action = np.asarray(_oa, dtype=np.float64).reshape(4)
                        oa_step["oracle"] = _oname
                        oa_step["oracle_cost"] = float(_ocost)
                        oa_step["oracle_prog"] = float(_oprog)
                        if step < 3 or step % 25 == 0:
                            logger.info(
                                "ORACLE step=%d choice=%s cost=%.3f prog=%.3f",
                                step,
                                _oname,
                                float(_ocost),
                                float(_oprog),
                            )
                    except Exception:
                        logger.exception("ORACLE select failed")

                if os.environ.get("STREET") == "1" and policy._latent is not None:
                    _cones_st = obs.info.get("depth_cones_pred") or {}
                    _yaw_err = float(obs.info.get("yaw_err_rad") or 0.0)
                    try:
                        _sa, _sname, _scost, _sprog = select_receding_primitive(
                            dynamics,
                            policy._latent,
                            g_rel_body,
                            cur_limits,
                            horizon=8,
                            cones=_cones_st,
                            yaw_err=_yaw_err,
                            allow_hits=False,
                        )
                        action = np.asarray(_sa, dtype=np.float64).reshape(4)
                        action[2] = 0.0
                        _sf = _cones_st.get("forward")
                        _sl = _cones_st.get("left")
                        _sr = _cones_st.get("right")
                        _su = _cones_st.get("up")
                        try:
                            _sf, _sl, _sr = float(_sf), float(_sl), float(_sr)
                        except (TypeError, ValueError):
                            _sf = _sl = _sr = None
                        try:
                            _su = float(_su)
                        except (TypeError, ValueError):
                            _su = None
                        _skirt_left = int(street_mem.get("skirt_left") or 0)
                        _skirt_on = False
                        if _sf is not None and _sl is not None and _sr is not None:
                            _locked = float(street_mem.get("skirt") or 0.0)
                            _open_locked = (
                                _sl if _locked > 0.0 else _sr if _locked < 0.0 else 0.0
                            )
                            # Stay on the side chosen at trigger. Do not flip:
                            # flipping yawed the nose back and forth in place.
                            if (
                                _skirt_left > 0
                                and _open_locked >= 8.0
                                and _sf >= 6.0
                            ):
                                _skirt_on = True
                            elif (
                                _skirt_left <= 0
                                and 8.0 <= _sf <= 16.0
                                and min(_sl, _sr) >= 8.0
                                and abs(_yaw_err) < 0.35
                            ):
                                street_mem["skirt"] = 1.0 if _sl >= _sr else -1.0
                                street_mem["skirt_left"] = 16
                                street_mem["skirt_mode"] = "commit"
                                _skirt_left = 16
                                _skirt_on = True
                        if _skirt_on:
                            _side = float(street_mem.get("skirt") or 1.0)
                            _open = _sl if _side > 0.0 else _sr
                            _lims = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                            _yaw_cap = min(0.12, float(_lims[3]))
                            if _sf >= 8.0 and abs(_yaw_err) <= 0.50:
                                action = np.array(
                                    [min(0.50, float(_lims[0])), 0.0, 0.0, 0.0],
                                    dtype=np.float64,
                                )
                                _sname = "skirt_go"
                                if _sf >= 14.0:
                                    street_mem["skirt_left"] = 0
                                    street_mem["skirt_mode"] = ""
                            elif _open >= _sf + 2.0 and abs(_yaw_err) < 0.90:
                                action = np.array(
                                    [0.0, 0.0, 0.0, _side * _yaw_cap],
                                    dtype=np.float64,
                                )
                                _sname = "skirt_yaw"
                            elif _sf >= 6.0:
                                action = np.array(
                                    [min(0.35, float(_lims[0])), 0.0, 0.0, 0.0],
                                    dtype=np.float64,
                                )
                                _sname = "skirt_go"
                            else:
                                action = np.zeros(4, dtype=np.float64)
                                _sname = "hold"
                                street_mem["skirt_left"] = 0
                                street_mem["skirt_mode"] = ""
                            if street_mem.get("skirt_left", 0):
                                street_mem["skirt_left"] = max(0, _skirt_left - 1)
                        # Corner: goal bearing goes through the facade and every
                        # cone is short. Yaw off that bearing, then fly the new
                        # heading. Do not yaw back toward the goal until it is
                        # ahead again. One attempt per episode.
                        if _sf is not None and _sl is not None and _sr is not None:
                            if max(_sl, _sr) >= 15.0 and abs(_sl - _sr) >= 5.0:
                                street_mem["open_side"] = 1.0 if _sl >= _sr else -1.0
                            _phase = int(street_mem.get("escape_phase") or 0)
                            _pocket = (
                                _sf < 8.0
                                and _sl < 10.0
                                and _sr < 10.0
                                and float(d_to_goal) > 20.0
                                and not _skirt_on
                            )
                            if _phase == 0 and _pocket and not street_mem.get("escape_used"):
                                street_mem["escape_phase"] = 1
                                street_mem["escape_left"] = 14
                                street_mem["escape_side"] = float(
                                    street_mem.get("open_side") or 1.0
                                )
                                street_mem["escape_d0"] = float(d_to_goal)
                                street_mem["escape_used"] = 1.0
                                _phase = 1
                            if _phase == 1 or _phase == 2:
                                _lims = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                                _yside = float(street_mem.get("escape_side") or 1.0)
                                _d0 = float(street_mem.get("escape_d0") or d_to_goal)
                                _left = int(street_mem.get("escape_left") or 0)
                                _best = float(street_mem.get("escape_best") or _d0)
                                if float(d_to_goal) < _best:
                                    _best = float(d_to_goal)
                                    street_mem["escape_best"] = _best
                                _flew_away = float(d_to_goal) > min(_d0 + 30.0, _best + 8.0)
                                if _flew_away:
                                    street_mem["escape_phase"] = 0
                                    street_mem["escape_left"] = 0
                                    _face = math.copysign(
                                        min(0.12, float(_lims[3])),
                                        _yaw_err if abs(_yaw_err) > 0.2 else 1.0,
                                    )
                                    action = np.array([0.0, 0.0, 0.0, _face], dtype=np.float64)
                                    _sname = "escape_back"
                                elif _phase == 1 and _sf >= 12.0:
                                    street_mem["escape_phase"] = 2
                                    street_mem["escape_left"] = 80
                                    action = np.array(
                                        [min(0.45, float(_lims[0])), 0.0, 0.0, 0.0],
                                        dtype=np.float64,
                                    )
                                    _sname = "escape_go"
                                elif _phase == 1 and _left > 0:
                                    action = np.array(
                                        [0.0, 0.0, 0.0, _yside * min(0.12, float(_lims[3]))],
                                        dtype=np.float64,
                                    )
                                    _sname = "escape_yaw"
                                    street_mem["escape_left"] = _left - 1
                                elif _phase == 1:
                                    street_mem["escape_phase"] = 2
                                    street_mem["escape_left"] = 80
                                    action = np.zeros(4, dtype=np.float64)
                                    _sname = "escape_yaw"
                                else:
                                    _clear = abs(_yaw_err) < 0.45 and _sf >= 10.0
                                    if _clear or _left <= 0 or _sf < 6.0:
                                        street_mem["escape_phase"] = 0
                                        street_mem["escape_left"] = 0
                                        if _sf < 6.0 and not _clear:
                                            action = np.zeros(4, dtype=np.float64)
                                            _sname = "hold"
                                            # First yaw never opened a lane. Try the
                                            # other direction once before sitting.
                                            if (
                                                float(d_to_goal) > 20.0
                                                and not street_mem.get("escape_flipped")
                                            ):
                                                street_mem["escape_flipped"] = 1.0
                                                street_mem["escape_used"] = 0.0
                                                street_mem["escape_side"] = -float(
                                                    street_mem.get("escape_side") or 1.0
                                                )
                                                street_mem["open_side"] = street_mem["escape_side"]
                                    else:
                                        _longer = _sl if _sl >= _sr else _sr
                                        if _sf < 12.0 and _longer > _sf + 1.0:
                                            _turn = 1.0 if _sl >= _sr else -1.0
                                            action = np.array(
                                                [0.0, 0.0, 0.0, _turn * min(0.10, float(_lims[3]))],
                                                dtype=np.float64,
                                            )
                                            _sname = "escape_yaw"
                                        else:
                                            action = np.array(
                                                [min(0.45, float(_lims[0])), 0.0, 0.0, 0.0],
                                                dtype=np.float64,
                                            )
                                            _sname = "escape_go"
                                        street_mem["escape_left"] = _left - 1
                            # Last few metres: a zero command sinks, and the 3D
                            # distance grows even though the horizontal gap is small.
                            if (
                                int(street_mem.get("escape_phase") or 0) == 0
                                and 8.0 < float(d_to_goal) < 30.0
                                and _sf < 8.0
                            ):
                                _lims = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                                _dz = 0.0
                                if float(p_curr[2]) < float(goal_pos[2]) - 0.4:
                                    _dz = min(0.15, float(_lims[2]))
                                _slide = 0.0
                                if _sl >= _sf + 0.8 and _sl >= _sr and _sl >= 6.0:
                                    _slide = 1.0
                                elif _sr >= _sf + 0.8 and _sr > _sl and _sr >= 6.0:
                                    _slide = -1.0
                                _close_left = int(street_mem.get("close_left") or 0)
                                _rev_left = int(street_mem.get("rev_left") or 0)
                                _walled = max(_sf, _sl, _sr) < 6.0
                                _overhead_closed = (
                                    _su is not None and _su < 8.0 and _sf < 10.0 and float(d_to_goal) > 12.0
                                )
                                if _overhead_closed and (
                                    _rev_left > 0 or not street_mem.get("rev_used")
                                ):
                                    if _rev_left <= 0:
                                        street_mem["rev_left"] = 24
                                        street_mem["rev_used"] = 1.0
                                        street_mem["rev_side"] = -float(
                                            street_mem.get("escape_side") or 1.0
                                        )
                                        _rev_left = 24
                                    _turn = float(street_mem.get("rev_side") or 1.0)
                                    action = np.array(
                                        [0.0, 0.0, 0.0, _turn * min(0.12, float(_lims[3]))],
                                        dtype=np.float64,
                                    )
                                    _sname = "reverse_yaw"
                                    street_mem["rev_left"] = _rev_left - 1
                                elif _slide != 0.0 and _close_left <= 0:
                                    action = np.array(
                                        [
                                            min(0.12, float(_lims[0])),
                                            _slide * min(0.25, float(_lims[1])),
                                            _dz,
                                            0.0,
                                        ],
                                        dtype=np.float64,
                                    )
                                    _sname = "finish_slide"
                                elif (
                                    _walled
                                    and _su is not None
                                    and _su >= 10.0
                                    and float(p_curr[2]) < 40.0
                                    and int(street_mem.get("climb_left") or 0) > 0
                                    or (
                                        _walled
                                        and _su is not None
                                        and _su >= 10.0
                                        and float(p_curr[2]) < 40.0
                                        and not street_mem.get("climb_used")
                                    )
                                ):
                                    if int(street_mem.get("climb_left") or 0) <= 0:
                                        street_mem["climb_left"] = 20
                                        street_mem["climb_used"] = 1.0
                                    action = np.array(
                                        [0.0, 0.0, min(0.25, float(_lims[2])), 0.0],
                                        dtype=np.float64,
                                    )
                                    _sname = "climb_out"
                                    street_mem["climb_left"] = int(street_mem["climb_left"]) - 1
                                elif _walled and (
                                    _close_left > 0 or not street_mem.get("close_used")
                                ):
                                    if _close_left <= 0:
                                        street_mem["close_left"] = 18
                                        street_mem["close_side"] = 1.0 if _sl >= _sr else -1.0
                                        street_mem["close_used"] = 1.0
                                        _close_left = 18
                                    _turn = float(street_mem.get("close_side") or 1.0)
                                    action = np.array(
                                        [0.0, 0.0, _dz, _turn * min(0.12, float(_lims[3]))],
                                        dtype=np.float64,
                                    )
                                    _sname = "close_yaw"
                                    street_mem["close_left"] = _close_left - 1
                                elif _sf >= 6.0 and min(_sl, _sr) >= 6.0 and abs(_yaw_err) < 0.5:
                                    action = np.array(
                                        [min(0.22, float(_lims[0])), 0.0, _dz, 0.0],
                                        dtype=np.float64,
                                    )
                                    _sname = "finish_creep"
                                elif _dz > 0.0:
                                    action = np.array([0.0, 0.0, _dz, 0.0], dtype=np.float64)
                                    _sname = "finish_z"
                        oa_step["street"] = _sname
                        if step < 5 or step % 20 == 0:
                            logger.info(
                                "STREET step=%d choice=%s cost=%.2f d_fwd=%s L=%s R=%s U=%s yaw_err=%.2f z=%.1f d_to_g=%.1f",
                                step,
                                _sname,
                                float(_scost) if np.isfinite(_scost) else -1.0,
                                None if _cones_st.get("forward") is None else round(float(_cones_st["forward"]), 1),
                                None if _cones_st.get("left") is None else round(float(_cones_st["left"]), 1),
                                None if _cones_st.get("right") is None else round(float(_cones_st["right"]), 1),
                                None if _cones_st.get("up") is None else round(float(_cones_st["up"]), 1),
                                _yaw_err,
                                float(p_curr[2]),
                                float(d_to_goal),
                            )
                    except Exception:
                        logger.exception("STREET select failed")
                        _sa, _sname = select_street_action(
                            obs.info.get("depth_cones_pred") or {},
                            float(obs.info.get("yaw_err_rad") or 0.0),
                            cur_limits,
                        )
                        action = np.asarray(_sa, dtype=np.float64).reshape(4)
                        oa_step["street"] = _sname

                # One 15-step open-side peel, then back to the policy. No training.
                if os.environ.get("SCRIPT_PEEL") == "1":
                    _c_sp = obs.info.get("depth_cones_pred") or {}
                    _df_sp = _c_sp.get("forward")
                    _dl_sp, _dr_sp = _c_sp.get("left"), _c_sp.get("right")
                    if (
                        script_peel_left <= 0
                        and not script_peel_used
                        and _df_sp is not None
                        and _dl_sp is not None
                        and _dr_sp is not None
                        and float(_df_sp) < 8.0
                        and max(float(_dl_sp), float(_dr_sp)) - float(_df_sp) >= 2.0
                    ):
                        script_peel_left = 15
                        script_peel_sign = 1.0 if float(_dl_sp) >= float(_dr_sp) else -1.0
                        script_peel_used = True
                        logger.info(
                            "SCRIPT_PEEL start step=%d d_to_g=%.2f d_fwd=%.2f sign=%+.0f",
                            step,
                            float(d_to_goal),
                            float(_df_sp),
                            script_peel_sign,
                        )
                    if script_peel_left > 0:
                        _lim_sp = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                        action = np.clip(
                            np.array(
                                [0.2, script_peel_sign * 0.4, 0.0, 0.0],
                                dtype=np.float64,
                            ),
                            -_lim_sp,
                            _lim_sp,
                        )
                        action[0] = max(0.0, float(action[0]))
                        script_peel_left -= 1
                        oa_step["script_peel"] = True
                        oa_step["script_peel_last"] = script_peel_left == 0

                # Yaw toward the open side with no forward speed. Go forward only
                # after the forward cone has opened. One burst per episode.
                if os.environ.get("SCRIPT_YAWGO") == "1":
                    _c_y = obs.info.get("depth_cones_pred") or {}
                    _df_y = _c_y.get("forward")
                    _dl_y, _dr_y = _c_y.get("left"), _c_y.get("right")
                    if (
                        script_yaw_phase is None
                        and _df_y is not None
                        and _dl_y is not None
                        and _dr_y is not None
                        and float(_df_y) < 8.0
                        and max(float(_dl_y), float(_dr_y)) - float(_df_y) >= 2.0
                    ):
                        script_yaw_phase = "yaw"
                        script_yaw_sign = 1.0 if float(_dl_y) >= float(_dr_y) else -1.0
                        logger.info(
                            "SCRIPT_YAW start step=%d d_to_g=%.2f d_fwd=%.2f sign=%+.0f yaw=%.1f",
                            step,
                            float(d_to_goal),
                            float(_df_y),
                            script_yaw_sign,
                            float(np.degrees(curr_yaw)),
                        )
                    if script_yaw_phase == "yaw":
                        _opened = _df_y is not None and float(_df_y) >= 8.0
                        if _opened or script_yaw_n >= 10:
                            if _opened:
                                script_yaw_phase = "fwd"
                                logger.info(
                                    "SCRIPT_YAW opened step=%d d_fwd=%.2f yaw_steps=%d yaw=%.1f",
                                    step,
                                    float(_df_y),
                                    script_yaw_n,
                                    float(np.degrees(curr_yaw)),
                                )
                            else:
                                script_yaw_phase = "done"
                                logger.info(
                                    "SCRIPT_YAW no-open step=%d d_fwd=%.2f yaw_steps=%d",
                                    step,
                                    float(_df_y) if _df_y is not None else -1.0,
                                    script_yaw_n,
                                )
                        else:
                            _lim_y = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                            action = np.clip(
                                np.array(
                                    [0.0, 0.0, 0.0, script_yaw_sign * float(_lim_y[3])],
                                    dtype=np.float64,
                                ),
                                -_lim_y,
                                _lim_y,
                            )
                            script_yaw_n += 1
                            oa_step["script_yaw"] = True
                    if script_yaw_phase == "fwd":
                        if script_fwd_n >= 5:
                            script_yaw_phase = "done"
                        else:
                            _lim_y = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                            action = np.clip(
                                np.array([0.7, 0.0, 0.0, 0.0], dtype=np.float64),
                                -_lim_y,
                                _lim_y,
                            )
                            script_fwd_n += 1
                            oa_step["script_fwd"] = True
                            oa_step["script_fwd_last"] = script_fwd_n == 5
                            if script_fwd_n == 5:
                                script_yaw_phase = "done"

                # One open-loop burst at the first wall contact, then the policy.
                # climb: straight up. long: 40-step open-side peel. pull: strafe
                # clear, then yaw toward the opening, then a short forward.
                if script_open_mode in ("climb", "long", "pull"):
                    _c_o = obs.info.get("depth_cones_pred") or {}
                    _df_o = _c_o.get("forward")
                    _dl_o, _dr_o = _c_o.get("left"), _c_o.get("right")
                    _wall = (
                        _df_o is not None
                        and _dl_o is not None
                        and _dr_o is not None
                        and float(_df_o) < 8.0
                        and max(float(_dl_o), float(_dr_o)) - float(_df_o) >= 2.0
                    )
                    if script_open_phase is None and not script_open_used and _wall:
                        script_open_used = True
                        script_open_sign = 1.0 if float(_dl_o) >= float(_dr_o) else -1.0
                        if script_open_mode == "climb":
                            script_open_phase = "climb"
                            script_open_left = 25
                        elif script_open_mode == "long":
                            script_open_phase = "long"
                            script_open_left = 40
                        else:
                            script_open_phase = "pull"
                            script_open_left = 20
                        logger.info(
                            "SCRIPT_OPEN %s start step=%d d_to_g=%.2f d_fwd=%.2f sign=%+.0f",
                            script_open_mode,
                            step,
                            float(d_to_goal),
                            float(_df_o),
                            script_open_sign,
                        )
                    if script_open_phase in ("climb", "long", "pull", "yaw", "fwd"):
                        _lim_o = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                        if script_open_phase == "climb" and script_open_left > 0:
                            action = np.array([0.0, 0.0, float(_lim_o[2]), 0.0])
                            script_open_left -= 1
                            oa_step["script_open"] = "climb"
                            if script_open_left == 0:
                                script_open_phase = "done"
                        elif script_open_phase == "long" and script_open_left > 0:
                            action = np.array(
                                [0.2, script_open_sign * 0.4, 0.0, 0.0], dtype=np.float64
                            )
                            script_open_left -= 1
                            oa_step["script_open"] = "long"
                            if script_open_left == 0:
                                script_open_phase = "done"
                        elif script_open_phase == "pull" and script_open_left > 0:
                            action = np.array(
                                [0.0, script_open_sign * 0.4, 0.0, 0.0], dtype=np.float64
                            )
                            script_open_left -= 1
                            oa_step["script_open"] = "pull"
                            if script_open_left == 0:
                                script_open_phase = "yaw"
                        elif script_open_phase == "yaw":
                            _opened = _df_o is not None and float(_df_o) >= 8.0
                            if _opened or script_open_left >= 8:
                                script_open_phase = "fwd"
                                script_open_left = 10
                                logger.info(
                                    "SCRIPT_OPEN pull yaw done step=%d d_fwd=%.2f opened=%s",
                                    step,
                                    float(_df_o) if _df_o is not None else -1.0,
                                    _opened,
                                )
                            else:
                                action = np.array(
                                    [0.0, 0.0, 0.0, script_open_sign * float(_lim_o[3])],
                                    dtype=np.float64,
                                )
                                script_open_left += 1
                                oa_step["script_open"] = "pull_yaw"
                        if script_open_phase == "fwd" and script_open_left > 0:
                            action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float64)
                            script_open_left -= 1
                            oa_step["script_open"] = "pull_fwd"
                            if script_open_left == 0:
                                script_open_phase = "done"
                        if script_open_phase == "done" and oa_step.get("script_open"):
                            action = np.clip(action, -_lim_o, _lim_o)
                            action[0] = max(0.0, float(action[0]))
                        elif script_open_phase not in (None, "done"):
                            action = np.clip(
                                np.asarray(action, dtype=np.float64).reshape(4),
                                -_lim_o,
                                _lim_o,
                            )
                            action[0] = max(0.0, float(action[0]))

                # One latched yaw, then a short forward. Records a real turn.
                if yaw_collect:
                    _c_yc = obs.info.get("depth_cones_pred") or {}
                    _df_yc = _c_yc.get("forward")
                    _dl_yc, _dr_yc = _c_yc.get("left"), _c_yc.get("right")
                    _early = (
                        _df_yc is not None
                        and _dl_yc is not None
                        and _dr_yc is not None
                        and 8.0 < float(_df_yc) <= 18.0
                        and max(float(_dl_yc), float(_dr_yc)) >= float(_df_yc)
                        and max(float(_dl_yc), float(_dr_yc)) - min(float(_dl_yc), float(_dr_yc)) >= 2.0
                    )
                    if yaw_collect_phase is None and not yaw_collect_used and _early:
                        yaw_collect_used = True
                        yaw_collect_phase = "yaw"
                        yaw_collect_left = 10
                        yaw_collect_sign = 1.0 if float(_dl_yc) >= float(_dr_yc) else -1.0
                        logger.info(
                            "YAW_COLLECT start step=%d d_fwd=%.2f sign=%+.0f",
                            step,
                            float(_df_yc),
                            yaw_collect_sign,
                        )
                    if yaw_collect_phase == "yaw" and yaw_collect_left > 0:
                        _lim_yc = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                        action = np.array(
                            [0.0, 0.0, 0.0, yaw_collect_sign * float(_lim_yc[3])],
                            dtype=np.float64,
                        )
                        yaw_collect_left -= 1
                        if yaw_collect_left == 0:
                            yaw_collect_phase = "fwd"
                            yaw_collect_left = 8
                    elif yaw_collect_phase == "fwd" and yaw_collect_left > 0:
                        action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float64)
                        yaw_collect_left -= 1
                        if yaw_collect_left == 0:
                            yaw_collect_phase = "done"
                            logger.info("YAW_COLLECT end step=%d", step)

                # Frontal obstacle only: 10th-percentile forward, neither side
                # closer than that. Press while the range keeps falling, then
                # yaw only if the run is a stable approach.
                if early_press:
                    from experiments.aerial.rl.depth_geometry import (
                        approach_clearances,
                        frontal_obstacle_frame,
                        stable_approach,
                    )

                    _depth_ep = getattr(obs, "depth", None)
                    _df_ep = _dl_ep = _dr_ep = None
                    _ok_ep = False
                    if _depth_ep is not None:
                        try:
                            _df_ep, _dl_ep, _dr_ep = approach_clearances(_depth_ep)
                            _ok_ep = frontal_obstacle_frame(_df_ep, _dl_ep, _dr_ep)
                        except ValueError:
                            _ok_ep = False
                    if early_press_phase is None and _ok_ep:
                        early_press_phase = "press"
                        early_press_left = 8
                        early_press_hist = []
                        early_press_sign = 1.0 if float(_dl_ep) >= float(_dr_ep) else -1.0
                        logger.info(
                            "EARLY_PRESS start step=%d d_fwd=%.2f L=%.1f R=%.1f sign=%+.0f",
                            step,
                            float(_df_ep),
                            float(_dl_ep),
                            float(_dr_ep),
                            early_press_sign,
                        )
                    if early_press_phase == "press":
                        _prev_ep = early_press_hist[-1] if early_press_hist else None
                        _jump_ep = (
                            _prev_ep is not None
                            and _df_ep is not None
                            and abs(float(_df_ep) - float(_prev_ep)) > 2.0
                        )
                        _take_ep = bool(
                            _ok_ep and not _jump_ep and early_press_left > 0 and _df_ep is not None
                        )
                        if _take_ep:
                            action = np.array([0.7, 0.0, 0.0, 0.0], dtype=np.float64)
                            early_press_hist.append(float(_df_ep))
                            early_press_left -= 1
                        if (not _take_ep) or early_press_left == 0:
                            _drop = float(early_press_hist[0] - early_press_hist[-1]) if early_press_hist else 0.0
                            if stable_approach(early_press_hist):
                                early_press_phase = "yaw"
                                early_press_left = 8
                                logger.info(
                                    "EARLY_PRESS keep step=%d n=%d drop=%.2f",
                                    step,
                                    len(early_press_hist),
                                    _drop,
                                )
                            else:
                                early_press_phase = None
                                logger.info(
                                    "EARLY_PRESS drop step=%d n=%d drop=%.2f",
                                    step,
                                    len(early_press_hist),
                                    _drop,
                                )
                                early_press_hist = []
                    elif early_press_phase == "yaw" and early_press_left > 0:
                        _lim_ep = np.asarray(cur_limits, dtype=np.float64).reshape(4)
                        action = np.array(
                            [0.0, 0.0, 0.0, early_press_sign * float(_lim_ep[3])],
                            dtype=np.float64,
                        )
                        early_press_left -= 1
                        if early_press_left == 0:
                            early_press_phase = None
                            early_press_hist = []
                            logger.info("EARLY_PRESS end step=%d", step)

                # Keep the flight at the spawn height. Below it, climb back;
                # above it, descend back. Inside the band, do not accumulate climb.
                _z_now = float(np.asarray(obs.state, dtype=np.float64).reshape(-1)[2])
                if _z_now < z_hold - 0.5:
                    action[2] = max(float(action[2]), 0.25)
                elif _z_now > z_hold + 0.5:
                    action[2] = min(float(action[2]), -0.25)
                else:
                    action[2] = 0.0
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
                if oa_step.get("script_peel_last"):
                    logger.info(
                        "SCRIPT_PEEL end step=%d d_to_g=%.2f collided=%s",
                        step,
                        float(np.linalg.norm(goal_pos - p_curr)),
                        bool(getattr(obs, "collided", False)),
                    )
                if oa_step.get("script_fwd_last"):
                    logger.info(
                        "SCRIPT_YAW end step=%d d_to_g=%.2f collided=%s",
                        step,
                        float(np.linalg.norm(goal_pos - p_curr)),
                        bool(getattr(obs, "collided", False)),
                    )

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
                        "oc_press": oa_step.get("oc_press") if oa_step else None,
                        "oc_peel": oa_step.get("oc_peel") if oa_step else None,
                        "oc_back": oa_step.get("oc_back") if oa_step else None,
                        "oc_exec": oa_step.get("oc_exec") if oa_step else None,
                        "oc_peel_beats_press": (
                            oa_step.get("oc_peel_beats_press") if oa_step else None
                        ),
                        "oc_peel_beats_back": (
                            oa_step.get("oc_peel_beats_back") if oa_step else None
                        ),
                        "act_dx": oa_step.get("act_dx") if oa_step else None,
                        "act_dy": oa_step.get("act_dy") if oa_step else None,
                        "g0_strafe": oa_step.get("g0_strafe") if oa_step else None,
                        "g0_peel": oa_step.get("g0_peel") if oa_step else None,
                        "g0_yawgo": oa_step.get("g0_yawgo") if oa_step else None,
                        "g0_climb": oa_step.get("g0_climb") if oa_step else None,
                        "g0_climb_wins": oa_step.get("g0_climb_wins") if oa_step else None,
                        "g0_best": oa_step.get("g0_best") if oa_step else None,
                        "g0_leave_wins": oa_step.get("g0_leave_wins") if oa_step else None,
                        "script_peel": oa_step.get("script_peel") if oa_step else None,
                        "script_yaw": oa_step.get("script_yaw") if oa_step else None,
                        "script_fwd": oa_step.get("script_fwd") if oa_step else None,
                        "oracle": oa_step.get("oracle") if oa_step else None,
                        "street": oa_step.get("street") if oa_step else None,
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
                "veto_counts": dict(veto_counts),
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
                " | veto "
                + " ".join(f"{k}={v}" for k, v in (ep_result.get("veto_counts") or {}).items())
                if ep_result.get("veto_counts")
                else ""
            )
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
        "encode_current": bool(args.encode_current),
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
