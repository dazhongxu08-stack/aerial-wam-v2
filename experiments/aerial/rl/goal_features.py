"""Goal-relative features for the V1-② reward head.

``NavigationReward`` progress is Δdist(goal). r60 episode npz historically omitted
``goal``, so open-loop reward fidelity cannot see the quantity the label depends
on. This module:

  * builds body-frame ``goal_rel = (fwd, left, up, remaining_dist)`` from pose+goal
  * resolves a per-episode goal from stored npz / transition info, or (legacy)
    an arrived end-proprio proxy / Gauss–Newton fit to the reward channel

Goal is **reward-head conditioning only** (like action) — not an encoder / policy
input — so the RGB+proprio4 §1.2 boundary stays intact.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from experiments.aerial.rl.buffer import Transition

GOAL_REL_DIM = 4  # body-frame fwd, left, up, remaining_dist_m
GOAL_NORM_DIM = 4  # unit body dir (3) + log1p(dist) — actor/critic conditioning (F9)
BODY_VEL_DIM = 3  # body-frame linear velocity (fwd, left, up)
# Reward-head aux (not encoder input): unit goal dir + log1p(dist) + body vel +
# analytic Δdist from vel·dt (action alone ≠ realized displacement on r60 starts).
REWARD_AUX_DIM = 8
DEFAULT_REWARD_DT = 0.2  # 1/step_hz for r60 (step_hz=5)
DEFAULT_MANEUVER_W = 0.01


def goal_rel_body(
    pos: np.ndarray,
    yaw: float,
    goal: np.ndarray,
) -> np.ndarray:
    """World goal → body-frame delta + remaining distance ``[fwd, left, up, dist]``."""
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    delta = goal - pos
    dist = float(np.linalg.norm(delta))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    fwd = c * delta[0] + s * delta[1]
    left = -s * delta[0] + c * delta[1]
    up = float(delta[2])
    return np.array([fwd, left, up, dist], dtype=np.float32)


def g_norm_from_goal_rel(goal_rel: np.ndarray) -> np.ndarray:
    """Scale-stable goal features ``[û_xyz, log1p(d)]`` aligned with ``reward_aux``."""
    g = np.asarray(goal_rel, dtype=np.float64).reshape(GOAL_REL_DIM)
    dist = max(float(g[3]), 1e-6)
    unit = (g[:3] / dist).astype(np.float32)
    logd = np.float32(np.log1p(dist))
    return np.concatenate([unit, np.array([logd], dtype=np.float32)], axis=0)


def goal_rel_from_obs(obs: Any) -> np.ndarray:
    """``goal_rel`` for one observation; zeros when goal is missing."""
    goal = None
    info = getattr(obs, "info", None)
    if isinstance(info, dict):
        goal = info.get("goal")
    if goal is None:
        return np.zeros(GOAL_REL_DIM, dtype=np.float32)
    return goal_rel_body(obs.position, float(obs.yaw), goal)


def advance_goal_rel_body(
    goal_rel: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    """Update body-frame ``goal_rel`` after one body-delta action (imagination aux).

    Order matches one control step: subtract body displacement in the frame at
    the start of the step, then rotate the remaining vector by ``-dyaw`` so the
    goal stays in the body frame after the yaw change. Remaining distance is
    ``||g_body||``.
    """
    g = np.asarray(goal_rel, dtype=np.float64).reshape(GOAL_REL_DIM).copy()
    a = np.asarray(action, dtype=np.float64).reshape(4)
    disp = a[:3]
    dyaw = float(a[3]) if a.size > 3 else 0.0
    g[:3] = g[:3] - disp
    if abs(dyaw) > 1e-12:
        c = float(np.cos(dyaw))
        s = float(np.sin(dyaw))
        # Body yawed +dyaw (CCW) ⇒ fixed world vector rotates by -dyaw in body.
        fwd, left = float(g[0]), float(g[1])
        g[0] = c * fwd + s * left
        g[1] = -s * fwd + c * left
    g[3] = float(np.linalg.norm(g[:3]))
    return g.astype(np.float32, copy=False)


def body_vel_from_obs(obs: Any) -> np.ndarray:
    """World-frame ``obs.velocity`` → body-frame ``[fwd, left, up]`` (m/s)."""
    v = np.asarray(getattr(obs, "velocity", np.zeros(3)), dtype=np.float64).reshape(3)
    yaw = float(getattr(obs, "yaw", 0.0))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    fwd = c * v[0] + s * v[1]
    left = -s * v[0] + c * v[1]
    up = float(v[2])
    return np.array([fwd, left, up], dtype=np.float32)


def analytic_progress(
    goal_rel: np.ndarray,
    body_disp: np.ndarray,
    action: Optional[np.ndarray] = None,
    *,
    w_maneuver: float = DEFAULT_MANEUVER_W,
) -> float:
    """``||g|| − ||g − body_disp|| − w‖a‖`` — NavigationReward progress proxy."""
    g = np.asarray(goal_rel, dtype=np.float64).reshape(GOAL_REL_DIM)
    disp = np.asarray(body_disp, dtype=np.float64).reshape(3)
    dist = float(g[3])
    prog = dist - float(np.linalg.norm(g[:3] - disp))
    if action is None:
        return float(prog)
    man = float(np.linalg.norm(np.asarray(action, dtype=np.float64).reshape(-1)))
    return float(prog - float(w_maneuver) * man)


def reward_aux_features(
    goal_rel: np.ndarray,
    body_vel: np.ndarray,
    action: np.ndarray,
    *,
    dt: float = DEFAULT_REWARD_DT,
    w_maneuver: float = DEFAULT_MANEUVER_W,
) -> np.ndarray:
    """Scale-stable reward-head conditioning ``[û, log1p(d), v_body, analytic]``.

    r60 open-loop reward fails when the head sees raw metre-scale ``goal_rel`` and
    near-constant actions: early-horizon commanded ``a`` ≫ realized displacement
    (accel from rest). Body velocity × ``dt`` recovers that displacement; the
    analytic channel alone beats the constant-mean baseline on honest held-out
    windows (oracle beat_frac=1.0).
    """
    g = np.asarray(goal_rel, dtype=np.float64).reshape(GOAL_REL_DIM)
    v = np.asarray(body_vel, dtype=np.float64).reshape(BODY_VEL_DIM)
    a = np.asarray(action, dtype=np.float64).reshape(4)
    dist = max(float(g[3]), 1e-6)
    unit = (g[:3] / dist).astype(np.float32)
    logd = np.float32(np.log1p(dist))
    analytic = np.float32(
        analytic_progress(g, v * float(dt), a, w_maneuver=float(w_maneuver))
    )
    return np.concatenate(
        [unit, np.array([logd], dtype=np.float32), v.astype(np.float32),
         np.array([analytic], dtype=np.float32)],
        axis=0,
    ).astype(np.float32, copy=False)


def _goal_from_info(transitions: Sequence[Transition]) -> Optional[np.ndarray]:
    for tr in transitions:
        for bag in (tr.info, getattr(tr.obs, "info", {}) or {}):
            if not isinstance(bag, dict):
                continue
            g = bag.get("goal")
            if g is not None:
                arr = np.asarray(g, dtype=np.float64).reshape(-1)
                if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                    return arr[:3].astype(np.float64)
    return None


def fit_goal_from_progress(
    pos: np.ndarray,
    prog: np.ndarray,
    *,
    n_iter: int = 30,
    damp: float = 0.5,
) -> np.ndarray:
    """Gauss–Newton fit of a fixed goal to observed progress ``Δdist`` labels.

    ``prog[t] ≈ ||pos[t]−g|| − ||pos[t+1]−g||`` for ``t < N−1``. Used as a legacy
    npz recovery path when the collector did not persist ``goal``.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    prog = np.asarray(prog, dtype=np.float64).reshape(-1)
    if pos.shape[0] < 2:
        return pos[-1].copy() if pos.shape[0] else np.zeros(3, dtype=np.float64)
    g = pos[-1].copy()
    for _ in range(int(n_iter)):
        d = np.linalg.norm(pos - g[None, :], axis=1) + 1e-6
        pred = d[:-1] - d[1:]
        err = pred - prog[:-1]
        u = (pos - g[None, :]) / d[:, None]
        jac = -u[:-1] + u[1:]
        dg, *_ = np.linalg.lstsq(jac, err, rcond=None)
        g = g + float(damp) * dg
    return g.astype(np.float64)


def end_proprio_goal_proxy(transitions: Sequence[Transition]) -> Optional[np.ndarray]:
    """Last pre-step position when the episode terminated without collision.

    Arrived episodes end within ``success_dist_m`` of the true goal; the stored
    terminal proprio is one step early but still a usable proxy for r60.
    """
    if not transitions:
        return None
    last = transitions[-1]
    post = last.next_obs if last.next_obs is not None else last.obs
    collided = bool(getattr(post, "collided", False) or last.obs.collided)
    if not (last.done and not collided):
        return None
    return np.asarray(last.obs.position, dtype=np.float64).reshape(3)


def resolve_episode_goal(
    transitions: Sequence[Transition],
    *,
    allow_fit: bool = False,
    allow_end_proxy: bool = True,
    maneuver_w: float = 0.01,
) -> Optional[np.ndarray]:
    """Best-effort episode goal: info → optional fit → optional end-proprio proxy.

    Default ``allow_fit=False``: Gauss–Newton recovery from the reward channel is
    **not** reliable on r60 (reconstructed progress MAE ≫ mean baseline), so we
    refuse to inject a misleading goal into ``goal_rel`` unless the caller opts in.
    """
    stored = _goal_from_info(transitions)
    if stored is not None:
        return stored
    if not transitions:
        return None
    if allow_fit and len(transitions) >= 3:
        pos = np.stack([t.obs.position for t in transitions], axis=0)
        acts = np.stack([np.asarray(t.action, dtype=np.float64).reshape(4)
                         for t in transitions], axis=0)
        rew = np.asarray([t.reward for t in transitions], dtype=np.float64)
        prog = rew + float(maneuver_w) * np.linalg.norm(acts, axis=1)
        return fit_goal_from_progress(pos, prog)
    if allow_end_proxy:
        return end_proprio_goal_proxy(transitions)
    return None


def attach_goal(transitions: List[Transition], goal: Optional[np.ndarray]) -> None:
    """Stamp ``goal`` onto every transition / obs ``info`` (in-place)."""
    if goal is None:
        return
    g = np.asarray(goal, dtype=np.float32).reshape(3)
    for tr in transitions:
        tr.info["goal"] = g.copy()
        if tr.obs.info is None:
            tr.obs.info = {}
        tr.obs.info["goal"] = g.copy()
        if tr.next_obs is not None:
            if tr.next_obs.info is None:
                tr.next_obs.info = {}
            tr.next_obs.info["goal"] = g.copy()


def attach_goals_per_step(transitions: List[Transition], goals: np.ndarray) -> None:
    """Stamp per-frame ``goals[i]`` onto each transition (polyline carrot FT)."""
    g = np.asarray(goals, dtype=np.float32).reshape(-1, 3)
    if len(g) != len(transitions):
        raise ValueError(f"goals rows {len(g)} != transitions {len(transitions)}")
    for i, tr in enumerate(transitions):
        gi = g[i].copy()
        tr.info["goal"] = gi
        if tr.obs.info is None:
            tr.obs.info = {}
        tr.obs.info["goal"] = gi.copy()
        if tr.next_obs is not None:
            if tr.next_obs.info is None:
                tr.next_obs.info = {}
            tr.next_obs.info["goal"] = gi.copy()
