"""Batched imagined rollout over a ``LatentDynamics`` (Plan-A parallelism).

This is where sample *volume* comes from: instead of many real envs (the
renderer is single-consumer), we roll a batch of latent states forward through
the fast world model. ``imagine`` takes a batch of encoded start states ``z0``
and an imagination policy, rolls ``horizon`` steps, and returns per-step
latents / actions / rewards / p_coll / done masks for the RL update (V4).

The horizon is capped (spec §9): multi-step WM error compounds, so until the WM
is shown non-divergent the rollout length stays bounded. ``done`` masking stops
reward accrual after a trajectory terminates.

Pure numpy; works with ``StubLatentDynamics`` for offline tests. Imagination
policies expose ``act_latent(z) -> [4]`` (fallback: ``act(z)``).

ACTION SPACE (2026-08-18): pass ``action_limits`` to hold the imagined action
set equal to the deployed one (``body_delta_limits(1/step_hz)``, clipped in
``collector.py:167``). ``ImaginedRollout.n_action_clipped`` counts how often the
clip actually bit — with the C2 bounded policy it must stay **0**, which is how
the "clip is a no-op" claim gets measured instead of assumed. Note this clip
alone is NOT the consistency fix: the actor update is REINFORCE, so a policy
that can propose out-of-box actions needs its *sampling law* bounded (see
``actor_critic`` C2 / proposal §4.1), not its samples truncated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from experiments.aerial.rl.dynamics import LatentDynamics
from experiments.aerial.rl.goal_features import (
    BODY_VEL_DIM,
    GOAL_REL_DIM,
    advance_goal_rel_body,
    analytic_progress,
)
from experiments.aerial.rl.reward import (
    RewardConfig,
    clearance_m_along_action,
    clearance_risk_from_depth,
    efficiency_cost,
    path_shaping_terms,
    reward_terms,
)

MAX_IMAGINATION_HORIZON = 15  # §9 safety cap until WM error shown non-divergent


@dataclass
class ImaginedRollout:
    z: np.ndarray            # [B, H+1, latent_dim]
    actions: np.ndarray      # [B, H, 4]
    rewards: np.ndarray      # [B, H]
    p_coll: np.ndarray       # [B, H]
    progress: np.ndarray     # [B, H]
    done: np.ndarray         # [B, H] bool (cumulative)
    #: How many (b, t, axis) entries the deployed-action clip actually changed.
    #: 0 under the C2 bounded policy; > 0 means imagined != deployed action set.
    n_action_clipped: int = 0
    #: Body-frame goal relative features at each latent before the action
    #: ``[B, H+1, GOAL_REL_DIM]``. ``None`` when imagination ran without goals.
    goal_rel: Optional[np.ndarray] = None

    @property
    def returns(self) -> np.ndarray:
        """Per-trajectory summed reward (done steps already zero-masked)."""
        return self.rewards.sum(axis=1)


def _act_latent(
    policy: Any, z: np.ndarray, goal_rel: Optional[np.ndarray] = None
) -> np.ndarray:
    fn = getattr(policy, "act_latent", None) or getattr(policy, "act", None)
    if not callable(fn):
        raise TypeError("imagination policy must implement act_latent(z) or act(z)")
    try:
        out = fn(z, goal_rel=goal_rel)
    except TypeError:
        out = fn(z)
    return np.asarray(out, dtype=np.float64).reshape(4)


def imagine(
    dynamics: LatentDynamics,
    policy: Any,
    z0_batch: np.ndarray,
    horizon: int,
    *,
    reward_cfg: Optional[RewardConfig] = None,
    max_horizon: int = MAX_IMAGINATION_HORIZON,
    goal_rel0: Optional[np.ndarray] = None,
    body_vel0: Optional[np.ndarray] = None,
    propagate_goal_rel: bool = True,
    action_limits: Optional[np.ndarray] = None,
) -> ImaginedRollout:
    """Roll ``z0_batch`` forward ``horizon`` steps through ``dynamics``.

    When ``goal_rel0`` / ``body_vel0`` are provided (shapes ``[B,4]`` / ``[B,3]``),
    each imagined ``step`` receives aux conditioning for the torch reward head.
    ``propagate_goal_rel`` advances ``goal_rel`` with ``advance_goal_rel_body`` after
    each action (stub-consistent kinematics); ``body_vel`` is held at the start value.

    ``action_limits`` [4] (per-axis, positive) constrains imagined actions to the
    deployed set; ``None`` leaves the historical unbounded behaviour so pre-C2
    audit runs replay bit-for-bit.
    """
    cfg = reward_cfg or RewardConfig()
    if action_limits is None:
        lim = None
    else:
        lim = np.abs(np.asarray(action_limits, dtype=np.float64).reshape(-1))
        if lim.shape != (4,) or not np.all(lim > 0):
            raise ValueError(f"action_limits must be 4 positive values, got {action_limits!r}")
    n_clipped = 0
    z0 = np.atleast_2d(np.asarray(z0_batch, dtype=np.float64))
    batch, latent_dim = z0.shape
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if horizon > max_horizon:
        raise ValueError(
            f"horizon {horizon} exceeds cap {max_horizon} (spec §9: WM multi-step "
            "error is unbounded until validated); raise max_horizon explicitly"
        )

    zs = np.zeros((batch, horizon + 1, latent_dim), dtype=np.float64)
    acts = np.zeros((batch, horizon, 4), dtype=np.float64)
    rews = np.zeros((batch, horizon), dtype=np.float64)
    pcs = np.zeros((batch, horizon), dtype=np.float64)
    progs = np.zeros((batch, horizon), dtype=np.float64)
    dones = np.zeros((batch, horizon), dtype=bool)
    zs[:, 0] = z0

    use_aux = goal_rel0 is not None or body_vel0 is not None
    if use_aux:
        if goal_rel0 is None:
            goal_rel0 = np.zeros((batch, GOAL_REL_DIM), dtype=np.float32)
        else:
            goal_rel0 = np.asarray(goal_rel0, dtype=np.float32).reshape(batch, GOAL_REL_DIM)
        if body_vel0 is None:
            body_vel0 = np.zeros((batch, BODY_VEL_DIM), dtype=np.float32)
        else:
            body_vel0 = np.asarray(body_vel0, dtype=np.float32).reshape(batch, BODY_VEL_DIM)
        goal_rel_t = goal_rel0.copy()
        body_vel_t = body_vel0.copy()
        goals_hist = np.zeros((batch, horizon + 1, GOAL_REL_DIM), dtype=np.float32)
        goals_hist[:, 0] = goal_rel_t
    else:
        goal_rel_t = body_vel_t = None
        goals_hist = None

    alive = np.ones(batch, dtype=bool)
    level_flags = [[] for _ in range(batch)]
    for t in range(horizon):
        for b in range(batch):
            if not alive[b]:
                zs[b, t + 1] = zs[b, t]
                dones[b, t] = True
                continue
            a = _act_latent(
                policy, zs[b, t],
                None if goal_rel_t is None else goal_rel_t[b],
            )
            if lim is not None:
                a_clipped = np.clip(a, -lim, lim)
                n_clipped += int(np.count_nonzero(a_clipped != a))
                a = a_clipped
            if bool(getattr(cfg, "forbid_backward_motion", False)):
                from experiments.aerial.rl.env.action import forbid_backward_dx

                a = forbid_backward_dx(a)
            step_kw: dict = {}
            if use_aux:
                step_kw["goal_rel"] = goal_rel_t[b]
                step_kw["body_vel"] = body_vel_t[b]
            out = dynamics.step(zs[b, t], a, **step_kw)
            zs[b, t + 1] = np.asarray(out.z_next, dtype=np.float64).reshape(latent_dim)
            acts[b, t] = a
            pcs[b, t] = out.p_coll
            if use_aux:
                prog = analytic_progress(goal_rel_t[b], a[:3])
            else:
                prog = out.progress
            progs[b, t] = prog
            maneuver = float(np.linalg.norm(a))
            # F15: body yaw-to-carrot + analytic progress as along-track proxy.
            yaw_err = 0.0
            g_rel = np.zeros(GOAL_REL_DIM, dtype=np.float64)
            if goal_rel_t is not None:
                g_rel = np.asarray(goal_rel_t[b], dtype=np.float64).reshape(GOAL_REL_DIM)
                gxy = g_rel[:2]
                if float(np.linalg.norm(gxy)) > 1e-6:
                    yaw_err = float(np.arctan2(gxy[1], gxy[0]))
            eff = efficiency_cost(
                a, yaw_err_rad=yaw_err, ds_true_m=float(prog), cfg=cfg,
            )
            # Obstacle cost: prefer trained directional head; else legacy clearance.
            if bool(getattr(cfg, "use_learned_obstacle_cost", False)):
                oc = getattr(out, "obstacle_cost", None)
                if oc is None or not np.isfinite(float(oc)):
                    collision_risk = 0.0
                else:
                    collision_risk = float(oc)
                if bool(getattr(cfg, "blend_clearance_risk", False)):
                    # Match real NavigationReward: only blend clearance along the
                    # dominant motion axis. Imagination has d_fwd_hat only — so
                    # non-forward actions skip blend (trust directional head).
                    d_hat = getattr(out, "d_fwd_hat", None)
                    cones = (
                        {"forward": float(d_hat)}
                        if d_hat is not None and np.isfinite(float(d_hat))
                        else {}
                    )
                    d_along = clearance_m_along_action(a, cones, d_hat)
                    if d_along is not None:
                        clear_risk = clearance_risk_from_depth(
                            d_along,
                            d_danger=float(getattr(cfg, "near_miss_d_fwd_m", 3.0)),
                            d_clear=float(getattr(cfg, "near_miss_d_clear_m", 12.0)),
                        )
                        collision_risk = float(
                            max(collision_risk, float(clear_risk))
                        )
                # Hard-contact analogue: high p_coll / done → ceiling (matches real).
                ceiling = float(getattr(cfg, "obstacle_cost_ceiling", 1.0))
                if float(getattr(out, "p_coll", 0.0) or 0.0) >= 1.0 - 1e-6:
                    collision_risk = max(collision_risk, ceiling)
                elif bool(getattr(out, "done", False)) and float(
                    getattr(out, "p_coll", 0.0) or 0.0
                ) >= 0.5:
                    collision_risk = max(collision_risk, ceiling)
            else:
                d_hat = getattr(out, "d_fwd_hat", None)
                clear_risk = clearance_risk_from_depth(d_hat)
                collision_risk = float(max(float(out.p_coll), float(clear_risk)))
            a_arr = np.asarray(a, dtype=np.float64).reshape(-1)
            level_step = (
                abs(float(a_arr[2]) if a_arr.size > 2 else 0.0) <= float(cfg.level_dz_thr_m)
                and abs(float(a_arr[3]) if a_arr.size > 3 else 0.0) <= float(cfg.level_dyaw_thr_rad)
                and float(prog) <= 0.0
            )
            level_flags[b].append(bool(level_step))
            wlen = max(1, int(cfg.level_window))
            if len(level_flags[b]) > wlen:
                level_flags[b] = level_flags[b][-wlen:]
            hist_level = len(level_flags[b]) >= wlen and all(level_flags[b])
            shaping = path_shaping_terms(
                a_arr, g_rel, float(prog), hist_level=hist_level, cfg=cfg,
            )
            prog_eff = float(shaping.get("progress_eff", prog))
            r = reward_terms(
                prog_eff, collision_risk, maneuver, cfg,
                efficiency_cost_val=float(eff["efficiency_cost"]),
                path_shaping_val=float(shaping["path_shaping"]),
            )["reward"]
            # Intervention proxy only on legacy clearance path (shield analogue).
            hard_m = float(getattr(cfg, "hard_brake_depth_m", 3.0) or 3.0)
            d_hat = getattr(out, "d_fwd_hat", None)
            if (
                not bool(getattr(cfg, "use_learned_obstacle_cost", False))
                and float(cfg.w_intervention) > 0.0
                and d_hat is not None
            ):
                try:
                    if float(d_hat) <= hard_m:
                        r -= float(cfg.w_intervention)
                except (TypeError, ValueError):
                    pass
            if getattr(out, "arrived", False):
                r += cfg.success_bonus
            rews[b, t] = r
            if out.done:
                alive[b] = False
                dones[b, t] = True
            if use_aux and propagate_goal_rel and alive[b]:
                goal_rel_t[b] = advance_goal_rel_body(goal_rel_t[b], a)
            if goal_rel_t is not None:
                goals_hist[b, t + 1] = goal_rel_t[b]

    return ImaginedRollout(
        z=zs, actions=acts, rewards=rews, p_coll=pcs, progress=progs, done=dones,
        n_action_clipped=int(n_clipped),
        goal_rel=(goals_hist if use_aux else None),
    )
