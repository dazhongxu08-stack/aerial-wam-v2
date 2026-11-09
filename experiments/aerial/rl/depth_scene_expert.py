"""Depth-aware scene expert for automatic obstacle-avoidance demos.

Unlike PathExpert (polyline chase, no OA), this expert:

  1. Uses ``SceneIntentPlanner`` yaw-fan + depth cones to pick a local subgoal
  2. Steers body-delta toward that subgoal
  3. Commands yaw toward the subgoal (human-like heading, not side-fly)

Privileged only in that it uses proprio + depth predictions (same D̂ deploy uses),
not GT mesh. Intended for offline densify / BC buffer preload — not deploy π.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.aerial.rl.scene_intent import SceneIntentPlanner


class DepthSceneExpertPolicy:
    """Collector policy: scene-intent subgoal + forward-biased body step + yaw."""

    def __init__(
        self,
        goal_getter: Any,
        *,
        r_m: float = 100.0,
        cruise_speed: float = 10.0,
        step_m: float = 1.0,
        max_dyaw: float = 0.314,
        max_ddyaw: float = 0.12,
        d_clear: float = 40.0,
        d_danger: float = 3.0,
        descent_radius_m: float = 0.0,
        replan_period_s: float = 2.0,
        w_jump: float = 0.05,
        stuck_escape_after_s: float = 12.0,
        stuck_escape_hold_s: float = 5.0,
        yaw_first_bearing_rad: float = 0.5,
        yaw_first_trans_scale: float = 0.35,
    ) -> None:
        self._goal_getter = goal_getter
        self.step_m = float(step_m)
        self.max_dyaw = float(max_dyaw)
        self.max_ddyaw = float(max_ddyaw)
        self.yaw_first_bearing_rad = float(yaw_first_bearing_rad)
        self.yaw_first_trans_scale = float(yaw_first_trans_scale)
        self.intent = SceneIntentPlanner(
            r_m=float(r_m),
            cruise_speed=float(cruise_speed),
            d_clear=float(d_clear),
            d_danger=float(d_danger),
            descent_radius_m=float(descent_radius_m),
            replan_period_s=float(replan_period_s),
            w_jump=float(w_jump),
            stuck_escape_after_s=float(stuck_escape_after_s),
            stuck_escape_hold_s=float(stuck_escape_hold_s),
        )
        self.last_info: Dict[str, Any] = {}
        self._prev_dyaw: Optional[float] = None

    def reset(self) -> None:
        self.intent.reset()
        self.last_info = {}
        self._prev_dyaw = None

    def act(self, obs: Any) -> np.ndarray:
        goal = self._goal_getter()
        if goal is None:
            return np.zeros(4, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64).reshape(3)
        pos = np.asarray(obs.position, dtype=np.float64).reshape(3)
        yaw = float(obs.yaw)

        info = getattr(obs, "info", None) or {}
        d_fwd = info.get("depth_min_pred")
        cones = info.get("depth_cones_pred")
        if isinstance(cones, dict):
            cf = cones.get("forward")
            if cf is not None and np.isfinite(float(cf)):
                d_fwd = float(cf)
        azimuth = info.get("depth_azimuth_pred")

        g_rel, s_info = self.intent.compute(
            curr_pos=pos,
            curr_yaw=yaw,
            goal=goal,
            d_fwd_hat=float(d_fwd) if d_fwd is not None else None,
            depth_cones=cones if isinstance(cones, dict) else None,
            depth_azimuth=azimuth if isinstance(azimuth, dict) else None,
        )
        self.last_info = dict(s_info)
        target = np.asarray(s_info["target_world"], dtype=np.float64).reshape(3)

        # Expose subgoal so collector / planner / goal_rel stay consistent.
        if isinstance(info, dict):
            info["goal"] = target.tolist()
            info["goal_rel"] = np.asarray(g_rel, dtype=np.float32).reshape(4).tolist()
            info["expert_chosen_idx"] = int(s_info.get("chosen_idx", 0))
            info["expert_offaxis"] = bool(int(s_info.get("chosen_idx", 0)) != 0)
            info["expert_in_peel"] = bool(s_info.get("in_peel"))
            info["expert_in_escape"] = bool(s_info.get("in_escape"))

        d_world = target - pos
        c, s = np.cos(yaw), np.sin(yaw)
        dx = c * d_world[0] + s * d_world[1]
        dy = -s * d_world[0] + c * d_world[1]
        dz = float(d_world[2])
        vec = np.array([dx, dy, dz], dtype=np.float64)
        n = float(np.linalg.norm(vec))
        if n > self.step_m and n > 1e-6:
            vec = vec / n * self.step_m

        # Face the subgoal (human-like); PathExpert / Heuristic often left dyaw=0.
        bearing = float(np.arctan2(dy, dx)) if n > 1e-4 else 0.0
        # Yaw-first when far off heading: cut translation so we don't crab-strafe.
        # During peel/escape allow faster yaw (max_ddyaw was starving 90° turns
        # and the vehicle crashed into the facade ~50 steps in on R1).
        in_evade = bool(s_info.get("in_peel") or s_info.get("in_escape"))
        dyaw_lim = float(self.max_dyaw) * (1.35 if in_evade else 1.0)
        ddyaw_lim = float(self.max_ddyaw)
        if in_evade and ddyaw_lim > 0.0:
            ddyaw_lim = max(ddyaw_lim, 0.25)
        if abs(bearing) > float(self.yaw_first_bearing_rad):
            scale = float(self.yaw_first_trans_scale)
            if in_evade:
                scale = max(scale, 0.55)  # keep some lateral while turning
            vec = vec * scale
        dyaw = float(np.clip(bearing, -dyaw_lim, dyaw_lim))
        if self._prev_dyaw is not None and float(ddyaw_lim) > 0.0:
            dd = float(np.clip(dyaw - self._prev_dyaw, -ddyaw_lim, ddyaw_lim))
            dyaw = float(self._prev_dyaw + dd)
        self._prev_dyaw = float(dyaw)
        return np.array([vec[0], vec[1], vec[2], dyaw], dtype=np.float64)
