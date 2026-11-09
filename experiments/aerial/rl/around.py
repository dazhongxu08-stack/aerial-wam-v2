"""Go around a building that sits on the straight line to the goal.

Face the goal. If that view is blocked, fly along the longer side, then look
at the goal again. Repeat until the goal direction is open. No policy update.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _wrap(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass
class AroundDetour:
    open_m: float = 25.0
    block_m: float = 6.0
    align_rad: float = 0.30
    recheck_m: float = 8.0
    cruise: float = 0.8

    def reset(self) -> None:
        self.phase = "face"
        self.detour_heading = None
        self.travel = 0.0

    def act(
        self,
        pos,
        yaw: float,
        goal,
        d_fwd: float,
        d_left: float,
        d_right: float,
        limits,
    ) -> tuple[np.ndarray, str]:
        if not hasattr(self, "phase"):
            self.reset()
        lim = np.abs(np.asarray(limits, dtype=np.float64).reshape(4))
        p = np.asarray(pos, dtype=np.float64).reshape(-1)
        g = np.asarray(goal, dtype=np.float64).reshape(-1)
        goal_heading = math.atan2(float(g[1] - p[1]), float(g[0] - p[0]))
        yaw = float(yaw)
        fwd = float(d_fwd)
        left = float(d_left)
        right = float(d_right)

        def yaw_toward(target: float) -> tuple[float, bool]:
            err = _wrap(target - yaw)
            if abs(err) <= float(self.align_rad):
                return 0.0, True
            step = min(float(lim[3]), abs(err))
            return math.copysign(step, err), False

        def latch(tag: str) -> tuple[np.ndarray, str]:
            # Keep the side already chosen. Re-picking the longer cone after a
            # few metres turns the detour around, because that side is now nearer.
            if self.detour_heading is None:
                sign = 1.0 if left >= right else -1.0
                self.detour_heading = _wrap(yaw + sign * math.pi / 2.0)
            self.travel = 0.0
            self.phase = "detour"
            return np.zeros(4), tag

        if self.phase == "face":
            dyaw, aligned = yaw_toward(goal_heading)
            if not aligned:
                return np.array([0.0, 0.0, 0.0, dyaw]), "face"
            if fwd >= float(self.open_m):
                self.detour_heading = None
                self.phase = "goal"
            else:
                return latch("latch")

        if self.phase == "goal":
            dyaw, aligned = yaw_toward(goal_heading)
            if not aligned:
                return np.array([0.0, 0.0, 0.0, dyaw]), "goal_turn"
            if fwd < float(self.open_m):
                return latch("relatch")
            dx = min(float(self.cruise), float(lim[0]))
            return np.array([dx, 0.0, 0.0, 0.0]), "goal"

        dyaw, aligned = yaw_toward(float(self.detour_heading))
        if not aligned:
            return np.array([0.0, 0.0, 0.0, dyaw]), "detour_turn"
        if fwd < float(self.block_m):
            self.detour_heading = _wrap(yaw + math.pi)
            self.travel = 0.0
            return np.zeros(4), "flip"
        dx = min(float(self.cruise), float(lim[0]))
        self.travel += dx
        if self.travel >= float(self.recheck_m):
            self.phase = "face"
            self.travel = 0.0
            return np.array([dx, 0.0, 0.0, 0.0]), "recheck"
        return np.array([dx, 0.0, 0.0, 0.0]), "detour"
