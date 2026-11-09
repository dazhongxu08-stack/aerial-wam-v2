"""Where this flight continues past the local horizon.

The point handed to the consequence model stays inside the usual horizon.
The remembered place may lie farther ahead. Survival is not read, and no
candidate is removed.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

HORIZON_M = 8.0
FAR_M = 40.0
_OFFSETS_DEG = tuple(range(15, 91, 15))


def _heading(here: np.ndarray, there: np.ndarray) -> float:
    return math.atan2(float(there[1] - here[1]), float(there[0] - here[0]))


def _point(here: np.ndarray, heading: float, dist: float) -> np.ndarray:
    return np.array([
        here[0] + dist * math.cos(heading),
        here[1] + dist * math.sin(heading),
        here[2],
    ], np.float64)


def _extend(here: np.ndarray, heading: float, far_m: float,
            hits: Callable[[np.ndarray, np.ndarray], Optional[bool]]) -> Optional[np.ndarray]:
    """Walk until a hit or an unseen sample after a clear one.

    ``hits`` is True on a surface, False when the segment is clear, None when
    no stored view contains it yet. The opening gap before the first clear
    sample does not count as open space.
    """
    last = None
    dist = 0.5
    while dist <= far_m + 1e-9:
        aim = _point(here, heading, dist)
        status = hits(here, aim)
        if status is None:
            if last is not None:
                break
            dist += 0.5
            continue
        if status:
            break
        last = aim
        dist += 0.5
    return last


def _clamp(here: np.ndarray, target: np.ndarray, horizon_m: float) -> np.ndarray:
    delta = np.asarray(target, np.float64)[:2] - here[:2]
    dist = float(np.linalg.norm(delta))
    step = min(max(dist, 0.0), float(horizon_m))
    if dist < 1e-6:
        return here.copy()
    xy = here[:2] + delta / dist * step
    return np.array([xy[0], xy[1], here[2]], np.float64)


class FlightContinuation:
    """One flight's far place, and the horizon point along the way there."""

    def __init__(self, horizon_m: float = HORIZON_M, far_m: float = FAR_M) -> None:
        self.horizon_m = float(horizon_m)
        self.far_m = float(far_m)
        self.far_point: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.far_point = None

    def _usable(self, here: np.ndarray, hits: Callable[[np.ndarray, np.ndarray], Optional[bool]]) -> bool:
        if self.far_point is None:
            return False
        if float(np.linalg.norm(self.far_point[:2] - here[:2])) <= 1.0:
            return False
        return hits(here, self.far_point) is False

    def _search(self, here: np.ndarray, goal: np.ndarray,
                hits: Callable[[np.ndarray, np.ndarray], Optional[bool]]) -> np.ndarray:
        base = _heading(here, goal)
        offsets = [0.0]
        for deg in _OFFSETS_DEG:
            offsets.append(float(deg))
            offsets.append(-float(deg))
        best: Optional[tuple] = None
        chosen = _point(here, base, 0.5)
        for off in offsets:
            heading = base + math.radians(off)
            end = _extend(here, heading, self.far_m, hits)
            if end is None:
                continue
            dist_goal = float(np.linalg.norm(end[:2] - goal[:2]))
            travel = float(np.linalg.norm(end[:2] - here[:2]))
            key = (dist_goal, -travel)
            if best is None or key < best[0]:
                best = (key, end)
                chosen = end
        return chosen

    def update(self, here: np.ndarray, goal: np.ndarray,
               hits: Callable[[np.ndarray, np.ndarray], Optional[bool]]) -> np.ndarray:
        """Subgoal on the remembered continuation, at most one horizon away."""
        here = np.asarray(here, np.float64)[:3].copy()
        goal = np.asarray(goal, np.float64)[:3]
        if not self._usable(here, hits):
            self.far_point = self._search(here, goal, hits)
        return _clamp(here, self.far_point, self.horizon_m)
