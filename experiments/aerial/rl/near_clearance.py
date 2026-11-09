"""Distance from one step to a surface an earlier depth frame still sees.

NanoMap keeps each depth image with the pose that saw it and answers a query
from the newest image whose camera contains that point. Images are not fused.
A forward step's center stays inside the current 90 degree picture, so the body
shell is queried too: a point on the shell can fall outside the current picture
and still lie in an earlier frame. Samples sit on the camera plane, one meter
below the body pose, which is the plane the depth image measures. The distance
is from the step center to that surface. Nothing here deletes a candidate,
reads progress, or changes a weight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from experiments.aerial.rl.body_surround import BODY_RADIUS_M, CAM_DZ, depth_to_world
from experiments.aerial.rl.scene_memory import project_pixel

HISTORY_S = 1.0
STEP_HZ = 5.0
MAX_RANGE_M = 40.0


@dataclass(frozen=True)
class StepClearance:
    """Nearest remembered surface along this step. ``inf`` when no view contains it."""

    distance_m: float
    seen: bool
    age_s: float


class NearClearance:
    """About one second of depth frames, each left in its own camera."""

    def __init__(self, history_s: float = HISTORY_S, radius_m: float = BODY_RADIUS_M) -> None:
        self.history_s = float(history_s)
        self.radius_m = float(radius_m)
        self._views: list[tuple[float, np.ndarray, np.ndarray, tuple[int, int]]] = []

    def add(self, depth: np.ndarray, state: np.ndarray, t: float) -> None:
        image = np.asarray(depth, np.float64)
        st = np.asarray(state, np.float64).copy()
        pts = depth_to_world(image, st, MAX_RANGE_M, stride=1)
        self._views.append((float(t), st, pts, (int(image.shape[0]), int(image.shape[1]))))
        keep = float(t) - self.history_s
        self._views = [v for v in self._views if v[0] >= keep - 1e-9]

    def step(self, state: np.ndarray, action: np.ndarray, t: float) -> StepClearance:
        """Clearance of the first-step body. A step that does not translate is unseen."""
        centers = _centers(state, action)
        if not centers:
            return StepClearance(math.inf, False, 0.0)
        best = math.inf
        best_age = 0.0
        seen = False
        for center, queries in centers:
            for sample in queries:
                found = self._surface(sample)
                if found is None:
                    continue
                surface, when = found
                seen = True
                dist = float(np.linalg.norm(surface - center))
                if dist < best:
                    best = dist
                    best_age = max(0.0, float(t) - when)
        if not seen:
            return StepClearance(math.inf, False, 0.0)
        return StepClearance(best, True, best_age)

    def _surface(self, sample: np.ndarray) -> tuple[np.ndarray, float] | None:
        point = np.asarray(sample, np.float64)
        for when, state, pts, shape in reversed(self._views):
            if len(pts) == 0:
                continue
            if project_pixel(point, state[:3], float(state[6]), shape) is None:
                continue
            nearest = pts[int(np.argmin(np.linalg.norm(pts - point, axis=1)))]
            return nearest, when
        return None


def _centers(state: np.ndarray, action: np.ndarray) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Body centers along the step, each with the center and the two shell points."""
    st = np.asarray(state, np.float64)
    a = np.asarray(action, np.float64)
    end_f, end_l = float(a[0]), -float(a[1])
    length = math.hypot(end_f, end_l)
    if length < 0.05:
        return []
    n = max(2, int(math.ceil(length / 0.1)) + 1)
    yaw = float(st[6])
    c, s = math.cos(yaw), math.sin(yaw)
    out: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for frac in np.linspace(min(0.1, length) / length, 1.0, n):
        fwd, left = end_f * float(frac), end_l * float(frac)
        z = float(st[2] + CAM_DZ + float(a[2]) * float(frac))
        center = _world(st, c, s, fwd, left, z)
        shell_l = _world(st, c, s, fwd, left + BODY_RADIUS_M, z)
        shell_r = _world(st, c, s, fwd, left - BODY_RADIUS_M, z)
        out.append((center, [center, shell_l, shell_r]))
    return out


def _world(state: np.ndarray, c: float, s: float, forward: float, left: float, z: float) -> np.ndarray:
    return np.array([
        float(state[0]) + forward * c + left * s,
        float(state[1]) + forward * s - left * c,
        z,
    ], np.float64)
