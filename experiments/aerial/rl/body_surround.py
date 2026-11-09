"""Body-frame surround map for the survival head's initial state.

Seen depth is stored as world cells. The head later reads a 32 by 32 crop in
the current body frame. Nothing here deletes a candidate or reads progress.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from experiments.aerial.rl.consequence_model import DEPTH_MAX_M, depth_input

CELL_M = 0.5
HALF_M = 8.0
N = 32
KEEP_M = 40.0
HEIGHT_M = 8.0
BODY_RADIUS_M = 0.4
AGE_CAP_S = 10.0
HFOV_DEG = 90.0
VFOV_DEG = 58.7
CAM_DZ = -1.0


def _yaw(state: np.ndarray) -> float:
    return float(np.asarray(state, np.float64)[6])


def world_from_body(state: np.ndarray, forward: float, left: float, z: float | None = None) -> np.ndarray:
    """Body +x is forward, body +y is left. Image right is negative left."""
    st = np.asarray(state, np.float64)
    yaw = _yaw(st)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([
        st[0] + forward * c + left * s,
        st[1] + forward * s - left * c,
        st[2] if z is None else float(z),
    ], np.float64)


def body_from_world(state: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = np.asarray(state, np.float64)
    q = np.asarray(pts, np.float64).reshape(-1, 3)
    yaw = _yaw(st)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = q[:, 0] - st[0], q[:, 1] - st[1]
    forward = c * dx + s * dy
    left = s * dx - c * dy
    return forward, left, q[:, 2] - st[2]


def in_view(forward: np.ndarray, left: np.ndarray, half_deg: float = HFOV_DEG / 2) -> np.ndarray:
    """True for points inside the forward camera, using the horizontal field only."""
    ang = np.degrees(np.arctan2(np.abs(left), np.maximum(forward, 1e-6)))
    return (forward > 0.05) & (ang <= float(half_deg))


def _bin(forward: np.ndarray, left: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = np.floor((forward + HALF_M) / CELL_M).astype(np.int64)
    j = np.floor((left + HALF_M) / CELL_M).astype(np.int64)
    ok = (i >= 0) & (i < N) & (j >= 0) & (j < N)
    return ok, i, j


def depth_to_world(depth: np.ndarray, state: np.ndarray, max_range: float = KEEP_M,
                   stride: int = 1) -> np.ndarray:
    """World points of a depth map. Same axes as the flight depth projection."""
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    step = max(1, int(stride))
    fx = (w / 2) / math.tan(math.radians(HFOV_DEG / 2))
    fy = (h / 2) / math.tan(math.radians(VFOV_DEG / 2))
    v, u = np.mgrid[0:h:step, 0:w:step]
    d = d[::step, ::step]
    ok = np.isfinite(d) & (d > 0.05) & (d < max_range)
    fwd = d[ok]
    right = (u[ok] + 0.5 - w / 2) / fx * fwd
    down = (v[ok] + 0.5 - h / 2) / fy * fwd
    yaw = _yaw(state)
    c, s = math.cos(yaw), math.sin(yaw)
    st = np.asarray(state, np.float64)
    return np.stack([
        st[0] + fwd * c - right * s,
        st[1] + fwd * s + right * c,
        st[2] + CAM_DZ - down,
    ], -1)


class BodySurround:
    """One point per world cell. A later view of the same cell replaces it."""

    def __init__(self) -> None:
        self.cells: Dict[Tuple[int, int, int], Tuple[np.ndarray, float]] = {}

    def _key(self, p: np.ndarray) -> Tuple[int, int, int]:
        q = np.floor(np.asarray(p, np.float64) / CELL_M).astype(int)
        return int(q[0]), int(q[1]), int(q[2])

    def add_world(self, pts: np.ndarray, t: float) -> None:
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            return
        keys = np.floor(pts / CELL_M).astype(np.int64)
        packed = np.ascontiguousarray(keys).view(np.dtype((np.void, keys.dtype.itemsize * 3))).ravel()
        _, first = np.unique(packed[::-1], return_index=True)
        pick = len(pts) - 1 - first
        for i in pick:
            key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            self.cells[key] = (pts[i].copy(), float(t))

    def add_body(self, state: np.ndarray, forward: float, left: float, z: float | None, t: float) -> None:
        self.add_world(world_from_body(state, forward, left, z)[None, :], t)

    def add_depth(self, depth: np.ndarray, state: np.ndarray, t: float, max_range: float = KEEP_M,
                  stride: int = 1) -> None:
        pts = depth_to_world(depth, state, max_range, stride)
        if len(pts):
            self.add_world(pts, t)

    def forget(self, state: np.ndarray) -> None:
        st = np.asarray(state, np.float64)
        drop = []
        for key, (p, _) in self.cells.items():
            if math.hypot(p[0] - st[0], p[1] - st[1]) > KEEP_M:
                drop.append(key)
        for key in drop:
            del self.cells[key]

    def render(self, state: np.ndarray, t: float) -> np.ndarray:
        """``[2, 32, 32]``. Axis 1 is forward, axis 2 is left. Empty is far and old."""
        self.forget(state)
        dist = np.full((N, N), DEPTH_MAX_M, np.float64)
        age = np.ones((N, N), np.float64)
        fwd, left, dz = body_from_world(state, np.stack([p for p, _ in self.cells.values()]) if self.cells else np.zeros((0, 3)))
        if len(fwd):
            times = np.array([ts for _, ts in self.cells.values()], np.float64)
            pts_z = dz
            near = np.abs(pts_z) <= HEIGHT_M
            ok, i, j = _bin(fwd, left)
            use = near & ok
            rng = np.sqrt(fwd * fwd + left * left + dz * dz)
            for n in np.flatnonzero(use):
                ii, jj = int(i[n]), int(j[n])
                if rng[n] < dist[ii, jj]:
                    dist[ii, jj] = rng[n]
                age[ii, jj] = min(age[ii, jj], min(AGE_CAP_S, max(0.0, float(t) - times[n])) / AGE_CAP_S)
        out = np.stack([depth_input(dist), age.astype(np.float32)], 0)
        return out

    def occupied(self, state: np.ndarray, t: float, near_m: float = HALF_M) -> set[Tuple[int, int]]:
        """Body cells that hold a surface closer than ``near_m``."""
        img = self.render(state, t)
        # depth_input is 0 at DEPTH_MIN and 1 at DEPTH_MAX. Invert by comparing the stored render
        # against the log of near_m.
        limit = float(depth_input(np.array([near_m]))[0])
        ii, jj = np.nonzero(img[0] < limit - 1e-6)
        return {(int(a), int(b)) for a, b in zip(ii, jj)}


def step_cells(state: np.ndarray, action: np.ndarray, radius: float = BODY_RADIUS_M) -> set[Tuple[int, int]]:
    """Body cells the vehicle body covers along the first step.

    ``action`` is ``[dx, dy, dz, dyaw]`` with ``dy`` toward image right.
    This does not remove the action from the allowed set.
    """
    a = np.asarray(action, np.float64)
    end_f, end_l = float(a[0]), -float(a[1])
    length = math.hypot(end_f, end_l)
    n = max(2, int(math.ceil(length / 0.1)) + 1)
    fwd = np.linspace(0.0, end_f, n)
    left = np.linspace(0.0, end_l, n)
    span = int(math.ceil(radius / CELL_M)) + 1
    found: set[Tuple[int, int]] = set()
    for f, l in zip(fwd, left):
        for di in range(-span, span + 1):
            for dj in range(-span, span + 1):
                cf = (math.floor((f + HALF_M) / CELL_M) + di + 0.5) * CELL_M - HALF_M
                cl = (math.floor((l + HALF_M) / CELL_M) + dj + 0.5) * CELL_M - HALF_M
                if math.hypot(cf - f, cl - l) > radius:
                    continue
                ok, i, j = _bin(np.array([cf]), np.array([cl]))
                if bool(ok[0]):
                    found.add((int(i[0]), int(j[0])))
    return found


def step_sweeps(mem: BodySurround, state: np.ndarray, action: np.ndarray, t: float,
                radius: float = BODY_RADIUS_M) -> bool:
    """True when the first step's body meets a remembered surface cell."""
    return bool(step_cells(state, action, radius) & mem.occupied(state, t))
