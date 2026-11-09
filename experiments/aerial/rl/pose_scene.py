"""Scenes already seen, placed in the current body frame.

Each depth frame is projected with the camera that took it, then moved by the
pose change into the body frame at the latest pose. The grid is what the
survival head reads. It does not delete a candidate, and an empty grid is all
zeros. Progress does not read it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from experiments.aerial.rl.body_surround import (
    CELL_M,
    HALF_M,
    N,
    body_from_world,
    depth_to_world,
)
from experiments.aerial.rl.consequence_model import depth_input

# The camera sits 1 m below the body. Keep the slice that includes that height
# and the body, and leave out the ground below it.
Z_LO = -1.5
Z_HI = 1.0


def _paint(dist: np.ndarray, height: np.ndarray, fwd: np.ndarray, left: np.ndarray, dz: np.ndarray) -> None:
    use = (dz >= Z_LO) & (dz <= Z_HI)
    fwd, left, dz = fwd[use], left[use], dz[use]
    if len(fwd) == 0:
        return
    i = np.floor((fwd + HALF_M) / CELL_M).astype(np.int64)
    j = np.floor((left + HALF_M) / CELL_M).astype(np.int64)
    inside = (i >= 0) & (i < N) & (j >= 0) & (j < N)
    i, j, fwd, left, dz = i[inside], j[inside], fwd[inside], left[inside], dz[inside]
    if len(i) == 0:
        return
    rng = np.hypot(fwd, left)
    order = np.argsort(rng, kind="mergesort")
    i, j, rng, dz = i[order], j[order], rng[order], dz[order]
    cell = i * N + j
    _, first = np.unique(cell, return_index=True)
    i, j, rng, dz = i[first], j[first], rng[first], dz[first]
    closer = rng < dist[i, j]
    ii, jj = i[closer], j[closer]
    dist[ii, jj] = rng[closer]
    height[ii, jj] = dz[closer]


def _grid_from_dist(dist: np.ndarray, height: np.ndarray) -> np.ndarray:
    seen = np.isfinite(dist)
    near = np.zeros((N, N), np.float32)
    if np.any(seen):
        near[seen] = depth_input(dist[seen])
    rel = np.zeros((N, N), np.float32)
    rel[seen] = (height[seen] / HALF_M).astype(np.float32)
    return np.stack([near, rel], 0)


def scene_in_body(depths: Sequence[np.ndarray], states: Sequence[np.ndarray], stride: int = 2) -> np.ndarray:
    """``[2, 32, 32]`` in the last pose's body frame.

    Axis 1 is forward, axis 2 is left, each cell is 0.5 m, covering 8 m.
    Channel 0 is the log range of the nearest surface in the cell, and 0 when
    the cell was never seen. Channel 1 is that surface's height relative to
    the body, divided by 8, and 0 when the cell is empty.
    """
    dist = np.full((N, N), np.inf, np.float64)
    height = np.zeros((N, N), np.float64)
    if len(depths) == 0:
        return np.zeros((2, N, N), np.float32)
    here = np.asarray(states[-1], np.float64)
    for depth, state in zip(depths, states):
        pts = depth_to_world(np.asarray(depth), np.asarray(state, np.float64), max_range=HALF_M * 2, stride=stride)
        if len(pts) == 0:
            continue
        fwd, left, dz = body_from_world(here, pts)
        use = (dz >= Z_LO) & (dz <= Z_HI)
        if not np.any(use):
            continue
        _paint(dist, height, fwd, left, dz)
    return _grid_from_dist(dist, height)


def scenes_along(depths: Sequence[np.ndarray], states: Sequence[np.ndarray], stride: int = 2) -> np.ndarray:
    """``[T,2,32,32]``. Frame ``t`` is every scene up to ``t``, in the body frame at ``t``."""
    frames = [np.asarray(d) for d in depths]
    poses = [np.asarray(s, np.float64) for s in states]
    worlds = [depth_to_world(d, s, max_range=HALF_M * 2, stride=stride) for d, s in zip(frames, poses)]
    out = np.zeros((len(frames), 2, N, N), np.float32)
    kept: list[np.ndarray] = []
    for t, pts in enumerate(worlds):
        if len(pts):
            kept.append(pts)
        dist = np.full((N, N), np.inf, np.float64)
        height = np.zeros((N, N), np.float64)
        if kept:
            fwd, left, dz = body_from_world(poses[t], np.concatenate(kept, axis=0))
            _paint(dist, height, fwd, left, dz)
        out[t] = _grid_from_dist(dist, height)
    return out
