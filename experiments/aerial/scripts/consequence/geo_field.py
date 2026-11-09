"""Geodesic (around-buildings) distance to the goal at flight altitude, from a topdown_height.py map.

A cell blocks when its surface is within ``clear_m`` of the flight altitude or higher; blocked cells are dilated by
``inflate_m``. Unobserved cells count as free. Invisible collision meshes are absent from the maps, so they are free
here too. Distances come from an 8-connected Dijkstra seeded at the goal cell (seeded even when blocked).
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, Tuple

import numpy as np


def blocked_mask(top_ned_z: np.ndarray, res: float, z_up: float, clear_m: float = 1.0,
                 inflate_m: float = 1.0) -> np.ndarray:
    h = -np.asarray(top_ned_z, np.float64)
    occ = np.isfinite(h) & (h >= z_up - clear_m)
    r = int(math.ceil(inflate_m / res))
    if r <= 0:
        return occ
    out = occ.copy()
    nx, ny = occ.shape
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj > r * r or (di == 0 and dj == 0):
                continue
            src = occ[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj)]
            out[max(0, di):nx - max(0, -di), max(0, dj):ny - max(0, -dj)] |= src
    return out


def dijkstra(blocked: np.ndarray, seed: Tuple[int, int], res: float) -> np.ndarray:
    nx, ny = blocked.shape
    dist = np.full((nx, ny), np.inf)
    si, sj = seed
    if not (0 <= si < nx and 0 <= sj < ny):
        return dist
    free = ~blocked
    free[si, sj] = True
    dist[si, sj] = 0.0
    steps = [(-1, 0, res), (1, 0, res), (0, -1, res), (0, 1, res),
             (-1, -1, res * math.sqrt(2)), (-1, 1, res * math.sqrt(2)),
             (1, -1, res * math.sqrt(2)), (1, 1, res * math.sqrt(2))]
    heap = [(0.0, si, sj)]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di, dj, w in steps:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[a, b]:
                nd = d + w
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    heapq.heappush(heap, (nd, a, b))
    return dist


class GeoField:
    """Geodesic distance field for one (map, goal, altitude); ``lookup`` returns NaN off-map or in blocked cells."""

    def __init__(self, top: Dict[str, np.ndarray], goal_xy, z_up: float, clear_m: float = 1.0,
                 inflate_m: float = 1.0) -> None:
        self.x0, self.y0, self.res = float(top["x0"]), float(top["y0"]), float(top["res"])
        blocked = blocked_mask(top["top_ned_z"], self.res, z_up, clear_m, inflate_m)
        self.dist = dijkstra(blocked, self.cell(goal_xy[0], goal_xy[1]), self.res)

    def cell(self, x: float, y: float) -> Tuple[int, int]:
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def lookup(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, np.float64))
        i = ((xy[:, 0] - self.x0) / self.res).astype(int)
        j = ((xy[:, 1] - self.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.dist.shape[0]) & (j >= 0) & (j < self.dist.shape[1])
        out = np.full(len(xy), np.nan)
        out[ok] = self.dist[i[ok], j[ok]]
        out[~np.isfinite(out)] = np.nan
        return out
