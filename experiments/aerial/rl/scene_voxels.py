"""Seen surfaces in a 3D body grid.

Each past depth pixel stays at the range where it was seen. The capture pose
puts it in the world and the current pose brings it into the body. The grid
indexes forward, left, and up. A cell is 0.5 m on each side. Points already
inside the body are not written. A step meets an occupied cell when the
cell center's horizontal distance to the body path is within the body radius.
"""
from __future__ import annotations

import torch

from experiments.aerial.rl.body_surround import BODY_RADIUS_M, CELL_M, HALF_M, N
from experiments.aerial.rl.scene_points import (
    Z_HI,
    Z_LO,
    body_points,
    outside_view,
    path_distance,
    step_ends,
    _usable,
)

NZ = int(round((Z_HI - Z_LO) / CELL_M))


def voxel_index(fwd: torch.Tensor, left: torch.Tensor, dz: torch.Tensor):
    """Cell indices. ``i`` is forward, ``j`` is left, ``k`` is up."""
    i = torch.floor((fwd + HALF_M) / CELL_M).long()
    j = torch.floor((left + HALF_M) / CELL_M).long()
    k = torch.floor((dz - Z_LO) / CELL_M).long()
    ok = (i >= 0) & (i < N) & (j >= 0) & (j < N) & (k >= 0) & (k < NZ)
    return ok, i, j, k


def _centers(i: torch.Tensor, j: torch.Tensor, k: torch.Tensor):
    fwd = (i.to(torch.float32) + 0.5) * CELL_M - HALF_M
    left = (j.to(torch.float32) + 0.5) * CELL_M - HALF_M
    up = Z_LO + (k.to(torch.float32) + 0.5) * CELL_M
    return fwd, left, up


def nearest_voxel(depth_m: torch.Tensor, state: torch.Tensor, here: torch.Tensor, action):
    """Nearest occupied cell to this step, outside the camera and still inside it.

    Returns ``((gap_m, up_m), (gap_m, up_m))``. Either side is ``None`` when
    that set of cells is empty. ``gap_m`` is the horizontal distance from the
    cell center to the body path. ``up_m`` is the cell center's height.
    """
    if depth_m.shape[0] == 0:
        return None, None
    if here.shape[0] == 1 and depth_m.shape[0] > 1:
        here = here.expand(depth_m.shape[0], -1)
    fwd, left, dz, dist = body_points(depth_m, state, here)
    outside_body = torch.hypot(fwd, left) > BODY_RADIUS_M
    ok, i, j, k = voxel_index(fwd, left, dz)
    band = _usable(dist, dz) & outside_body & ok
    cf, cl, up = _centers(i, j, k)
    end_f, end_l = step_ends(action)
    gap = path_distance(cf, cl, end_f, end_l)
    gone = band & outside_view(fwd, left)
    stay = band & ~outside_view(fwd, left)

    def closest(mask: torch.Tensor):
        if not bool(mask.any()):
            return None
        pick = torch.argmin(torch.where(mask, gap, torch.full_like(gap, 1e9)))
        return float(gap.reshape(-1)[pick]), float(up.reshape(-1)[pick])

    return closest(gone), closest(stay)
