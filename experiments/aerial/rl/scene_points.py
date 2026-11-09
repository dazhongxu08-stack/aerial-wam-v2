"""Seen surfaces kept as points, moved by the current pose. No grid.

Each past depth pixel stays at the range where it was seen. The capture pose
puts it in the world and the current pose brings it into the body. A point
already inside the body is dropped: the body has reached it, and it does not
say whether this step enters a surface. A step meets a remaining point when
that point's horizontal distance to the body path is within the body radius.
The ray length is not that distance.
"""
from __future__ import annotations

import math

import torch

from experiments.aerial.rl.body_surround import BODY_RADIUS_M, CAM_DZ, HFOV_DEG, VFOV_DEG

Z_LO = -1.5
Z_HI = 1.0
MAX_RANGE_M = 16.0


def path_distance(fwd: torch.Tensor, left: torch.Tensor, end_f: float, end_l: float) -> torch.Tensor:
    """Horizontal distance from body points to the segment (0, 0) -> (end_f, end_l)."""
    length2 = end_f * end_f + end_l * end_l
    if length2 < 1e-12:
        return torch.hypot(fwd, left)
    t = ((fwd * end_f + left * end_l) / length2).clamp(0.0, 1.0)
    return torch.hypot(fwd - t * end_f, left - t * end_l)


def outside_view(fwd: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    """Outside the current 90° camera. Same test as ``body_surround.in_view``."""
    ang = torch.atan2(left.abs(), fwd.clamp(min=1e-6))
    return (fwd <= 0.05) | (ang > math.radians(HFOV_DEG / 2))


def body_points(depth_m: torch.Tensor, state: torch.Tensor, here: torch.Tensor):
    """Every depth pixel in the current body.

    ``depth_m`` is meters ``[B,H,W]``. Returns ``fwd``, ``left``, ``dz``,
    ``dist`` in that shape.
    """
    b, h, w = depth_m.shape
    v = torch.arange(h, device=depth_m.device, dtype=depth_m.dtype)[:, None]
    u = torch.arange(w, device=depth_m.device, dtype=depth_m.dtype)[None, :]
    fx = (w / 2) / math.tan(math.radians(HFOV_DEG / 2))
    fy = (h / 2) / math.tan(math.radians(VFOV_DEG / 2))
    right = (u + 0.5 - w / 2) / fx * depth_m
    down = (v + 0.5 - h / 2) / fy * depth_m
    yaw = state[:, 6]
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    world_x = state[:, 0, None, None] + depth_m * c - right * s
    world_y = state[:, 1, None, None] + depth_m * s + right * c
    world_z = state[:, 2, None, None] + CAM_DZ - down
    hy = here[:, 6]
    cy, sy = torch.cos(hy)[:, None, None], torch.sin(hy)[:, None, None]
    dx = world_x - here[:, 0, None, None]
    dy = world_y - here[:, 1, None, None]
    fwd = cy * dx + sy * dy
    left = sy * dx - cy * dy
    dz = world_z - here[:, 2, None, None]
    return fwd, left, dz, depth_m


def _usable(dist: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(dist) & (dist > 0.05) & (dist < MAX_RANGE_M) & (dz >= Z_LO) & (dz <= Z_HI)


def step_ends(action) -> tuple[float, float]:
    """Body path of one step. Action ``dy`` points to image right, body left is opposite."""
    return float(action[0]), -float(action[1])


def nearest(depth_m: torch.Tensor, state: torch.Tensor, here: torch.Tensor, action) -> tuple[float | None, float | None]:
    """Nearest past point to this step, outside the current camera and still inside it.

    ``depth_m`` ``[P,H,W]`` are past frames only. ``here`` is ``[1,7]`` or ``[P,7]``.
    Distances are horizontal meters. Points inside the body radius are dropped.
    ``None`` means no remaining point in the height band.
    """
    if depth_m.shape[0] == 0:
        return None, None
    if here.shape[0] == 1 and depth_m.shape[0] > 1:
        here = here.expand(depth_m.shape[0], -1)
    fwd, left, dz, dist = body_points(depth_m, state, here)
    outside_body = torch.hypot(fwd, left) > BODY_RADIUS_M
    band = _usable(dist, dz) & outside_body
    end_f, end_l = step_ends(action)
    gap = path_distance(fwd, left, end_f, end_l)
    gone = band & outside_view(fwd, left)
    stay = band & ~outside_view(fwd, left)

    def mind(mask: torch.Tensor) -> float | None:
        if not bool(mask.any()):
            return None
        return float(gap[mask].min())

    return mind(gone), mind(stay)
