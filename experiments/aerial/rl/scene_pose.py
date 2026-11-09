"""Each seen frame keeps its depth image and the pose it was taken from.

The pose is the reference: at scoring time it is rewritten as the position and
heading of that camera in the current body frame. The image is not resampled
into a grid. An empty record list reads as zeros. Progress does not use this.
"""
from __future__ import annotations

import math

import torch
from torch import nn

LENGTH = 8
POSE_SCALE_M = 8.0


def relative_pose(current: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
    """Where each stored camera sits in the current body frame.

    ``current`` is ``[B,7]`` and ``seen`` is ``[B,T,7]`` (x, y, z, ..., yaw).
    Returns ``[B,T,5]``: forward/8, left/8, dz/8, sin(heading difference),
    cos(heading difference) - 1. The same place and heading is all zeros.
    Body +x is forward and body +y is left.
    """
    now = current[:, None, :]
    dx = seen[..., 0] - now[..., 0]
    dy = seen[..., 1] - now[..., 1]
    yaw = now[..., 6]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    forward = cy * dx + sy * dy
    left = sy * dx - cy * dy
    dz = seen[..., 2] - now[..., 2]
    dyaw = seen[..., 6] - yaw
    return torch.stack([
        forward / POSE_SCALE_M,
        left / POSE_SCALE_M,
        dz / POSE_SCALE_M,
        torch.sin(dyaw),
        torch.cos(dyaw) - 1.0,
    ], dim=-1)


class ScenePoseMemory(nn.Module):
    """Read up to ``length`` stored views. The current image features are the query."""

    def __init__(self, dim: int, hidden: int, length: int = LENGTH) -> None:
        super().__init__()
        self.dim = int(dim)
        self.length = int(length)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, self.dim),
        )
        self.pose = nn.Linear(5, self.dim, bias=False)
        self.q = nn.Linear(hidden, self.dim, bias=False)
        self.k = nn.Linear(self.dim, self.dim, bias=False)
        self.v = nn.Linear(self.dim, self.dim, bias=False)
        self.out = nn.Linear(self.dim, self.dim, bias=False)

    def encode(self, depth: torch.Tensor) -> torch.Tensor:
        """``depth`` ``[B,T,56,56]`` in ``depth_input`` units. Returns ``[B,T,dim]``."""
        b, steps, _, _ = depth.shape
        return self.enc(depth.reshape(b * steps, 1, 56, 56)).view(b, steps, self.dim)

    def forward(
        self,
        current: torch.Tensor,
        current_depth: torch.Tensor,
        depth: torch.Tensor,
        seen_state: torch.Tensor,
        now: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B,dim]``. A stored frame that repeats the current view from the same pose is zero."""
        feat = self.encode(depth) - self.encode(current_depth[:, None])
        rel = relative_pose(now, seen_state)
        token = feat + self.pose(rel)
        query = self.q(current).unsqueeze(1)
        key = self.k(token)
        value = self.v(token)
        score = (query * key).sum(-1) / math.sqrt(self.dim)
        if mask is None:
            mask = torch.ones(score.shape, dtype=torch.bool, device=score.device)
        score = score.masked_fill(~mask, -1e9)
        weight = torch.softmax(score, dim=-1) * mask.to(score.dtype)
        none = ~mask.any(dim=-1)
        weight = weight / weight.sum(-1, keepdim=True).clamp(min=1e-6)
        weight = torch.where(none.unsqueeze(-1), torch.zeros_like(weight), weight)
        read = (weight.unsqueeze(-1) * value).sum(1)
        return self.out(read)


class SceneHazard(nn.Module):
    """Per-step change of the hazard logit. A zero scene vector adds nothing."""

    def __init__(self, hidden: int, dim: int) -> None:
        super().__init__()
        self.h = nn.Linear(hidden, dim, bias=False)
        self.s = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.s.weight)

    def forward(self, hidden: torch.Tensor, posed: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``[B,T,H]``, ``posed`` ``[B,dim]`` -> ``[B,T]``."""
        return (self.h(hidden) * self.s(posed).unsqueeze(1)).sum(-1)
