"""Correction from forward/left/right clearance plus the goal direction.

The frozen actor does not use the picture. These distances do change between
an open street and a wall, and they move smoothly while the aircraft yaws.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.aerial.rl.depth_geometry import approach_clearances
from experiments.aerial.rl.gap_passage import LIMITS
from experiments.aerial.rl.goal_features import goal_rel_from_obs


def clearance_features(obs) -> np.ndarray:
    depth = getattr(obs, "depth", None)
    if depth is None:
        fwd = left = right = 80.0
    else:
        arr = np.asarray(depth, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[0]
        fwd, left, right = approach_clearances(arr, center_frac=0.12)
    def _log(v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            v = 80.0
        return math.log1p(min(float(v), 80.0))

    yaw = float(getattr(obs, "yaw", 0.0))
    goal = goal_rel_from_obs(obs).astype(np.float32)
    return np.array(
        [_log(fwd), _log(left), _log(right), math.sin(yaw), math.cos(yaw), *goal.tolist()],
        dtype=np.float32,
    )


class ClearanceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.register_buffer(
            "limits", torch.tensor(LIMITS, dtype=torch.float32), persistent=True
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        return torch.tanh(self.net(feat.float())) * self.limits

    @torch.no_grad()
    def residual(self, obs) -> np.ndarray:
        feat = torch.as_tensor(clearance_features(obs), device=self.limits.device)
        return self.forward(feat).squeeze(0).detach().cpu().numpy().astype(np.float64)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ClearanceHead":
        model = cls()
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device)
        model.load_state_dict(payload["net"])
        model.to(device)
        model.eval()
        return model
