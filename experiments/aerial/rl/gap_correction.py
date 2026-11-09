"""Gated residual on the frozen route-0 policy.

The correction is added only while the aircraft is still east of the west
gap and south of y=80. Route 0 starts at y=101, so the gate is off there.
Outside the gate the residual is forced to zero.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

GATE_Y = 80.0
GATE_X = -960.0


def correction_gate(x: float, y: float) -> float:
    """1 inside the route-1 prefix, 0 everywhere else."""
    if float(y) < GATE_Y and float(x) > GATE_X:
        return 1.0
    return 0.0


def correction_features(pos, yaw: float, goal) -> np.ndarray:
    x = float(pos[0])
    y = float(pos[1])
    gx = float(goal[0])
    gy = float(goal[1])
    return np.array([x, y, float(yaw), gx, gy, gx - x, gy - y], dtype=np.float32)


class GapCorrection:
    """Frozen-base residual. ``apply`` returns the flight action."""

    def __init__(self, net, mean: np.ndarray, std: np.ndarray, limits: np.ndarray):
        self.net = net
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.limits = np.asarray(limits, dtype=np.float64)

    def residual(self, pos, yaw: float, goal) -> np.ndarray:
        import torch

        feat = (correction_features(pos, yaw, goal) - self.mean) / self.std
        with torch.no_grad():
            out = self.net(torch.as_tensor(feat).view(1, -1))
        return out.squeeze(0).cpu().numpy().astype(np.float64)

    def apply(self, base_action, pos, goal, yaw: float) -> np.ndarray:
        base = np.asarray(base_action, dtype=np.float64).reshape(4)
        if correction_gate(float(pos[0]), float(pos[1])) == 0.0:
            return base
        action = base + self.residual(pos, yaw, goal)
        return np.clip(action, -self.limits, self.limits)

    def save(self, path: Path) -> None:
        import torch

        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "mean": self.mean,
                "std": self.std,
                "limits": self.limits,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "GapCorrection":
        import torch

        payload = torch.load(str(path), map_location=device, weights_only=False)
        net = _net()
        net.load_state_dict(payload["state_dict"])
        net.to(device)
        net.eval()
        return cls(net, payload["mean"], payload["std"], payload["limits"])


def _net():
    import torch
    from torch import nn

    return nn.Sequential(
        nn.Linear(7, 32),
        nn.Tanh(),
        nn.Linear(32, 4),
    )
