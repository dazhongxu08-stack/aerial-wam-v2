"""Sticks from the frozen clearance code and this frame's goal direction."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

YAW_LIMIT = 0.3141592653589793
OPEN_DX = 0.92


def _opening(mean_m: float, left: float, right: float) -> bool:
    gap = float(left) - float(right)
    return abs(gap) >= 8.0 and max(float(left), float(right)) >= float(mean_m) + 8.0


def stick_target(mean_m: float, left: float, right: float, goal_rel: np.ndarray) -> np.ndarray:
    """Stop when the front is near and neither side is open. Yaw when the goal agrees with that opening."""
    bearing = math.atan2(float(goal_rel[1]), max(float(goal_rel[0]), 1e-3))
    gap = float(left) - float(right)
    dyaw = float(np.clip(0.12 * bearing, -0.08, 0.08))
    if float(mean_m) < 10.0 and not _opening(mean_m, left, right):
        return np.array([0.0, 0.0, 0.0, dyaw], dtype=np.float32)
    blocked = (
        float(mean_m) < 18.0
        and abs(gap) >= 12.0
        and max(float(left), float(right)) >= float(mean_m) + 8.0
        and abs(bearing) >= 0.46
        and gap * bearing > 0.0
    )
    if blocked:
        sign = 1.0 if gap >= 0.0 else -1.0
        return np.array([0.0, 0.0, 0.0, sign * YAW_LIMIT], dtype=np.float32)
    return np.array([OPEN_DX, 0.0, 0.0, dyaw], dtype=np.float32)


def measured_clearances(depth: np.ndarray) -> tuple[float, float, float]:
    """Forward mean and side medians on a depth map, same crop as the training labels."""
    from experiments.aerial.rl.depth_geometry import approach_clearances

    arr = np.asarray(depth, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    _fwd, left, right = approach_clearances(arr, center_frac=0.12)
    height, width = arr.shape
    band_h = max(1, int(round(height * 0.12)))
    band_w = max(1, int(round(width * 0.12)))
    row0 = (height - band_h) // 2
    col0 = (width - band_w) // 2
    crop = arr[row0 : row0 + band_h, col0 : col0 + band_w]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    mean_m = float(min(float(valid.mean()), 80.0)) if valid.size else 80.0
    return mean_m, float(left), float(right)


class DepthStickDeploy:
    """Current frame through the depth head, then sticks from that map and the goal."""

    def __init__(self, predictor, device: str) -> None:
        self.predictor = predictor
        self.device = device

    @classmethod
    def load(cls, path: str, device: str) -> "DepthStickDeploy":
        from experiments.aerial.rl.depth_predictor import DepthMinPredictor

        return cls(DepthMinPredictor.from_checkpoint(path, device=device), device)

    def act(self, obs) -> np.ndarray:
        from experiments.aerial.rl.goal_features import goal_rel_from_obs

        self.predictor.reset()
        depth = self.predictor.predict_depth(obs)
        if depth is None:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mean_m, left, right = measured_clearances(depth)
        return stick_target(mean_m, left, right, goal_rel_from_obs(obs))


class ClearanceStick(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, clearance_m: torch.Tensor, goal_rel: torch.Tensor) -> torch.Tensor:
        code = torch.log1p(clearance_m.clamp(min=0.0, max=80.0))
        goal = torch.cat([goal_rel[:, 0:2], goal_rel[:, 3:4]], dim=-1)
        raw = self.net(torch.cat([code, goal], dim=-1))
        dx = (torch.tanh(raw[:, 0:1]) + 1.0) * 0.5
        dyaw = torch.tanh(raw[:, 1:2]) * YAW_LIMIT
        zeros = torch.zeros_like(dx)
        return torch.cat([dx, zeros, zeros, dyaw], dim=-1)


class ClearanceStickDeploy:
    """Flight policy: current picture clearance, then this frame's goal direction."""

    def __init__(self, view: nn.Module, stick: ClearanceStick, device: str, near: nn.Module | None = None) -> None:
        self.view = view
        self.stick = stick
        self.device = device
        self.near = near

    @classmethod
    def load(cls, spec: str, device: str) -> "ClearanceStickDeploy":
        from experiments.aerial.rl.clearance_view import ClearanceView
        from experiments.aerial.rl.near_stop import NearStop

        parts = spec.split(",")
        view_path, stick_path = parts[0], parts[1]
        view = ClearanceView().to(device)
        view.load_state_dict(torch.load(view_path, map_location=device)["model"])
        view.eval()
        stick = ClearanceStick().to(device)
        stick.load_state_dict(torch.load(stick_path, map_location=device)["model"])
        stick.eval()
        near = None
        if len(parts) > 2 and parts[2]:
            near = NearStop().to(device)
            near.load_state_dict(torch.load(parts[2], map_location=device)["model"])
            near.eval()
        return cls(view, stick, device, near)

    def act(self, obs) -> np.ndarray:
        from experiments.aerial.rl.goal_features import goal_rel_from_obs

        rgb = torch.as_tensor(np.asarray(obs.rgb))
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        goal = torch.as_tensor(goal_rel_from_obs(obs), dtype=torch.float32).view(1, 4)
        with torch.no_grad():
            code = torch.expm1(self.view(rgb.to(self.device)))
            action = self.stick(code, goal.to(self.device))[0].detach().cpu().numpy()
        fwd, left, right = (float(code[0, 0]), float(code[0, 1]), float(code[0, 2]))
        self.last_logit = None
        if self.near is not None:
            device = next(self.near.parameters()).device
            with torch.no_grad():
                self.last_logit = float(self.near.logit(rgb.to(device)).reshape(-1)[0].item())
            if self.last_logit > 0.0:
                action[0] = 0.0
        elif fwd < 12.0 and not _opening(fwd, left, right):
            action[0] = 0.0
        return action
