"""Policy that reads a metric depth map and the current goal.

The depth map comes from the depth head. This network does not choose among
forward, stop, and yaw.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

YAW_LIMIT = 0.3141592653589793


class DepthPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 + 3, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, depth: torch.Tensor, goal_rel: torch.Tensor) -> torch.Tensor:
        image = depth
        if image.dim() == 3:
            image = image.unsqueeze(1)
        image = torch.log1p(image.clamp(min=0.0, max=80.0))
        feat = self.stem(image).mean(dim=(-1, -2))
        goal = torch.cat([goal_rel[:, 0:2], goal_rel[:, 3:4]], dim=-1)
        raw = self.head(torch.cat([feat, goal], dim=-1))
        dx = (torch.tanh(raw[:, 0:1]) + 1.0) * 0.5
        dyaw = torch.tanh(raw[:, 1:2]) * YAW_LIMIT
        zeros = torch.zeros_like(dx)
        return torch.cat([dx, zeros, zeros, dyaw], dim=-1)


def resize_depth(depth: np.ndarray, size: int = 64) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    tensor = torch.as_tensor(arr).view(1, 1, arr.shape[0], arr.shape[1])
    out = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return out.view(size, size).cpu().numpy()


class DepthPolicyDeploy:
    """Current frame through the depth head, then a network on that map and the goal."""

    def __init__(self, predictor, policy: DepthPolicy, device: str) -> None:
        self.predictor = predictor
        self.policy = policy
        self.device = device

    @classmethod
    def load(cls, spec: str, device: str) -> "DepthPolicyDeploy":
        from experiments.aerial.rl.depth_predictor import DepthMinPredictor

        depth_path, policy_path = spec.split(",")
        predictor = DepthMinPredictor.from_checkpoint(depth_path, device=device)
        policy = DepthPolicy().to(device)
        policy.load_state_dict(torch.load(policy_path, map_location=device)["model"])
        policy.eval()
        return cls(predictor, policy, device)

    def act(self, obs) -> np.ndarray:
        from experiments.aerial.rl.goal_features import goal_rel_from_obs

        self.predictor.reset()
        depth = self.predictor.predict_depth(obs)
        if depth is None:
            return np.zeros(4, dtype=np.float32)
        image = torch.as_tensor(resize_depth(depth)).view(1, 1, 64, 64)
        goal = torch.as_tensor(goal_rel_from_obs(obs), dtype=torch.float32).view(1, 4)
        with torch.no_grad():
            action = self.policy(image.to(self.device), goal.to(self.device))
        return action[0].detach().cpu().numpy()
