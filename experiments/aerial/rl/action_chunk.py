"""Predict the next few sticks from the current picture and run them in order.

One matched frame then carries the aircraft through the yaw change, instead of
asking a new picture to be recognized at every step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.aerial.rl.gap_passage import LIMITS


class ActionChunkNet(nn.Module):
    def __init__(self, horizon: int = 8) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat = int(self.conv(torch.zeros(1, 3, 224, 224)).view(1, -1).shape[1])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.horizon * 4),
        )
        self.register_buffer(
            "limits", torch.tensor(LIMITS, dtype=torch.float32), persistent=True
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = rgb
        if x.dtype == torch.uint8:
            x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        raw = self.fc(self.conv(x.float() / 255.0))
        out = torch.tanh(raw).view(-1, self.horizon, 4) * self.limits
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.state_dict(), "horizon": self.horizon}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ActionChunkNet":
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device)
        model = cls(horizon=int(payload.get("horizon", 8)))
        model.load_state_dict(payload["net"])
        model.to(device)
        model.eval()
        return model


class ActionChunkRunner:
    def __init__(self, net: ActionChunkNet) -> None:
        self.net = net
        self.queue: list[np.ndarray] = []

    def reset(self) -> None:
        self.queue = []

    @torch.no_grad()
    def next_action(self, rgb: np.ndarray) -> np.ndarray:
        if not self.queue:
            batch = torch.as_tensor(np.asarray(rgb), device=self.net.limits.device)
            chunk = self.net(batch).squeeze(0).detach().cpu().numpy()
            self.queue = [chunk[i].astype(np.float64) for i in range(chunk.shape[0])]
        return self.queue.pop(0)
