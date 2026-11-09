"""Image residual for the route-1 side passage.

The frozen actor barely changes its stick when the picture changes. This head
reads the current frame and adds a correction. On the open-street frames it is
trained to add nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.aerial.rl.gap_passage import LIMITS


class VisualResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.register_buffer(
            "limits", torch.tensor(LIMITS, dtype=torch.float32), persistent=True
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: uint8 or float, NHWC or NCHW. Returns a correction inside the body box.
        x = rgb
        if x.dtype == torch.uint8:
            x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        return torch.tanh(self.net(x)) * self.limits

    @torch.no_grad()
    def residual(self, rgb: np.ndarray) -> np.ndarray:
        device = self.limits.device
        batch = torch.as_tensor(np.asarray(rgb), device=device)
        out = self.forward(batch)
        return out.squeeze(0).detach().cpu().numpy().astype(np.float64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "VisualResidual":
        model = cls()
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device)
        model.load_state_dict(payload["net"])
        model.to(device)
        model.eval()
        return model
