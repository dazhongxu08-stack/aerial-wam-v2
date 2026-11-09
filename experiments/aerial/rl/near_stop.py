"""Stop when the front is near and neither side is an opening.

The clearance distance head stays unchanged. This head only answers whether
the current picture is a blocked front.
"""
from __future__ import annotations

import torch
from torch import nn


class NearStop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Linear(32, 1)

    def logit(self, rgb: torch.Tensor) -> torch.Tensor:
        image = rgb.to(dtype=torch.float32)
        if image.dim() == 4 and image.shape[-1] in (1, 3):
            image = image.permute(0, 3, 1, 2)
        if float(image.detach().max()) > 1.5:
            image = image / 255.0
        return self.head(self.stem(image).mean(dim=(-1, -2))).squeeze(-1)

    def blocked(self, rgb: torch.Tensor) -> bool:
        device = next(self.parameters()).device
        with torch.no_grad():
            return bool(self.logit(rgb.to(device)).reshape(-1)[0].item() > 0.0)
