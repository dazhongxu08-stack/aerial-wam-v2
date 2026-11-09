"""Read forward, left, and right clearance from the current picture.

Depth is a training label only. Forward distance uses the whole frame plus the
same narrow center band as the label, so an opening beside the nose does not
replace the near pixels in front. The side gap is left minus right.
"""
from __future__ import annotations

import torch
from torch import nn


class ClearanceView(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.fwd = nn.Linear(128, 1)
        self.level = nn.Linear(64, 1)
        self.side = nn.Linear(64, 1, bias=False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        image = _as_nchw(rgb)
        feat = self.stem(image)
        height, width = feat.shape[-2:]
        row0, row1 = height // 4, (3 * height) // 4
        band = feat[:, :, row0:row1]
        side_w = max(1, width // 3)
        left = band[..., :side_w].mean(dim=(-1, -2))
        right = band[..., width - side_w :].mean(dim=(-1, -2))
        center = band[..., side_w : width - side_w].mean(dim=(-1, -2))
        context = feat.mean(dim=(-1, -2))
        forward = self.fwd(torch.cat([center, context], dim=-1)).squeeze(-1)
        gap = self.side(left - right).squeeze(-1)
        level = self.level((left + right) * 0.5).squeeze(-1)
        return torch.stack([forward, level + 0.5 * gap, level - 0.5 * gap], dim=-1)


def _as_nchw(rgb: torch.Tensor) -> torch.Tensor:
    image = rgb.to(dtype=torch.float32)
    if image.dim() == 4 and image.shape[-1] in (1, 3):
        image = image.permute(0, 3, 1, 2)
    if float(image.detach().max()) > 1.5:
        image = image / 255.0
    return image
