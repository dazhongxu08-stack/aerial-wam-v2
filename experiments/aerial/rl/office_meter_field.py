"""Local scale on a frozen metric depth map.

The depth network stays fixed. The scale is read from this picture only:
local texture, plus a summary of the whole frame, so two surfaces the
frozen network places at the same depth can still take different scales.
Pixels under 1.5 m, and the far plane, stay as the frozen network wrote
them. An untrained field reproduces that map.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficeMeterField(nn.Module):
    NEAR_M = 1.5
    FAR_M = 15.0
    LOG_LO = math.log(0.45)
    LOG_HI = math.log(1.25)

    def __init__(self) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.head.weight)
        span = self.LOG_HI - self.LOG_LO
        unit = -self.LOG_LO / span
        nn.init.constant_(self.head.bias, math.log(unit / (1.0 - unit)))
        self.head.bias.requires_grad_(False)
        for param in self.local.parameters():
            if param.ndim == 1:
                nn.init.constant_(param, 0.1)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``rgb`` [B,H,W,3], frozen ``depth`` [B,H,W] → corrected depth and scale."""
        image = rgb.float()
        if float(image.detach().max()) > 1.5:
            image = image / 255.0
        image = image.permute(0, 3, 1, 2)
        d = depth.float().clamp(0.05, 30.0)
        feat = self.local(image)
        scene = feat.mean(dim=(2, 3), keepdim=True).expand_as(feat)
        raw = self.head(torch.cat([feat, scene], dim=1))
        span = self.LOG_HI - self.LOG_LO
        log_s = self.LOG_LO + span * torch.sigmoid(raw)
        scale = F.interpolate(log_s.exp(), size=d.shape[-2:], mode="bilinear", align_corners=False)
        scale = scale.squeeze(1)
        mid = (d >= self.NEAR_M) & (d < self.FAR_M)
        corrected = torch.where(mid, d * scale, d)
        return corrected, scale
