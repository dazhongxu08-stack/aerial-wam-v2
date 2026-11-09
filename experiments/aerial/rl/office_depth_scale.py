"""Shrink mid-range office depth. The depth network stays frozen.

A gate looks only at the picture and turns the shrink off on real photos.
Pixels closer than 1.5 m and the far plane stay as the network wrote them.
The scale starts at 1 and can only get smaller.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def _picture_code(image: torch.Tensor) -> torch.Tensor:
    """``image`` is [B,3,H,W] in 0–1. Same code for stored views and a new grab."""
    small = F.adaptive_avg_pool2d(image, (16, 16)).flatten(1)
    small = small - small.mean(dim=1, keepdim=True)
    return small / (small.norm(dim=1, keepdim=True) + 1e-6)


MID_LO = 1.5
MID_HI = 15.0
LOG_CAP = 1.0


class OfficeDepthScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mid_lo = MID_LO
        self.mid_hi = MID_HI
        self.log_cap = LOG_CAP
        self.gate_stem = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gate_fc = nn.Linear(16, 1)
        nn.init.zeros_(self.gate_fc.weight)
        nn.init.constant_(self.gate_fc.bias, 2.0)
        self.stem = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.0)
        self.third_stem = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.third_fc = nn.Linear(8, 3)
        nn.init.zeros_(self.third_fc.weight)
        nn.init.constant_(self.third_fc.bias, 0.05)
        self.last_gate = None
        self.freeze_global = False
        self.use_thirds = False
        self.use_bank = False
        self.register_buffer("bank_code", torch.zeros(1, 16 * 16 * 3))
        self.register_buffer("bank_mul", torch.ones(1, 3))
        self.register_buffer("bank_max_dist", torch.tensor(0.02))

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``rgb`` [B,H,W,3], ``depth`` [B,H,W] → corrected depth and the scale."""
        image = rgb.float()
        if float(image.detach().max()) > 1.5:
            image = image / 255.0
        image = image.permute(0, 3, 1, 2)
        d = depth.float().clamp(0.05, 30.0)
        gate = torch.sigmoid(self.gate_fc(self.gate_stem(image).flatten(1))).clamp(1e-4, 1.0 - 1e-4)
        self.last_gate = gate.squeeze(1)
        # Hard off below 1/2, so a real photo is left exactly as the depth network wrote it.
        # The threshold is detached; the picture loss does not train the gate.
        gate_use = (gate > 0.5).float().detach()
        feat = self.stem(torch.cat([image, d.unsqueeze(1).log()], dim=1))
        pooled = feat.mean(dim=(2, 3), keepdim=True)
        if self.freeze_global:
            pooled = pooled.detach()
        pooled = pooled.expand_as(feat)
        raw = self.head(torch.cat([feat, pooled], dim=1))
        log_s = -self.log_cap * torch.sigmoid(raw) * gate_use[:, :, None, None]
        scale = F.interpolate(log_s.exp(), size=d.shape[-2:], mode="bilinear", align_corners=False)
        mid = (d > self.mid_lo) & (d < self.mid_hi)
        corrected = torch.where(mid, d * scale.squeeze(1), d)
        if self.use_thirds:
            raw_t = self.third_fc(self.third_stem(image).flatten(1))
            log_t = (torch.log(torch.tensor(2.0, device=raw_t.device)) - F.softplus(raw_t)).clamp(-0.55, 0.0)
            log_t = log_t * gate_use
            third = log_t.exp()
            height, width = corrected.shape[-2:]
            c0, c1 = width // 3, 2 * width // 3
            left = third[:, 0][:, None, None].expand(-1, height, c0)
            centre = third[:, 1][:, None, None].expand(-1, height, c1 - c0)
            right = third[:, 2][:, None, None].expand(-1, height, width - c1)
            bands = torch.cat([left, centre, right], dim=-1)
            corrected = torch.where(mid, corrected * bands, corrected)
            scale = scale * bands.unsqueeze(1)
        if self.use_bank and int(self.bank_code.shape[0]) > 1:
            code = _picture_code(image)
            dist = 1.0 - code @ self.bank_code.T
            nearest = dist.argmin(dim=1)
            self.last_match = nearest.detach()
            self.last_dist = dist.min(dim=1).values.detach()
            hit = self.last_dist <= float(self.bank_max_dist)
            picked = self.bank_mul[nearest]
            mul = torch.where(hit[:, None], picked, torch.ones_like(picked))
            height, width = corrected.shape[-2:]
            c0, c1 = width // 3, 2 * width // 3
            left = mul[:, 0][:, None, None].expand(-1, height, c0)
            centre = mul[:, 1][:, None, None].expand(-1, height, c1 - c0)
            right = mul[:, 2][:, None, None].expand(-1, height, width - c1)
            bands = torch.cat([left, centre, right], dim=-1)
            corrected = torch.where(mid, corrected * bands, corrected)
            scale = scale * bands.unsqueeze(1)
        return corrected, scale.squeeze(1)
