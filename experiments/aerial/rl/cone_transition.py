"""Next-frame depth cones from the current cones and the body action.

The frozen encoder does not separate left from right, so the cones that
imagination scores come from the image depth head. This module only predicts
how those three numbers change after an action. It does not touch the RSSM.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn


class ConeTransition(nn.Module):
    """Map ``(log cones, action)`` to the next three cones, in meters."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        # Zero residual: with no evidence yet, the next cones stay the current ones.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, cones: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        log_c = torch.log1p(cones.clamp(min=0.0))
        delta = self.net(torch.cat([log_c, action], dim=-1))
        return torch.expm1(log_c + delta).clamp(min=0.05, max=80.0)

    @torch.no_grad()
    def predict_np(self, cones: np.ndarray, action: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        c = torch.as_tensor(np.asarray(cones, dtype=np.float32).reshape(1, 3), device=device)
        a = torch.as_tensor(np.asarray(action, dtype=np.float32).reshape(1, 4), device=device)
        return self.forward(c, a).squeeze(0).float().cpu().numpy()


def column_profile(depth: np.ndarray, *, center_frac: float = 0.5) -> np.ndarray:
    """Per-column min depth on the same center-row band as the left/right cones."""
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"column_profile expects 2-D depth, got {d.shape}")
    h, w = d.shape
    dh = max(1, int(h * float(np.clip(center_frac, 0.05, 1.0))))
    r0 = (h - dh) // 2
    band = d[r0 : r0 + dh]
    valid = np.isfinite(band) & (band > 0.05)
    filled = np.where(valid, band, np.inf)
    mins = filled.min(axis=0)
    mins[~np.isfinite(mins)] = np.nan
    return mins.astype(np.float64)


def stamp_column_profile(obs: object, predictor: object) -> None:
    """Write this step's image-depth column profile onto ``obs.info``.

    Reads the depth map already produced by the predictor. Does not run the
    depth head again.
    """
    info = getattr(obs, "info", None)
    if not isinstance(info, dict):
        return
    depth = getattr(predictor, "last_depth", None)
    if depth is None:
        info.pop("depth_profile_pred", None)
        return
    info["depth_profile_pred"] = column_profile(depth)


def yaw_shift_profile(profile: np.ndarray, dyaw_rad: float, *, hfov_deg: float = 90.0) -> np.ndarray:
    """Shift a column depth profile by one yaw step.

    ``profile[c]`` is the center-band depth at column ``c`` (left = 0). A
    positive ``dyaw_rad`` is the same sign as an increase in logged yaw: the
    view looks toward the old image-right. Columns that rotate in from outside
    the field of view are NaN.
    """
    old = np.asarray(profile, dtype=np.float64).reshape(-1)
    w = int(old.shape[0])
    shift_px = -math.degrees(float(dyaw_rad)) / float(hfov_deg) * w
    src = np.arange(w, dtype=np.float64) - shift_px
    return np.interp(src, np.arange(w, dtype=np.float64), old, left=np.nan, right=np.nan)


def cones_from_column_profile(profile: np.ndarray) -> np.ndarray:
    """Forward / left / right mins from a center-band column profile."""
    row = np.asarray(profile, dtype=np.float64).reshape(-1)
    w = int(row.shape[0])
    mid = w // 2
    dw = max(1, int(w * 0.5))
    c0 = (w - dw) // 2

    def vmin(sl: np.ndarray) -> float:
        v = sl[np.isfinite(sl) & (sl > 0.05)]
        return float(np.min(v)) if v.size else float("nan")

    return np.asarray([vmin(row[c0 : c0 + dw]), vmin(row[:mid]), vmin(row[mid:])], dtype=np.float64)


def cones_from_obs(obs: object) -> Optional[np.ndarray]:
    """Read the image depth head's three cones off an observation, if present."""
    info = getattr(obs, "info", None) or {}
    raw = info.get("depth_cones_pred") if isinstance(info, dict) else None
    if not isinstance(raw, dict):
        return None
    vals = [raw.get("forward"), raw.get("left"), raw.get("right")]
    if any(v is None or not np.isfinite(float(v)) for v in vals):
        return None
    return np.asarray([float(v) for v in vals], dtype=np.float64)
