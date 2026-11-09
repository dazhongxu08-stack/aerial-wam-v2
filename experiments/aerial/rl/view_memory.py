"""Short view memory for the survival head.

A depth frame is compressed to a latent, and a recurrent state reads the
sequence. Training asks that state to redraw the current depth and the depth
from ``span`` steps earlier, so an obstacle that has left the picture can
remain in the state. The decoder is not used when scoring a candidate.
Progress does not read this state.

The span of 10 steps is the setting that peaked in the later of the two
source papers. 20 steps was best in the earlier one; pass ``span`` to use it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SPAN = 10
LATENT = 256
LOOKBACK = 4
NEAR_CENTER_M = 4.0


def recent_past_index(depth_m: np.ndarray, lookback: int = LOOKBACK, near_m: float = NEAR_CENTER_M) -> np.ndarray:
    """Latest earlier frame whose center is still near, else the previous frame.

    ``depth_m`` is ``[S,H,W]`` in meters. Index ``t`` is ``-1`` at the first frame.
    A center nearer than ``near_m`` within ``lookback`` steps wins over a fixed
    offset, so a wall that just slid out of the middle is kept and the open
    frame from the start of the flight is not.
    """
    depth = np.asarray(depth_m, np.float32)
    s, h, w = depth.shape
    band = depth[:, h // 4:h - h // 4, w // 3:w - w // 3]
    valid = np.isfinite(band) & (band > 0.05)
    filled = np.where(valid, band, np.inf)
    mins = filled.reshape(s, -1).min(axis=1)
    mins = np.where(np.isfinite(mins), mins, 100.0)
    idx = np.full(s, -1, np.int64)
    for t in range(1, s):
        pick = t - 1
        for k in range(1, min(int(lookback), t) + 1):
            if mins[t - k] <= float(near_m):
                pick = t - k
                break
        idx[t] = pick
    return idx


class ViewMemory(nn.Module):
    def __init__(self, latent: int = LATENT, span: int = SPAN) -> None:
        super().__init__()
        self.latent = int(latent)
        self.span = int(span)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, self.latent * 2),
        )
        self.lstm = nn.LSTM(self.latent, self.latent, batch_first=True)
        self.to_pair = nn.Linear(self.latent, self.latent * 2)
        self.dec_fc = nn.Linear(self.latent, 128 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def encode_state(self, depth_seq: torch.Tensor) -> torch.Tensor:
        """Latent after the newest frame. ``depth_seq`` is [S,56,56] or [B,S,56,56]."""
        seq = depth_seq.unsqueeze(0) if depth_seq.dim() == 3 else depth_seq
        state, _, _ = self._states(seq)
        return state[:, -1]

    def loss(self, depth_seq: torch.Tensor, past_at: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MSE of the current frame and the chosen earlier frame, plus a small KL term.

        ``past_at`` is ``[S]`` or ``[B,S]``. Negative means that step is not
        supervised. When omitted, each step redraws the previous frame.
        ``drawn`` in the log is the pair from the newest step.
        """
        seq = depth_seq.unsqueeze(0) if depth_seq.dim() == 3 else depth_seq
        state, mu, logvar = self._states(seq)
        b, steps, height, width = seq.shape
        if past_at is None:
            past_at = torch.full((b, steps), -1, device=seq.device, dtype=torch.long)
            if steps > 1:
                past_at[:, 1:] = torch.arange(steps - 1, device=seq.device)
        else:
            past_at = past_at.to(device=seq.device, dtype=torch.long)
            if past_at.dim() == 1:
                past_at = past_at.unsqueeze(0).expand(b, -1)
        valid = past_at >= 0
        if not bool(valid.any()):
            raise ValueError("no past frame to redraw")
        clamped = past_at.clamp(min=0)
        flat = seq.reshape(b, steps, height * width)
        past = torch.gather(flat, 1, clamped.unsqueeze(-1).expand(b, steps, height * width))
        past = past.view(b, steps, height, width)
        drawn_all = self._draw(state.reshape(b * steps, self.latent)).view(b, steps, 2, height, width)
        cur_err = (drawn_all[:, :, 0] - seq).pow(2).flatten(2).mean(-1)
        past_err = (drawn_all[:, :, 1] - past).pow(2).flatten(2).mean(-1)
        rec = cur_err[valid].mean() + past_err[valid].mean()
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        drawn = drawn_all[:, -1]
        return rec + 1e-4 * kl, {"rec": rec.detach(), "kl": kl.detach(), "drawn": drawn}

    def _states(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, _, _ = seq.shape
        mu, logvar = self.enc(seq.reshape(b * s, 1, 56, 56)).chunk(2, dim=-1)
        if self.training:
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z = mu
        state, _ = self.lstm(z.view(b, s, self.latent))
        return state, mu.view(b, s, self.latent), logvar.view(b, s, self.latent)

    def _draw(self, state: torch.Tensor) -> torch.Tensor:
        """``[B,2,56,56]``: index 0 is the current frame, index 1 is the earlier frame."""
        b = state.shape[0]
        code = self.to_pair(state).view(b * 2, self.latent)
        flat = self.dec_fc(code).view(b * 2, 128, 7, 7)
        return self.dec(flat).view(b, 2, 56, 56)
