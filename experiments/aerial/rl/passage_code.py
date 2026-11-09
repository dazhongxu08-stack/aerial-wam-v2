"""Two image codes: open street, and the wall in front of route 1.

The street code leaves the frozen actor alone. The wall code starts the
recorded passage and keeps those sticks until the south-street handoff,
without asking the picture to be recognized again on every degree of yaw.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PassageEncoder(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
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
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flat, dim))
        self.street = nn.Parameter(torch.randn(dim))
        self.wall = nn.Parameter(torch.randn(dim))

    def embed(self, rgb: torch.Tensor) -> torch.Tensor:
        x = rgb
        if x.dtype == torch.uint8:
            x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        return F.normalize(self.fc(self.conv(x.float() / 255.0)), dim=-1)

    def proto(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(self.street, dim=0), F.normalize(self.wall, dim=0)

    def scores(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embed(rgb)
        street, wall = self.proto()
        return z @ street, z @ wall


class LearnedPassage(nn.Module):
    """Each frame picks street, wall, or gap. Only wall and gap replace the stick."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = PassageEncoder()
        self.gap = nn.Parameter(torch.randn(32))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(inplace=True), nn.Linear(64, 4))
        from experiments.aerial.rl.gap_passage import LIMITS

        self.register_buffer("limits", torch.tensor(LIMITS, dtype=torch.float32))
        self.margin = 0.05

    def scores(self, rgb: torch.Tensor):
        z = self.encoder.embed(rgb)
        street, wall = self.encoder.proto()
        gap = F.normalize(self.gap, dim=0)
        return z, z @ street, z @ wall, z @ gap

    def stick(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(z)) * self.limits

    def reset(self) -> None:
        return None

    @torch.no_grad()
    def maybe(self, rgb: np.ndarray):
        batch = torch.as_tensor(np.asarray(rgb), device=self.limits.device)
        z, ss, sw, sg = self.scores(batch)
        ss_v, sw_v, sg_v = float(ss[0]), float(sw[0]), float(sg[0])
        if max(sw_v, sg_v) <= ss_v + float(self.margin):
            return None
        code = "wall" if sw_v >= sg_v else "gap"
        action = self.stick(z).squeeze(0).detach().cpu().numpy().astype(np.float64)
        return action, code

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.state_dict(), "margin": float(self.margin)}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "LearnedPassage":
        model = cls()
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device)
        model.load_state_dict(payload["net"])
        model.margin = float(payload.get("margin", 0.05))
        model.to(device)
        model.eval()
        return model


class PassagePlayer:
    """Play the recorded passage after the wall code wins. Street code returns None."""

    def __init__(
        self,
        encoder: PassageEncoder,
        embeds: np.ndarray,
        actions: np.ndarray,
        release_idx: int,
        margin: float,
    ) -> None:
        self.encoder = encoder
        self.embeds = np.asarray(embeds, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float64)
        self.release_idx = int(release_idx)
        self.margin = float(margin)
        self.idx: int | None = None
        self.steps = 0

    def reset(self) -> None:
        self.idx = None
        self.steps = 0

    @torch.no_grad()
    def wall_open(self, rgb: np.ndarray) -> bool:
        batch = torch.as_tensor(np.asarray(rgb), device=self.encoder.street.device)
        z = self.encoder.embed(batch).squeeze(0)
        street, wall = self.encoder.proto()
        return float(z @ wall) > float(z @ street) + self.margin

    @torch.no_grad()
    def maybe(self, rgb: np.ndarray):
        if self.idx is not None:
            if self.idx >= self.release_idx or self.idx >= len(self.actions):
                self.idx = None
                return None
            action = self.actions[self.idx]
            self.idx += 1
            self.steps += 1
            return action
        batch = torch.as_tensor(np.asarray(rgb), device=self.encoder.street.device)
        z = self.encoder.embed(batch).squeeze(0).detach().cpu().numpy()
        street, wall = self.encoder.proto()
        street_n = street.detach().cpu().numpy()
        wall_n = wall.detach().cpu().numpy()
        if float(z @ wall_n) <= float(z @ street_n) + self.margin:
            return None
        sims = self.embeds @ z.astype(np.float32)
        self.idx = int(np.argmax(sims))
        action = self.actions[self.idx]
        self.idx += 1
        self.steps += 1
        return action

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "embeds": self.embeds,
                "actions": self.actions,
                "release_idx": self.release_idx,
                "margin": self.margin,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PassagePlayer":
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device)
        encoder = PassageEncoder()
        encoder.load_state_dict(payload["encoder"])
        encoder.to(device)
        encoder.eval()
        return cls(
            encoder,
            payload["embeds"],
            payload["actions"],
            int(payload["release_idx"]),
            float(payload["margin"]),
        )


class LatchedStick:
    """Open only on the west-wall code. Then each frame the head outputs the stick.

    Route 0 never shows that code, so the head is never asked to fly the canyon.
    """

    def __init__(self, gate: PassagePlayer, head) -> None:
        self.gate = gate
        self.head = head
        self.latched = False
        self.steps = 0

    def reset(self) -> None:
        self.latched = False
        self.steps = 0

    def maybe(self, rgb: np.ndarray):
        if not self.latched:
            if not self.gate.wall_open(rgb):
                return None
            self.latched = True
        if self.steps >= self.gate.release_idx:
            return None
        self.steps += 1
        action = self.head.residual(rgb)
        return action, self.steps
