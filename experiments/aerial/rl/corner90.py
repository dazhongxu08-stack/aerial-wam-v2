"""90° corner turn: frozen image encoder, forward and inverse heads, policy head.

The encoder is the street depth model's DINOv2 and is not updated. The forward
head predicts the next feature and the next forward cone. The inverse head
reads the action, especially yaw, back out of that change. The policy head
maps the current feature and the goal to the demonstrated action. Flight uses
only the encoder and the policy head.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments.aerial.rl.goal_features import goal_rel_body

HOLDOUT_ROUTES = (2, 6, 10)
# Route 10 never arrived; it is not a demonstration of the turn.
DEMO_SKIP = (10,)
# One control step of the demonstration yaws about 8°. A 90° corner is a
# run of those steps. The policy predicts the run and flies it before looking again.
CHUNK = 8
EXEC = 6


def demo_routes() -> list[int]:
    return [i for i in range(13) if i not in HOLDOUT_ROUTES and i not in DEMO_SKIP]


def pool_tokens(feat: torch.Tensor) -> torch.Tensor:
    """Mean-pool a ViT layer to ``[B, C]``."""
    if isinstance(feat, (tuple, list)):
        tensors = [item for item in feat if torch.is_tensor(item)]
        if not tensors:
            return pool_tokens(feat[0])
        feat = max(tensors, key=lambda item: item.numel())
    if feat.dim() == 4:
        # [B, C, H, W] or [B, N, C] with a channel-last layout is not used.
        if feat.shape[1] <= 64:
            return feat.mean(dim=(2, 3))
        return feat.mean(dim=1)
    if feat.dim() == 3:
        tokens = feat[:, 1:] if feat.shape[1] > 1 else feat
        return tokens.mean(dim=1)
    raise ValueError(f"unexpected feature shape {tuple(feat.shape)}")


class Corner90Heads(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.forward_head = nn.Sequential(
            nn.Linear(dim + 4, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
        )
        self.cone_head = nn.Sequential(
            nn.Linear(dim + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(dim + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def predict_next(self, h: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([h, action], dim=-1)
        return self.forward_head(inp), self.cone_head(inp).squeeze(-1)

    def predict_action(self, h: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
        return self.inverse_head(torch.cat([h, h_next], dim=-1))

    def chunk(self, h: torch.Tensor, goal_rel: torch.Tensor) -> torch.Tensor:
        # One rate for the whole segment. Independent steps were cancelling
        # inside the chunk, so the sum looked small while the nose hunted.
        step = self.policy_head(torch.cat([h, goal_rel], dim=-1))
        return step.unsqueeze(1).expand(-1, CHUNK, -1).contiguous()

    def act(self, h: torch.Tensor, goal_rel: torch.Tensor) -> torch.Tensor:
        return self.chunk(h, goal_rel)[:, 0]


class BearingTurnHead(nn.Module):
    """Yaw only after the goal has swung well out to the side.

    On these L demos the aircraft keeps the street heading while the goal
    bearing is still moderate, and the turn starts near 80°. The forward cone
    is still long at that moment, so the trigger is the bearing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.2))
        self.gain = nn.Parameter(torch.tensor(0.8))
        self.dx = nn.Parameter(torch.tensor(1.0))

    def forward(self, goal_rel: torch.Tensor) -> torch.Tensor:
        bearing = torch.atan2(goal_rel[:, 1], goal_rel[:, 0])
        mag = torch.relu(bearing.abs() - self.bias)
        dyaw = torch.sign(bearing) * self.gain * mag
        dyaw = dyaw.clamp(-0.31, 0.31)
        dx = self.dx.clamp(0.2, 1.0).expand_as(dyaw)
        zeros = torch.zeros_like(dyaw)
        return torch.stack([dx, zeros, zeros, dyaw], dim=-1)


def chunk_policy_loss(
    heads: Corner90Heads,
    h: torch.Tensor,
    chunk: torch.Tensor,
    goal_rel: torch.Tensor,
) -> torch.Tensor:
    """Behavior cloning of the next corner segment. Turn segments count more."""
    pred = heads.chunk(h.detach(), goal_rel.detach())
    err = (pred - chunk.detach()).abs()
    # On the approach the demonstrated yaw is about 0 even though the goal is
    # already to the side. Those steps need a heavier yaw penalty or the head
    # spends the error on a turn.
    straight = chunk.detach()[..., 3].abs().mean(dim=-1) < 0.05
    yaw_err = err[..., 3].mean(dim=-1)
    base = err[..., :3].mean(dim=(-1, -2)) + 5.0 * yaw_err
    return (base + 20.0 * straight.to(base.dtype) * yaw_err).mean()


def corner90_losses(
    heads: Corner90Heads,
    h: torch.Tensor,
    h_next: torch.Tensor,
    action: torch.Tensor,
    goal_rel: torch.Tensor,
    log_fwd_next: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Forward, inverse, cone, and behavior-cloning losses. Targets are detached."""
    h_hat, cone_hat = heads.predict_next(h, action)
    fwd = (h_hat - h_next.detach()).abs().mean()
    cone = (cone_hat - log_fwd_next.detach()).abs().mean()
    inv = heads.predict_action(h.detach(), h_next.detach())
    inv_err = (inv - action.detach()).abs()
    inv_loss = inv_err[:, :3].mean() + 5.0 * inv_err[:, 3].mean()
    pol = heads.act(h.detach(), goal_rel.detach())
    act = (pol - action.detach()).abs().mean()
    total = act + 0.1 * fwd + inv_loss + 0.1 * cone
    return {
        "loss": total,
        "act": act,
        "fwd": fwd,
        "inv": inv_loss,
        "cone": cone,
        "inv_yaw": inv[:, 3],
    }


class BearingLatch:
    """Hold the street until the goal bearing opens, then yaw onto the goal.

    On the L demonstrations the inbound heading stays put until the bearing
    reaches about 82° (a few metres before the corner). After that the turn
    keeps going until the goal is in front; forward speed drops while the
    nose is still off the exit, so the arc does not leave the intersection.
    """

    def __init__(self, open_deg: float = 82.0, rate_deg: float = 16.0, dx: float = 1.0) -> None:
        self.open_rad = math.radians(float(open_deg))
        self.rate = math.radians(float(rate_deg))
        self.dx = float(dx)
        self.latched = False

    def act_bearing(self, bearing: float) -> np.ndarray:
        b = float(bearing)
        if not self.latched and abs(b) >= self.open_rad:
            self.latched = True
        if not self.latched:
            return np.array([self.dx, 0.0, 0.0, 0.0], dtype=np.float64)
        dyaw = max(-self.rate, min(self.rate, b))
        dx = self.dx * max(0.0, math.cos(b))
        return np.array([dx, 0.0, 0.0, dyaw], dtype=np.float64)

    def act(self, rgb: np.ndarray, pos, yaw: float, goal) -> np.ndarray:
        del rgb
        grel = goal_rel_body(np.asarray(pos, dtype=np.float64), float(yaw), goal)
        bearing = math.atan2(float(grel[1]), float(grel[0]))
        return self.act_bearing(bearing)


class BearingTurnController:
    """Flies from the goal bearing only. The image encoder is not used."""

    def __init__(self, head: BearingTurnHead, device: str) -> None:
        self.head = head.eval()
        self.device = device

    def act(self, rgb: np.ndarray, pos, yaw: float, goal) -> np.ndarray:
        del rgb
        grel = goal_rel_body(np.asarray(pos, dtype=np.float64), float(yaw), goal)
        g = torch.as_tensor(grel, dtype=torch.float32, device=self.device).view(1, 4)
        with torch.no_grad():
            action = self.head(g)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float64)


class Corner90Controller:
    """Frozen encoder plus the policy head. No forward or inverse head at flight."""

    def __init__(self, heads: Corner90Heads, model, device: str, exec_horizon: int = EXEC) -> None:
        self.heads = heads.eval()
        self.model = model.eval()
        self.device = device
        self.exec_horizon = int(exec_horizon)
        self._queue: list[np.ndarray] = []
        for p in self.model.parameters():
            p.requires_grad_(False)

    @classmethod
    def load(cls, path: Path | str, model, device: str):
        payload = torch.load(str(path), map_location=device)
        if payload.get("kind") == "bearing_latch":
            return BearingLatch(
                float(payload.get("open_deg", 82.0)),
                float(payload.get("rate_deg", 16.0)),
                float(payload.get("dx", 1.0)),
            )
        if payload.get("kind") == "bearing":
            head = BearingTurnHead()
            head.load_state_dict(payload["head"])
            head.to(device)
            return BearingTurnController(head, device)
        heads = Corner90Heads(int(payload["dim"]))
        heads.load_state_dict(payload["heads"])
        heads.to(device)
        return cls(heads, model, device, int(payload.get("exec_horizon", EXEC)))

    def encode(self, rgb: np.ndarray) -> torch.Tensor:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.shape[0] != self.model.image_size or image.shape[1] != self.model.image_size:
            import cv2

            image = cv2.resize(
                image,
                (self.model.image_size, self.model.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
        x = torch.from_numpy(image).to(self.device)
        window = x.view(1, 1, *x.shape)
        frame = self.model._prep_last_frame(window)
        with torch.no_grad():
            feats, _aux = self.model.encoder(
                frame.unsqueeze(1), cam_token=None, export_feat_layers=[]
            )
            return pool_tokens(feats[-1])

    def act(self, rgb: np.ndarray, pos, yaw: float, goal) -> np.ndarray:
        if self._queue:
            return self._queue.pop(0)
        h = self.encode(rgb)
        grel = goal_rel_body(np.asarray(pos, dtype=np.float64), float(yaw), goal)
        g = torch.as_tensor(grel, dtype=torch.float32, device=self.device).view(1, 4)
        with torch.no_grad():
            chunk = self.heads.chunk(h, g).squeeze(0).detach().cpu().numpy()
        self._queue = [np.asarray(a, dtype=np.float64) for a in chunk[: self.exec_horizon]]
        return self._queue.pop(0)
