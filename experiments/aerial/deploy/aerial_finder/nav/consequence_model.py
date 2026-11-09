"""Inference-only copy of the consequence model and its candidate machinery.

Kept numerically identical to ``experiments/aerial/rl/consequence_model.py``,
``collect_consequence_dataset.sample_macro`` and the MPC eval's candidate
families, so training checkpoints load and score exactly as in evaluation.
Training-side changes to those must be mirrored here.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LIMITS = np.array([1.0, 0.4, 0.4, 0.3141592653589793], dtype=np.float32)
DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 100.0
DGOAL_SCALE_M = 10.0
DEFAULT_KEEP_RADIUS_M = 0.5
BACK_LIMIT = 0.6
FAMILIES = (
    "fwd", "yaw_fwd", "yaw_fwd", "yaw_fwd", "yaw_fwd", "goal_turn_fwd",
    "strafe", "strafe", "climb", "smooth_random", "yaw_only",
)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def depth_input(depth56: np.ndarray) -> np.ndarray:
    d = np.asarray(depth56, dtype=np.float32)
    d = np.where(np.isfinite(d) & (d > 0), d, DEPTH_MAX_M)
    d = np.clip(d, DEPTH_MIN_M, DEPTH_MAX_M)
    return (np.log(d / DEPTH_MIN_M) / math.log(DEPTH_MAX_M / DEPTH_MIN_M)).astype(np.float32)


def pool_min(depth: np.ndarray, out: int = 56) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    h, w = d.shape
    fh, fw = h // out, w // out
    d = d[: fh * out, : fw * out].reshape(out, fh, out, fw)
    return d.min(axis=(1, 3)).astype(np.float16)


def goal_features(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    dx, dy = g[0] - s[0], g[1] - s[1]
    b = math.atan2(dy, dx) - float(s[6])
    dist = math.hypot(dx, dy)
    return np.array([math.sin(b), math.cos(b), math.log1p(dist) / 5.0, (g[2] - s[2]) / 20.0], dtype=np.float32)


def goal_bearing_body(pos: np.ndarray, yaw: float, goal: np.ndarray) -> float:
    d = np.asarray(goal[:2], dtype=np.float64) - np.asarray(pos[:2], dtype=np.float64)
    return wrap(math.atan2(d[1], d[0]) - yaw)


def history_inputs(depth_seq: List[np.ndarray], act_seq: List[np.ndarray], frame_offsets: Tuple[int, ...],
                   hist_actions: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(depth_seq)
    depth = np.stack([depth_input(depth_seq[max(0, n - 1 - int(o))]) for o in frame_offsets])
    acts = np.zeros((int(hist_actions), 4), np.float32)
    past = [np.asarray(a, np.float32) for a in act_seq[-int(hist_actions):]] if hist_actions else []
    if past:
        acts[len(acts) - len(past):] = np.stack(past)
    return depth.astype(np.float32), acts


class ConsequenceModel(nn.Module):
    def __init__(self, horizon: int = 10, hidden: int = 256, dropout: float = 0.0,
                 frame_offsets: Tuple[int, ...] = (0,), hist_actions: int = 0,
                 sem_channels: int = 0, env_dim: int = 0, env_hidden: int = 64,
                 env_drop: float = 0.0) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.hidden = int(hidden)
        self.frame_offsets = tuple(int(o) for o in frame_offsets)
        self.hist_actions = int(hist_actions)
        self.sem_channels = int(sem_channels)
        self.env_dim = int(env_dim)
        self.env_hidden = int(env_hidden) if self.env_dim else 0
        self.env_drop = float(env_drop)
        self.enc = nn.Sequential(
            nn.Conv2d(len(self.frame_offsets) + self.sem_channels, 32, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.SiLU(),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(128 * 4 * 4, hidden), nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        if self.env_dim:
            self.env_proj = nn.Sequential(nn.LayerNorm(self.env_dim), nn.Dropout(float(dropout)),
                                          nn.Linear(self.env_dim, self.env_hidden), nn.SiLU())
        self.ctx = nn.Linear(hidden + 4 + 4 + 4 * self.hist_actions + (1 if self.sem_channels else 0)
                             + self.env_hidden, hidden)
        self.act_in = nn.Linear(4, 64)
        self.gru = nn.GRU(64, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 5 + (1 if self.sem_channels else 0)))

    def forward(self, depth: torch.Tensor, goal: torch.Tensor, prev_action: torch.Tensor, actions: torch.Tensor,
                hist_act: Optional[torch.Tensor] = None, sem: Optional[torch.Tensor] = None,
                keep_radius: Optional[torch.Tensor] = None, env: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        lim = torch.as_tensor(LIMITS, device=actions.device)
        x = depth.unsqueeze(1) if depth.dim() == 3 else depth
        if self.sem_channels:
            if sem is None:
                sem = torch.zeros_like(x[:, :1])
            x = torch.cat([x, sem.unsqueeze(1) if sem.dim() == 3 else sem], dim=1)
        f = self.enc(x)
        ctx = [f, goal, prev_action / lim]
        if self.hist_actions:
            ctx.append((hist_act / lim).flatten(1))
        if self.sem_channels:
            if keep_radius is None:
                keep_radius = torch.full((x.shape[0],), DEFAULT_KEEP_RADIUS_M, device=x.device)
            ctx.append(keep_radius.reshape(-1, 1).to(f.dtype))
        if self.env_dim:
            ctx.append(f.new_zeros((f.shape[0], self.env_hidden)) if env is None else self.env_proj(env.to(f.dtype)))
        h0 = torch.tanh(self.ctx(torch.cat(ctx, dim=-1)))
        out, _ = self.gru(F.silu(self.act_in(actions / lim)), h0.unsqueeze(0))
        y = self.head(out)
        res = {"hazard_logit": y[..., 0], "log_cones": y[..., 1:4], "dgoal": y[..., 4] * DGOAL_SCALE_M}
        if self.sem_channels:
            res["keepout_logit"] = y[..., 5]
        return res


def survival(hazard_logit: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(1.0 - torch.sigmoid(hazard_logit), dim=-1)


def load_checkpoint(path: Path | str, device: str = "cpu") -> ConsequenceModel:
    from experiments.aerial.deploy.rtw_blob import unpack

    p = unpack(path)
    if p is None:
        p = torch.load(str(path), map_location="cpu")
    sd = p["model"]
    if p.get("arch") != "dropout_v2":
        sd = {k.replace("enc.9.", "enc.10.", 1) if k.startswith("enc.9.") else k: v for k, v in sd.items()}
    m = ConsequenceModel(horizon=int(p["horizon"]), hidden=int(p["hidden"]),
                         frame_offsets=tuple(p.get("frame_offsets", (0,))),
                         hist_actions=int(p.get("hist_actions") or 0), sem_channels=int(p.get("sem_channels") or 0),
                         env_dim=int(p.get("env_dim") or 0), env_hidden=int(p.get("env_hidden", 64) or 64),
                         env_drop=float(p.get("env_drop", 0.0)))
    m.load_state_dict(sd)
    return m.to(device).eval()


def select_candidate(surv: np.ndarray, pred_progress: np.ndarray, allowed: np.ndarray, surv_min: float) -> int:
    surv = np.asarray(surv, np.float64)
    allowed = np.asarray(allowed, bool)
    ok = allowed & (surv >= float(surv_min))
    if ok.any():
        return int(np.argmax(np.where(ok, np.asarray(pred_progress, np.float64), -np.inf)))
    return int(np.argmax(np.where(allowed, surv, -np.inf)))


def select_open_candidate(surv: np.ndarray, pred_progress: np.ndarray, forward_m: np.ndarray,
                          allowed: np.ndarray, surv_min: float, margin_m: float = 0.15) -> int:
    """Among survivors, the action whose predicted forward clearance is largest.

    A gap smaller than ``margin_m`` falls back to progress, so a tie stays straight.
    """
    surv = np.asarray(surv, np.float64)
    allowed = np.asarray(allowed, bool)
    forward = np.asarray(forward_m, np.float64)
    ok = allowed & (surv >= float(surv_min)) & np.isfinite(forward)
    if not ok.any():
        return select_candidate(surv, pred_progress, allowed, surv_min)
    best = float(np.max(np.where(ok, forward, -np.inf)))
    tied = ok & (forward >= best - float(margin_m))
    prog = np.where(tied, np.asarray(pred_progress, np.float64), -np.inf)
    return int(np.argmax(prog))


def altitude_allowed(candidates: np.ndarray, state: np.ndarray, goal: np.ndarray, *, goal_band_m: float = 1.0,
                     net_tol_m: float = 0.5) -> np.ndarray:
    c = np.asarray(candidates, dtype=np.float64)
    net_dz = c[..., 2].sum(axis=-1)
    dz_goal = float(np.asarray(goal)[2]) - float(np.asarray(state)[2])
    ok = np.ones(net_dz.shape, dtype=bool)
    if dz_goal <= goal_band_m:
        ok &= net_dz <= net_tol_m
    if dz_goal >= -goal_band_m:
        ok &= net_dz >= -net_tol_m
    return ok


def altitude_band_allowed(candidates: np.ndarray, z_now: float, z_lo: float, z_hi: float, prefix: int,
                          return_per_step: float = 0.0) -> np.ndarray:
    c = np.asarray(candidates, dtype=np.float64)
    z_after = float(z_now) + c[:, :int(prefix), 2].sum(axis=-1)
    ok = (z_after >= z_lo) & (z_after <= z_hi)
    back = float(return_per_step) * int(prefix)
    if z_now > z_hi:
        ok |= z_after <= z_now - back
    elif z_now < z_lo:
        ok |= z_after >= z_now + back
    return ok


def visible_motion_allowed(candidates: np.ndarray, prefix: int, max_deg: float, min_speed: float = 0.1) -> np.ndarray:
    c = np.asarray(candidates, dtype=np.float64)[:, :int(prefix)]
    vx, vy = c[..., 0], c[..., 1]
    off = np.degrees(np.arctan2(np.abs(vy), vx))
    return ((np.hypot(vx, vy) < float(min_speed)) | (off <= float(max_deg))).all(axis=-1)


def heading_yaw_allowed(candidates: np.ndarray, bearing: float, prefix: int) -> np.ndarray:
    """True when the yaw that will actually be driven does not increase the bearing error.

    Bearing and model dyaw are clockwise-positive, so a positive step turns toward a goal
    on the right. ``yaw_fwd`` draws its sign at random; only the executed prefix counts,
    because later steps of that macro are thrown away when replanning every step.
    """
    c = np.asarray(candidates, dtype=np.float64)
    dyaw = c[:, : max(1, int(prefix)), 3].sum(axis=-1)
    err = abs(float(bearing))
    return np.abs(float(bearing) - dyaw) <= err + 1e-6


def sample_macro(rng: np.random.Generator, k: int, bearing: float, name: str) -> Tuple[str, np.ndarray]:
    lim = LIMITS.astype(np.float64)
    a = np.zeros((k, 4), dtype=np.float64)
    if name == "fwd":
        a[:, 0] = rng.uniform(0.4, 1.0)
    elif name == "yaw_fwd":
        m = int(rng.integers(1, 5))
        a[:m, 3] = rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 1.0) * lim[3]
        a[:m, 0] = rng.uniform(0.0, 0.25)
        a[m:, 0] = rng.uniform(0.4, 1.0)
    elif name == "goal_turn_fwd":
        left = bearing
        for t in range(k):
            step = float(np.clip(left, -lim[3], lim[3]))
            if abs(left) > 0.05:
                a[t, 3] = step
                a[t, 0] = rng.uniform(0.0, 0.25)
                left -= step
            else:
                a[t, 0] = rng.uniform(0.4, 1.0)
    elif name == "yaw_only":
        a[:, 3] = rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 1.0) * lim[3]
        a[:, 0] = rng.uniform(0.0, 0.2)
    elif name == "strafe":
        a[:, 1] = rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 1.0) * lim[1]
        a[:, 0] = rng.uniform(0.0, 0.4)
    elif name == "climb":
        a[:, 2] = rng.uniform(0.25, 1.0) * lim[2]
        a[:, 0] = rng.uniform(0.0, 0.5)
    elif name == "descend":
        a[:, 2] = -rng.uniform(0.25, 0.75) * lim[2]
        a[:, 0] = rng.uniform(0.0, 0.5)
    elif name == "hover":
        pass
    elif name == "spin":
        a[:, 3] = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0) * lim[3]
    elif name == "back":
        m = int(rng.integers(3, k + 1))
        a[:m, 0] = -rng.uniform(0.2, 0.6) * BACK_LIMIT
    else:
        u = np.zeros(4)
        for t in range(k):
            u = 0.7 * u + 0.3 * rng.normal(0.0, 1.0, size=4)
            a[t] = np.tanh(u) * lim
    a[:, 0] = np.clip(a[:, 0], -BACK_LIMIT if name == "back" else 0.0, lim[0])
    return name, np.clip(a, -lim, lim)


DOG_FAMILIES = ("fwd", "yaw_fwd", "yaw_fwd", "yaw_fwd", "goal_turn_fwd", "yaw_only", "smooth_random", "hover")


def sample_dog_macro(rng: np.random.Generator, k: int, bearing: float, name: str,
                     rad_per_step: float) -> Tuple[str, np.ndarray]:
    """Same macros as ``rl/collect_dog_consequence.sample_dog_macro``: actions are dog commands in the model box,
    ``rad_per_step`` the nominal heading change of a full yaw command."""
    lim = LIMITS.astype(np.float64)
    a = np.zeros((k, 4), dtype=np.float64)
    if name == "fwd":
        a[:, 0] = rng.uniform(0.4, 1.0)
    elif name == "yaw_fwd":
        m = int(rng.integers(2, k))
        a[:m, 3] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * lim[3]
        a[:m, 0] = rng.uniform(0.0, 0.3)
        a[m:, 0] = rng.uniform(0.4, 1.0)
    elif name == "goal_turn_fwd":
        m = min(k - 1, int(round(abs(bearing) / max(rad_per_step, 1e-3))))
        a[:m, 3] = math.copysign(lim[3], bearing)
        a[:m, 0] = rng.uniform(0.0, 0.3)
        a[m:, 0] = rng.uniform(0.4, 1.0)
    elif name == "yaw_only":
        a[:, 3] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * lim[3]
    elif name == "smooth_random":
        u = np.zeros(2)
        for t in range(k):
            u = 0.7 * u + 0.3 * rng.normal(0.0, 1.0, size=2)
            a[t, 0] = 0.5 * (1.0 + math.tanh(u[0]))
            a[t, 3] = math.tanh(u[1]) * lim[3]
    elif name != "hover":
        raise ValueError(f"unknown dog macro {name!r}")
    a[:, 0] = np.clip(a[:, 0], 0.0, lim[0])
    a[:, 3] = np.clip(a[:, 3], -lim[3], lim[3])
    return name, a
