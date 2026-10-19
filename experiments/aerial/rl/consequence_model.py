"""Action-sequence -> consequence predictor (restart Stage 2).

Input is what the deployed drone has: the image depth head's map at the
current step (56x56 block-min, no GT), the goal in the body frame, and the
previous action. A GRU rolls the candidate action sequence forward and each
step emits

* a collision hazard logit (collide on this step given alive so far),
* GT forward / left / right clearance (log meters) after the step,
* change of distance to goal (meters / 10) since the start.

With ``sem_channels=1`` the model also takes a keep-out mask (pixels of scene
objects to fly around, e.g. the red stool) plus the keep-out radius, and emits
a per-step keep-out logit (enter the zone on this step given outside so far).
Keep-out labels are synthesised on the fly (``keepout_synth``) from the logged
future positions, so no semantic simulator data is needed.

Labels come only from the simulator rollouts in ``collect_consequence_dataset``.
"""

from __future__ import annotations

import json
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


def depth_input(depth56: np.ndarray) -> np.ndarray:
    """Log-scaled depth in [0, 1]; missing pixels read as far."""
    d = np.asarray(depth56, dtype=np.float32)
    d = np.where(np.isfinite(d) & (d > 0), d, DEPTH_MAX_M)
    d = np.clip(d, DEPTH_MIN_M, DEPTH_MAX_M)
    return (np.log(d / DEPTH_MIN_M) / math.log(DEPTH_MAX_M / DEPTH_MIN_M)).astype(np.float32)


def cones3_from_depth56(depth56: np.ndarray) -> np.ndarray:
    """Forward / left / right mins with no shared pixels (meters).

    Middle row band; forward = middle half of the columns, left / right = the
    outer quarters. The older ``cone_clearances`` halves overlap the forward
    box, so a wall dead ahead also set both side values.
    """
    d = np.asarray(depth56, dtype=np.float32)
    h, w = d.shape
    band = d[h // 4: h - h // 4]
    c0, c1 = w // 4, w - w // 4

    def vmin(x: np.ndarray) -> float:
        v = x[np.isfinite(x) & (x > 0.05)]
        return float(v.min()) if v.size else DEPTH_MAX_M

    return np.array([vmin(band[:, c0:c1]), vmin(band[:, :c0]), vmin(band[:, c1:])], dtype=np.float32)


def goal_features(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """[sin bearing, cos bearing, log1p(dist_xy)/5, dz/20] in the body frame."""
    s = np.asarray(state, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    dx, dy = g[0] - s[0], g[1] - s[1]
    b = math.atan2(dy, dx) - float(s[6])
    dist = math.hypot(dx, dy)
    return np.array(
        [math.sin(b), math.cos(b), math.log1p(dist) / 5.0, (g[2] - s[2]) / 20.0],
        dtype=np.float32,
    )


def history_inputs(
    depth_seq: List[np.ndarray],
    act_seq: List[np.ndarray],
    frame_offsets: Tuple[int, ...],
    hist_actions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Model inputs from a frame history (oldest first, newest = current step).

    ``act_seq[i]`` moved frame i -> i+1. Frame offset ``o`` takes the frame
    ``o`` steps back, clamped to the oldest one (a flight start has no past).
    Returns depth [K,56,56] (``depth_input``) and the last ``hist_actions``
    executed actions [A,4], zero-padded at the front.
    """
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

    def forward(
        self,
        depth: torch.Tensor,
        goal: torch.Tensor,
        prev_action: torch.Tensor,
        actions: torch.Tensor,
        hist_act: Optional[torch.Tensor] = None,
        sem: Optional[torch.Tensor] = None,
        keep_radius: Optional[torch.Tensor] = None,
        env: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """depth [B,56,56] or [B,K,56,56] (``depth_input``), goal [B,4], actions in limit units.

        ``hist_act`` [B,A,4] is required when the model was built with ``hist_actions`` > 0.
        ``sem`` [B,56,56] keep-out mask in [0, 1] and ``keep_radius`` [B] meters are read
        only by a ``sem_channels`` model; missing means no object / ``DEFAULT_KEEP_RADIUS_M``.
        ``env`` [B,env_dim] is the depth head's encoder scene vector (``da3_env_vector``),
        read only by an ``env_dim`` model; missing means the same as a dropped vector.
        """
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
            if env is None:
                e = f.new_zeros((f.shape[0], self.env_hidden))
            else:
                e = self.env_proj(env.to(f.dtype))
                if self.training and self.env_drop > 0:
                    e = e * (torch.rand(e.shape[0], 1, device=e.device) >= self.env_drop).to(e.dtype)
            ctx.append(e)
        h0 = torch.tanh(self.ctx(torch.cat(ctx, dim=-1)))
        out, _ = self.gru(F.silu(self.act_in(actions / lim)), h0.unsqueeze(0))
        y = self.head(out)
        res = {
            "hazard_logit": y[..., 0],
            "log_cones": y[..., 1:4],
            "dgoal": y[..., 4] * DGOAL_SCALE_M,
        }
        if self.sem_channels:
            res["keepout_logit"] = y[..., 5]
        return res


def altitude_allowed(
    candidates: np.ndarray,
    state: np.ndarray,
    goal: np.ndarray,
    *,
    goal_band_m: float = 1.0,
    net_tol_m: float = 0.5,
) -> np.ndarray:
    """Which candidate sequences may be chosen under the hold-altitude rule.

    Net climb over the sequence is not allowed unless the goal is more than
    ``goal_band_m`` above; net descent is not allowed unless it is more than
    ``goal_band_m`` below. Reverse flight is excluded upstream (dx >= 0).
    """
    c = np.asarray(candidates, dtype=np.float64)
    net_dz = c[..., 2].sum(axis=-1)
    dz_goal = float(np.asarray(goal)[2]) - float(np.asarray(state)[2])
    ok = np.ones(net_dz.shape, dtype=bool)
    if dz_goal <= goal_band_m:
        ok &= net_dz <= net_tol_m
    if dz_goal >= -goal_band_m:
        ok &= net_dz >= -net_tol_m
    return ok


def altitude_band_allowed(
    candidates: np.ndarray,
    z_now: float,
    z_lo: float,
    z_hi: float,
    prefix: int,
    return_per_step: float = 0.0,
) -> np.ndarray:
    """Hold-altitude rule on the executed prefix against a fixed band.

    Per-sequence net limits integrate under receding-horizon replanning (each
    executed first step is always within tolerance), so the drone drifts. Here
    the altitude after the executed prefix must stay inside [z_lo, z_hi]; when
    already outside, the prefix must move back by at least
    ``return_per_step`` per step (AirSim climbs ~0.02 m/step on a zero dz
    command, so "not further away" alone does not stop the drift).
    """
    c = np.asarray(candidates, dtype=np.float64)
    z_after = float(z_now) + c[:, :int(prefix), 2].sum(axis=-1)
    ok = (z_after >= z_lo) & (z_after <= z_hi)
    back = float(return_per_step) * int(prefix)
    if z_now > z_hi:
        ok |= z_after <= z_now - back
    elif z_now < z_lo:
        ok |= z_after >= z_now + back
    return ok


def visible_motion_allowed(
    candidates: np.ndarray, prefix: int, max_deg: float, min_speed: float = 0.1,
) -> np.ndarray:
    """Executed-prefix horizontal motion must point inside the camera view.

    The forward camera sees +-45 deg; the model has no evidence about what lies
    outside it, so a step translating faster than ``min_speed`` at more than
    ``max_deg`` off the nose is disallowed. Turning in place and hovering pass.
    """
    c = np.asarray(candidates, dtype=np.float64)[:, :int(prefix)]
    vx, vy = c[..., 0], c[..., 1]
    off = np.degrees(np.arctan2(np.abs(vy), vx))
    return ((np.hypot(vx, vy) < float(min_speed)) | (off <= float(max_deg))).all(axis=-1)


def survival(hazard_logit: torch.Tensor, keepout_logit: Optional[torch.Tensor] = None) -> torch.Tensor:
    """P(no collision, and no keep-out entry when given, through step k) for each k."""
    p_ok = 1.0 - torch.sigmoid(hazard_logit)
    if keepout_logit is not None:
        p_ok = p_ok * (1.0 - torch.sigmoid(keepout_logit))
    return torch.cumprod(p_ok, dim=-1)


def score_candidates(out: Dict[str, torch.Tensor], collision_cost_m: float = 20.0) -> torch.Tensor:
    """Expected goal progress over the horizon minus a collision price (meters)."""
    s = survival(out["hazard_logit"])[..., -1]
    progress = -out["dgoal"][..., -1]
    return s * progress - (1.0 - s) * float(collision_cost_m)


def select_candidate(surv: np.ndarray, pred_progress: np.ndarray, allowed: np.ndarray,
                     surv_min: float = 0.7) -> int:
    """Most goal progress among allowed candidates predicted to survive; else the safest allowed."""
    surv = np.asarray(surv, np.float64)
    allowed = np.asarray(allowed, bool)
    ok = allowed & (surv >= float(surv_min))
    if ok.any():
        return int(np.argmax(np.where(ok, np.asarray(pred_progress, np.float64), -np.inf)))
    return int(np.argmax(np.where(allowed, surv, -np.inf)))


def mirror_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Left-right mirror: flip the image, negate dy and yaw, swap left/right."""
    m = dict(batch)
    sign = torch.tensor([1.0, -1.0, 1.0, -1.0], device=batch["actions"].device)
    m["depth"] = torch.flip(batch["depth"], dims=[-1])
    if "sem" in batch:
        m["sem"] = torch.flip(batch["sem"], dims=[-1])
    if "env" in batch:
        m["env"], m["env_flip"] = batch["env_flip"], batch["env"]
    if "fut_xy" in batch:
        m["fut_xy"] = batch["fut_xy"] * torch.tensor([1.0, -1.0], device=batch["fut_xy"].device)
    m["actions"] = batch["actions"] * sign
    m["prev_action"] = batch["prev_action"] * sign
    if "hist_act" in batch:
        m["hist_act"] = batch["hist_act"] * sign
    g = batch["goal"].clone()
    g[:, 0] = -g[:, 0]
    m["goal"] = g
    c = batch["log_cones"].clone()
    c[..., 1], c[..., 2] = batch["log_cones"][..., 2], batch["log_cones"][..., 1]
    m["log_cones"] = c
    return m


def consequence_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                     keepout_weight: float = 2.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Hazard / cones / dgoal, plus keep-out when the batch carries ``keepout_synth`` labels.

    Once a trajectory enters a synthetic keep-out zone the logged physics no
    longer matches the pasted scene, so hazard and dgoal stop at that step; the
    cones of a window with a pasted object are not scored at all (the GT cones
    do not contain it).
    """
    alive = batch["alive"]
    valid = batch["valid"]
    cone_valid = valid
    has_ko = "keepout_logit" in out and "ko_alive" in batch
    if has_ko:
        before = batch["ko_alive"] * (1.0 - batch["ko_hit"])
        alive = alive * before
        valid = valid * before
        cone_valid = cone_valid * (1.0 - batch["ko_obj"])[:, None]
    hz = F.binary_cross_entropy_with_logits(out["hazard_logit"], batch["hit"], reduction="none")
    hz = (hz * alive).sum() / alive.sum().clamp(min=1.0)
    cone = F.smooth_l1_loss(out["log_cones"], batch["log_cones"], reduction="none").mean(-1)
    cone = (cone * cone_valid).sum() / cone_valid.sum().clamp(min=1.0)
    dg = F.smooth_l1_loss(out["dgoal"] / DGOAL_SCALE_M, batch["dgoal"] / DGOAL_SCALE_M, reduction="none")
    dg = (dg * valid).sum() / valid.sum().clamp(min=1.0)
    total = 2.0 * hz + cone + dg
    parts = {"hazard": float(hz.detach()), "cones": float(cone.detach()), "dgoal": float(dg.detach())}
    if has_ko:
        ka = batch["ko_alive"]
        ko = F.binary_cross_entropy_with_logits(out["keepout_logit"], batch["ko_hit"], reduction="none")
        ko = (ko * ka).sum() / ka.sum().clamp(min=1.0)
        total = total + float(keepout_weight) * ko
        parts["keepout"] = float(ko.detach())
    return total, parts


# -- dataset -----------------------------------------------------------------


def load_episode(path: Path) -> Dict[str, Any]:
    z = np.load(path)
    ep = {k: z[k] for k in z.files if k not in ("jpeg", "jpeg_offsets")}
    ep["meta"] = json.loads(bytes(z["meta"]).decode("utf-8"))
    side = path.with_suffix(".da3.npz")
    if side.is_file():
        ep["da3_depth56"] = np.load(side)["depth56"]
    env_side = path.with_suffix(".env.npz")
    if env_side.is_file():
        ze = np.load(env_side)
        ep["env"], ep["env_flip"] = ze["env"], ze["env_flip"]
    return ep


def body_xy(world_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    """World (NED) xy -> body frame at ``origin`` facing ``yaw``: x forward, y right."""
    d = np.asarray(world_xy, np.float64) - np.asarray(origin_xy, np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)


def dead_reckon_xy(actions: np.ndarray) -> np.ndarray:
    """Body-frame xy after each body-delta action [H,4], from the start pose."""
    p = np.zeros(2, np.float64)
    psi = 0.0
    out = np.zeros((len(actions), 2), np.float32)
    for k, a in enumerate(np.asarray(actions, np.float64)):
        c, s = math.cos(psi), math.sin(psi)
        p = p + np.array([c * a[0] - s * a[1], s * a[0] + c * a[1]])
        psi += a[3]
        out[k] = p
    return out


def episode_windows(ep: Dict[str, Any], horizon: int, frame_offsets: Tuple[int, ...] = (0,),
                    hist_actions: int = 0) -> Dict[str, np.ndarray]:
    """Every start step t with labels for the next ``horizon`` executed actions.

    ``action[i]`` moves state i -> i+1 and ``collided[i+1]`` flags contact on
    that step. Steps past the episode end are masked; the step that collides
    is the last alive step. ``fut_xy`` is the logged position after each step
    in the body frame at t (for ``keepout_synth``).
    """
    state = ep["state"].astype(np.float64)
    action = ep["action"].astype(np.float32)
    coll = ep["collided"].astype(bool)
    cones = np.stack([cones3_from_depth56(d) for d in ep["depth56"]])
    goal = np.asarray(ep["meta"]["goal"], dtype=np.float64)
    depth = ep["da3_depth56"]
    n_trans = len(state) - 1
    dist = np.linalg.norm(state[:, :3] - goal[None], axis=1)
    H = int(horizon)
    T = n_trans
    hist = tuple(frame_offsets) != (0,) or hist_actions > 0
    out: Dict[str, List[np.ndarray]] = {k: [] for k in (
        "depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal", "macro",
        "fut_xy")}
    if hist_actions:
        out["hist_act"] = []
    has_env = "env" in ep
    if has_env:
        out["env"], out["env_flip"] = [], []
    for t in range(T):
        if has_env:
            out["env"].append(ep["env"][t])
            out["env_flip"].append(ep["env_flip"][t])
        fut = np.zeros((H, 2), np.float32)
        n_f = min(H, T - t)
        fut[:n_f] = body_xy(state[t + 1:t + 1 + n_f, :2], state[t, :2], float(state[t, 6]))
        out["fut_xy"].append(fut)
        acts = np.zeros((H, 4), np.float32)
        hit = np.zeros(H, np.float32)
        alive = np.zeros(H, np.float32)
        lc = np.zeros((H, 3), np.float32)
        dg = np.zeros(H, np.float32)
        for k in range(H):
            i = t + k
            if i >= T:
                break
            acts[k] = action[i]
            alive[k] = 1.0
            c = np.clip(np.nan_to_num(cones[i + 1, :3], nan=DEPTH_MAX_M, posinf=DEPTH_MAX_M), 0.05, DEPTH_MAX_M)
            lc[k] = np.log(c)
            dg[k] = dist[i + 1] - dist[t]
            if coll[i + 1]:
                hit[k] = 1.0
                break
        if hist:
            d_k, a_k = history_inputs(list(depth[max(0, t - max(frame_offsets)):t + 1]),
                                      list(action[max(0, t - hist_actions):t]), frame_offsets, hist_actions)
            out["depth"].append(d_k.astype(np.float16))
            if hist_actions:
                out["hist_act"].append(a_k)
        else:
            out["depth"].append(depth_input(depth[t]))
        out["goal"].append(goal_features(state[t], goal))
        out["prev_action"].append(action[t - 1] if t > 0 else np.zeros(4, np.float32))
        out["actions"].append(acts)
        out["hit"].append(hit)
        out["alive"].append(alive)
        out["valid"].append(alive.copy())
        out["log_cones"].append(lc)
        out["dgoal"].append(dg)
        out["macro"].append(np.int16(ep["macro"][t]))
    return {k: np.stack(v) for k, v in out.items()}


def branch_windows(z: Any, depth56: Any, horizon: int, frame_offsets: Tuple[int, ...] = (0,),
                   hist_actions: int = 0, env: Optional[Tuple[np.ndarray, np.ndarray]] = None
                   ) -> Dict[str, np.ndarray]:
    """One window per candidate from a branching probe (all share the start state).

    ``z`` is a ``probe_*.npz`` from ``probe_counterfactual_airsim``; ``depth56``
    is the depth head's map on the last history frame, or the list of maps on
    every history frame (oldest first) for a history model. ``env`` is the
    (env, env_flip) vector pair of the last history frame.
    """
    cands = np.asarray(z["candidates"], np.float32)
    M, H0, _ = cands.shape
    H = int(horizon)
    if H0 < H:
        raise ValueError(f"probe horizon {H0} < model horizon {H}")
    coll = np.asarray(z["coll_step"])
    dist = np.asarray(z["dist"], np.float64)
    sd = np.asarray(z["step_depth56"], np.float32)
    cones = np.full((M, H0 + 1, 3), np.nan, np.float32)
    for m in range(M):
        for k in range(H0 + 1):
            if np.isfinite(dist[m, k]):
                cones[m, k] = cones3_from_depth56(sd[m, k])
    hist_a = np.asarray(z["hist_action"], np.float32)
    prev = hist_a[-2] if len(hist_a) >= 2 else np.zeros(4, np.float32)
    out = {k: [] for k in ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal",
                           "fut_xy")}
    hist = tuple(frame_offsets) != (0,) or hist_actions > 0
    if hist:
        seq = list(depth56) if isinstance(depth56, (list, tuple)) else [depth56]
        d_in, a_in = history_inputs(seq, list(hist_a[:len(seq) - 1]), frame_offsets, hist_actions)
        d_in = d_in.astype(np.float16)
        if hist_actions:
            out["hist_act"] = [a_in] * M
    else:
        d_in = depth_input(depth56[-1] if isinstance(depth56, (list, tuple)) else depth56)
    g = goal_features(z["state"], z["goal"])
    for m in range(M):
        hit = np.zeros(H, np.float32)
        alive = np.zeros(H, np.float32)
        lc = np.zeros((H, 3), np.float32)
        dg = np.zeros(H, np.float32)
        for k in range(H):
            if not np.isfinite(dist[m, k + 1]):
                break
            alive[k] = 1.0
            c = np.clip(np.nan_to_num(cones[m, k + 1, :3], nan=DEPTH_MAX_M, posinf=DEPTH_MAX_M), 0.05, DEPTH_MAX_M)
            lc[k] = np.log(c)
            dg[k] = dist[m, k + 1] - dist[m, 0]
            if coll[m] == k:
                hit[k] = 1.0
                break
        out["depth"].append(d_in)
        out["goal"].append(g)
        out["prev_action"].append(prev)
        out["actions"].append(cands[m, :H])
        out["fut_xy"].append(dead_reckon_xy(cands[m, :H]))
        out["hit"].append(hit)
        out["alive"].append(alive)
        out["valid"].append(alive.copy())
        out["log_cones"].append(lc)
        out["dgoal"].append(dg)
    if env is not None:
        out["env"] = [env[0]] * M
        out["env_flip"] = [env[1]] * M
    return {k: np.stack(v) for k, v in out.items()}


def group_ranking_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], group: torch.Tensor) -> torch.Tensor:
    """Within each start state, a candidate that survived must out-score one that hit."""
    alive = batch["alive"]
    log_s = (torch.nn.functional.logsigmoid(-out["hazard_logit"]) * alive).sum(-1)
    did_hit = batch["hit"].sum(-1) > 0
    same = group[:, None] == group[None, :]
    pair = same & (~did_hit)[:, None] & did_hit[None, :]
    if not bool(pair.any()):
        return log_s.sum() * 0.0
    diff = log_s[:, None] - log_s[None, :]
    return F.softplus(-diff[pair]).mean()


def save_checkpoint(path: Path, model: ConsequenceModel, extra: Optional[Dict[str, Any]] = None) -> None:
    torch.save({"model": model.state_dict(), "horizon": model.horizon, "hidden": model.hidden,
                "arch": "dropout_v2", "frame_offsets": list(model.frame_offsets),
                "hist_actions": model.hist_actions, "sem_channels": model.sem_channels,
                "env_dim": model.env_dim, "env_hidden": model.env_hidden, "env_drop": model.env_drop,
                **(extra or {})}, str(path))


def _checkpoint_state(p: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    sd = p["model"]
    if p.get("arch") != "dropout_v2":
        # v0 had no dropout: its encoder Linear sat at enc.9 instead of enc.10.
        sd = {k.replace("enc.9.", "enc.10.", 1) if k.startswith("enc.9.") else k: v for k, v in sd.items()}
    return sd


def load_checkpoint(path: Path, device: str = "cpu") -> ConsequenceModel:
    p = torch.load(str(path), map_location="cpu")
    m = ConsequenceModel(horizon=int(p["horizon"]), hidden=int(p["hidden"]),
                         frame_offsets=tuple(p.get("frame_offsets", (0,))),
                         hist_actions=int(p.get("hist_actions", 0)),
                         sem_channels=int(p.get("sem_channels", 0)),
                         env_dim=int(p.get("env_dim", 0)),
                         env_hidden=int(p.get("env_hidden", 64) or 64),
                         env_drop=float(p.get("env_drop", 0.0)))
    m.load_state_dict(_checkpoint_state(p))
    return m.to(device).eval()


def load_into_semantic(model: ConsequenceModel, path: Path) -> None:
    """Start a ``sem_channels`` / ``env_dim`` model from a checkpoint without them, keeping its outputs.

    The extra encoder input, the keep-radius and env context columns and the
    keep-out head row start at zero, so hazard / cones / dgoal match the old
    model exactly and the keep-out logit starts at a low prior. The env
    projection keeps its fresh init (its output is gated by the zero columns).
    """
    p = torch.load(str(path), map_location="cpu")
    old = _checkpoint_state(p)
    if (int(p.get("sem_channels", 0)) == model.sem_channels
            and int(p.get("env_dim", 0)) == model.env_dim):
        model.load_state_dict(old)
        return
    if int(p.get("env_dim", 0)) not in (0, model.env_dim):
        raise ValueError(f"env_dim {p.get('env_dim')} -> {model.env_dim}")
    if int(p.get("env_dim", 0)) and model.sem_channels != int(p.get("sem_channels", 0)):
        raise ValueError("adding sem_channels to an env model would misplace the env context columns")
    new = model.state_dict()
    for k, v in old.items():
        t = new[k]
        if t.shape == v.shape:
            new[k] = v
        elif k == "enc.0.weight":
            t = torch.zeros_like(t)
            t[:, : v.shape[1]] = v
            new[k] = t
        elif k == "ctx.weight":
            t = torch.zeros_like(t)
            t[:, : v.shape[1]] = v
            new[k] = t
        elif k.startswith("head.2."):
            t = t.clone()
            t[: v.shape[0]] = v
            if k.endswith("weight"):
                t[v.shape[0]:] = 0.0
            else:
                t[v.shape[0]:] = -4.0
            new[k] = t
        else:
            raise ValueError(f"cannot expand {k}: {tuple(v.shape)} -> {tuple(t.shape)}")
    model.load_state_dict(new)
