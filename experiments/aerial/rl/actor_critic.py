"""DreamerV3-style λ-return actor-critic for imagination RL (V4-MVP).

Pure imagination training: sample ``z0`` → ``imagine(H)`` → ``update(rollout)``.
Uses λ-return targets, REINFORCE policy gradient with value baseline,
return normalization, entropy regularization, and stop-grad on value targets
for the actor (DreamerV3 §2.5 — **no PPO**).

The actor exposes ``act_latent(z) -> [4]`` for ``imagine()``; the critic
estimates ``V(z)`` on latent states.

ACTION SPACE (C2, 2026-08-18 — see ``V4_SIGNAL1_STRUCTURAL_REFREEZE_PROPOSAL``
§4.1): the policy distribution is **bounded** by construction —
``u ~ N(mean, std)``, ``a = action_limits ⊙ tanh(u)`` with the tanh Jacobian
correction in ``log_prob`` — so every sampled action already lies inside the
deployed set ``body_delta_limits(1/step_hz)`` and the deployment-side clip
(``collector.py:167``) is a no-op.

Why not "just clip in ``imagine()``" (option C1, rejected): the actor update is
REINFORCE / score-function (``update()`` feeds ``rollout.actions`` back through
``evaluate_actions``; no gradient crosses ``dynamics.step``, ``imagine()`` being
pure numpy). A literal clip would (a) score boundary atoms under an unclipped
Gaussian — a biased likelihood — and (b) collapse exploration, since the single
isotropic ``std = exp(-0.5) = 0.607`` exceeds three of the four per-axis limits
at 5 Hz (0.4 / 0.4 / 0.314), leaving P(all four dims interior) ≈ 8.6% at
``mean = 0``.

``policy_class="unbounded_gaussian_legacy"`` reproduces the pre-C2 sampling law
**for audit replay only** (§A / §A.3 / §A.4 numbers); ``update()`` refuses it, so
the invalidated policy class can never be warm-started into training.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from experiments.aerial.rl.env.action import DEFAULT_STEP_HZ, body_delta_limits
from experiments.aerial.rl.goal_features import GOAL_NORM_DIM, GOAL_REL_DIM, g_norm_from_goal_rel
from experiments.aerial.rl.imagination import ImaginedRollout

logger = logging.getLogger(__name__)

#: Bounded (tanh-squashed) policy distribution — C2, the training default.
POLICY_TANH_BOUNDED = "tanh_bounded_v1"
#: Pre-C2 unbounded diagonal Gaussian. Load-and-evaluate only (audit replay).
POLICY_UNBOUNDED_LEGACY = "unbounded_gaussian_legacy"
_POLICY_CLASSES = (POLICY_TANH_BOUNDED, POLICY_UNBOUNDED_LEGACY)

#: Keeps ``atanh`` finite when an off-policy action sits on/outside the box.
_TANH_EPS = 1.0e-6

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - H100 has torch; stub host may not
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


@dataclass
class ActorCriticConfig:
    """MVP hyperparameters (design spec §3)."""

    latent_dim: int = 8
    action_dim: int = 4
    hidden_dim: int = 256
    lambda_gae: float = 0.95
    gamma: float = 0.997
    entropy_scale: float = 3.0e-4
    actor_lr: float = 1.0e-4
    critic_lr: float = 1.0e-4
    grad_clip: float = 100.0
    #: Pre-tanh gain on the actor head under C2 (it was a raw output gain, and
    #: thus an unbounded magnitude multiplier, before 2026-08-18). Keep at 1.0:
    #: larger values pre-saturate tanh and kill the gradient.
    action_scale: float = 1.0
    #: Sampling law; see module docstring. Legacy is replay-only.
    policy_class: str = POLICY_TANH_BOUNDED
    #: Control rate the action box is derived from — must match ``env.step_hz``,
    #: which is what ``collector`` clips against. Measured, never guessed.
    step_hz: float = DEFAULT_STEP_HZ
    #: Per-axis action bound; ``None`` -> ``body_delta_limits(1/step_hz)``, the
    #: single source of truth shared with the deployed path.
    action_limits: Optional[Tuple[float, ...]] = None
    #: When True, actor/critic input is ``concat(z, goal_feat)``.
    #: ``goal_feat_mode`` selects encoding (Phase-2 re-anchor 2026-08-30):
    #:   * ``meter`` — raw body ``goal_rel`` [fwd,left,up,dist_m] (Step E / 基线)
    #:   * ``g_norm`` — ``[û_xyz, log1p(d)]`` (F9; R1/R2 ckpts only)
    condition_on_goal: bool = True
    goal_feat_mode: str = "meter"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.policy_class not in _POLICY_CLASSES:
            raise ValueError(
                f"policy_class must be one of {_POLICY_CLASSES}, got {self.policy_class!r}"
            )
        mode = str(self.goal_feat_mode).strip().lower()
        if mode not in ("meter", "g_norm"):
            raise ValueError(
                f"goal_feat_mode must be 'meter' or 'g_norm', got {self.goal_feat_mode!r}"
            )
        self.goal_feat_mode = mode
        ad = int(self.action_dim)
        if self.action_limits is None:
            lim = body_delta_limits(1.0 / float(self.step_hz))
            self.action_limits = tuple(float(x) for x in np.asarray(lim).reshape(-1)[:ad])
        else:
            self.action_limits = tuple(float(x) for x in self.action_limits)
        if len(self.action_limits) != ad:
            raise ValueError(
                f"action_limits {self.action_limits} does not match action_dim={ad}"
            )
        if min(self.action_limits) <= 0.0:
            raise ValueError(f"action_limits must be positive, got {self.action_limits}")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "actor_critic requires torch (install with pip install 'aerial-wam-v2[gpu]')"
        )


def compute_lambda_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    *,
    gamma: float = 0.997,
    lambda_gae: float = 0.95,
    done: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Backward λ-return (GAE-style) over imagined trajectories.

    ``rewards`` [B, H], ``values`` [B, H+1] (bootstrap at H), ``done`` [B, H].
    Returns [B, H] targets for the critic and advantages for the actor.
    """
    rews = np.asarray(rewards, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if rews.ndim != 2 or vals.ndim != 2:
        raise ValueError("rewards must be [B,H] and values [B,H+1]")
    batch, horizon = rews.shape
    if vals.shape != (batch, horizon + 1):
        raise ValueError(f"values shape {vals.shape} != ({batch}, {horizon + 1})")

    dones = np.zeros_like(rews, dtype=bool) if done is None else np.asarray(done, dtype=bool)
    returns = np.zeros_like(rews)
    adv = np.zeros_like(rews)
    gae = np.zeros(batch, dtype=np.float64)
    for t in reversed(range(horizon)):
        nonterminal = (~dones[:, t]).astype(np.float64)
        delta = rews[:, t] + gamma * vals[:, t + 1] * nonterminal - vals[:, t]
        gae = delta + gamma * lambda_gae * nonterminal * gae
        adv[:, t] = gae
        returns[:, t] = adv[:, t] + vals[:, t]
    return returns


def normalize_advantage(adv: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-batch advantage normalization (DreamerV3 return norm)."""
    a = np.asarray(adv, dtype=np.float64)
    std = float(np.std(a))
    if std < eps:
        return a - float(np.mean(a))
    return (a - float(np.mean(a))) / (std + eps)


class _MLP(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class ImaginationActorPolicy:
    """Adapter wrapping ``LatentActorCritic`` for ``imagine()``."""

    def __init__(self, ac: "LatentActorCritic", *, deterministic: bool = False) -> None:
        self._ac = ac
        self.deterministic = bool(deterministic)

    def act_latent(
        self, z: np.ndarray, goal_rel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self._ac.act_latent(
            z, goal_rel=goal_rel, deterministic=self.deterministic
        )


@dataclass
class LatentActorCritic:
    """λ-return actor-critic on latent states (V4 deliverable)."""

    config: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    _actor: Any = field(default=None, repr=False)
    _critic: Any = field(default=None, repr=False)
    _actor_opt: Any = field(default=None, repr=False)
    _critic_opt: Any = field(default=None, repr=False)
    _log_std: Any = field(default=None, repr=False)
    _device: Any = field(default=None, repr=False)
    _limits: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require_torch()
        cfg = self.config
        self._device = torch.device(str(cfg.device))
        ld, ad, hd = int(cfg.latent_dim), int(cfg.action_dim), int(cfg.hidden_dim)
        in_dim = ld + (GOAL_REL_DIM if bool(cfg.condition_on_goal) else 0)
        self._actor = _MLP(in_dim, ad, hd).to(self._device)
        self._critic = _MLP(in_dim, 1, hd).to(self._device)
        self._limits = torch.as_tensor(
            np.asarray(cfg.action_limits, dtype=np.float32), device=self._device,
        )
        self._log_std = nn.Parameter(
            torch.zeros(ad, device=self._device) - 0.5,
        )
        self._actor_opt = torch.optim.Adam(
            list(self._actor.parameters()) + [self._log_std],
            lr=float(cfg.actor_lr),
        )
        self._critic_opt = torch.optim.Adam(self._critic.parameters(), lr=float(cfg.critic_lr))

    def _feat_tensor(
        self, z: np.ndarray, goal_rel: Optional[np.ndarray] = None
    ) -> "torch.Tensor":
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        z = z.reshape(-1, self.config.latent_dim)
        if not bool(self.config.condition_on_goal):
            return torch.as_tensor(z, device=self._device)
        if goal_rel is None:
            g = np.zeros((z.shape[0], GOAL_REL_DIM), dtype=np.float32)
        else:
            g = np.asarray(goal_rel, dtype=np.float32).reshape(-1, GOAL_REL_DIM)
            if g.shape[0] == 1 and z.shape[0] > 1:
                g = np.repeat(g, z.shape[0], axis=0)
            if g.shape[0] != z.shape[0]:
                raise ValueError(
                    f"goal_rel batch {g.shape[0]} != z batch {z.shape[0]}"
                )
        if str(self.config.goal_feat_mode) == "g_norm":
            g_feat = np.stack(
                [g_norm_from_goal_rel(g[i]) for i in range(g.shape[0])], axis=0
            )
        else:
            g_feat = g
        return torch.as_tensor(np.concatenate([z, g_feat], axis=-1), device=self._device)

    @classmethod
    def from_config(cls, cfg: Any) -> "LatentActorCritic":
        """Build from a plain dict / OmegaConf-like ``v4`` block."""
        if isinstance(cfg, ActorCriticConfig):
            return cls(config=cfg)
        if isinstance(cfg, dict):
            ac_cfg = ActorCriticConfig(
                latent_dim=int(cfg.get("latent_dim", 8)),
                lambda_gae=float(cfg.get("lambda_gae", 0.95)),
                gamma=float(cfg.get("gamma", 0.997)),
                entropy_scale=float(cfg.get("entropy_scale", 3.0e-4)),
                actor_lr=float(cfg.get("actor_lr", 1.0e-4)),
                critic_lr=float(cfg.get("critic_lr", 1.0e-4)),
                action_scale=float(cfg.get("action_scale", ActorCriticConfig.action_scale)),
                step_hz=float(cfg.get("step_hz", DEFAULT_STEP_HZ)),
                condition_on_goal=bool(cfg.get("condition_on_goal", True)),
                goal_feat_mode=str(cfg.get("goal_feat_mode", "meter")),
                device=str(cfg.get("device", "cpu")),
            )
            return cls(config=ac_cfg)
        return cls(
            config=ActorCriticConfig(
                lambda_gae=float(getattr(cfg, "lambda_gae", 0.95)),
                gamma=float(getattr(cfg, "gamma", 0.997)),
                entropy_scale=float(getattr(cfg, "entropy_scale", 3.0e-4)),
                actor_lr=float(getattr(cfg, "actor_lr", 1.0e-4)),
                critic_lr=float(getattr(cfg, "critic_lr", 1.0e-4)),
                action_scale=float(
                    getattr(cfg, "action_scale", ActorCriticConfig.action_scale)
                ),
                step_hz=float(getattr(cfg, "step_hz", DEFAULT_STEP_HZ)),
                condition_on_goal=bool(getattr(cfg, "condition_on_goal", True)),
                goal_feat_mode=str(getattr(cfg, "goal_feat_mode", "meter")),
                device=str(getattr(cfg, "device", "cpu")),
            )
        )

    def _z_tensor(self, z: np.ndarray) -> "torch.Tensor":
        """Deprecated alias: goal-blind features only."""
        return self._feat_tensor(z, goal_rel=None)

    def value(
        self, z: np.ndarray, goal_rel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        z_arr = np.asarray(z, dtype=np.float32)
        orig = z_arr.shape
        if z_arr.ndim == 1:
            z_arr = z_arr.reshape(1, -1)
        if z_arr.ndim == 3:
            b, t, d = z_arr.shape
            z_flat = z_arr.reshape(b * t, d)
            g_flat = None
            if goal_rel is not None:
                g_flat = np.asarray(goal_rel, dtype=np.float32).reshape(b * t, GOAL_REL_DIM)
            with torch.no_grad():
                v = self._critic(self._feat_tensor(z_flat, g_flat)).squeeze(-1)
            return v.cpu().numpy().reshape(b, t)
        flat = z_arr.reshape(-1, self.config.latent_dim)
        with torch.no_grad():
            v = self._critic(self._feat_tensor(flat, goal_rel)).squeeze(-1)
        out = v.cpu().numpy()
        if len(orig) == 1:
            return out
        if len(orig) == 2:
            return out.reshape(orig[0])
        return out.reshape(orig[0], orig[1])

    @property
    def bounded(self) -> bool:
        """True when the policy distribution is the C2 tanh-squashed one."""
        return self.config.policy_class == POLICY_TANH_BOUNDED

    @property
    def action_limits(self) -> np.ndarray:
        """Per-axis action bound (= deployed ``body_delta_limits(1/step_hz)``)."""
        return np.asarray(self.config.action_limits, dtype=np.float64)

    def _pre_dist(self, feat_t: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Pre-squash Gaussian ``(mean, std)``. Under legacy this IS the action dist."""
        mean = self._actor(feat_t) * float(self.config.action_scale)
        std = torch.exp(self._log_std).clamp(min=1e-4, max=2.0)
        return mean, std

    def act_latent(
        self,
        z: np.ndarray,
        goal_rel: Optional[np.ndarray] = None,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        feat_t = self._feat_tensor(z, goal_rel)
        with torch.no_grad():
            mean, std = self._pre_dist(feat_t)
            u = mean if deterministic else mean + std * torch.randn_like(mean)
            # ``deterministic`` returns the squashed mode/median, not E[a].
            act = self._limits * torch.tanh(u) if self.bounded else u
        a = act.squeeze(0).cpu().numpy().astype(np.float64)
        if self.bounded:
            # float32 tanh*limits can overshoot the float64 box by 1 ulp
            # (0.40000001 vs 0.4); clamp so the emitted action IS the deployed set.
            a = np.clip(a, -self.action_limits, self.action_limits)
        return a

    def evaluate_actions(
        self,
        z: np.ndarray,
        actions: np.ndarray,
        goal_rel: Optional[np.ndarray] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Log-prob and entropy for taken actions.

        Bounded (C2): ``a = limits ⊙ tanh(u)``, so
        ``log p(a) = log N(u; mean, std) - Σ log(limits ⊙ (1 - tanh²u))`` and the
        entropy carries the same Jacobian term (a single-sample estimate; it is
        gradient-free w.r.t. the actor, so ``entropy_scale`` keeps its meaning as
        the coefficient on ``Σ log_std``).

        Off-box actions (other policies' arms, or replayed corpora) are clamped
        into the open box before ``atanh`` — that is a diagnosis path only; the
        C2 policy cannot emit them.
        """
        feat_t = self._feat_tensor(z, goal_rel)
        a_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        if a_t.ndim == 1:
            a_t = a_t.unsqueeze(0)
        mean, std = self._pre_dist(feat_t)
        dist = torch.distributions.Normal(mean, std)
        if not self.bounded:
            return (
                dist.log_prob(a_t).sum(-1),
                dist.entropy().sum(-1),
                self._critic(feat_t).squeeze(-1),
            )
        y = (a_t / self._limits).clamp(-1.0 + _TANH_EPS, 1.0 - _TANH_EPS)
        u = torch.atanh(y)
        log_det = (torch.log(self._limits) + torch.log1p(-y * y)).sum(-1)
        logp = dist.log_prob(u).sum(-1) - log_det
        ent = dist.entropy().sum(-1) + log_det
        return logp, ent, self._critic(feat_t).squeeze(-1)

    def update(self, rollout: ImaginedRollout) -> Dict[str, Any]:
        """One imagination AC step on a batched ``ImaginedRollout``."""
        cfg = self.config
        if not self.bounded:
            # Red line: the pre-C2 unbounded policy class is invalidated (its
            # imagined action space is not the deployed one — §A.4). It stays
            # loadable for audit replay, but must never be warm-started.
            raise RuntimeError(
                "refusing to train policy_class="
                f"{cfg.policy_class!r}: the pre-C2 unbounded Gaussian is invalidated "
                "(imagined action space != deployed set; see proposal §4.1). "
                "Retrain from scratch with policy_class=" + POLICY_TANH_BOUNDED
            )
        z = np.asarray(rollout.z, dtype=np.float64)
        acts = np.asarray(rollout.actions, dtype=np.float64)
        rews = np.asarray(rollout.rewards, dtype=np.float64)
        dones = np.asarray(rollout.done, dtype=bool)
        batch, horizon_p1, ld = z.shape
        horizon = horizon_p1 - 1

        g_all = getattr(rollout, "goal_rel", None)
        if bool(cfg.condition_on_goal):
            if g_all is None:
                g_all = np.zeros((batch, horizon_p1, GOAL_REL_DIM), dtype=np.float32)
            else:
                g_all = np.asarray(g_all, dtype=np.float32).reshape(
                    batch, horizon_p1, GOAL_REL_DIM
                )

        # Flatten [B, H] steps; mask out post-done padding.
        z_flat = z[:, :horizon].reshape(-1, ld)
        a_flat = acts.reshape(-1, acts.shape[-1])
        alive = (~dones).reshape(-1)
        if not np.any(alive):
            return {"status": "skipped", "reason": "all trajectories terminated at t=0"}

        z_alive = z_flat[alive]
        a_alive = a_flat[alive]
        g_alive = None
        if bool(cfg.condition_on_goal):
            g_flat = g_all[:, :horizon].reshape(-1, GOAL_REL_DIM)
            g_alive = g_flat[alive]

        # Reconstruct per-trajectory alive slices for λ-return.
        vals_all = self.value(z, goal_rel=g_all if bool(cfg.condition_on_goal) else None)
        lam_targets = compute_lambda_returns(
            rews, vals_all, gamma=cfg.gamma, lambda_gae=cfg.lambda_gae, done=dones,
        )
        lam_flat = lam_targets.reshape(-1)[alive]

        target_t = torch.as_tensor(lam_flat, dtype=torch.float32, device=self._device)

        logp, ent, v_pred = self.evaluate_actions(z_alive, a_alive, goal_rel=g_alive)

        # Critic: MSE to λ-return targets.
        critic_loss = F.mse_loss(v_pred, target_t)
        self._critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self._critic.parameters(), cfg.grad_clip)
        self._critic_opt.step()

        # Actor: REINFORCE with stop-grad baseline + entropy bonus.
        with torch.no_grad():
            baseline = self._critic(self._feat_tensor(z_alive, g_alive)).squeeze(-1)
        adv_actor = target_t.detach() - baseline  # stop-grad on value for actor
        adv_actor = (adv_actor - adv_actor.mean()) / (adv_actor.std() + 1e-8)
        actor_loss = -(logp * adv_actor).mean() - float(cfg.entropy_scale) * ent.mean()
        self._actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self._actor.parameters()) + [self._log_std], cfg.grad_clip,
        )
        self._actor_opt.step()

        return {
            "status": "updated",
            "batch": int(batch),
            "horizon": int(horizon),
            "n_steps": int(alive.sum()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "mean_return": float(rews.sum(axis=1).mean()),
            "mean_entropy": float(ent.mean().item()),
            "condition_on_goal": bool(cfg.condition_on_goal),
            "goal_feat_mode": str(cfg.goal_feat_mode),
        }

    def update_bc(
        self,
        z: np.ndarray,
        actions: np.ndarray,
        goal_rel: Optional[np.ndarray] = None,
        *,
        loss_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Behavior-clone actor mean toward expert actions (MSE in action space).

        Does not touch the critic. Expert actions are clipped into the deployed
        box before the loss so off-box demo frames cannot explode gradients.
        """
        cfg = self.config
        if not self.bounded:
            raise RuntimeError(
                "refusing BC on policy_class="
                f"{cfg.policy_class!r}: only {POLICY_TANH_BOUNDED} is trainable"
            )
        z_arr = np.asarray(z, dtype=np.float64)
        a_arr = np.asarray(actions, dtype=np.float64)
        if z_arr.ndim == 1:
            z_arr = z_arr.reshape(1, -1)
        if a_arr.ndim == 1:
            a_arr = a_arr.reshape(1, -1)
        if z_arr.shape[0] != a_arr.shape[0]:
            raise ValueError(
                f"z batch {z_arr.shape[0]} != actions batch {a_arr.shape[0]}"
            )
        if a_arr.shape[-1] != int(cfg.action_dim):
            raise ValueError(
                f"actions dim {a_arr.shape[-1]} != action_dim={cfg.action_dim}"
            )
        g_alive = None
        if bool(cfg.condition_on_goal):
            if goal_rel is None:
                g_alive = np.zeros((z_arr.shape[0], GOAL_REL_DIM), dtype=np.float32)
            else:
                g_alive = np.asarray(goal_rel, dtype=np.float32).reshape(
                    -1, GOAL_REL_DIM
                )
                if g_alive.shape[0] != z_arr.shape[0]:
                    raise ValueError(
                        f"goal_rel batch {g_alive.shape[0]} != z batch {z_arr.shape[0]}"
                    )

        feat_t = self._feat_tensor(z_arr, g_alive)
        mean, _std = self._pre_dist(feat_t)
        pred = self._limits * torch.tanh(mean)
        a_t = torch.as_tensor(a_arr, dtype=torch.float32, device=self._device)
        a_t = a_t.clamp(-self._limits, self._limits)
        bc_loss = F.mse_loss(pred, a_t) * float(loss_scale)
        self._actor_opt.zero_grad()
        bc_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self._actor.parameters()) + [self._log_std], cfg.grad_clip,
        )
        self._actor_opt.step()
        with torch.no_grad():
            mae = (pred - a_t).abs().mean()
        return {
            "status": "updated",
            "bc_loss": float(bc_loss.item()),
            "bc_mae": float(mae.item()),
            "n_steps": int(z_arr.shape[0]),
            "loss_scale": float(loss_scale),
        }

    @classmethod
    def load_from_checkpoint(
        cls,
        path: Any,
        *,
        device: Optional[str] = None,
    ) -> "LatentActorCritic":
        """Restore actor/critic weights saved by ``train_v4_ac``.

        A payload with no ``policy_class`` predates C2 (2026-08-18) and is
        restored as ``POLICY_UNBOUNDED_LEGACY`` so §A/§A.3/§A.4 replay exactly.
        Such a checkpoint is evaluate-only: ``update()`` refuses it. The tensor
        shapes are identical across the two classes, so this auto-detect is the
        only thing standing between an old ckpt and a silently-wrong warm start.
        """
        _require_torch()
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        raw_cfg = dict(payload.get("config") or {})
        if "policy_class" not in raw_cfg:
            raw_cfg["policy_class"] = POLICY_UNBOUNDED_LEGACY
            logger.warning(
                "%s has no policy_class -> loading as %s (pre-C2 unbounded Gaussian, "
                "audit replay only; training on it is refused)",
                path, POLICY_UNBOUNDED_LEGACY,
            )
        if "condition_on_goal" not in raw_cfg:
            raw_cfg["condition_on_goal"] = False
            logger.warning(
                "%s has no condition_on_goal -> loading goal-blind "
                "(retrain from scratch for imagination-to-goal navigation)",
                path,
            )
        # Step E / pre-F9 ckpts expect metre goal_rel; F9 R1/R2 must set g_norm explicitly.
        if "goal_feat_mode" not in raw_cfg:
            raw_cfg["goal_feat_mode"] = "meter"
            if bool(raw_cfg.get("condition_on_goal", False)):
                logger.info(
                    "%s has no goal_feat_mode -> default 'meter' (Phase-1 / Step E contract)",
                    path,
                )
        fields = set(ActorCriticConfig.__dataclass_fields__)
        ac_cfg = ActorCriticConfig(
            **{k: raw_cfg[k] for k in fields if k in raw_cfg}
        )
        if device is not None:
            ac_cfg.device = str(device)
        ac = cls(config=ac_cfg)
        ac._actor.load_state_dict(payload["actor"])
        ac._critic.load_state_dict(payload["critic"])
        ac._log_std.data = payload["log_std"].to(ac._device)
        return ac


class LatentActorDeployPolicy:
    """Real-env policy: streaming ``z`` + ``act_latent(z, goal_rel)`` (V4 M5 rollout)."""

    def __init__(
        self,
        dynamics: Any,
        actor_critic: LatentActorCritic,
        *,
        deterministic: bool = True,
        stream_latent: bool = False,
    ) -> None:
        self._dynamics = dynamics
        self._ac = actor_critic
        self.deterministic = bool(deterministic)
        self.stream_latent = bool(stream_latent)
        self._prev_pos: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        self._latent: Optional[np.ndarray] = None
        self._prev_act: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_pos = None
        self._prev_t = None
        self._latent = None
        self._prev_act = None

    def act_latent(
        self, z: np.ndarray, goal_rel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if goal_rel is None:
            gr = np.zeros(GOAL_REL_DIM, dtype=np.float32)
        else:
            gr = np.asarray(goal_rel, dtype=np.float32).reshape(GOAL_REL_DIM)
        return self._ac.act_latent(
            z, goal_rel=gr, deterministic=self.deterministic
        )

    def act(self, view: Any) -> np.ndarray:
        from experiments.aerial.rl.env.obs import Observation
        from experiments.aerial.rl.goal_features import goal_rel_body, goal_rel_from_obs

        if isinstance(view, Observation):
            obs = view
            pos = np.asarray(obs.position, dtype=np.float32)
            vel = np.asarray(obs.velocity, dtype=np.float32)
            rgb = obs.rgb
            t = float(obs.t)
            yaw = float(obs.yaw)
            goal = None
            if isinstance(obs.info, dict):
                goal = obs.info.get("goal")
        else:
            pos = np.asarray(view.position, dtype=np.float32)
            vel = np.zeros(3, dtype=np.float32)
            if self._prev_pos is not None and self._prev_t is not None:
                dt = float(view.t) - float(self._prev_t)
                if 1e-4 < dt < 1.0:
                    vel = (pos - self._prev_pos) / float(dt)
            rgb = view.rgb
            t = float(view.t)
            yaw = float(view.yaw)
            goal = getattr(view, "goal", None)
            obs = Observation(
                rgb=rgb,
                state=np.array(
                    [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], yaw],
                    dtype=np.float32,
                ),
                t=t,
                info={"goal": np.asarray(goal, dtype=np.float32).reshape(3)} if goal is not None else {},
            )

        self._prev_pos = pos.copy()
        self._prev_t = t

        if (
            self.stream_latent
            and self._latent is not None
            and self._prev_act is not None
            and hasattr(self._dynamics, "observe_and_advance")
        ):
            z = self._dynamics.observe_and_advance(self._latent, self._prev_act, obs)
        else:
            z = self._dynamics.encode(obs)
        self._latent = np.asarray(z, dtype=np.float64).copy()

        if goal is not None:
            gr = goal_rel_body(pos, yaw, np.asarray(goal, dtype=np.float64).reshape(3))
        else:
            gr = goal_rel_from_obs(obs)
        act = self._ac.act_latent(
            z, goal_rel=gr, deterministic=self.deterministic
        )
        self._prev_act = np.asarray(act, dtype=np.float64).copy()
        return act
