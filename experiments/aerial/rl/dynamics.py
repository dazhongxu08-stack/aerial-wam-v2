"""World-model latent-dynamics interface (spec §4.4 / §8 contract).

The online RL loop runs on a **fast latent** dynamics model, not on the Wan2.2
pixel model (spec §4.4, §11). The contract every implementation honours:

    encode(obs)  -> z                                    latent state
    step(z, a)   -> DynamicsOutput(z_next, p_coll,       one imagined step
                                   progress, done)
    decode(z)    -> rgb   (optional)                     viz / distillation

Implementations here:

  * ``StubLatentDynamics`` — cheap analytic latent (position-carrying) so the
    imagination + corrector scaffolding is fully testable offline. This is the
    slot the real distilled WM (V1) drops into; it is NOT a trained model.
  * ``WanImaginationDynamics`` — wraps a ``FastWAM`` (VAE encode/decode +
    ``infer_joint``) as the pixel-level reference / distillation *source*. It is
    guarded ``offline_only``: calling ``step`` in an online loop raises, because
    stepping Wan2.2 pixels online is an explicit non-goal.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from experiments.aerial.rl.env.obs import Observation


@dataclass
class DynamicsOutput:
    """One imagined transition (§4.4: ``(z_{t+1}, p_coll, progress, done) = f(z,a)``).

    ``arrived`` distinguishes a *goal-reached* termination from a collision one so
    the imagination reward can mirror the real ``NavigationReward`` (which adds a
    success bonus on arrival). ``done`` is true for either terminal cause.
    """

    z_next: np.ndarray
    p_coll: float
    progress: float
    done: bool
    arrived: bool = False
    #: Optional forward/center clearance (metres) decoded from the latent
    #: (depth-aux WM). Legacy imagination path; main OA reward prefers
    #: ``obstacle_cost`` when the directional head is trained.
    d_fwd_hat: Optional[float] = None
    #: Action-conditioned obstacle cost from ``feature=[h‖z]`` + action.
    #: ``None`` when the head is absent / not yet trained (do not invent risk).
    obstacle_cost: Optional[float] = None


class LatentDynamics(abc.ABC):
    """Abstract fast-latent world model used for imagination."""

    #: latent dimensionality (implementations set this)
    latent_dim: int = 0

    @abc.abstractmethod
    def encode(self, obs: Observation) -> np.ndarray:
        """Map a real observation to a latent state ``z``."""

    @abc.abstractmethod
    def step(
        self,
        z: np.ndarray,
        action: np.ndarray,
        goal_rel: Optional[np.ndarray] = None,
        body_vel: Optional[np.ndarray] = None,
    ) -> DynamicsOutput:
        """Imagine one step forward from ``z`` under ``action``.

        ``goal_rel`` / ``body_vel`` are optional reward-head conditioning (V1-②);
        stubs may ignore them.
        """

    def decode(self, z: np.ndarray) -> np.ndarray:  # pragma: no cover - optional
        raise NotImplementedError("this dynamics model does not implement decode()")

    # -- V1 gate ----------------------------------------------------------
    def update(self, windows: Any) -> dict:
        """Fit the dynamics on replay windows. GATED (V1): no-op stub.

        Real WM training lands at V1; until then this returns a skip marker so
        the corrector loop can call it unconditionally.
        """
        return {"skipped": True, "reason": "world-model training is V1-gated"}


class StubLatentDynamics(LatentDynamics):
    """Analytic latent = [x, y, z, yaw] + a small learned-shaped tail.

    Transition simply integrates the body delta (via the shared body->world map)
    and computes a monotone ``p_coll`` from proximity to an optional obstacle and
    ``progress`` toward an optional goal. Deterministic; no torch.
    """

    def __init__(
        self,
        goal: Optional[np.ndarray] = None,
        obstacle: Optional[np.ndarray] = None,
        latent_dim: int = 8,
        collide_radius_m: float = 2.0,
        success_dist_m: float = 3.0,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self._goal = None if goal is None else np.asarray(goal, dtype=np.float64).reshape(3)
        self._obstacle = None if obstacle is None else np.asarray(obstacle, dtype=np.float64).reshape(3)
        self._collide_radius = float(collide_radius_m)
        self._success_dist = float(success_dist_m)

    def set_goal(self, goal: Optional[np.ndarray]) -> None:
        """Point imagination at the current episode's goal (corrector calls this
        before each imagine so imagined progress/arrival track the real task)."""
        self._goal = None if goal is None else np.asarray(goal, dtype=np.float64).reshape(3)

    def encode(self, obs: Observation) -> np.ndarray:
        z = np.zeros(self.latent_dim, dtype=np.float64)
        base = obs.proprio4()  # x, y, z, yaw
        z[: min(4, self.latent_dim)] = base[: min(4, self.latent_dim)]
        return z

    def observe_and_advance(
        self,
        prev_latent: np.ndarray,
        action: np.ndarray,
        current_obs: Observation,
    ) -> np.ndarray:
        """Test stub: carry streaming state (distinct from fresh ``encode``)."""
        z = self.encode(current_obs)
        prev = np.asarray(prev_latent, dtype=np.float64).reshape(self.latent_dim)
        tail = min(4, self.latent_dim)
        if tail > 0:
            z[:tail] = 0.7 * prev[:tail] + 0.3 * z[:tail]
        return z

    def step(
        self,
        z: np.ndarray,
        action: np.ndarray,
        goal_rel: Optional[np.ndarray] = None,
        body_vel: Optional[np.ndarray] = None,
    ) -> DynamicsOutput:
        from experiments.aerial.eval.run_closed_loop import apply_body_delta

        del goal_rel, body_vel  # analytic stub uses set_goal(); aux is TorchRSSM-only
        z = np.asarray(z, dtype=np.float64).reshape(self.latent_dim)
        pos = z[:3].copy()
        yaw = float(z[3]) if self.latent_dim > 3 else 0.0
        prev_goal_dist = self._dist(pos, self._goal)

        new_pos, new_yaw = apply_body_delta(pos, yaw, np.asarray(action, dtype=np.float64).reshape(4))
        z_next = z.copy()
        z_next[:3] = new_pos
        if self.latent_dim > 3:
            z_next[3] = new_yaw

        p_coll = 0.0
        if self._obstacle is not None:
            d = float(np.linalg.norm(new_pos - self._obstacle))
            p_coll = float(np.clip(1.0 - d / (2.0 * self._collide_radius), 0.0, 1.0))
        new_goal_dist = self._dist(new_pos, self._goal)
        progress = 0.0
        if prev_goal_dist is not None and new_goal_dist is not None:
            progress = prev_goal_dist - new_goal_dist
        arrived = new_goal_dist is not None and new_goal_dist < self._success_dist
        done = bool(p_coll >= 1.0 or arrived)
        return DynamicsOutput(
            z_next=z_next, p_coll=p_coll, progress=progress, done=done, arrived=arrived,
        )

    @staticmethod
    def _dist(pos: np.ndarray, target: Optional[np.ndarray]) -> Optional[float]:
        if target is None:
            return None
        return float(np.linalg.norm(pos - target))


class WanImaginationDynamics(LatentDynamics):
    """Pixel-level reference wrapping FastWAM — OFFLINE / distillation source only.

    ``encode`` uses the VAE, ``step`` runs ``model.infer_joint`` (jointly denoise
    video + action = imagination) and ``decode`` runs the VAE decoder. This is
    intended for producing distillation targets for the fast latent WM, NOT for
    online stepping — hence ``offline_only`` guards ``step``. Torch/model are
    lazy so importing this module never requires a GPU.
    """

    def __init__(self, model: Any, offline_only: bool = True, latent_dim: int = 48) -> None:
        self.model = model
        self.offline_only = bool(offline_only)
        self.latent_dim = int(latent_dim)  # VAE z_dim = 48
        self._online_context = False

    def enter_online(self) -> None:
        """Marker the corrector sets to assert Wan2.2 is never stepped online."""
        self._online_context = True

    def encode(self, obs: Observation) -> np.ndarray:  # pragma: no cover - needs model
        raise NotImplementedError(
            "WanImaginationDynamics.encode requires a loaded FastWAM VAE; use it "
            "offline to build distillation targets, not in unit tests."
        )

    def step(
        self,
        z: np.ndarray,
        action: np.ndarray,
        goal_rel: Optional[np.ndarray] = None,
        body_vel: Optional[np.ndarray] = None,
    ) -> DynamicsOutput:  # pragma: no cover
        del goal_rel, body_vel
        if self.offline_only and self._online_context:
            raise RuntimeError(
                "Wan2.2 pixel model must not be stepped online (spec §4.4/§11); "
                "distill a fast latent WM (StubLatentDynamics slot) for the RL loop."
            )
        raise NotImplementedError(
            "WanImaginationDynamics.step wraps model.infer_joint — offline only."
        )
