"""Collector dual-latent + pre-shield p_coll wiring (2026-09-21 audit)."""
from __future__ import annotations

import numpy as np

from experiments.aerial.rl.actor_critic import (
    ActorCriticConfig,
    LatentActorCritic,
    LatentActorDeployPolicy,
)
from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.collector import RolloutCollector
from experiments.aerial.rl.dynamics import DynamicsOutput, StubLatentDynamics
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.reward import RewardConfig
from experiments.aerial.rl.train_rl import Phase2CollectionPolicy


def _obs(pos, *, t=0.0, info=None):
    p = np.asarray(pos, dtype=np.float32).reshape(3)
    state = np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    return Observation(
        rgb=rgb, state=state, t=float(t), info=info or {"goal": [20.0, 0.0, 0.0]}
    )


class _StubEnv:
    def __init__(self):
        self.config = type("C", (), {"step_hz": 5.0})()
        self.goal = np.array([20.0, 0.0, 0.0], dtype=np.float64)
        self._t = 0.0
        self._pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def reset(self, episode=None):
        self._t = 0.0
        self._pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        return _obs(self._pos, t=self._t)

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(4)
        self._pos = self._pos + a[:3]
        self._t += 0.2
        return _obs(self._pos, t=self._t), {"cmd": a.tolist()}


class _ProbeDyn(StubLatentDynamics):
    """p_coll encodes action magnitude so post-shield zero would under-report."""

    def __init__(self):
        super().__init__(goal=np.array([20.0, 0.0, 0.0]), latent_dim=8)
        self.step_calls: list[np.ndarray] = []
        self.n_step = 0

    def step(self, z, action, goal_rel=None, body_vel=None):
        self.n_step += 1
        a = np.asarray(action, dtype=np.float64).reshape(4)
        self.step_calls.append(a.copy())
        out = super().step(z, action, goal_rel=goal_rel, body_vel=body_vel)
        mag = float(np.linalg.norm(a[:3]))
        return DynamicsOutput(
            z_next=out.z_next,
            p_coll=float(np.clip(mag / 2.0, 0.0, 1.0)),
            progress=out.progress,
            done=out.done,
            arrived=out.arrived,
        )


class _AlwaysIntervene:
    def should_override(self, obs, wm_out=None):
        return True

    def override_action(self, obs):
        return np.zeros(4, dtype=np.float64)


def _fwd_deploy(dyn: _ProbeDyn) -> LatentActorDeployPolicy:
    ac = LatentActorCritic(
        config=ActorCriticConfig(latent_dim=8, condition_on_goal=True, device="cpu")
    )
    pol = LatentActorDeployPolicy(dyn, ac, deterministic=True, stream_latent=True)

    def _fixed_act_latent(z, goal_rel=None, deterministic=True):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    pol._ac.act_latent = _fixed_act_latent  # type: ignore[method-assign]
    return pol


def test_streaming_policy_and_collector_share_one_latent():
    dyn = _ProbeDyn()
    env = _StubEnv()
    deploy = _fwd_deploy(dyn)
    policy = Phase2CollectionPolicy(deploy, goal_getter=lambda: env.goal)
    col = RolloutCollector(
        env,
        policy,
        ReplayBuffer(capacity_episodes=2, seed=0),
        dynamics=dyn,
        safety=_AlwaysIntervene(),
        max_steps=3,
        target_hz=0.0,
        skip_reset_collision=False,
        reward_cfg=RewardConfig(w_progress=0.0, w_collision=1.0, w_maneuver=0.0),
    )
    col.collect_episode()
    assert deploy._latent is not None
    np.testing.assert_allclose(col._latent, deploy._latent)
    # Streaming: one pre-shield probe per step only (no second prior-roll).
    assert dyn.n_step == 3
    # Shield rewrote executed action onto policy._prev_act for next advance.
    np.testing.assert_allclose(deploy._prev_act, np.zeros(4))


def test_reward_uses_pre_shield_intent_p_coll():
    dyn = _ProbeDyn()
    policy = _fwd_deploy(dyn)
    col = RolloutCollector(
        _StubEnv(),
        policy,
        ReplayBuffer(capacity_episodes=2, seed=0),
        dynamics=dyn,
        safety=_AlwaysIntervene(),
        max_steps=1,
        target_hz=0.0,
        skip_reset_collision=False,
        reward_cfg=RewardConfig(w_progress=0.0, w_collision=1.0, w_maneuver=0.0),
    )
    ep, _ = col.collect_episode()
    assert len(ep) == 1
    assert float(np.linalg.norm(ep[0].action[:3])) == 0.0
    assert dyn.step_calls and float(np.linalg.norm(dyn.step_calls[0][:3])) > 0.05
    assert float(ep[0].info["collision_risk"]) > 0.05
