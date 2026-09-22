"""Torch-free test: depth predictor fills obs.info BEFORE the shield sees it.

Frozen §4 ④ wiring. Uses only stubs (no torch), so it runs on Mac — it must NOT
live in ``test_depth_head.py`` where a module-level ``importorskip('torch')``
would skip it on GPU-less hosts.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.collector import RolloutCollector
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.safety import ThresholdSafetyShield


def _obs(size=16, depth_val=5.0, info=None):
    state = np.zeros(7, dtype=np.float32)
    rgb = np.random.default_rng(0).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    depth = np.full((size, size), depth_val, dtype=np.float32)
    return Observation(rgb=rgb, state=state, depth=depth, info=info or {})


def test_collector_writes_depth_min_pred_before_shield():
    """Predictor fills info before should_override sees the obs → intervention."""

    class _StubEnv:
        def __init__(self):
            self.config = type("C", (), {"step_hz": 5.0})()
            self.goal = np.array([10.0, 0.0, 0.0])

        def reset(self, episode=None):
            return _obs(depth_val=10.0)

        def step(self, action):
            return _obs(depth_val=10.0), {"cmd": action.tolist()}

    class _Pred:
        def reset(self):
            pass

        def predict_min(self, obs):
            return 0.5  # below ThresholdSafetyShield.min_depth_m=1.5

    class _Policy:
        def act(self, view):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    buf = ReplayBuffer(capacity_episodes=2, seed=0)
    shield = ThresholdSafetyShield(min_depth_m=1.5)
    col = RolloutCollector(
        _StubEnv(), _Policy(), buf,
        safety=shield, max_steps=3, target_hz=0.0,
        depth_predictor=_Pred(),
        skip_reset_collision=False,
    )
    ep, stats = col.collect_episode()
    assert stats.interventions >= 1
    # The stored transition's obs.info must carry the prediction.
    assert ep[0].obs.info.get("depth_min_pred") == 0.5
    # ATTR: same pred must also land on transition.info for P7/attr harness.
    assert ep[0].info.get("depth_min_pred") == 0.5


def test_collector_carries_decision_depth_into_real_reward():
    """env.step rebuilds next_obs.info without depth — collector must copy it
    so NavigationReward can fold clearance_risk (2026-09-21 audit)."""
    from experiments.aerial.rl.reward import RewardConfig, clearance_risk_from_depth

    class _StubEnv:
        def __init__(self):
            self.config = type("C", (), {"step_hz": 5.0})()
            self.goal = np.array([100.0, 0.0, 0.0])

        def reset(self, episode=None):
            return _obs(depth_val=10.0, info={"goal": [100.0, 0.0, 0.0]})

        def step(self, action):
            # Mimic airsim_env.observe(): fresh info with only goal, no depth.
            return _obs(depth_val=10.0, info={"goal": [100.0, 0.0, 0.0]}), {}

    class _Pred:
        def reset(self):
            pass

        def predict_min(self, obs):
            return 5.0

    class _Policy:
        def act(self, view):
            return np.zeros(4, dtype=np.float64)

    col = RolloutCollector(
        _StubEnv(),
        _Policy(),
        ReplayBuffer(capacity_episodes=1, seed=0),
        max_steps=1,
        target_hz=0.0,
        depth_predictor=_Pred(),
        skip_reset_collision=False,
        reward_cfg=RewardConfig(w_progress=0.0, w_collision=10.0, w_maneuver=0.0),
    )
    ep, _ = col.collect_episode()
    expect = clearance_risk_from_depth(5.0)
    assert ep[0].info.get("clearance_risk") == pytest.approx(expect)
    assert ep[0].info.get("collision_risk") == pytest.approx(expect)


def test_collector_copies_tau_pred_into_ep_info():
    class _StubEnv:
        def __init__(self):
            self.config = type("C", (), {"step_hz": 5.0})()
            self.goal = np.array([10.0, 0.0, 0.0])

        def reset(self, episode=None):
            return _obs(depth_val=10.0)

        def step(self, action):
            return _obs(depth_val=10.0), {"cmd": action.tolist()}

    class _Tau:
        def reset(self):
            pass

        def predict_tau(self, obs):
            return 2.5

    class _Policy:
        def act(self, view):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    buf = ReplayBuffer(capacity_episodes=2, seed=0)
    col = RolloutCollector(
        _StubEnv(), _Policy(), buf,
        max_steps=2, target_hz=0.0,
        tau_predictor=_Tau(),
        skip_reset_collision=False,
    )
    ep, _ = col.collect_episode()
    assert ep[0].obs.info.get("tau_pred") == 2.5
    assert ep[0].info.get("tau_pred") == 2.5
