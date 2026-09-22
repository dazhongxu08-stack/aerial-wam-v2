"""Phase-2 toward_g must stamp live obs.info (match long_eval), not only PolicyObservation."""
from __future__ import annotations

import numpy as np

from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.collector import RolloutCollector, act_delta
from experiments.aerial.rl.env.action import body_delta_limits
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.goal_features import goal_rel_from_obs
from experiments.aerial.rl.train_rl import Phase2CollectionPolicy


def _obs(pos, *, goal=None):
    p = np.asarray(pos, dtype=np.float32).reshape(3)
    state = np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    info = {}
    if goal is not None:
        info["goal"] = list(np.asarray(goal, dtype=np.float64).reshape(3))
    return Observation(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        state=state,
        info=info,
    )


def test_phase2_stamp_local_goal_mutates_live_obs_info():
    """act_delta alone used to leave obs.info at terminal G — stamp must fix that."""

    class _Inner:
        def __init__(self):
            self.seen = None

        def act(self, view):
            self.seen = None if view.goal is None else np.asarray(view.goal, dtype=np.float64)
            return np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float64)

    env_goal = np.array([200.0, 0.0, 0.0])
    inner = _Inner()
    pol = Phase2CollectionPolicy(inner, goal_getter=lambda: env_goal, r_m=100.0)
    obs = _obs([0.0, 0.0, 0.0], goal=[200.0, 0.0, 0.0])

    pol.stamp_local_goal(obs)
    np.testing.assert_allclose(obs.info["goal"], [100.0, 0.0, 0.0])
    assert float(goal_rel_from_obs(obs)[3]) == 100.0

    act_delta(pol, obs, "", body_delta_limits(0.2))
    np.testing.assert_allclose(inner.seen, [100.0, 0.0, 0.0])
    # Live obs must still carry the carrot after act_delta (not revert to 200).
    np.testing.assert_allclose(obs.info["goal"], [100.0, 0.0, 0.0])


def test_collector_phase2_planner_and_buffer_see_toward_g():
    class _TrackingPlanner:
        def __init__(self):
            self.last_goal = None

        def set_goal(self, goal):
            self.last_goal = None if goal is None else np.asarray(goal, dtype=np.float64).reshape(3)

        def plan(self, obs, action, latent=None):
            return np.asarray(action, dtype=np.float64)

    class _Env:
        def __init__(self):
            self.config = type("C", (), {"step_hz": 5.0})()
            self.goal = np.array([200.0, 0.0, 0.0], dtype=np.float64)
            self._pos = np.zeros(3, dtype=np.float64)

        def reset(self, episode=None):
            self._pos = np.zeros(3, dtype=np.float64)
            return _obs(self._pos, goal=self.goal)

        def step(self, action):
            self._pos = self._pos + np.asarray(action, dtype=np.float64)[:3]
            return _obs(self._pos, goal=self.goal), {}

    class _Inner:
        def act(self, view):
            return np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float64)

    env = _Env()
    planner = _TrackingPlanner()
    pol = Phase2CollectionPolicy(_Inner(), goal_getter=lambda: env.goal, r_m=100.0)
    col = RolloutCollector(
        env,
        pol,
        ReplayBuffer(capacity_episodes=1, seed=0),
        max_steps=1,
        target_hz=0.0,
        planner=planner,
        skip_reset_collision=False,
    )
    ep, _ = col.collect_episode()
    assert planner.last_goal is not None
    np.testing.assert_allclose(planner.last_goal, [100.0, 0.0, 0.0])
    np.testing.assert_allclose(ep[0].info["goal"], [100.0, 0.0, 0.0])
    # Stored obs must carry carrot for corrector imagination goal_rel.
    np.testing.assert_allclose(ep[0].obs.info["goal"], [100.0, 0.0, 0.0])
