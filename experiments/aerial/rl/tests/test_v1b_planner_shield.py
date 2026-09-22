"""Tests for V1b planner + DepthTauShield."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.collector import RolloutCollector
from experiments.aerial.rl.dynamics import StubLatentDynamics
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.planner import (
    ImaginationPlanner,
    default_candidates,
    drop_backward_if_subgoal_ahead,
    drop_strafe_if_yaw_off_axis,
    face_goal_candidates,
)
from experiments.aerial.rl.reward import RewardConfig
from experiments.aerial.rl.safety import DepthTauShield
from experiments.aerial.rl import v1_metrics


def _obs(pos, depth_val=10.0, info=None):
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    depth = np.full((16, 16), depth_val, dtype=np.float32)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    return Observation(rgb=rgb, state=state, depth=depth, info=info or {})


def test_planner_prefers_forward_toward_goal():
    dyn = StubLatentDynamics(goal=np.array([50.0, 0.0, 0.0]), latent_dim=8)
    planner = ImaginationPlanner(dyn, horizon=3, reward_cfg=RewardConfig())
    obs = _obs([0.0, 0.0, 0.0], info={"goal": np.array([50.0, 0.0, 0.0])})
    hover = np.zeros(4)
    planned = planner.plan(obs, hover)
    assert planned[0] > 0.0


def test_drop_backward_when_subgoal_ahead():
    cands = default_candidates(np.zeros(4))
    assert any(c[0] < -0.01 for c in cands)
    gr = np.array([15.0, 0.0, 0.0, 15.0], dtype=np.float32)
    filtered = drop_backward_if_subgoal_ahead(cands, gr)
    assert all(not (c[0] < -0.01 and np.all(np.abs(c[1:]) < 1e-6)) for c in filtered)


def test_drop_backward_motion_always():
    from experiments.aerial.rl.planner import drop_backward_motion

    cands = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([-0.5, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.3, 0.0, 0.1]),
        np.array([-0.2, 0.2, 0.0, 0.0]),
    ]
    kept = drop_backward_motion(cands)
    assert all(float(c[0]) >= 0.0 for c in kept)
    assert len(kept) == 2


def test_drop_strafe_helper_removes_pure_lateral():
    gr = np.array([0.0, 40.0, 0.0, 40.0], dtype=np.float32)
    cands = [
        np.array([0.0, 0.4, 0.0, 0.0]),
        np.array([0.3, 0.0, 0.0, 0.3]),
    ]
    kept = drop_strafe_if_yaw_off_axis(cands, gr)
    assert all(not (abs(c[1]) > 0.2 and abs(c[3]) < 0.05) for c in kept)
    assert any(abs(c[3]) > 0.05 for c in kept)


def test_planner_h1_faces_when_forward_is_open():
    """Open nose + carrot on the left: turn, do not pick pure crab."""
    dyn = StubLatentDynamics(goal=np.array([0.0, 40.0, 0.0]), latent_dim=8)
    cfg = RewardConfig(w_progress=1.0, w_collision=0.0, w_maneuver=0.0)
    planner = ImaginationPlanner(dyn, horizon=1, reward_cfg=cfg)
    obs = _obs(
        [0.0, 0.0, 0.0],
        info={
            "goal": np.array([0.0, 40.0, 0.0]),
            "depth_cones_pred": {"forward": 30.0},
        },
    )
    planned = planner.plan(obs, np.array([0.0, 0.4, 0.0, 0.0]))
    assert planned[3] > 0.05, planned
    assert abs(float(planned[1])) < abs(float(planned[3])) + 0.15, planned


def test_planner_h1_mid_discourages_strafe_keeps_ability():
    """Mid clearance (urban default): prefer face, but pure strafe still legal."""
    dyn = StubLatentDynamics(goal=np.array([0.0, 40.0, 0.0]), latent_dim=8)
    cfg = RewardConfig(w_progress=1.0, w_collision=0.0, w_maneuver=0.0)
    planner = ImaginationPlanner(dyn, horizon=1, reward_cfg=cfg)
    obs = _obs(
        [0.0, 0.0, 0.0],
        info={
            "goal": np.array([0.0, 40.0, 0.0]),
            "depth_cones_pred": {"forward": 5.0},
        },
    )
    planned = planner.plan(obs, np.array([0.0, 0.4, 0.0, 0.0]))
    assert planned[3] > 0.05, planned
    # Ability not banned: zero face/strafe weights → crab can still win.
    planner2 = ImaginationPlanner(
        dyn,
        horizon=1,
        reward_cfg=cfg,
        face_yaw_score_w_mid=0.0,
        strafe_score_w_mid=0.0,
    )
    planned2 = planner2.plan(obs, np.array([0.0, 0.4, 0.0, 0.0]))
    assert abs(float(planned2[1])) > 0.2, planned2


def test_planner_h1_keeps_strafe_when_forward_blocked():
    """Blocked nose: no pure crab; escape uses lateral+turn toward clearer cone."""
    dyn = StubLatentDynamics(goal=np.array([0.0, 40.0, 0.0]), latent_dim=8)
    cfg = RewardConfig(w_progress=1.0, w_collision=0.0, w_maneuver=0.0)
    planner = ImaginationPlanner(dyn, horizon=1, reward_cfg=cfg)
    obs = _obs(
        [0.0, 0.0, 0.0],
        info={
            "goal": np.array([0.0, 40.0, 0.0]),
            "depth_cones_pred": {"forward": 2.0, "left": 20.0, "right": 3.0},
        },
    )
    strafe = np.array([0.0, 0.4, 0.0, 0.0])
    planned = planner.plan(obs, strafe)
    # Escape should turn (and usually strafe) toward clearer left — not pure dy.
    assert abs(float(planned[3])) > 0.05, planned
    assert abs(float(planned[1])) > 0.05 or abs(float(planned[3])) > 0.05, planned


def test_drop_pure_lateral_helper():
    from experiments.aerial.rl.planner import drop_pure_lateral_strafe

    cands = [
        np.array([0.0, 0.4, 0.0, 0.0]),
        np.array([0.1, 0.3, 0.0, 0.2]),
    ]
    kept = drop_pure_lateral_strafe(cands)
    assert all(not (abs(c[1]) > 0.2 and abs(c[3]) < 0.05) for c in kept)
    assert any(abs(c[3]) > 0.05 for c in kept)


def test_escape_clearer_cone_fallback_without_side_cones():
    """Missing L/R cones must still yield peel atoms (goal-side + backup)."""
    from experiments.aerial.rl.planner import escape_clearer_cone_candidates

    obs = _obs(
        [0.0, 0.0, 0.0],
        info={"depth_cones_pred": {"forward": 2.0}},  # no left/right
    )
    cands = escape_clearer_cone_candidates(
        obs, goal_rel=np.array([0.0, 40.0, 0.0, 40.0])
    )
    assert len(cands) >= 2
    assert all(abs(float(c[3])) > 0.05 for c in cands)
    # Goal is left (+y) → first peel should turn left.
    assert float(cands[0][3]) > 0.0


def test_escape_side_clearer_and_fallback():
    from experiments.aerial.rl.planner import escape_side

    hug_left = _obs(
        [0.0, 0.0, 0.0],
        info={"depth_cones_pred": {"forward": 2.0, "left": 2.5, "right": 18.0}},
    )
    assert escape_side(hug_left) == pytest.approx(-1.0)
    both_ambig = _obs(
        [0.0, 0.0, 0.0],
        info={"depth_cones_pred": {"forward": 12.0, "left": 10.0, "right": 10.4}},
    )
    assert escape_side(both_ambig) == pytest.approx(0.0)
    no_lr = _obs([0.0, 0.0, 0.0], info={"depth_cones_pred": {"forward": 2.0}})
    assert escape_side(no_lr, goal_rel=np.array([0.0, 40.0, 0.0])) == pytest.approx(1.0)


def test_planner_escape_beats_freeze_when_nose_blocked():
    """Blocked nose: must not pick zero/freeze; peel toward clearer cone."""
    dyn = StubLatentDynamics(goal=np.array([0.0, 40.0, 0.0]), latent_dim=8)
    cfg = RewardConfig(w_progress=1.0, w_collision=0.0, w_maneuver=0.0)
    planner = ImaginationPlanner(dyn, horizon=1, reward_cfg=cfg)
    obs = _obs(
        [0.0, 0.0, 0.0],
        info={
            "goal": np.array([0.0, 40.0, 0.0]),
            "depth_cones_pred": {"forward": 2.0, "left": 20.0, "right": 3.0},
        },
    )
    planned = planner.plan(obs, np.zeros(4))
    assert abs(float(planned[3])) > 0.05, planned
    assert float(planned[3]) > 0.0, planned


def test_planner_rules_mock_still_peels_when_blocked():
    """rules ablation: no WM, hand biases alone must still leave the wall."""
    dyn = StubLatentDynamics(goal=np.array([0.0, 40.0, 0.0]), latent_dim=8)
    cfg = RewardConfig(w_progress=1.0, w_collision=0.0, w_maneuver=0.0)
    planner = ImaginationPlanner(dyn, horizon=1, reward_cfg=cfg, mock_mode="rules")
    obs = _obs(
        [0.0, 0.0, 0.0],
        info={
            "goal": np.array([0.0, 40.0, 0.0]),
            "depth_cones_pred": {"forward": 2.0, "left": 20.0, "right": 3.0},
        },
    )
    planned = planner.plan(obs, np.zeros(4))
    assert float(planned[3]) > 0.05, planned


def test_planner_wm_bare_skips_hand_escape_bias():
    """wm_bare: score from WM only; stub progress may pick freeze over peel."""
    dyn = StubLatentDynamics(goal=np.array([40.0, 0.0, 0.0]), latent_dim=8)
    cfg = RewardConfig(w_progress=1.0, w_collision=0.0, w_maneuver=0.0)
    # Goal straight ahead + blocked nose: hand mode peels left; wm_bare with
    # zero collision may prefer +dx (progress) from default candidates.
    obs = _obs(
        [0.0, 0.0, 0.0],
        info={
            "goal": np.array([40.0, 0.0, 0.0]),
            "depth_cones_pred": {"forward": 2.0, "left": 20.0, "right": 3.0},
        },
    )
    hand = ImaginationPlanner(dyn, horizon=1, reward_cfg=cfg)
    bare = ImaginationPlanner(dyn, horizon=1, reward_cfg=cfg, mock_mode="wm_bare")
    a_hand = hand.plan(obs, np.zeros(4))
    a_bare = bare.plan(obs, np.array([1.0, 0.0, 0.0, 0.0]))
    assert float(a_hand[3]) > 0.05, a_hand
    # Bare keeps escape *candidates* but without stuck/escape bias may keep +dx.
    assert abs(float(a_bare[1])) < 0.05 or float(a_bare[0]) > 0.2, a_bare


def test_planner_open_lane_off_when_nose_blocked():
    """Hard-block: escape peels toward clearer cone (aligned with off-axis goal)."""
    dyn = StubLatentDynamics(goal=np.array([0.0, 40.0, 0.0]), latent_dim=8)
    cfg = RewardConfig(w_progress=1.0, w_collision=0.0, w_maneuver=0.0)
    planner = ImaginationPlanner(dyn, horizon=1, reward_cfg=cfg)
    obs = _obs(
        [0.0, 0.0, 0.0],
        info={
            "goal": np.array([0.0, 40.0, 0.0]),
            "depth_cones_pred": {"forward": 2.0, "left": 20.0, "right": 3.0},
        },
    )
    planned = planner.plan(obs, np.array([0.0, 0.4, 0.0, 0.0]))
    assert abs(float(planned[3])) > 0.05, planned
    assert float(planned[3]) > 0.0, planned

def test_planner_tight_mid_offers_escape_without_side_cones():
    """Tight mid (d_fwd~4) + off-axis: escape fallback, not face-into-wall."""
    dyn = StubLatentDynamics(goal=np.array([0.0, 40.0, 0.0]), latent_dim=8)
    cfg = RewardConfig(w_progress=1.0, w_collision=0.0, w_maneuver=0.0)
    planner = ImaginationPlanner(dyn, horizon=1, reward_cfg=cfg)
    obs = _obs(
        [0.0, 0.0, 0.0],
        info={
            "goal": np.array([0.0, 40.0, 0.0]),
            "depth_cones_pred": {"forward": 4.0},  # tight mid, no L/R
        },
    )
    planned = planner.plan(obs, np.array([0.0, 0.4, 0.0, 0.0]))
    assert abs(float(planned[3])) > 0.05, planned


def test_planner_near_goal_skips_tight_mid_escape():
    """Within 25m of goal + tight mid: face goal, do not peel-away (R4 terminal)."""
    dyn = StubLatentDynamics(goal=np.array([0.0, 12.0, 0.0]), latent_dim=8)
    cfg = RewardConfig(w_progress=1.0, w_collision=0.0, w_maneuver=0.0)
    planner = ImaginationPlanner(
        dyn, horizon=1, reward_cfg=cfg, face_yaw_score_w_mid=2.0
    )
    obs = _obs(
        [0.0, 0.0, 0.0],
        info={
            "goal": np.array([0.0, 12.0, 0.0]),
            "depth_cones_pred": {"forward": 4.0, "left": 20.0, "right": 3.0},
        },
    )
    planned = planner.plan(obs, np.array([0.5, 0.0, 0.0, 0.0]))
    # Prefer +dyaw toward goal (left), not a hard left peel dominating progress.
    assert float(planned[3]) > 0.0, planned


def test_planner_h1_weak_face_when_depth_unknown():
    """Unknown depth: soft face + strafe penalty; ability not hard-banned."""
    dyn = StubLatentDynamics(goal=np.array([0.0, 40.0, 0.0]), latent_dim=8)
    cfg = RewardConfig(w_progress=1.0, w_collision=0.0, w_maneuver=0.0)
    planner = ImaginationPlanner(
        dyn, horizon=1, reward_cfg=cfg, face_yaw_score_w_unknown=2.0
    )
    obs = _obs([0.0, 0.0, 0.0], info={"goal": np.array([0.0, 40.0, 0.0])})
    planned = planner.plan(obs, np.array([0.0, 0.4, 0.0, 0.0]))
    assert planned[3] > 0.05, planned
    planner2 = ImaginationPlanner(
        dyn,
        horizon=1,
        reward_cfg=cfg,
        face_yaw_score_w_unknown=0.0,
        strafe_score_w_mid=0.0,
    )
    planned2 = planner2.plan(obs, np.array([0.0, 0.4, 0.0, 0.0]))
    assert abs(float(planned2[1])) > 0.2, planned2


def test_face_goal_candidates_turn_sign():
    left = face_goal_candidates(np.array([1.0, 10.0, 0.0, 10.0]))
    right = face_goal_candidates(np.array([1.0, -10.0, 0.0, 10.0]))
    assert left and all(c[3] > 0 for c in left)
    assert right and all(c[3] < 0 for c in right)


def test_planner_mock_pass_returns_base_action():
    dyn = StubLatentDynamics(goal=np.array([20.0, 0.0, 0.0]), latent_dim=8)
    planner = ImaginationPlanner(
        dyn, horizon=3, reward_cfg=RewardConfig(), mock_mode="pass"
    )
    obs = _obs([0.0, 0.0, 0.0], info={"goal": np.array([20.0, 0.0, 0.0])})
    base = np.array([0.3, -0.2, 0.0, 0.1], dtype=np.float64)
    assert np.allclose(planner.plan(obs, base), base)


def test_closed_loop_uses_actor_after_first_action(monkeypatch):
    """Step 0 is the candidate; later steps come from the actor. No hand bias."""
    seen: list = []

    def _spy(dynamics, policy, z0, horizon, **kwargs):
        del dynamics, z0, kwargs
        for _ in range(int(horizon)):
            seen.append(np.asarray(policy.act_latent(np.zeros(8), np.zeros(4))).copy())
        return type("Roll", (), {"returns": np.zeros(1)})()

    monkeypatch.setattr("experiments.aerial.rl.planner.imagine", _spy)

    class _Tail:
        def act_latent(self, z, goal_rel=None):
            del z, goal_rel
            return np.array([0.1, 0.0, 0.0, 0.25])

    dyn = StubLatentDynamics(goal=np.array([20.0, 0.0, 0.0]), latent_dim=8)
    planner = ImaginationPlanner(
        dyn,
        horizon=3,
        reward_cfg=RewardConfig(),
        rollout_mode="closed_loop",
        tail_policy=_Tail(),
    )
    obs = _obs(
        [0.0, 0.0, 0.0],
        info={
            "goal": np.array([0.0, 40.0, 0.0]),
            "depth_cones_pred": {"forward": 2.0, "left": 20.0, "right": 3.0},
        },
    )
    planner.plan(obs, np.zeros(4))
    assert len(seen) >= 3
    assert np.allclose(seen[0], 0.0)
    assert np.allclose(seen[1], [0.1, 0.0, 0.0, 0.25])
    assert np.allclose(seen[2], [0.1, 0.0, 0.0, 0.25])


def test_closed_loop_requires_tail_policy():
    dyn = StubLatentDynamics(goal=np.array([20.0, 0.0, 0.0]), latent_dim=8)
    planner = ImaginationPlanner(
        dyn, horizon=2, reward_cfg=RewardConfig(), rollout_mode="closed_loop"
    )
    obs = _obs([0.0, 0.0, 0.0], info={"goal": np.array([20.0, 0.0, 0.0])})
    with pytest.raises(RuntimeError):
        planner.plan(obs, np.zeros(4))


def test_planner_passes_goal_rel_into_imagine(monkeypatch):
    captured: dict = {}

    def _spy_imagine(*args, **kwargs):
        captured.update(kwargs)
        from experiments.aerial.rl.imagination import imagine as real_imagine

        return real_imagine(*args, **kwargs)

    monkeypatch.setattr("experiments.aerial.rl.planner.imagine", _spy_imagine)
    dyn = StubLatentDynamics(goal=np.array([20.0, 0.0, 0.0]), latent_dim=8)
    planner = ImaginationPlanner(dyn, horizon=2, reward_cfg=RewardConfig())
    obs = _obs([0.0, 0.0, 0.0], info={"goal": np.array([20.0, 0.0, 0.0])})
    planner.plan(obs, np.zeros(4))
    assert captured.get("goal_rel0") is not None
    assert float(captured["goal_rel0"][0, 0]) > 0.0
    assert captured.get("body_vel0") is not None
    assert captured.get("propagate_goal_rel") is True


def test_depth_tau_shield_records_independent_channel():
    shield = DepthTauShield(min_depth_m=3.0, min_tau_s=2.0)
    obs = _obs([0.0, 0.0, 0.0], info={"tau_pred": 0.5})
    assert shield.should_override(obs)
    assert shield.last_channels == ("tau",)
    assert obs.info["shield_channels"] == ["tau"]


def test_dual_channel_independence_metric():
    d = np.array([True, False, True, False, False])
    t = np.array([False, True, False, False, True])
    out = v1_metrics.check_dual_channel_independence(d, t, max_both_fail_frac=0.5)
    assert out["ok"] is True
    assert out["both_fail_frac"] == 0.0


def test_collector_runs_with_planner():
    class _Env:
        def __init__(self):
            self.config = type("C", (), {"step_hz": 5.0})()
            self.goal = np.array([20.0, 0.0, 0.0])

        def reset(self, episode=None):
            return _obs([0.0, 0.0, 0.0])

        def step(self, action):
            return _obs([1.0, 0.0, 0.0]), {}

    class _Policy:
        def act(self, view):
            return np.zeros(4, dtype=np.float64)

    dyn = StubLatentDynamics(goal=np.array([20.0, 0.0, 0.0]), latent_dim=8)
    planner = ImaginationPlanner(dyn, horizon=2)
    buf = ReplayBuffer(capacity_episodes=1, seed=0)
    col = RolloutCollector(
        _Env(), _Policy(), buf,
        max_steps=2, target_hz=0.0, planner=planner,
        skip_reset_collision=False,
    )
    ep, stats = col.collect_episode()
    assert stats.steps == 2
    assert len(ep) == 2


def test_collector_planner_uses_obs_info_subgoal():
    """toward_g collect: planner.set_goal must follow clipped obs.info['goal']."""

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
            self.goal = np.array([100.0, 0.0, 0.0])

        def reset(self, episode=None):
            o = _obs([0.0, 0.0, 0.0])
            o.info["goal"] = [25.0, 0.0, 0.0]
            return o

        def step(self, action):
            o = _obs([1.0, 0.0, 0.0])
            o.info["goal"] = [24.0, 0.0, 0.0]
            return o, {}

    class _Policy:
        def act(self, view):
            return np.zeros(4, dtype=np.float64)

    planner = _TrackingPlanner()
    buf = ReplayBuffer(capacity_episodes=1, seed=0)
    col = RolloutCollector(
        _Env(), _Policy(), buf,
        max_steps=1, target_hz=0.0, planner=planner,
        skip_reset_collision=False,
    )
    col.collect_episode()
    assert planner.last_goal is not None
    # obs.info subgoal (25 m), not env.goal terminal G (100 m)
    np.testing.assert_allclose(planner.last_goal, [25.0, 0.0, 0.0])
