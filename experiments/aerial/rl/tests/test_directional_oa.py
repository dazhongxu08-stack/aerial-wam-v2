"""Directional OA: yaw advance, depth labels, obstacle head, path shaping."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from experiments.aerial.rl.depth_geometry import (
    clearance_to_obstacle_label,
    directional_clearance_m,
)
from experiments.aerial.rl.goal_features import advance_goal_rel_body
from experiments.aerial.rl.reward import (
    OBSTACLE_COST_EMPTY,
    OBSTACLE_COST_NEAR,
    NavigationReward,
    directional_oa_reward_cfg,
    path_shaping_terms,
    reward_terms,
)
from experiments.aerial.rl.env.obs import Observation


def test_advance_goal_rel_body_rotates_on_yaw():
    g0 = np.array([10.0, 0.0, 0.0, 10.0], dtype=np.float32)
    # Pure +90° yaw (CCW): goal that was ahead moves to body -left (right).
    g1 = advance_goal_rel_body(g0, np.array([0.0, 0.0, 0.0, np.pi / 2], dtype=np.float64))
    np.testing.assert_allclose(g1[:3], [0.0, -10.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(g1[3], 10.0, atol=1e-5)


def test_advance_goal_rel_body_disp_then_yaw():
    g0 = np.array([5.0, 0.0, 0.0, 5.0], dtype=np.float32)
    # 1 m forward then +90°: remaining [4,0] → [0,-4].
    g1 = advance_goal_rel_body(g0, np.array([1.0, 0.0, 0.0, np.pi / 2], dtype=np.float64))
    np.testing.assert_allclose(g1[:3], [0.0, -4.0, 0.0], atol=1e-5)


def _synthetic_depth(h=48, w=64, *, wall_az_deg: float, wall_m: float, far_m: float = 40.0):
    """Flat far depth with a near wall wedge around ``wall_az_deg`` (+az = left)."""
    d = np.full((h, w), far_m, dtype=np.float64)
    half_h = 45.0
    cols = (np.arange(w, dtype=np.float64) + 0.5) / w
    az = (0.5 - cols) * 2.0 * half_h
    for j in range(w):
        if abs(az[j] - wall_az_deg) <= 20.0:
            d[:, j] = wall_m
    return d


def test_directional_clearance_fwd_left_up():
    # Wall ahead at 4 m; sides far.
    depth = _synthetic_depth(wall_az_deg=0.0, wall_m=4.0)
    c_fwd = directional_clearance_m(depth, np.array([1.0, 0.0, 0.0]))
    c_left = directional_clearance_m(depth, np.array([0.0, 1.0, 0.0]))
    assert c_fwd < 6.0
    assert c_left > 20.0

    # Near patch only in upper-centre; fwd+up sees it, pure left stays far.
    depth_up = np.full((48, 64), 40.0, dtype=np.float64)
    depth_up[:16, 20:44] = 3.0
    c_up = directional_clearance_m(depth_up, np.array([1.0, 0.0, 1.0]))
    c_left_u = directional_clearance_m(depth_up, np.array([0.0, 1.0, 0.0]))
    assert c_up < 5.0
    assert c_left_u > 20.0

    assert clearance_to_obstacle_label(c_fwd) > clearance_to_obstacle_label(c_left)
    assert clearance_to_obstacle_label(3.0) == pytest.approx(1.0)
    assert clearance_to_obstacle_label(22.0) == pytest.approx(0.0)


def test_obstacle_cost_head_interface_no_depth_kw():
    torch = pytest.importorskip("torch")
    from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics

    sig = inspect.signature(TorchRSSMDynamics.predict_obstacle_cost)
    assert "depth" not in sig.parameters
    dyn = TorchRSSMDynamics(
        image_size=16,
        action_dim=4,
        recurrent_dim=8,
        stoch_dim=4,
        stoch_classes=4,
        hidden_dim=16,
        device="cpu",
    )
    dyn.obstacle_cost_trained = True
    feat = np.zeros(dyn.latent_dim, dtype=np.float32)
    a_fwd = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    a_left = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    c0 = dyn.predict_obstacle_cost(feat, a_fwd)
    c1 = dyn.predict_obstacle_cost(feat, a_left)
    assert np.isfinite(c0) and np.isfinite(c1) and c0 >= 0.0 and c1 >= 0.0
    # Untrained must not leak into reward.
    dyn.obstacle_cost_trained = False
    assert dyn.predict_obstacle_cost(feat, a_fwd) == 0.0


def _step_reward(action, goal_rel, progress, *, obstacle, cfg, hist_level=False):
    shaping = path_shaping_terms(
        action, goal_rel, progress, hist_level=hist_level, cfg=cfg
    )
    return reward_terms(
        float(shaping.get("progress_eff", progress)),
        float(obstacle),
        float(np.linalg.norm(action)),
        cfg,
        path_shaping_val=float(shaping["path_shaping"]),
    )["reward"]


def test_path_shaping_empty_forward_beats_side_climb_hover_away():
    cfg = directional_oa_reward_cfg()
    g = np.array([20.0, 0.0, 0.0, 20.0], dtype=np.float64)
    fwd = np.array([1.0, 0.0, 0.0, 0.0])
    side = np.array([0.7, 0.7, 0.0, 0.0])  # similar Δdist toward goal, less pure
    climb = np.array([0.7, 0.0, 0.7, 0.0])
    hover = np.array([0.0, 0.0, 0.0, 0.0])
    away = np.array([-1.0, 0.0, 0.0, 0.0])
    r_fwd = _step_reward(fwd, g, 1.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    r_side = _step_reward(side, g, 0.7, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    r_climb = _step_reward(climb, g, 0.7, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    r_hover = _step_reward(hover, g, 0.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    r_away = _step_reward(away, g, -1.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    assert r_fwd > r_side > r_hover
    assert r_fwd > r_climb
    assert r_fwd > r_away
    assert r_hover > r_away


def test_aligning_yaw_between_hover_and_straight():
    cfg = directional_oa_reward_cfg()
    # Goal to the left → need +yaw (CCW) to face it.
    g = np.array([0.0, 10.0, 0.0, 10.0], dtype=np.float64)
    yaw = np.array([0.0, 0.0, 0.0, 0.4])
    hover = np.zeros(4)
    away = np.array([-1.0, 0.0, 0.0, 0.0])
    # After facing: carrot ahead + nose-forward is the only "straight".
    g_ahead = np.array([10.0, 0.0, 0.0, 10.0], dtype=np.float64)
    straight = np.array([1.0, 0.0, 0.0, 0.0])
    r_yaw = _step_reward(yaw, g, 0.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    r_hover = _step_reward(hover, g, 0.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    r_away = _step_reward(away, g, -1.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    r_str = _step_reward(straight, g_ahead, 1.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    assert r_str > r_yaw > r_hover
    assert r_yaw > r_away


def test_reverse_closing_does_not_beat_nose_forward():
    """Reverse-SR exploit: closing while nose-away / dx<0 must lose to fwd."""
    cfg = directional_oa_reward_cfg()
    g_ahead = np.array([20.0, 0.0, 0.0, 20.0], dtype=np.float64)
    g_behind = np.array([-20.0, 0.0, 0.0, 20.0], dtype=np.float64)
    fwd = np.array([1.0, 0.0, 0.0, 0.0])
    back = np.array([-1.0, 0.0, 0.0, 0.0])
    yaw_align = np.array([0.0, 0.0, 0.0, 0.5])  # pure yaw toward behind carrot
    r_fwd = _step_reward(fwd, g_ahead, 1.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    # Carrot ahead but body reverse → away + backward tax.
    r_back = _step_reward(back, g_ahead, -1.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    assert r_fwd > r_back
    # Carrot behind: reverse that closes Euclidean distance still pays w_backward
    # (no ahead-only exemption — that hole made nose-away reverse a local opt).
    r_rev_close = _step_reward(back, g_behind, 1.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    assert r_fwd > r_rev_close
    assert r_rev_close <= 0.0
    from experiments.aerial.rl.reward import backward_flight_cost

    assert backward_flight_cost(back, g_behind, cfg=cfg) == pytest.approx(
        float(cfg.w_backward) * 1.0
    )
    # Pure yaw while goal behind: no backward tax; beat reverse closing.
    r_yaw = _step_reward(yaw_align, g_behind, 0.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg)
    assert backward_flight_cost(yaw_align, g_behind, cfg=cfg) == 0.0
    assert r_yaw > r_rev_close


def test_clearance_along_action_skips_backward():
    """Backward-dominant motion must not inherit forward-cone clearance tax."""
    from experiments.aerial.rl.reward import clearance_m_along_action

    cones = {"forward": 2.0, "left": 20.0, "right": 20.0}
    assert clearance_m_along_action(
        np.array([1.0, 0.0, 0.0, 0.0]), cones, 2.0
    ) == pytest.approx(2.0)
    assert clearance_m_along_action(
        np.array([-1.0, 0.0, 0.0, 0.0]), cones, 2.0
    ) is None
    assert clearance_m_along_action(
        np.array([0.1, 1.0, 0.0, 0.0]), cones, 2.0
    ) == pytest.approx(20.0)


def test_directional_oa_forbids_backward_by_default():
    cfg = directional_oa_reward_cfg()
    assert cfg.forbid_backward_motion is True
    assert float(cfg.w_backward) >= 5.0


def test_near_wall_straight_loses_to_empty_approach():
    cfg = directional_oa_reward_cfg()
    g = np.array([20.0, 0.0, 0.0, 20.0], dtype=np.float64)
    into_wall = np.array([1.0, 0.0, 0.0, 0.0])
    empty_close = np.array([0.8, 0.0, 0.0, 0.0])
    r_wall = _step_reward(
        into_wall, g, 1.0, obstacle=OBSTACLE_COST_NEAR, cfg=cfg
    )
    r_empty = _step_reward(
        empty_close, g, 0.8, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg
    )
    assert r_empty > r_wall


def test_level_flight_window_beats_by_straight_segment():
    cfg = directional_oa_reward_cfg()
    g = np.array([30.0, 0.0, 0.0, 30.0], dtype=np.float64)
    level_a = np.array([1.0, 0.0, 0.0, 0.0])
    # Sustained level + no progress → hist_level True charges w_level_flight.
    r_level = _step_reward(
        level_a, g, 0.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg, hist_level=True
    )
    r_str = _step_reward(
        level_a, g, 1.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg, hist_level=False
    )
    assert r_str > r_level
    # Straight closing step must not pay level tax even if hist was level.
    r_str_hist = _step_reward(
        level_a, g, 1.0, obstacle=OBSTACLE_COST_EMPTY, cfg=cfg, hist_level=True
    )
    assert r_str_hist == pytest.approx(r_str)


def test_offline_label_batch_and_gate_helpers():
    from experiments.aerial.rl.train_obstacle_cost_labels import (
        gate_sort_checks,
        label_batch,
        score_candidate,
    )

    depth = _synthetic_depth(wall_az_deg=0.0, wall_m=4.0)
    depths = np.stack([depth, depth], axis=0)
    acts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    out = label_batch(depths, acts)
    assert out["obstacle_label"][0] > out["obstacle_label"][1]

    cfg = directional_oa_reward_cfg()
    g = np.array([20.0, 0.0, 0.0, 20.0])
    r_fwd = score_candidate(
        np.array([1.0, 0, 0, 0]), g, 1.0, OBSTACLE_COST_EMPTY, cfg=cfg
    )
    r_side = score_candidate(
        np.array([0.7, 0.7, 0, 0]), g, 0.7, OBSTACLE_COST_EMPTY, cfg=cfg
    )
    r_hover = score_candidate(np.zeros(4), g, 0.0, OBSTACLE_COST_EMPTY, cfg=cfg)
    r_away = score_candidate(
        np.array([-1.0, 0, 0, 0]), g, -1.0, OBSTACLE_COST_EMPTY, cfg=cfg
    )
    r_wall = score_candidate(
        np.array([1.0, 0, 0, 0]), g, 1.0, OBSTACLE_COST_NEAR, cfg=cfg
    )
    r_empty = score_candidate(
        np.array([0.8, 0, 0, 0]), g, 0.8, OBSTACLE_COST_EMPTY, cfg=cfg
    )
    # Constant-cost stand-in for left-near: left action pays near, fwd pays empty.
    r_fwd_ln = score_candidate(
        np.array([1.0, 0, 0, 0]), g, 1.0, OBSTACLE_COST_EMPTY, cfg=cfg
    )
    r_left_ln = score_candidate(
        np.array([0.0, 1.0, 0, 0]), g, 0.0, OBSTACLE_COST_NEAR, cfg=cfg
    )
    checks = gate_sort_checks(
        r_fwd_empty=r_fwd,
        r_side_empty=r_side,
        r_hover=r_hover,
        r_away=r_away,
        r_fwd_near=r_wall,
        r_empty_approach=r_empty,
        r_fwd_left_near=r_fwd_ln,
        r_left_left_near=r_left_ln,
    )
    assert all(checks.values()), checks


def test_navigation_reward_uses_stamped_obstacle_cost():
    cfg = directional_oa_reward_cfg(w_maneuver=0.0)
    r = NavigationReward(goal=np.array([10.0, 0.0, 0.0]), cfg=cfg)
    r.reset(goal=np.array([10.0, 0.0, 0.0]), start_pos=np.zeros(3))
    state = np.array([1.0, 0.0, 0.0, 0, 0, 0, 0], dtype=np.float32)
    obs = Observation(
        rgb=np.zeros((4, 4, 3), np.uint8),
        state=state,
        info={
            "obstacle_cost": OBSTACLE_COST_NEAR,
            "goal_rel": np.array([9.0, 0.0, 0.0, 9.0]),
            "depth_min_pred": 50.0,  # legacy path would see this as empty
        },
    )
    reward, _, terms = r.step(obs, np.array([1.0, 0.0, 0.0, 0.0]))
    assert terms["collision_risk"] == pytest.approx(OBSTACLE_COST_NEAR)
    assert reward < 1.0  # near cost bites into progress+straight


def test_learned_path_missing_oc_ignores_p_coll():
    cfg = directional_oa_reward_cfg(w_maneuver=0.0, w_progress=0.0, w_straight=0.0)
    r = NavigationReward(goal=np.array([10.0, 0.0, 0.0]), cfg=cfg)
    r.reset(goal=np.array([10.0, 0.0, 0.0]), start_pos=np.zeros(3))
    obs = Observation(
        rgb=np.zeros((4, 4, 3), np.uint8),
        state=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        info={"goal_rel": np.array([10.0, 0.0, 0.0, 10.0])},
    )
    reward, _, terms = r.step(obs, np.zeros(4), p_coll=1.0)
    assert terms["collision_risk"] == pytest.approx(0.0)


def test_carrot_progress_matches_advance_goal_rel():
    cfg = directional_oa_reward_cfg(w_maneuver=0.0, w_collision=0.0)
    r = NavigationReward(goal=np.array([100.0, 0.0, 0.0]), cfg=cfg)
    r.reset(goal=np.array([100.0, 0.0, 0.0]), start_pos=np.zeros(3))
    # Carrot 5 m ahead; 1 m forward → progress 1 even if world pos unchanged.
    obs = Observation(
        rgb=np.zeros((4, 4, 3), np.uint8),
        state=np.zeros(7, dtype=np.float32),
        info={
            "goal_rel": np.array([5.0, 0.0, 0.0, 5.0]),
            "obstacle_cost": 0.0,
        },
    )
    _, _, terms = r.step(obs, np.array([1.0, 0.0, 0.0, 0.0]))
    assert terms["progress"] == pytest.approx(1.0, abs=1e-5)


def test_level_flight_window_on_same_trajectory():
    """W consecutive level+no-progress steps charge; a closing step clears tax."""
    cfg = directional_oa_reward_cfg(w_maneuver=0.0, level_window=3)
    r = NavigationReward(goal=np.array([30.0, 0.0, 0.0]), cfg=cfg)
    r.reset(goal=np.array([30.0, 0.0, 0.0]), start_pos=np.zeros(3))
    hover = np.zeros(4)
    level_move = np.array([1.0, 0.0, 0.0, 0.0])  # still no carrot progress if g fixed wrong
    # Use goal_rel that does not shrink under hover; under forward it shrinks.
    g_far = np.array([30.0, 0.0, 0.0, 30.0])
    rewards = []
    for _ in range(3):
        obs = Observation(
            rgb=np.zeros((4, 4, 3), np.uint8),
            state=np.zeros(7, dtype=np.float32),
            info={"goal_rel": g_far.copy(), "obstacle_cost": 0.0},
        )
        rew, _, terms = r.step(obs, hover)
        rewards.append(float(rew))
    assert terms["level_flight_cost"] > 0.0
    # Closing straight step on same episode must not pay level tax.
    obs2 = Observation(
        rgb=np.zeros((4, 4, 3), np.uint8),
        state=np.zeros(7, dtype=np.float32),
        info={"goal_rel": g_far.copy(), "obstacle_cost": 0.0},
    )
    _, _, terms2 = r.step(obs2, level_move)
    assert terms2["level_flight_cost"] == pytest.approx(0.0)
    assert terms2["straight_bonus"] > 0.0


def test_obstacle_cost_trained_requires_explicit_flag(tmp_path):
    torch = pytest.importorskip("torch")
    from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics

    dyn = TorchRSSMDynamics(
        image_size=16, action_dim=4, recurrent_dim=8, stoch_dim=4,
        stoch_classes=4, hidden_dim=16, device="cpu",
    )
    assert dyn.obstacle_cost_trained is False
    # Keys always present after architecture land; must NOT auto-enable.
    p = str(tmp_path / "wm_random_head.pt")
    dyn.save_checkpoint(p, step=0)
    dyn2 = TorchRSSMDynamics(
        image_size=16, action_dim=4, recurrent_dim=8, stoch_dim=4,
        stoch_classes=4, hidden_dim=16, device="cpu",
    )
    payload = dyn2.load_checkpoint(p)
    assert payload["obstacle_cost_keys_loaded"] > 0
    assert dyn2.obstacle_cost_trained is False
    # Explicit mark + save → reload True.
    dyn.mark_obstacle_cost_trained(True)
    p2 = str(tmp_path / "wm_trained.pt")
    dyn.save_checkpoint(p2, step=1)
    dyn3 = TorchRSSMDynamics(
        image_size=16, action_dim=4, recurrent_dim=8, stoch_dim=4,
        stoch_classes=4, hidden_dim=16, device="cpu",
    )
    payload3 = dyn3.load_checkpoint(p2)
    assert dyn3.obstacle_cost_trained is True
    assert payload3["obstacle_cost_trained"] is True


def test_predict_obstacle_cost_rejects_wrong_feature_dim():
    torch = pytest.importorskip("torch")
    from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics

    dyn = TorchRSSMDynamics(
        image_size=16, action_dim=4, recurrent_dim=8, stoch_dim=4,
        stoch_classes=4, hidden_dim=16, device="cpu",
    )
    dyn.mark_obstacle_cost_trained(True)
    with pytest.raises(ValueError, match="latent_dim"):
        dyn.predict_obstacle_cost(np.zeros(3, dtype=np.float32), np.zeros(4))


def test_collector_stamps_obstacle_cost_from_feature_t():
    """Executed action + pre-step latent → next_obs.info['obstacle_cost']."""
    from experiments.aerial.rl.buffer import ReplayBuffer
    from experiments.aerial.rl.collector import RolloutCollector
    from experiments.aerial.rl.dynamics import DynamicsOutput, StubLatentDynamics
    from experiments.aerial.rl.train_rl import HeuristicPolicy

    class _Env:
        def __init__(self):
            self.config = type("C", (), {"step_hz": 5.0})()
            self.goal = np.array([20.0, 0.0, 0.0], dtype=np.float64)
            self._pos = np.zeros(3)

        def reset(self, episode=None):
            self._pos = np.zeros(3)
            return Observation(
                rgb=np.zeros((8, 8, 3), np.uint8),
                state=np.zeros(7, np.float32),
                info={"goal": [20.0, 0.0, 0.0]},
            )

        def step(self, action):
            self._pos = self._pos + np.asarray(action, dtype=np.float64)[:3]
            st = np.array(
                [self._pos[0], self._pos[1], self._pos[2], 0, 0, 0, 0], np.float32
            )
            return Observation(rgb=np.zeros((8, 8, 3), np.uint8), state=st), {}

    class _Dyn(StubLatentDynamics):
        def __init__(self):
            super().__init__(goal=np.array([20.0, 0.0, 0.0]), latent_dim=8)
            self.obstacle_cost_trained = True
            self.last_feat = None
            self.last_act = None

        def predict_obstacle_cost(self, feature, action):
            self.last_feat = np.asarray(feature, dtype=np.float64).copy()
            self.last_act = np.asarray(action, dtype=np.float64).copy()
            return 0.42

        def step(self, z, action, goal_rel=None, body_vel=None):
            out = super().step(z, action, goal_rel=goal_rel, body_vel=body_vel)
            return DynamicsOutput(
                z_next=out.z_next, p_coll=0.0, progress=out.progress,
                done=False, arrived=False,
            )

    dyn = _Dyn()
    env = _Env()
    col = RolloutCollector(
        env,
        HeuristicPolicy(goal_getter=lambda: env.goal),
        ReplayBuffer(capacity_episodes=4, seed=0),
        dynamics=dyn,
        safety=None,
        max_steps=2,
        target_hz=0.0,
        skip_reset_collision=False,
        reward_cfg=directional_oa_reward_cfg(),
    )
    ep, _ = col.collect_episode()
    assert dyn.last_feat is not None
    assert dyn.last_act is not None
    assert any(
        isinstance(t.next_obs.info, dict)
        and t.next_obs.info.get("obstacle_cost") == 0.42
        for t in ep
    )


def test_train_obstacle_head_marks_trained():
    torch = pytest.importorskip("torch")
    from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics
    from experiments.aerial.rl.train_obstacle_cost_labels import train_obstacle_cost_head

    dyn = TorchRSSMDynamics(
        image_size=16, action_dim=4, recurrent_dim=8, stoch_dim=4,
        stoch_classes=4, hidden_dim=16, device="cpu",
    )
    n, d = 64, dyn.latent_dim
    feat = np.random.randn(n, d).astype(np.float32)
    act = np.zeros((n, 4), dtype=np.float32)
    act[:, 0] = 1.0
    labels = np.linspace(0.0, 1.0, n).astype(np.float32)
    stats = train_obstacle_cost_head(dyn, feat, act, labels, steps=40, batch=16)
    assert dyn.obstacle_cost_trained is True
    assert np.isfinite(stats["loss"])


def test_pack_features_from_frames_encodes_latent_dim():
    torch = pytest.importorskip("torch")
    from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics
    from experiments.aerial.rl.train_obstacle_cost_labels import pack_features_from_frames

    dyn = TorchRSSMDynamics(
        image_size=16, action_dim=4, recurrent_dim=8, stoch_dim=4,
        stoch_classes=4, hidden_dim=16, device="cpu",
    )
    n = 4
    rgbs = np.random.randint(0, 256, size=(n, 16, 16, 3), dtype=np.uint8)
    depth = np.full((n, 16, 16), 40.0, dtype=np.float64)
    depth[1, :, 6:10] = 3.0
    acts = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    groups = np.array(["fwd_empty", "fwd_near", "left_near", "fwd_empty"])
    out = pack_features_from_frames(dyn, rgbs, acts, depths=depth, groups=groups)
    assert out["feature"].shape == (n, dyn.latent_dim)
    assert out["action"].shape == (n, 4)
    assert out["obstacle_label"].shape == (n,)
    assert "group" in out


def test_run_task5_gate_api():
    torch = pytest.importorskip("torch")
    from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics
    from experiments.aerial.rl.train_obstacle_cost_labels import (
        run_task5_gate,
        train_obstacle_cost_head,
    )

    dyn = TorchRSSMDynamics(
        image_size=16, action_dim=4, recurrent_dim=8, stoch_dim=4,
        stoch_classes=4, hidden_dim=16, device="cpu",
    )
    rng = np.random.default_rng(0)
    rows, labels, acts, groups = [], [], [], []
    for _ in range(20):
        f = rng.normal(size=dyn.latent_dim).astype(np.float32)
        for a, y, g in (
            ([1.0, 0.0, 0.0, 0.0], 0.0, "fwd_empty"),
            ([1.0, 0.0, 0.0, 0.0], 1.0, "fwd_near"),
            ([0.0, 1.0, 0.0, 0.0], 1.0, "left_near"),
            ([1.0, 0.0, 0.0, 0.0], 0.0, "left_near"),
        ):
            rows.append(f.copy())
            acts.append(a)
            labels.append(y)
            groups.append(g)
    feat = np.stack(rows)
    act = np.asarray(acts, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    train_obstacle_cost_head(dyn, feat, act, y, steps=80, batch=32, seed=0)
    g_feat = np.stack([
        feat[groups.index("fwd_empty")],
        feat[groups.index("fwd_near")],
        feat[groups.index("left_near")],
    ])
    report = run_task5_gate(dyn, g_feat, np.array(["fwd_empty", "fwd_near", "left_near"]))
    assert set(report["checks"]) >= {
        "fwd_empty_wins", "fwd_near_loses", "left_near_prefers_fwd"
    }
    assert "numbers" in report


def test_deep_merge_overlay():
    from experiments.aerial.rl.train_v4_ac import _deep_merge

    base = {"reward": {"w_collision": 10.0, "w_progress": 1.0}, "safety": {"kind": "three_zone"}}
    over = {
        "reward": {"w_collision": 1.0, "use_learned_obstacle_cost": True},
        "safety": {"kind": "null"},
    }
    m = _deep_merge(base, over)
    assert m["reward"]["w_collision"] == 1.0
    assert m["reward"]["w_progress"] == 1.0
    assert m["reward"]["use_learned_obstacle_cost"] is True
    assert m["safety"]["kind"] == "null"


def test_collector_stamp_raises_on_predict_error():
    from experiments.aerial.rl.buffer import ReplayBuffer
    from experiments.aerial.rl.collector import RolloutCollector
    from experiments.aerial.rl.dynamics import DynamicsOutput, StubLatentDynamics
    from experiments.aerial.rl.train_rl import HeuristicPolicy

    class _Env:
        def __init__(self):
            self.config = type("C", (), {"step_hz": 5.0})()
            self.goal = np.array([20.0, 0.0, 0.0], dtype=np.float64)

        def reset(self, episode=None):
            return Observation(
                rgb=np.zeros((8, 8, 3), np.uint8),
                state=np.zeros(7, np.float32),
                info={"goal": [20.0, 0.0, 0.0]},
            )

        def step(self, action):
            return Observation(
                rgb=np.zeros((8, 8, 3), np.uint8), state=np.zeros(7, np.float32)
            ), {}

    class _Dyn(StubLatentDynamics):
        def __init__(self):
            super().__init__(goal=np.array([20.0, 0.0, 0.0]), latent_dim=8)
            self.obstacle_cost_trained = True

        def predict_obstacle_cost(self, feature, action):
            raise ValueError("latent_dim mismatch")

        def step(self, z, action, goal_rel=None, body_vel=None):
            out = super().step(z, action, goal_rel=goal_rel, body_vel=body_vel)
            return DynamicsOutput(
                z_next=out.z_next, p_coll=0.0, progress=0.0, done=False, arrived=False
            )

    env = _Env()
    col = RolloutCollector(
        env,
        HeuristicPolicy(goal_getter=lambda: env.goal),
        ReplayBuffer(capacity_episodes=2, seed=0),
        dynamics=_Dyn(),
        max_steps=1,
        target_hz=0.0,
        skip_reset_collision=False,
        reward_cfg=directional_oa_reward_cfg(),
    )
    with pytest.raises(ValueError, match="latent_dim"):
        col.collect_episode()


def test_blend_clearance_risk_plugs_head_underreport():
    """Learned oc=0 but d_fwd near + forward action → still charges via blend."""
    from experiments.aerial.rl.reward import clearance_risk_from_depth

    cfg = directional_oa_reward_cfg(
        blend_clearance_risk=True, w_maneuver=0.0, w_progress=0.0, w_straight=0.0,
    )
    r = NavigationReward(goal=np.array([10.0, 0.0, 0.0]), cfg=cfg)
    r.reset(goal=np.array([10.0, 0.0, 0.0]), start_pos=np.zeros(3))
    obs = Observation(
        rgb=np.zeros((4, 4, 3), np.uint8),
        state=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        info={
            "obstacle_cost": 0.0,
            "depth_cones_pred": {"forward": 2.0, "left": 20.0, "right": 20.0},
            "goal_rel": np.array([10.0, 0.0, 0.0, 10.0]),
        },
    )
    _reward, _done, terms = r.step(obs, np.array([1.0, 0.0, 0.0, 0.0]))
    expect = clearance_risk_from_depth(2.0, d_danger=3.0, d_clear=12.0)
    assert float(terms["collision_risk"]) == pytest.approx(expect, abs=1e-5)
    assert float(terms["collision_risk"]) > 0.5


def test_blend_clearance_does_not_tax_side_escape_with_fwd_cliff():
    """Left-wall / open-fwd: sideways action must NOT inherit forward clearance tax."""
    cfg = directional_oa_reward_cfg(
        blend_clearance_risk=True, w_maneuver=0.0, w_progress=0.0, w_straight=0.0,
    )
    r = NavigationReward(goal=np.array([10.0, 0.0, 0.0]), cfg=cfg)
    r.reset(goal=np.array([10.0, 0.0, 0.0]), start_pos=np.zeros(3))
    obs = Observation(
        rgb=np.zeros((4, 4, 3), np.uint8),
        state=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        info={
            "obstacle_cost": 0.05,
            "depth_cones_pred": {"forward": 2.0, "left": 20.0, "right": 20.0},
            "goal_rel": np.array([10.0, 0.0, 0.0, 10.0]),
        },
    )
    _r, _d, terms = r.step(obs, np.array([0.2, 1.0, 0.0, 0.0]))  # left-dominant
    # Left cone open → blend ≈ 0; keep oc, do not jump to fwd cliff (~1).
    assert float(terms["collision_risk"]) == pytest.approx(0.05, abs=1e-5)
    assert float(terms["clearance_risk"]) < 0.1
