import numpy as np
import pytest

from experiments.aerial.rl.reward import (
    DEFAULT_ONLINE_SUCCESS_DIST_M,
    EVAL_SUCCESS_DIST_M,
    NavigationReward,
    RewardConfig,
    maneuver_weight_at,
    reward_terms,
)
from experiments.aerial.rl.env.obs import Observation


def test_bare_config_default_is_tight_online_radius_not_eval():
    # A RewardConfig()/NavigationReward() built WITHOUT the YAML path must still
    # use the tight online radius (3 m), never the loose 20 m eval SR radius.
    assert RewardConfig().success_dist_m == pytest.approx(DEFAULT_ONLINE_SUCCESS_DIST_M)
    assert DEFAULT_ONLINE_SUCCESS_DIST_M < EVAL_SUCCESS_DIST_M
    r = NavigationReward(goal=np.array([10.0, 0.0, 0.0]))
    r.reset(goal=np.array([10.0, 0.0, 0.0]), start_pos=np.zeros(3))
    # 5 m out: arrived under a 20 m radius, NOT under the 3 m default.
    _, done, terms = r.step(_obs([5.0, 0.0, 0.0]), np.zeros(4))
    assert not done
    assert terms["arrived"] == 0.0


def _obs(pos, collided=False, info=None):
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return Observation(
        rgb=np.zeros((4, 4, 3), np.uint8),
        state=state,
        collided=collided,
        info=info or {},
    )


def test_reward_terms_signs():
    cfg = RewardConfig(w_progress=1.0, w_collision=10.0, w_maneuver=0.1)
    t = reward_terms(progress=2.0, collision_risk=0.0, maneuver_cost=0.0, cfg=cfg)
    assert t["reward"] == pytest.approx(2.0)
    # collision subtracts; maneuver subtracts
    t2 = reward_terms(progress=0.0, collision_risk=1.0, maneuver_cost=3.0, cfg=cfg)
    assert t2["reward"] == pytest.approx(-10.0 - 0.3)


def test_progress_positive_when_approaching_goal():
    r = NavigationReward(goal=None, cfg=RewardConfig(success_dist_m=1.0))
    r.reset(goal=np.array([10.0, 0.0, 0.0]), start_pos=np.array([0.0, 0.0, 0.0]))
    reward, done, terms = r.step(_obs([3.0, 0.0, 0.0]), np.zeros(4))
    assert terms["progress"] == pytest.approx(3.0)
    assert reward > 0
    assert not done


def test_collision_makes_done_and_penalizes():
    cfg = RewardConfig(w_progress=1.0, w_collision=10.0, w_maneuver=0.0, success_dist_m=1.0)
    r = NavigationReward(goal=np.array([10.0, 0.0, 0.0]), cfg=cfg)
    r.reset(goal=np.array([10.0, 0.0, 0.0]), start_pos=np.array([0.0, 0.0, 0.0]))
    reward, done, terms = r.step(_obs([1.0, 0.0, 0.0], collided=True), np.zeros(4))
    assert done
    assert terms["collision_risk"] == 1.0
    assert reward < 0  # collision penalty dominates 1m of progress


def test_arrival_bonus_and_done():
    cfg = RewardConfig(success_dist_m=1.0, success_bonus=10.0, w_maneuver=0.0)
    r = NavigationReward(goal=np.array([5.0, 0.0, 0.0]), cfg=cfg)
    r.reset(goal=np.array([5.0, 0.0, 0.0]), start_pos=np.array([0.0, 0.0, 0.0]))
    reward, done, terms = r.step(_obs([4.8, 0.0, 0.0]), np.zeros(4))
    assert done
    assert terms["arrived"] == 1.0
    assert reward >= cfg.success_bonus  # bonus applied on top of progress


def test_maneuver_cost_is_action_norm():
    cfg = RewardConfig(w_progress=0.0, w_collision=0.0, w_maneuver=1.0)
    r = NavigationReward(goal=None, cfg=cfg)
    r.reset(goal=None, start_pos=np.zeros(3))
    action = np.array([3.0, 4.0, 0.0, 0.0])  # norm 5
    reward, done, terms = r.step(_obs([0.0, 0.0, 0.0]), action)
    assert terms["maneuver_cost"] == pytest.approx(5.0)
    assert reward == pytest.approx(-5.0)


def test_no_goal_yields_zero_progress():
    r = NavigationReward(goal=None)
    r.reset(goal=None, start_pos=np.zeros(3))
    _, _, terms = r.step(_obs([9.0, 9.0, 9.0]), np.zeros(4))
    assert terms["progress"] == 0.0


# --- maneuver-penalty curriculum --------------------------------------------

def test_curriculum_is_noop_when_final_equals_start():
    # default config: final == start -> flat regardless of the metric
    cfg = RewardConfig(w_maneuver=0.01)
    for m in (-100.0, 0.0, 5.0, 1000.0):
        assert maneuver_weight_at(m, cfg) == pytest.approx(0.01)


def test_curriculum_flat_below_threshold_then_ramps():
    cfg = RewardConfig(
        w_maneuver=0.01, w_maneuver_final=0.05,
        maneuver_curriculum_threshold=10.0, maneuver_curriculum_ramp=10.0,
    )
    assert maneuver_weight_at(0.0, cfg) == pytest.approx(0.01)   # below threshold
    assert maneuver_weight_at(10.0, cfg) == pytest.approx(0.01)  # at threshold
    assert maneuver_weight_at(15.0, cfg) == pytest.approx(0.03)  # halfway up the ramp
    assert maneuver_weight_at(20.0, cfg) == pytest.approx(0.05)  # top of the ramp
    assert maneuver_weight_at(999.0, cfg) == pytest.approx(0.05)  # never exceeds final


def test_curriculum_monotone_nondecreasing():
    cfg = RewardConfig(
        w_maneuver=0.01, w_maneuver_final=0.05,
        maneuver_curriculum_threshold=0.0, maneuver_curriculum_ramp=20.0,
    )
    ws = [maneuver_weight_at(m, cfg) for m in range(0, 30)]
    assert all(b >= a for a, b in zip(ws, ws[1:]))
    assert all(0.01 <= w <= 0.05 for w in ws)


def test_curriculum_zero_ramp_is_a_step():
    cfg = RewardConfig(
        w_maneuver=0.01, w_maneuver_final=0.05,
        maneuver_curriculum_threshold=10.0, maneuver_curriculum_ramp=0.0,
    )
    assert maneuver_weight_at(9.9, cfg) == pytest.approx(0.01)
    assert maneuver_weight_at(10.0, cfg) == pytest.approx(0.05)  # jumps at threshold


def test_curriculum_w_start_override_prevents_feedback():
    # simulate the corrector mutating cfg.w_maneuver: passing the snapshot as
    # w_start keeps the schedule anchored to the base, not the last output.
    cfg = RewardConfig(
        w_maneuver=0.03,  # already ramped up in a prior iter
        w_maneuver_final=0.05,
        maneuver_curriculum_threshold=0.0, maneuver_curriculum_ramp=10.0,
    )
    # metric=5 -> halfway; anchored to the true start 0.01, not the mutated 0.03
    assert maneuver_weight_at(5.0, cfg, w_start=0.01) == pytest.approx(0.03)


# --- F15 efficiency ----------------------------------------------------------

def test_efficiency_default_weights_are_noop():
    from experiments.aerial.rl.reward import efficiency_cost

    out = efficiency_cost(
        np.array([1.0, 10.0, 0.0, 0.0]),
        yaw_err_rad=1.0,
        ds_true_m=0.0,
    )
    assert out["efficiency_cost"] == pytest.approx(0.0)
    assert out["strafe_ratio"] > 1.0
    assert out["idle"] == 1.0


def test_efficiency_strafe_and_idle_penalize_when_weighted():
    from experiments.aerial.rl.reward import efficiency_cost

    cfg = RewardConfig(w_eff_strafe=1.0, w_eff_idle=2.0, eff_strafe_thr=0.5)
    # |dy|/|dx| = 2 → excess 1.5; ds≈0 → idle 1
    out = efficiency_cost(
        np.array([1.0, 2.0, 0.0, 0.0]),
        yaw_err_rad=0.0,
        ds_true_m=0.0,
        cfg=cfg,
    )
    assert out["strafe_excess"] == pytest.approx(1.5)
    assert out["efficiency_cost"] == pytest.approx(1.5 + 2.0)


def test_reward_terms_subtract_efficiency():
    cfg = RewardConfig(w_progress=0.0, w_collision=0.0, w_maneuver=0.0)
    t = reward_terms(0.0, 0.0, 0.0, cfg=cfg, efficiency_cost_val=3.0)
    assert t["reward"] == pytest.approx(-3.0)
    assert t["efficiency_cost"] == pytest.approx(3.0)


def test_efficiency_heading_penalizes_yaw_err_while_maneuvering():
    """F15: |yaw_err| × 1{|dx|+|dy|+|dyaw|>ε} — pure turn also sees heading."""
    from experiments.aerial.rl.reward import efficiency_cost

    cfg = RewardConfig(w_eff_heading=1.0, w_eff_strafe=0.0, w_eff_idle=0.0)
    # Forward thrust, no side slip, large yaw error → must still cost.
    thrust = efficiency_cost(
        np.array([1.0, 0.0, 0.0, 0.0]),
        yaw_err_rad=0.5,
        ds_true_m=1.0,
        cfg=cfg,
    )
    assert thrust["heading_term"] == pytest.approx(0.5)
    assert thrust["efficiency_cost"] == pytest.approx(0.5)
    # Pure yaw turn also pays heading (face-goal path).
    turn = efficiency_cost(
        np.array([0.0, 0.0, 0.0, 0.1]),
        yaw_err_rad=0.5,
        ds_true_m=0.0,
        cfg=cfg,
    )
    assert turn["heading_term"] == pytest.approx(0.5)
    assert turn["efficiency_cost"] == pytest.approx(0.5)
    # True idle (zero action) → heading term off.
    idle = efficiency_cost(
        np.zeros(4),
        yaw_err_rad=0.5,
        ds_true_m=0.0,
        cfg=cfg,
    )
    assert idle["heading_term"] == pytest.approx(0.0)
    assert idle["efficiency_cost"] == pytest.approx(0.0)


def test_clearance_risk_from_depth_piecewise():
    from experiments.aerial.rl.reward import clearance_risk_from_depth

    assert clearance_risk_from_depth(None) == 0.0
    assert clearance_risk_from_depth(1.0) == pytest.approx(1.0)   # < d_danger
    assert clearance_risk_from_depth(3.0) == pytest.approx(1.0)
    assert clearance_risk_from_depth(22.0) == pytest.approx(0.0)
    assert clearance_risk_from_depth(100.0) == pytest.approx(0.0)
    # Mid-band: (22-12.5)/(22-3) = 9.5/19
    assert clearance_risk_from_depth(12.5) == pytest.approx(9.5 / 19.0)
    # At d≈5 (hard-route median) risk is still high → imag penalty usable
    r5 = clearance_risk_from_depth(5.0)
    assert r5 > 0.8
    assert RewardConfig().w_collision * r5 > 5.0


def test_navigation_reward_folds_depth_min_clearance():
    """Real path mirrors imag: collision_risk = max(p_coll, clearance(depth_min))."""
    from experiments.aerial.rl.reward import clearance_risk_from_depth

    cfg = RewardConfig(w_progress=0.0, w_collision=10.0, w_maneuver=0.0)
    r = NavigationReward(goal=None, cfg=cfg)
    r.reset(goal=None, start_pos=np.zeros(3))
    obs = _obs([0.0, 0.0, 0.0], info={"depth_min_pred": 5.0})
    _, _, terms = r.step(obs, np.zeros(4), p_coll=0.0)
    expect = clearance_risk_from_depth(5.0)
    assert terms["clearance_risk"] == pytest.approx(expect)
    assert terms["collision_risk"] == pytest.approx(expect)
    assert terms["reward"] == pytest.approx(-10.0 * expect)

