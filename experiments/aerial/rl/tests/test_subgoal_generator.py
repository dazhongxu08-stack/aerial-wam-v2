# experiments/aerial/rl/tests/test_subgoal_generator.py
import numpy as np
import pytest
from experiments.aerial.rl.subgoal_generator import (
    hold_carrot_before_bend,
    project_to_polyline,
    sample_point_along_polyline,
    AdaptiveSubgoalGenerator,
)


def test_project_to_polyline_straight_line():
    path = np.array(
        [
            [0.0, 0.0, 10.0],
            [100.0, 0.0, 10.0],
            [200.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    pos = np.array([50.0, 5.0, 10.0], dtype=np.float64)
    proj, seg_idx, s_curr, rem_dist = project_to_polyline(pos, path, prev_s_max=0.0)
    np.testing.assert_allclose(proj, [50.0, 0.0, 10.0], atol=1e-5)
    assert seg_idx == 0
    assert pytest.approx(s_curr, abs=1e-3) == 50.0
    assert pytest.approx(rem_dist, abs=1e-3) == 150.0


def test_anti_backtracking_lock():
    path = np.array(
        [
            [0.0, 0.0, 10.0],
            [100.0, 0.0, 10.0],
            [200.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    pos_back = np.array([70.0, 0.0, 10.0], dtype=np.float64)
    proj, seg_idx, s_curr, rem_dist = project_to_polyline(pos_back, path, prev_s_max=80.0)
    assert s_curr >= 80.0
    assert pytest.approx(rem_dist, abs=1e-3) == 120.0


def test_sample_point_along_polyline():
    path = np.array(
        [
            [0.0, 0.0, 10.0],
            [100.0, 0.0, 10.0],
            [100.0, 100.0, 10.0],
        ],
        dtype=np.float64,
    )
    proj_point = np.array([80.0, 0.0, 10.0], dtype=np.float64)
    target = sample_point_along_polyline(
        path, segment_idx=0, proj_point=proj_point, r_lookahead=40.0
    )
    np.testing.assert_allclose(target, [100.0, 20.0, 10.0], atol=1e-5)


def test_adaptive_subgoal_generator_straight_clear():
    generator = AdaptiveSubgoalGenerator(
        r_base=55.0, r_min=20.0, cruise_speed=10.0, segment_length_m=0.0
    )
    path = np.array([[0.0, 0.0, 10.0], [200.0, 0.0, 10.0]], dtype=np.float64)
    g_rel, info = generator.compute_subgoal(
        np.array([0.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=30.0
    )
    assert g_rel.shape == (4,)
    assert pytest.approx(g_rel[0], abs=1.0) == 55.0
    assert pytest.approx(g_rel[1], abs=0.1) == 0.0
    assert pytest.approx(g_rel[3], abs=1.0) == 55.0
    assert info["alpha_clearance"] == 1.0
    assert info["safe_speed_limit"] >= 8.0


def test_default_r_base_is_local_pi_scale():
    """Mainline default carrot stays near step_e local-goal band (not 55 m)."""
    generator = AdaptiveSubgoalGenerator(cruise_speed=10.0)
    assert generator.r_base <= 30.0
    assert generator.r_min <= generator.r_base
    path = np.array([[0.0, 0.0, 10.0], [200.0, 0.0, 10.0]], dtype=np.float64)
    g_rel, info = generator.compute_subgoal(
        np.array([0.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    assert info["r_lookahead"] == pytest.approx(generator.r_base, abs=0.5)
    assert float(g_rel[3]) == pytest.approx(generator.r_base, abs=1.0)


def test_soft_cte_reentry_shrinks_before_5m():
    generator = AdaptiveSubgoalGenerator(cruise_speed=10.0)
    path = np.array([[0.0, 0.0, 10.0], [200.0, 0.0, 10.0]], dtype=np.float64)
    _, info = generator.compute_subgoal(
        np.array([40.0, 4.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    assert 3.0 < info["cte_m"] < 5.0
    assert info["r_lookahead"] < generator.r_base
    # default cte_reentry=2 must already be active at CTE≈4
    assert generator.cte_reentry_m <= 2.0 + 1e-6


def test_heading_peel_shrinks_lookahead_before_cte_grows():
    """F15 baseline: yaw dies early while CTE still small — shorten carrot for 汇入角."""
    generator = AdaptiveSubgoalGenerator(cruise_speed=10.0, r_base=25.0, r_min=15.0)
    path = np.array([[0.0, 0.0, 10.0], [200.0, 0.0, 10.0]], dtype=np.float64)
    # On-track (CTE≈0) but heading +90° vs +x tangent
    g_aligned, info_ok = generator.compute_subgoal(
        np.array([10.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    g_peel, info_peel = generator.compute_subgoal(
        np.array([10.0, 0.0, 10.0]), float(np.pi / 2.0), path, d_fwd_hat=40.0
    )
    assert info_ok["cte_m"] < 0.5
    assert info_peel["cte_m"] < 0.5
    assert info_ok["cos_heading_tang"] > 0.9
    assert info_peel["cos_heading_tang"] < 0.2
    assert info_peel["heading_reentry"] is True
    assert info_peel["r_lookahead"] <= generator.heading_reentry_r_m + 1e-6
    assert info_peel["r_lookahead"] < info_ok["r_lookahead"]
    # Peeled: carrot should have large body-lateral vs forward-aligned case
    assert abs(float(g_peel[1])) > abs(float(g_aligned[1])) + 5.0


def test_carrot_stays_on_the_inbound_leg_until_the_bend_is_close():
    path = np.array(
        [[0.0, 0.0, 10.0], [40.0, 0.0, 10.0], [40.0, 80.0, 10.0]],
        dtype=np.float64,
    )
    generator = AdaptiveSubgoalGenerator(r_base=25.0, r_min=15.0, cruise_speed=8.0)
    _, early = generator.compute_subgoal(
        np.array([20.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=30.0
    )
    # 20 m before the corner: carrot stays on the inbound leg.
    early_tw = np.asarray(early["target_world"], dtype=np.float64)
    assert early_tw[0] <= 40.0 + 1e-6
    assert abs(early_tw[1]) < 1.0
    _, late = generator.compute_subgoal(
        np.array([36.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=30.0
    )
    # 4 m before the corner: carrot is released onto the exit leg.
    late_tw = np.asarray(late["target_world"], dtype=np.float64)
    assert late_tw[1] > 8.0
    assert hold_carrot_before_bend(path, 20.0, 25.0) == pytest.approx(20.0)
    assert hold_carrot_before_bend(path, 36.0, 25.0) == pytest.approx(16.0)


def test_adaptive_subgoal_generator_tight_turn_and_obstacle():
    generator = AdaptiveSubgoalGenerator(r_base=55.0, r_min=20.0, cruise_speed=10.0)
    path = np.array(
        [[0.0, 0.0, 10.0], [20.0, 0.0, 10.0], [20.0, 100.0, 10.0]],
        dtype=np.float64,
    )
    g_rel, info = generator.compute_subgoal(
        np.array([10.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=3.0
    )
    assert info["r_lookahead"] <= 25.0
    assert info["alpha_clearance"] == pytest.approx(0.4, abs=1e-3)
    # Near danger: creep floor, not 4 m/s hard floor
    assert info["safe_speed_limit"] <= 1.0


def test_cte_recovery_shrinks_lookahead():
    generator = AdaptiveSubgoalGenerator(r_base=55.0, r_min=20.0, cruise_speed=10.0)
    path = np.array([[0.0, 0.0, 10.0], [200.0, 0.0, 10.0]], dtype=np.float64)
    # Large cross-track error
    g_rel, info = generator.compute_subgoal(
        np.array([50.0, 12.0, 10.0]), 0.0, path, d_fwd_hat=30.0
    )
    assert info["cte_m"] > 5.0
    assert info["r_lookahead"] < 55.0


def test_monotone_lock_updates_seg_idx():
    path = np.array(
        [[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [200.0, 0.0, 10.0]],
        dtype=np.float64,
    )
    # Orthogonal nearest is still on seg0, but lock forces s>=150 → seg1
    pos = np.array([40.0, 0.0, 10.0], dtype=np.float64)
    proj, seg_idx, s_curr, rem = project_to_polyline(pos, path, prev_s_max=150.0)
    assert s_curr >= 150.0
    assert seg_idx == 1
    np.testing.assert_allclose(proj[0], 150.0, atol=1e-5)


def test_terminal_creep_caps_v_safe():
    gen = AdaptiveSubgoalGenerator(
        r_base=55.0, r_min=20.0, cruise_speed=10.0, terminal_creep_rem_m=8.0
    )
    path = np.array([[0.0, 0.0, 10.0], [100.0, 0.0, 10.0]], dtype=np.float64)
    # Near end of path with clear depth
    _, info = gen.compute_subgoal(
        np.array([96.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    assert info["rem_dist"] < 8.0
    assert info["safe_speed_limit"] < 10.0
    assert info["safe_speed_limit"] <= gen.min_creep_speed + 1e-6 + (
        info["rem_dist"] / 8.0
    ) * (10.0 - gen.min_creep_speed)


def test_cte_uses_true_projection_not_locked_point():
    """Off-track + high s-lock must not report CTE to the locked arc point."""
    gen = AdaptiveSubgoalGenerator(r_base=55.0, r_min=20.0, cruise_speed=10.0)
    path = np.array(
        [[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [200.0, 0.0, 10.0]],
        dtype=np.float64,
    )
    gen._prev_s_max = 150.0
    # True nearest is ~(40,0); lateral CTE=30. Locked point would be ~(150,0).
    _, info = gen.compute_subgoal(
        np.array([40.0, 30.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    assert info["cte_m"] == pytest.approx(30.0, abs=0.5)
    # Carrot must pull from true projection, not from locked s=150.
    target = np.asarray(info["target_world"], dtype=np.float64)
    assert target[0] < 100.0


def test_offtrack_freezes_monotone_progress():
    """CTE above freeze threshold must not inflate s_progress via monotone lock."""
    gen = AdaptiveSubgoalGenerator(
        r_base=55.0, r_min=20.0, cruise_speed=10.0, cte_lock_freeze_m=5.0
    )
    path = np.array(
        [[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [200.0, 0.0, 10.0]],
        dtype=np.float64,
    )
    gen._prev_s_max = 150.0
    _, info = gen.compute_subgoal(
        np.array([40.0, 30.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    assert info["cte_m"] > 5.0
    assert info["s_progress"] == pytest.approx(40.0, abs=1.0)
    assert info.get("s_lock_frozen") is True
    # Next on-track step should not re-jump to 150.
    assert gen._prev_s_max == pytest.approx(info["s_progress"], abs=1e-3)


def test_ontrack_monotone_lock_still_advances():
    gen = AdaptiveSubgoalGenerator(
        r_base=55.0, r_min=20.0, cruise_speed=10.0, cte_lock_freeze_m=5.0
    )
    path = np.array(
        [[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [200.0, 0.0, 10.0]],
        dtype=np.float64,
    )
    gen._prev_s_max = 80.0
    _, info = gen.compute_subgoal(
        np.array([70.0, 1.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    assert info["cte_m"] < 5.0
    assert info["s_progress"] >= 80.0
    assert not info.get("s_lock_frozen")


def test_terminal_pin_uses_route_goal():
    gen = AdaptiveSubgoalGenerator(
        r_base=25.0,
        r_min=15.0,
        cruise_speed=10.0,
        segment_length_m=0.0,
        terminal_pin_rem_m=40.0,  # opt-in for this unit test
    )
    path = np.array([[0.0, 0.0, 10.0], [100.0, 0.0, 10.0]], dtype=np.float64)
    g_far, info_far = gen.compute_subgoal(
        np.array([50.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    assert info_far["rem_dist"] == pytest.approx(50.0, abs=0.1)
    assert info_far.get("terminal_pin") is False

    g_near, info_near = gen.compute_subgoal(
        np.array([70.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    assert info_near["rem_dist"] == pytest.approx(30.0, abs=0.1)
    assert info_near.get("terminal_pin") is True
    np.testing.assert_allclose(info_near["target_world"], path[-1], atol=1e-5)
    # ‖g_rel‖ equals remaining distance to route goal (converging endgame).
    assert float(g_near[3]) == pytest.approx(30.0, abs=0.5)


def test_segment_caps_carrot_before_segment_end():
    gen = AdaptiveSubgoalGenerator(
        r_base=25.0,
        r_min=15.0,
        cruise_speed=10.0,
        segment_length_m=40.0,
        terminal_pin_rem_m=40.0,
    )
    path = np.array([[0.0, 0.0, 10.0], [200.0, 0.0, 10.0]], dtype=np.float64)
    # Mid first segment: s=0 → segment end 40; carrot 25 m, not past 40.
    _, info = gen.compute_subgoal(
        np.array([0.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    assert info.get("terminal_pin") is False
    assert info.get("segment_end_s") == pytest.approx(40.0, abs=0.1)
    target = np.asarray(info["target_world"], dtype=np.float64)
    assert target[0] == pytest.approx(25.0, abs=0.5)
    assert info.get("segment_pin") is False

    # Near segment end: pin to s=40.
    _, info2 = gen.compute_subgoal(
        np.array([30.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    assert info2.get("segment_pin") is True
    target2 = np.asarray(info2["target_world"], dtype=np.float64)
    assert target2[0] == pytest.approx(40.0, abs=0.5)


def test_lookahead_feedback_default_off():
    gen = AdaptiveSubgoalGenerator(cruise_speed=10.0)
    assert gen.lookahead_feedback is False
    path = np.array([[0.0, 0.0, 10.0], [200.0, 0.0, 10.0]], dtype=np.float64)
    _, info = gen.compute_subgoal(
        np.array([0.0, 0.0, 10.0]), 0.0, path, d_fwd_hat=40.0
    )
    assert info.get("lookahead_feedback") is False
    assert info.get("no_progress_shrink") is False
    assert info.get("fb_r_mul") == pytest.approx(1.0)


def test_lookahead_feedback_shrinks_after_stall():
    gen = AdaptiveSubgoalGenerator(
        cruise_speed=10.0,
        r_base=25.0,
        r_min=15.0,
        lookahead_feedback=True,
        no_progress_steps=5,
        no_progress_ds_m=0.5,
        no_progress_r_scale=0.6,
    )
    path = np.array([[0.0, 0.0, 10.0], [200.0, 0.0, 10.0]], dtype=np.float64)
    pos = np.array([10.0, 0.0, 10.0], dtype=np.float64)
    r_last = None
    for _ in range(6):
        _, info = gen.compute_subgoal(pos, 0.0, path, d_fwd_hat=40.0)
        r_last = float(info["r_lookahead"])
    assert info.get("no_progress_shrink") is True
    assert float(info["fb_r_mul"]) == pytest.approx(0.6, abs=1e-6)
    assert r_last <= 25.0 * 0.6 + 1e-3

