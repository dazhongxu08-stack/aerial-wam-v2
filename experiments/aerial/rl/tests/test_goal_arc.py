"""goal_arc is a scored candidate, not a override: cruise forward while closing heading error."""
from __future__ import annotations

import math

import numpy as np

from experiments.aerial.rl.mpc_consequence_eval import (
    carrot_yaw_hold,
    choose_align,
    choose_near,
    cruise_yaw_cap,
    goal_arc,
    hold_straight,
    fit_slide,
    open_slide,
    alt_hold_dz,
    corner_turn,
    leaves_point,
    stall_yaw,
    prefer_motion,
    prefer_safer_turn,
    side_arc,
    gap_turn_sign,
    upper_gap_sign,
    choose_skirt_side,
    skirt_hold,
    skirt_gate,
    skirt_tie,
    smooth_turn,
    step_seen_clear,
    step_seen_gap,
)


def test_in_place_turn_holds_altitude():
    state = np.array([0.0, 0.0, 22.0, 0, 0, 0, 0.0])
    assert alt_hold_dz(state, None) == 0.0
    assert alt_hold_dz(state, 24.0) == 0.2
    assert alt_hold_dz(np.array([0, 0, 24.0]), 24.0) == 0.0


def test_a_corner_on_one_side_turns_toward_the_open_side():
    # unseen_c_01 step 8: the building is on image left at 3 m, the gap is on the right at 15 m.
    depth = np.full((56, 56), 15.0)
    depth[8:40, :18] = 3.0
    assert corner_turn(depth) == 1.0
    depth = np.full((56, 56), 15.0)
    depth[8:40, 38:] = 2.5
    assert corner_turn(depth) == -1.0
    assert corner_turn(np.full((56, 56), 15.0)) is None
    both = np.full((56, 56), 15.0)
    both[8:40, :18] = 2.0
    both[8:40, 38:] = 2.0
    assert corner_turn(both) is None


def _fwd(dx: float = 0.8) -> np.ndarray:
    cands = np.zeros((1, 1, 4), np.float64)
    cands[0, 0, 0] = dx
    return cands


def test_one_meter_check_keeps_a_forward_step_beside_a_three_meter_corner():
    # unseen_c_01: image-left about 3 m, center about 15 m, the flown step is 0.8 m forward.
    # The nearest left-band pixel stays more than 1 m off that step, so the step is kept.
    depth = np.full((56, 56), 15.0)
    depth[8:40, :18] = 3.0
    state = np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0])
    gap = float(step_seen_gap([(depth, state)], state, _fwd())[0])
    assert gap > 1.0
    assert step_seen_clear([(depth, state)], state, _fwd())[0]
    depth[8:40, :18] = 2.3
    assert float(step_seen_gap([(depth, state)], state, _fwd())[0]) > 1.0


def test_one_meter_check_drops_a_step_that_reaches_the_surface():
    depth = np.full((56, 56), 15.0)
    depth[20:36, 22:34] = 0.9
    state = np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0])
    assert not step_seen_clear([(depth, state)], state, _fwd())[0]


def test_a_newer_view_replaces_a_false_near_point():
    near = np.full((56, 56), 15.0)
    near[20:36, 22:34] = 0.5
    far = np.full((56, 56), 15.0)
    state = np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0])
    assert step_seen_clear([(near, state), (far, state)], state, _fwd())[0]
    assert not step_seen_clear([(near, state)], state, _fwd())[0]


def test_a_wall_outside_the_new_view_still_blocks_the_step():
    # Seen straight ahead at 0.7 m, then the nose yaws past the field of view.
    # The forward step on the new heading still passes that remembered point.
    seen = np.full((56, 56), 30.0)
    seen[24:32, 26:30] = 0.7
    turned = np.array([0.0, 0.0, 24.0, 0, 0, 0, 1.2])
    empty = np.full((56, 56), 30.0)
    assert not step_seen_clear([(seen, np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0])), (empty, turned)],
                               turned, _fwd())[0]


def test_a_view_farther_than_40m_is_dropped():
    depth = np.full((56, 56), 80.0)
    depth[20:36, 22:34] = 50.0
    far_pose = np.array([-50.0, 0.0, 24.0, 0, 0, 0, 0.0])
    here = np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0])
    assert step_seen_clear([(depth, far_pose)], here, _fwd(), keep_m=40.0, max_range=80.0)[0]
    assert not step_seen_clear([(depth, far_pose)], here, _fwd(), keep_m=80.0, max_range=80.0)[0]


def test_a_surface_far_below_the_aircraft_does_not_block():
    depth = np.full((56, 56), 30.0)
    depth[52:, 26:30] = 2.0
    state = np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0])
    assert step_seen_clear([(depth, state)], state, _fwd())[0]


def test_a_south_step_leaves_the_north_point_and_a_closing_step_does_not():
    # Nose south. The schematic point is north. The flown step is what matters.
    pos = np.array([126.5, 122.3])
    point = np.array([121.0, 127.0])
    yaw = -math.pi / 2
    assert leaves_point(pos, yaw, point, np.array([0.4, 0.0, 0.0, 0.0]))
    assert not leaves_point(pos, yaw, point, np.array([0.0, 0.0, 0.0, 0.3]))
    assert not leaves_point(pos, math.pi / 2, point, np.array([0.4, 0.0, 0.0, 0.0]))


def test_aligned_is_straight_cruise():
    a = goal_arc(0.0, k=10, dx=0.8, yaw_cap=0.12)
    assert a.shape == (10, 4)
    np.testing.assert_allclose(a[:, 0], 0.8)
    np.testing.assert_allclose(a[:, 1:], 0.0)


def test_yaw_caps_then_flies_straight():
    cap = 0.12
    bearing = 0.30
    a = goal_arc(bearing, k=10, dx=0.8, yaw_cap=cap)
    assert np.allclose(a[:2, 3], cap)
    assert abs(a[2, 3] - (bearing - 2 * cap)) < 1e-9
    np.testing.assert_allclose(a[3:, 3], 0.0)
    assert abs(a[:, 3].sum() - bearing) < 1e-9
    assert np.abs(a[:, 3]).max() <= cap + 1e-9


def test_large_bearing_uses_the_whole_horizon_at_the_cap():
    cap = 0.12
    a = goal_arc(math.pi / 2, k=10, dx=0.8, yaw_cap=cap)
    np.testing.assert_allclose(a[:, 3], cap)
    np.testing.assert_allclose(a[:, 0], 0.8)
    assert a[:, 3].sum() < math.pi / 2


def test_smooth_turn_keeps_the_yaw_step_moving_and_capped():
    seq = np.zeros((4, 4))
    seq[0, 3] = 0.31
    seq[0, 0] = 0.1
    seq[2:, 0] = 0.8
    a = smooth_turn(seq, move_dx=0.6, yaw_cap=0.12)
    assert a[0, 0] == 0.6
    assert abs(a[0, 3] - 0.12) < 1e-9
    np.testing.assert_allclose(a[2:, 0], 0.8)
    np.testing.assert_allclose(a[1], 0.0)


def test_hold_straight_ignores_a_tiny_detour_and_keeps_a_real_one():
    names = ["fwd", "goal_arc"]
    surv = np.array([0.99, 0.99])
    allowed = np.array([True, True])
    assert hold_straight(names, surv, np.array([2.0, 2.05]), allowed, 1, 0.95, 0.5) == 0
    assert hold_straight(names, surv, np.array([2.0, 2.8]), allowed, 1, 0.95, 0.5) == 1
    assert hold_straight(names, surv, np.array([0.2, 1.0]), np.array([False, True]), 1, 0.95, 0.5) == 1


def test_align_waits_until_the_nose_is_off_and_the_arc_is_as_safe():
    names = ["fwd", "goal_arc"]
    allowed = np.array([True, True])
    assert choose_align(names, np.array([0.99, 0.98]), allowed, 0.2, 0.95, math.radians(25)) == -1
    assert choose_align(names, np.array([0.99, 0.98]), allowed, 0.6, 0.95, math.radians(25)) == 1
    assert choose_align(names, np.array([0.99, 0.96]), allowed, 0.6, 0.95, math.radians(25)) == -1


def test_near_flies_the_arc_toward_the_goal_not_a_stop_turn():
    names = ["fwd", "yaw_fwd", "near_arc"]
    cands = np.zeros((3, 2, 4))
    cands[0, :, 0] = 0.8
    cands[1, 0, 3] = -0.3
    cands[1, 0, 0] = 0.1
    cands[2, 0, 0] = 0.8
    cands[2, 0, 3] = 0.18
    surv = np.array([0.99, 0.99, 0.97])
    allowed = np.array([True, True, True])
    assert choose_near(names, cands, surv, allowed, bearing=0.8, surv_min=0.95) == 2
    allowed[2] = False
    assert choose_near(names, cands, surv, allowed, bearing=0.8, surv_min=0.95) == -1


def test_small_heading_error_uses_the_shallow_cap():
    assert cruise_yaw_cap(0.10, 0.10, trim_yaw=0.0, trim_rad=0.0) == 0.10
    assert cruise_yaw_cap(0.14, 0.10, trim_yaw=0.04, trim_rad=math.radians(25)) == 0.04
    assert cruise_yaw_cap(0.60, 0.10, trim_yaw=0.04, trim_rad=math.radians(25)) == 0.10
    assert cruise_yaw_cap(-0.20, 0.10, trim_yaw=0.04, trim_rad=math.radians(25)) == 0.04


def test_stall_yaw_turns_toward_the_goal_and_stops_on_it():
    assert stall_yaw(0.0, 0.314) == 0.0
    assert abs(stall_yaw(-0.005, 0.314) + 0.005) < 1e-9
    assert abs(stall_yaw(0.10, 0.314) - 0.10) < 1e-9
    assert abs(stall_yaw(0.50, 0.314) - 0.314) < 1e-9
    assert abs(stall_yaw(-0.80, 0.314) + 0.314) < 1e-9


def test_fit_slide_stays_inside_the_camera_and_keeps_a_shallow_slide():
    dx, dy = fit_slide(0.35, 0.35, 32.0)
    assert math.degrees(math.atan2(dy, dx)) <= 32.0 + 1e-6
    assert dy < 0.35
    dx2, dy2 = fit_slide(0.60, 0.30, 32.0)
    assert abs(dx2 - 0.60) < 1e-9 and abs(dy2 - 0.30) < 1e-9
    assert math.degrees(math.atan2(dy2, dx2)) < 35.0


def test_open_slide_moves_toward_image_right_when_that_side_is_open():
    a = open_slide(side=1.0, k=4, dx=0.35, dy=0.35, yaw=0.08)
    np.testing.assert_allclose(a[:, 0], 0.35)
    np.testing.assert_allclose(a[:, 1], -0.35)
    np.testing.assert_allclose(a[:, 3], 0.08)
    left = open_slide(side=-1.0, k=2, dx=0.35, dy=0.35, yaw=0.08)
    assert left[0, 1] > 0 and left[0, 3] < 0


def test_safer_turn_replaces_a_sagging_straight_step_and_keeps_a_healthy_one():
    names = ["fwd", "yaw_fwd", "goal_turn_fwd", "yaw_only"]
    cands = np.zeros((4, 2, 4))
    cands[0, 0, 0] = 0.8
    cands[1, 0, 0] = 0.6
    cands[1, 0, 3] = 0.10
    cands[2, 0, 0] = 0.6
    cands[3, 0, 3] = 0.30
    allowed = np.array([True, True, True, True])
    # The logged facade step: straight 0.97 / 7.4 m, the moving turn 0.99.
    assert prefer_safer_turn(names, cands, np.array([0.97, 0.989, 0.982, 0.999]),
                             allowed, 0, 0.95, 0.98) == 1
    assert prefer_safer_turn(names, cands, np.array([0.994, 0.995, 0.990, 0.999]),
                             allowed, 0, 0.95, 0.98) == 0
    assert prefer_safer_turn(names, cands, np.array([0.97, 0.96, 0.96, 0.999]),
                             allowed, 0, 0.95, 0.98) == 0
    assert prefer_safer_turn(names, cands, np.array([0.96, 0.99, 0.97, 0.999]),
                             allowed, 0, 0.95, 0.98, min_step=0.3) == 1


def test_prefer_motion_replaces_a_stop_when_a_move_already_passes():
    names = ["yaw_only", "fwd", "open_slide"]
    cands = np.zeros((3, 2, 4))
    cands[0, 0, 3] = 0.3
    cands[1, 0, 0] = 0.8
    cands[2, 0, 0] = 0.35
    cands[2, 0, 1] = -0.35
    surv = np.array([0.99, 0.97, 0.96])
    allowed = np.array([True, True, True])
    assert prefer_motion(names, cands, surv, allowed, 0, 0.95) == 1
    surv[1] = 0.90
    assert prefer_motion(names, cands, surv, allowed, 0, 0.95) == 2
    surv[2] = 0.90
    assert prefer_motion(names, cands, surv, allowed, 0, 0.95) == 0


def test_skirt_holds_the_safer_side_then_flips_once():
    right = side_arc(1.0, k=4, dx=0.8, yaw=0.10)
    left = side_arc(-1.0, k=4, dx=0.8, yaw=0.10)
    np.testing.assert_allclose(right[:, 3], 0.10)
    np.testing.assert_allclose(left[:, 3], -0.10)
    assert choose_skirt_side(0.40, 0.91, tie_sign=-1.0) == 1.0
    assert choose_skirt_side(0.93, 0.55, tie_sign=1.0) == -1.0
    assert choose_skirt_side(0.90, 0.91, tie_sign=-1.0) == -1.0
    depth = np.ones((8, 8))
    depth[:, 4:] = 6.0
    assert gap_turn_sign(depth) == 1.0
    depth[:, :] = 6.0
    depth[:, 4:] = 1.0
    assert gap_turn_sign(depth) == -1.0
    flat = np.full((8, 8), 10.0)
    assert upper_gap_sign(flat, min_ratio=0.10) is None
    flat = flat.copy()
    flat[:, 4:] = 10.5
    assert upper_gap_sign(flat, min_ratio=0.10) is None
    opened = np.full((8, 8), 10.0)
    opened[:, 4:] = 20.0
    assert upper_gap_sign(opened, min_ratio=0.10) == 1.0
    opened[:, :] = 20.0
    opened[:, 4:] = 8.0
    assert upper_gap_sign(opened, min_ratio=0.10) == -1.0
    side, flips, gave = skirt_hold(0.0, 0.0, 0.6, 1.0, 0)
    assert (side, flips, gave) == (1.0, 0, False)
    side, flips, gave = skirt_hold(1.0, 0.2, 0.6, -1.0, 0)
    assert (side, flips, gave) == (1.0, 0, False)
    side, flips, gave = skirt_hold(1.0, 0.7, 0.6, -1.0, 0)
    assert (side, flips, gave) == (-1.0, 1, False)
    side, flips, gave = skirt_hold(-1.0, 0.7, 0.6, 1.0, 1)
    assert gave and side == 0.0


def test_skirt_starts_only_on_a_tie_and_releases_when_the_goal_reopens():
    # Both arcs alive. 0.025 is inside 0.03 and outside the old 0.02 line.
    assert skirt_tie(0.975, 0.950, 0.95, margin=0.03)
    assert not skirt_tie(0.975, 0.950, 0.95, margin=0.02)
    assert skirt_tie(0.980, 0.965, 0.95)
    # One arc under the line stays with the picker, including the corner and the billboard.
    assert not skirt_tie(0.971, 0.944, 0.95, margin=0.03)
    assert not skirt_tie(0.963, 0.937, 0.95, margin=0.03)
    assert not skirt_tie(0.97, 0.92, 0.95)
    assert not skirt_tie(0.989, 0.619, 0.95)
    assert not skirt_tie(0.40, 0.41, 0.95)
    assert not skirt_tie(float("nan"), 0.99, 0.95)
    # A clear preference does not start. A tie does, on that same step.
    skirt, clear, closed, _ = skirt_gate(False, True, 0, 0, 0.0, False, 1, start=False)
    assert (skirt, clear, closed) == (False, False, 1)
    skirt, clear, _, _ = skirt_gate(False, True, 0, 0, 0.0, False, 1, start=True)
    assert skirt and not clear
    # Once held, the side continues after the scores stop being a tie.
    skirt, clear, _, _ = skirt_gate(False, True, 1, 0, 1.0, False, 1, start=False)
    assert skirt and not clear
    # The first recovered step hands the nose back.
    skirt, clear, _, open_n = skirt_gate(True, True, 4, 0, 1.0, False, 1)
    assert not skirt and clear and open_n == 0
    # A longer release still keeps one recovered step.
    skirt, clear, _, open_n = skirt_gate(True, True, 4, 0, 1.0, False, 3)
    assert skirt and not clear and open_n == 1
    # After the cap gives up, a new tie does not restart it until the goal has been open.
    skirt, clear, _, _ = skirt_gate(False, True, 5, 0, 0.0, True, 1, start=True)
    assert not skirt and not clear
    skirt, clear, _, _ = skirt_gate(True, True, 5, 0, 0.0, True, 1)
    assert not skirt and clear
    # Leaving the region clears the latch.
    skirt, clear, closed, open_n = skirt_gate(False, False, 4, 2, 1.0, True, 1)
    assert (skirt, clear, closed, open_n) == (False, True, 0, 0)


def test_negative_bearing_yaws_negative():
    a = goal_arc(-0.5, k=8, dx=0.8, yaw_cap=0.12)
    assert np.all(a[:, 3] <= 0)
    assert abs(a[:, 3].sum() - -0.5) < 1e-9


def test_carrot_yaw_holds_side_until_forward_clears():
    yaw, latch = carrot_yaw_hold(-1.2, 0.0, False, 0.314)
    assert yaw == -0.314 and latch == -1.0
    # A one-frame sign flip does not reverse the turn.
    yaw, latch = carrot_yaw_hold(0.1, latch, False, 0.314)
    assert yaw < 0 and latch == -1.0
    # A clear crossing does.
    yaw, latch = carrot_yaw_hold(0.5, latch, False, 0.314)
    assert yaw == 0.314 and latch == 1.0
    assert carrot_yaw_hold(0.4, latch, True, 0.314) == (None, 0.0)
