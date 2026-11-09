"""Detour toward the more open side. Spin in place only when both sides are very close."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_choice = importlib.util.spec_from_file_location("office_choice", _ROOT / "scripts" / "office_choice.py")
office = importlib.util.module_from_spec(_choice)
_choice.loader.exec_module(office)

_wire = importlib.util.spec_from_file_location("yhs_wire", Path(__file__).with_name("yhs_wire.py"))
wire = importlib.util.module_from_spec(_wire)
_wire.loader.exec_module(wire)

_center = importlib.util.spec_from_file_location("steer_center", Path(__file__).with_name("steer_center.py"))
center = importlib.util.module_from_spec(_center)
_center.loader.exec_module(center)


def _reset():
    office.reset_passage()


def _names():
    names = ["fwd", "slow", "spin_left", "spin_right"]
    cands = np.zeros((4, 1, 4), np.float64)
    cands[0, 0, 0] = 0.8
    cands[1, 0, 0] = 0.35
    cands[2, 0, 3] = -0.3
    cands[3, 0, 3] = 0.3
    surv = np.array([0.20, 0.10, 0.1, 0.1])
    return names, cands, surv


def test_a_body_side_with_an_open_nose_does_not_drive():
    # Right low 0.98 m is beside the body. Straight creep pushed into it.
    action, family = office.side_passage((1.36, 1.96, 3.02), (2.89, None, 0.98))
    assert family == "arc_left"
    assert action[0] == 0.0 and action[3] == -office.ARC_DYAW
    action, family = office.side_passage((1.10, 1.96, 3.02), (2.89, None, 0.98))
    assert family == "arc_left"
    assert action[0] == 0.0 and action[3] == -office.ARC_DYAW


def test_a_corridor_does_not_drive_between_close_walls():
    # Both sides inside 1 m. Forward speed here is what clipped the body.
    action, family = office.side_passage((1.31, 0.75, 1.32), (0.70, 0.90, 0.80))
    assert family == "arc_right"
    assert action[0] == 0.0 and action[3] == office.ARC_DYAW
    action, family = office.side_passage((0.90, 0.70, 0.80), (0.70, 0.50, 0.80))
    assert family == "arc_right"
    assert action[0] == 0.0 and action[3] == office.ARC_DYAW


def test_a_far_pair_twelve_centimetres_apart_goes_straight():
    action, family = office.side_passage((3.2, 3.00, 3.13), (None, None, None))
    assert family == "through" and action[3] == 0.0
    assert action[0] == office.CRUISE_DX


def test_a_front_box_is_not_pushed():
    _reset()
    action, passage, _clear = office.commit_passage(
        (3.20, 3.00, 3.10), (None, 1.30, None), None)
    assert passage is None and action[1] == "stop"
    assert action[0][0] == 0.0 and action[0][3] == 0.0
    _reset()
    action, passage, _clear = office.commit_passage(
        (3.20, 2.00, 3.40), (None, 1.30, None), None)
    assert passage == "arc_right" and action[0][0] == 0.0
    assert action[0][3] == office.ARC_DYAW
    _reset()


def test_a_center_box_turns_back_toward_the_goal():
    _reset()
    office.commit_passage._bearing = math.radians(-83.0)
    action, passage, _clear = office.commit_passage(
        (1.91, 1.70, 2.21), (2.22, 0.91, 3.62), None)
    assert passage == "arc_left" and action[0][0] == 0.0
    assert action[0][3] == -office.ARC_DYAW
    _reset()
    office.commit_passage._bearing = math.radians(110.0)
    action, passage, _clear = office.commit_passage(
        (1.40, 2.40, 0.80), (2.40, 0.90, 0.80), None)
    assert passage is None and action[1] == "stop"
    assert action[0][0] == 0.0
    _reset()


def test_a_close_center_does_not_drive_straight():
    action, family = office.side_passage((1.80, 2.20, 2.30), (2.40, 0.40, 2.50))
    assert family == "stop" and action[0] == 0.0 and action[3] == 0.0
    action, family = office.side_passage((1.80, 1.50, 2.50), (1.60, 0.40, 2.60))
    assert family == "arc_right"
    assert action[0] == 0.0 and action[3] == office.ARC_DYAW


def test_a_boxed_spin_holds_until_a_side_clears_one_metre_twenty_five():
    _reset()
    action, passage, _clear = office.commit_passage(
        (0.90, 0.70, 0.80), (0.70, 0.50, 0.80), None)
    assert action[0][0] == 0.0 and passage == "arc_right"
    action, passage, _clear = office.commit_passage(
        (0.90, 1.10, 1.15), (1.10, 0.50, 1.15), passage)
    assert action[0][0] == 0.0 and passage == "arc_right"
    # One open frame does not leave the spin.
    action, passage, _clear = office.commit_passage(
        (1.80, 1.40, 2.20), (1.50, 1.60, 2.30), passage)
    assert passage == "arc_right" and action[0][0] == 0.0
    action, passage, _clear = office.commit_passage(
        (0.90, 0.70, 0.80), (0.70, 0.50, 0.80), passage)
    assert passage == "arc_right" and action[0][0] == 0.0
    for _ in range(office.ESCAPE_FLIP_FRAMES - 1):
        action, passage, _clear = office.commit_passage(
            (1.80, 1.40, 2.20), (1.50, 1.60, 2.30), passage)
        assert passage == "arc_right" and action[0][0] == 0.0
    action, passage, _clear = office.commit_passage(
        (1.80, 1.40, 2.20), (1.50, 1.60, 2.30), passage)
    assert passage == "arc_right" and action[1] == "arc_right"
    assert action[0][0] == office.SLOW_DX
    _reset()


def test_a_boxed_spin_stops_when_the_corner_stays_closed():
    _reset()
    boxed = ((0.90, 0.70, 0.80), (0.70, 0.50, 0.80))
    for _ in range(office.BOXED_SPIN_FRAMES - 1):
        action, passage, _clear = office.commit_passage(boxed[0], boxed[1], None)
        assert passage == "arc_right" and action[0][0] == 0.0
    action, passage, _clear = office.commit_passage(boxed[0], boxed[1], passage)
    assert passage is None and action[1] == "stop" and action[0][0] == 0.0
    action, passage, _clear = office.commit_passage(boxed[0], boxed[1], passage)
    assert action[1] == "stop"
    step = office.BOXED_SPIN_RAD / (office.BOXED_SPIN_MIN_FRAMES - 0.5)
    _reset()
    for _ in range(office.BOXED_SPIN_MIN_FRAMES - 1):
        action, passage, _clear = office.commit_passage(
            boxed[0], boxed[1], None, 0.0, 0.0, step)
        assert passage == "arc_right"
    action, passage, _clear = office.commit_passage(
        boxed[0], boxed[1], passage, 0.0, 0.0, step)
    assert action[1] == "stop"
    _reset()


def test_a_yaw_change_waits_three_frames():
    _reset()
    passage = "arc_left"
    for _ in range(office.ESCAPE_FLIP_FRAMES - 1):
        action, passage, _clear = office.commit_passage(
            (1.80, 1.20, 2.40), (1.50, 1.80, 2.50), passage)
        assert passage == "arc_left" and action[0][3] == -office.NUDGE_DYAW
    action, passage, _clear = office.commit_passage(
        (1.80, 1.20, 2.40), (1.50, 1.80, 2.50), passage)
    assert passage == "arc_right" and action[0][3] == office.NUDGE_DYAW
    assert action[0][0] == office.SLOW_DX
    _reset()


def test_a_straight_frame_does_not_allow_an_immediate_reverse():
    _reset()
    action, passage, _clear = office.commit_passage(
        (1.10, 2.50, 0.98), (2.40, 2.20, 0.98), None)
    assert passage == "arc_left"
    action, passage, _clear = office.commit_passage(
        (3.20, 3.00, 3.10), (None, None, None), passage)
    assert passage == "arc_left" and action[0][3] == -office.ARC_DYAW
    for _ in range(office.ESCAPE_FLIP_FRAMES - 1):
        action, passage, _clear = office.commit_passage(
            (1.10, 0.98, 2.50), (0.98, 2.20, 2.40), passage)
        assert passage is None and action[1] == "stop" and action[0][0] == 0.0
    action, passage, _clear = office.commit_passage(
        (1.10, 0.98, 2.50), (0.98, 2.20, 2.40), passage)
    assert passage == "arc_right" and action[0][3] == office.ARC_DYAW
    _reset()


def test_a_clear_view_turns_toward_the_goal():
    _reset()
    office.commit_passage._bearing = math.radians(110.0)
    action, passage, _clear = office.commit_passage(
        (3.20, 3.00, 3.10), (None, None, None), None)
    assert passage == "arc_right" and action[0][0] == 0.0
    assert action[0][3] == office.ARC_DYAW
    _reset()
    office.commit_passage._bearing = math.radians(-125.0)
    action, passage, _clear = office.commit_passage(
        (3.20, 3.00, 3.10), (None, None, None), None)
    assert passage == "arc_left" and action[0][0] == 0.0
    assert action[0][3] == -office.ARC_DYAW
    _reset()
    office.commit_passage._bearing = math.radians(20.0)
    action, passage, _clear = office.commit_passage(
        (3.20, 3.00, 3.10), (None, None, None), None)
    assert passage == "arc_right" and action[0][0] == office.CRUISE_DX
    assert action[0][3] == office.NUDGE_DYAW
    _reset()
    office.commit_passage._bearing = math.radians(-20.0)
    action, passage, _clear = office.commit_passage(
        (3.20, 3.00, 3.10), (None, None, None), None)
    assert passage == "arc_left" and action[0][3] == -office.NUDGE_DYAW
    _reset()
    office.commit_passage._bearing = math.radians(2.0)
    action, passage, _clear = office.commit_passage(
        (3.20, 3.00, 3.10), (None, None, None), None)
    assert passage is None and action[1] == "through" and action[0][3] == 0.0
    _reset()


def test_a_blocked_nose_turns_back_toward_the_goal_instead_of_stopping():
    _reset()
    office.commit_passage._bearing = math.radians(-86.0)
    action, passage, _clear = office.commit_passage(
        (0.97, 1.57, 1.56), (1.08, 1.44, 1.70), None)
    assert passage == "arc_left" and action[1] == "arc_left"
    assert action[0][0] == 0.0 and action[0][3] == -office.ARC_DYAW
    office.commit_passage._bearing = math.radians(-30.0)
    for _ in range(office.ESCAPE_FLIP_FRAMES - 1):
        action, passage, _clear = office.commit_passage(
            (0.97, 0.70, 2.40), (0.70, 2.00, 2.40), passage)
        assert passage is None and action[1] == "stop" and action[0][0] == 0.0
    action, passage, _clear = office.commit_passage(
        (0.97, 0.70, 2.40), (0.70, 2.00, 2.40), passage)
    assert passage == "arc_right" and action[0][0] == 0.0
    assert action[0][3] == office.ARC_DYAW
    _reset()


def test_a_turned_away_car_does_not_drive_into_the_near_side():
    _reset()
    office.commit_passage._bearing = math.radians(-101.0)
    action, passage, _clear = office.commit_passage(
        (2.60, 1.31, 2.10), (1.34, None, None), None)
    assert passage == "arc_left" and action[0][0] == 0.0
    assert action[0][3] == -office.ARC_DYAW
    _reset()
    office.commit_passage._bearing = 0.0
    action, passage, _clear = office.commit_passage(
        (3.20, 3.00, 3.10), (None, None, None), None)
    assert action[0][0] == office.CRUISE_DX
    office.commit_passage._bearing = math.radians(-101.0)
    action, passage, _clear = office.commit_passage(
        (2.60, 1.31, 2.10), (1.34, None, None), passage)
    assert passage == "arc_left" and action[0][0] == 0.0
    _reset()


def test_a_goal_side_on_the_body_turns_the_other_way():
    _reset()
    office.commit_passage._bearing = math.radians(68.0)
    action, passage, _clear = office.commit_passage(
        (1.69, 1.21, 2.04), (1.93, 2.74, 0.95), None)
    assert passage == "arc_left" and action[0][0] == 0.0
    assert action[0][3] == -office.ARC_DYAW
    _reset()


def test_a_low_that_jumps_from_far_to_near_does_not_count():
    _reset()
    office.commit_passage._bearing = math.radians(62.0)
    cones = (2.40, 2.20, 1.80)
    action, passage, _clear = office.commit_passage(
        cones, (1.30, 1.80, 2.08), None)
    assert action[0][0] > 0.0
    action, passage, _clear = office.commit_passage(
        cones, (1.30, 1.80, 0.56), passage)
    assert action[0][0] > 0.0
    assert action[0][3] != -office.ARC_DYAW
    _reset()
    office.commit_passage._bearing = math.radians(62.0)
    action, passage, _clear = office.commit_passage(
        cones, (1.30, 1.80, 0.56), None)
    assert action[0][0] == 0.0
    _reset()


def test_a_goal_side_is_not_turned_back_into_when_the_view_opens():
    _reset()
    office.commit_passage._bearing = math.radians(68.0)
    action, passage, _clear = office.commit_passage(
        (1.69, 1.21, 2.04), (1.93, 2.74, 0.95), None)
    assert passage == "arc_left" and action[0][0] == 0.0
    office.commit_passage._bearing = math.radians(80.0)
    opened = ((2.50, 2.80, 2.70), (2.40, 2.60, 2.50))
    for _ in range(office.ESCAPE_FLIP_FRAMES - 1):
        action, passage, _clear = office.commit_passage(*opened, passage)
        assert action[0][3] == -office.ARC_DYAW and action[0][0] == 0.0
    action, passage, _clear = office.commit_passage(*opened, passage)
    assert action[1] == "through" and action[0][0] == office.SLOW_DX
    assert action[0][3] == 0.0
    action, passage, _clear = office.commit_passage(*opened, passage)
    assert action[1] == "through" and action[0][3] == 0.0
    action, passage, _clear = office.commit_passage(
        (2.20, 2.40, 1.40), (2.40, 2.20, 0.92), passage)
    assert action[0][0] == 0.0
    _reset()


def test_a_detour_stops_turning_away_once_the_goal_is_off_to_the_side():
    _reset()
    office.commit_passage._bearing = math.radians(110.0)
    action, passage, _clear = office.commit_passage(
        (1.8, 2.5, 0.98), (2.4, 2.2, 0.98), None)
    assert passage is None and action[1] == "stop"
    assert action[0][0] == 0.0 and action[0][3] == 0.0
    _reset()
    office.commit_passage._bearing = 0.0
    action, passage, _clear = office.commit_passage(
        (1.10, 2.5, 0.98), (2.4, 2.2, 0.98), None)
    assert passage == "arc_left" and action[0][0] == 0.0
    assert action[0][3] == -office.ARC_DYAW
    _reset()
    office.commit_passage._bearing = math.radians(-37.0)
    action, passage, _clear = office.commit_passage(
        (1.81, 3.29, 3.56), (None, None, 0.95), None)
    assert passage == "arc_left" and action[0][0] == 0.0
    assert action[0][3] == -office.ARC_DYAW
    action, passage, _clear = office.commit_passage(
        (1.81, 0.70, 3.20), (0.70, None, 2.40), passage)
    assert passage is None and action[1] == "stop" and action[0][0] == 0.0
    action, passage, _clear = office.commit_passage(
        (1.81, 0.70, 3.20), (0.70, None, 2.40), passage)
    assert passage is None and action[1] == "stop" and action[0][0] == 0.0
    _reset()


def test_balanced_open_sides_drive_straight():
    _reset()
    action, passage, _clear = office.commit_passage(
        (2.40, 2.10, 2.20), (None, None, None), None)
    assert passage is None and action[1] == "through"
    assert action[0][0] == office.CRUISE_DX and action[0][3] == 0.0
    action, family = office.side_passage((1.60, 2.20, 2.30), (None, None, None))
    assert family == "through" and action[0] == office.SLOW_DX


def test_a_close_side_drops_forward_speed_immediately():
    _reset()
    action, passage, _clear = office.commit_passage(
        (2.40, 1.40, 2.60), (2.20, 2.00, 2.40), None)
    assert passage == "arc_right" and action[0][0] == office.SLOW_DX
    action, passage, _clear = office.commit_passage(
        (2.40, 2.60, 0.70), (2.40, 2.00, 0.70), passage)
    assert passage is None and action[1] == "stop"
    assert action[0][0] == 0.0 and action[0][3] == 0.0
    _reset()


def test_a_blocked_nose_with_open_sides_stops():
    action, family = office.side_passage((1.10, 2.40, 2.40), (None, None, None))
    assert family == "stop" and action[0] == 0.0


def test_a_blocked_nose_turns_toward_the_more_open_side():
    _reset()
    office.commit_passage._bearing = 0.0
    action, passage, _clear = office.commit_passage(
        (1.13, 2.61, 2.15), (None, 1.45, 3.12), None)
    assert passage == "arc_left" and action[0][0] == 0.0
    assert action[0][3] == -office.ARC_DYAW
    _reset()


def test_resolve_uses_the_same_detour():
    names, cands, surv = _names()
    _i, action, family, _n = office.resolve(
        names, cands, surv, None, math.radians(31.0), (1.54, 2.30, 2.40), None, None, None)
    assert family == "arc_right" and action[0] == office.SLOW_DX
    assert action[3] == office.NUDGE_DYAW
    _i, action, family, _n = office.resolve(
        names, cands, surv, None, 0.0, (2.40, 0.68, 1.90), None, None, None)
    assert family == "arc_right"
    assert action[0] == 0.0 and action[3] == office.ARC_DYAW


def test_hold_arc_releases_when_the_sides_are_balanced():
    assert office.hold_arc((2.40, 2.40, 2.30), (None, None, None), "arc_right", 0.2) is None
    still = office.hold_arc((1.10, 0.62, 1.80), (None, None, None), "arc_right", 0.9)
    assert still[1] == "arc_right" and still[0][0] == 0.0
    assert still[0][3] == office.ARC_DYAW


def test_side_cones_skip_the_open_floor():
    depth = np.full((56, 56), 3.0, np.float32)
    depth[14:42, 14:42] = 2.57
    depth[22:42, :14] = 0.99
    depth[22:42, 42:] = 0.99
    cones = wire._office_cones(depth, floor_row=100, src_h=224)
    assert abs(float(cones[0]) - 2.57) < 1e-3
    assert float(cones[1]) > 1.6 and float(cones[2]) > 1.6
    depth[16:20, :14] = 1.10
    cones = wire._office_cones(depth, floor_row=100, src_h=224)
    assert abs(float(cones[1]) - 1.10) < 1e-3
    assert abs(float(cones[0]) - 2.57) < 1e-3


def test_floor_band_is_not_a_side_object():
    height = 56
    depth = np.full((height, 56), 6.0, np.float32)
    depth[int(height * 0.26):int(height * 0.42), :] = 4.5
    fy = 0.5 * height / math.tan(math.radians(45.0))
    left, center, right = wire.low_clearance(depth, fy, height / 2.0, height)
    assert (left, center, right) == (None, None, None)
    depth[int(height * 0.30):int(height * 0.38), 40:] = 0.8
    left, center, right = wire.low_clearance(depth, fy, height / 2.0, height)
    assert right is not None and right < 1.25
    assert left is None and center is None


def _floor_depth():
    height, width = 480, 640
    rows = np.linspace(2.8, 0.6, height, dtype=np.float32)
    return np.repeat(rows[:, None], width, axis=1)


def test_an_open_cone_drops_a_floor_low_on_that_side():
    cones, low = wire.passage_view((1.30, 1.05, 2.20), (1.02, 2.30, 0.85), _floor_depth())
    assert low[2] is None
    assert cones[2] == 2.20
    assert low[0] == 1.02
    _reset()
    office.commit_passage._bearing = math.radians(45.0)
    action, passage, _clear = office.commit_passage(cones, low, None)
    assert action[0][0] > 0.0
    _reset()


def test_sensor_floor_is_not_a_box():
    left, right, left_floor, right_floor = wire.sensor_side_face(_floor_depth())
    assert left is None and right is None
    assert left_floor and right_floor


def test_sensor_right_box_drives_through_the_open_center():
    depth = _floor_depth()
    depth[int(480 * 0.34):int(480 * 0.62), 2 * 640 // 3:] = 1.05
    cones, low = wire.passage_view((1.66, 0.85, 1.70), (None, None, None), depth)
    action, family = office.side_passage(cones, low)
    assert family == "arc_left"
    assert action[0] == office.SLOW_DX and action[3] == -office.NUDGE_DYAW
    assert low[2] is not None and low[2] < 1.25
    assert low[1] is None
    assert cones[1] > 2.0


def test_a_low_obstacle_does_not_drive_through_the_point():
    assert wire.arrival_scan("point", 1.42, False, (1.28, None, 3.44), False)
    assert wire.arrival_scan("point", 0.12, False, None, False)
    assert not wire.arrival_scan("point", 1.51, False, (1.28, None, 3.44), False)
    assert not wire.arrival_scan("point", 0.12, True, None, False)
    assert not wire.arrival_scan("along", 0.12, False, None, False)


def test_open_floor_does_not_block_the_way():
    depth = _floor_depth()
    cones, low = wire.passage_view((1.29, 0.78, 2.33), (0.91, None, None), depth)
    assert cones[0] == 2.5 and cones[1] == 2.5
    assert low[0] is None
    _reset()
    action, family = office.side_passage(cones, low)
    assert family == "through"
    assert action[0] == office.CRUISE_DX and action[3] == 0.0


def test_sensor_near_floor_does_not_clear_a_real_cone():
    depth = np.full((480, 640), 0.45, np.float32)
    cones, low = wire.passage_view((1.66, 0.85, 1.70), (None, None, None), depth)
    assert cones[1] == 0.85 and cones[2] == 1.70
    assert low == (None, None, None)


def test_a_forward_step_waits_until_the_spin_wheels_center():
    crooked = (43.6, -43.6, -43.6, 43.6)
    nudge = (12.0, -11.0, 10.0, -9.0)
    straight = (2.0, -1.0, 1.5, -2.0)
    assert center.hold_for_center(0.8, 0.0, 0.40, crooked, 10.0, 0.0)
    assert center.hold_for_center(0.8, 0.0, 0.0, crooked, 10.0, 0.0)
    assert not center.hold_for_center(0.0, 0.0, 0.40, crooked, 10.0, 0.0)
    assert not center.hold_for_center(0.8, 0.0, 0.40, straight, 10.0, 0.0)
    assert not center.hold_for_center(0.8, 0.24, 0.13, nudge, 10.0, 0.0)
    assert center.hold_for_center(0.8, 0.0, 0.0, crooked, 10.0, 12.0)
    assert not center.hold_for_center(0.8, 0.0, 0.0, crooked, 13.0, 12.0)
    assert not center.hold_for_center(0.8, 0.24, 0.0, crooked, 10.0, 0.0, True)
    until, gave = center.next_center_state(True, 43.6, 10.0, 0.0, False)
    assert until == 12.0 and gave is False
    until, gave = center.next_center_state(False, 43.6, 13.0, 12.0, False)
    assert until == 0.0 and gave is True
    until, gave = center.next_center_state(False, 43.6, 14.0, 0.0, True)
    assert gave is True
    until, gave = center.next_center_state(False, 2.0, 15.0, 0.0, True)
    assert until == 0.0 and gave is False
