"""The early arc uses the obstacle band, and the vehicle in frame is not a side."""
from __future__ import annotations

import numpy as np

from aerial_finder.nav.navigator import corridor_passage, lead_guard, lead_open, passage_side


def _scene(left: float, center: float, right: float, body: float = 0.6) -> np.ndarray:
    d = np.full((56, 56), 8.0, np.float32)
    d[18:27, :22] = body
    d[18:27, 22:33] = left
    d[18:27, 33:44] = center
    d[18:27, 44:] = right
    return d


def test_vehicle_on_the_left_does_not_hide_an_open_left_side():
    center, side = lead_open(_scene(left=5.0, center=2.4, right=2.0), (18, 27), body_cols=22, margin_m=0.5)
    assert abs(center - 2.4) < 1e-5
    assert side == -1.0


def test_a_clear_right_gap_yaws_right():
    _center, side = lead_open(_scene(left=2.0, center=2.2, right=4.0), (18, 27), body_cols=22, margin_m=0.5)
    assert side == 1.0


def test_a_tie_or_a_missing_side_does_not_pick_one():
    _center, tied = lead_open(_scene(left=4.0, center=4.0, right=4.2), (18, 27), body_cols=22, margin_m=0.5)
    assert tied is None
    d = _scene(left=5.0, center=2.0, right=6.0)
    d[18:27, 22:33] = np.nan
    _center, missing = lead_open(d, (18, 27), body_cols=22, margin_m=0.5)
    assert missing is None


def test_shell_pixels_past_the_cut_are_not_a_blocked_side():
    d = _scene(left=5.0, center=2.5, right=2.0)
    d[18:27, 22:27] = 0.7
    _center, side = lead_open(d, (18, 27), body_cols=22, margin_m=0.5)
    assert side == -1.0


def test_close_guard_keeps_the_earlier_side_and_does_not_invent_one():
    action = np.array([0.8, 0.0, 0.0, 0.0])
    kept, phase = lead_guard(action, front=2.0, stop_m=0.8, turn_m=1.2, lead_side=-1.0, lead_yaw=0.08, fwd_dx=0.8)
    assert phase is None
    assert kept[0] == 0.8
    arc, phase = lead_guard(action, front=1.0, stop_m=0.8, turn_m=1.2, lead_side=-1.0, lead_yaw=0.08, fwd_dx=0.8)
    assert phase == "lead"
    assert arc[0] == 0.4
    assert arc[3] == -0.08
    stopped, phase = lead_guard(action, front=1.0, stop_m=0.8, turn_m=1.2, lead_side=0.0, lead_yaw=0.08, fwd_dx=0.8)
    assert phase == "stop"
    assert np.all(stopped == 0)
    hard, phase = lead_guard(action, front=0.5, stop_m=0.8, turn_m=1.2, lead_side=0.0, lead_yaw=0.08, fwd_dx=0.8)
    assert phase == "stop"
    assert np.all(hard == 0)
    arc_close, phase = lead_guard(
        action, front=0.5, stop_m=0.8, turn_m=1.2, lead_side=1.0, lead_yaw=0.08, fwd_dx=0.8)
    assert phase == "lead"
    assert arc_close[3] == 0.08
    opened, phase = lead_guard(action, front=1.0, stop_m=0.8, turn_m=1.2, lead_side=1.0, lead_yaw=0.08, fwd_dx=0.8)
    assert phase == "lead"
    assert opened[0] == 0.4
    assert opened[3] == 0.08
    straight, phase = lead_guard(
        action, front=1.0, stop_m=0.8, turn_m=1.2, lead_side=0.0, lead_yaw=0.08, fwd_dx=0.8, through=True)
    assert phase == "through"
    assert straight[0] == 0.4
    assert straight[3] == 0.0


def test_a_passable_right_side_is_taken_when_the_left_is_not():
    assert passage_side(left=None, right=1.4, pass_m=1.0, margin_m=0.5) == 1.0
    assert passage_side(left=0.7, right=1.4, pass_m=1.0, margin_m=0.5) == 1.0


def test_a_passable_left_side_is_taken_and_a_missing_side_is_not_invented():
    assert passage_side(left=1.6, right=None, pass_m=1.0, margin_m=0.5) == -1.0
    assert passage_side(left=None, right=None, pass_m=1.0, margin_m=0.5) is None
    assert passage_side(left=0.6, right=0.7, pass_m=1.0, margin_m=0.5) is None


def test_both_open_sides_go_ahead_unless_one_is_clearly_farther():
    assert passage_side(left=1.4, right=1.6, pass_m=1.0, margin_m=0.5) == 0.0
    assert passage_side(left=1.2, right=2.0, pass_m=1.0, margin_m=0.5) == 1.0


def test_the_vehicle_filling_one_side_is_not_a_passage():
    depth = _scene(left=0.55, center=0.7, right=1.8)
    left, right = corridor_passage(depth, (18, 27), body_cols=22, pass_m=1.0)
    assert left == 0.0
    assert right == 1.0
    assert passage_side(left, right, 0.6, 0.25) == 1.0


def test_a_side_already_open_stays_until_the_other_is_clearly_better():
    assert passage_side(left=1.3, right=1.5, pass_m=1.0, margin_m=0.5, held=1.0) == 1.0
    assert passage_side(left=2.2, right=1.4, pass_m=1.0, margin_m=0.5, held=1.0) == -1.0
