"""The car uses the drone's cruise rules: an explicit arc, not a random yaw."""
from __future__ import annotations

import math

import numpy as np

from aerial_finder.nav.navigator import (
    align_may_replace,
    allow_early_open,
    choose_align,
    choose_near,
    cruise_yaw_cap,
    goal_arc,
    hold_straight,
    open_side_arcs,
    smooth_turn,
)


def test_small_heading_error_uses_the_shallow_cap():
    assert cruise_yaw_cap(0.05, 0.10, trim_yaw=0.04, trim_rad=math.radians(25)) == 0.04
    assert cruise_yaw_cap(0.50, 0.10, trim_yaw=0.04, trim_rad=math.radians(25)) == 0.10
    assert cruise_yaw_cap(-0.20, 0.10, trim_yaw=0.04, trim_rad=math.radians(25)) == 0.04


def test_early_open_arc_may_leave_the_goal_line():
    names = ["fwd", "open_left", "open_right"]
    heading_ok = np.array([True, False, False])
    opened = allow_early_open(names, heading_ok, True)
    assert opened.tolist() == [True, True, True]
    assert allow_early_open(names, heading_ok, False).tolist() == [True, False, False]
    assert align_may_replace("open_left", True) is False
    assert align_may_replace("fwd", True) is True
    assert align_may_replace("open_right", False) is True


def test_open_side_arcs_are_steady_and_opposite():
    left, right = open_side_arcs(k=10, dx=0.8, yaw=0.16)
    assert left[0] == "open_left" and right[0] == "open_right"
    assert np.all(left[1][:, 0] == 0.8) and np.all(right[1][:, 0] == 0.8)
    assert np.all(left[1][:, 3] == -0.16)
    assert np.all(right[1][:, 3] == 0.16)
    again = open_side_arcs(k=10, dx=0.8, yaw=0.16)
    assert np.array_equal(left[1], again[0][1])
    assert np.array_equal(right[1], again[1][1])


def test_goal_arc_yaws_toward_the_bearing_while_moving():
    bearing = 0.30
    a = goal_arc(bearing, k=10, dx=0.8, yaw_cap=0.04)
    assert np.all(a[:, 0] == 0.8)
    assert np.all(a[:7, 3] > 0)
    assert abs(a[:, 3].sum() - bearing) < 1e-9
    left = goal_arc(-0.30, k=10, dx=0.8, yaw_cap=0.04)
    assert np.all(left[left[:, 3] != 0, 3] < 0)
    assert abs(left[:, 3].sum() + 0.30) < 1e-9


def test_smooth_turn_moves_the_executed_yaw_step():
    seq = np.zeros((4, 4))
    seq[0, 3] = -0.31
    seq[0, 0] = 0.1
    seq[2:, 0] = 0.8
    a = smooth_turn(seq, move_dx=0.6, yaw_cap=0.10)
    assert a[0, 0] == 0.6
    assert abs(a[0, 3] + 0.10) < 1e-9


def test_high_survival_left_yaw_loses_to_the_arc():
    names = ["fwd", "yaw_fwd", "goal_arc"]
    surv = np.array([0.98, 0.99, 0.97])
    prog = np.array([2.0, 2.2, 2.1])
    allowed = np.array([True, True, True])
    chosen = hold_straight(names, surv, prog, allowed, 1, 0.95, 0.5)
    assert chosen == 0
    chosen = choose_align(names, surv, allowed, 0.30, 0.95, math.radians(4))
    assert chosen == 2
    # Inside the align threshold the arc is not forced.
    assert choose_align(names, surv, allowed, 0.02, 0.95, math.radians(4)) == -1
    # A clearly less safe arc stays with the progress pick.
    weak = surv.copy()
    weak[2] = 0.95
    assert choose_align(names, weak, allowed, 0.30, 0.95, math.radians(4)) == -1


def test_near_picks_the_moving_arc_toward_the_goal():
    names = ["fwd", "yaw_fwd", "near_arc"]
    cands = np.zeros((3, 2, 4))
    cands[0, :, 0] = 0.8
    cands[1, 0, 0] = 0.6
    cands[1, 0, 3] = -0.10
    cands[2, 0, 0] = 0.8
    cands[2, 0, 3] = 0.18
    surv = np.array([0.99, 0.99, 0.97])
    allowed = np.array([True, True, True])
    assert choose_near(names, cands, surv, allowed, bearing=0.4, surv_min=0.95) == 2
