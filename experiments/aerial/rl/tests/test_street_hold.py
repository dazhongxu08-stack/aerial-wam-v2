"""Street-hold action: stay facing the goal, and do not climb."""

import numpy as np

from experiments.aerial.scripts.wam_phase2_long_eval import select_street_action

LIMITS = np.array([1.0, 0.4, 0.4, 0.3])


def test_cruise_goes_straight_while_the_goal_is_in_front():
    action, name = select_street_action(
        {"forward": 20.0, "left": 9.0, "right": 18.0},
        yaw_err=0.8,
        limits=LIMITS,
    )
    assert name == "cruise"
    assert action[0] > 0.0
    assert action[1] == 0.0
    assert action[2] == 0.0
    assert action[3] == 0.0


def test_clear_street_behind_the_goal_does_not_cruise():
    action, name = select_street_action(
        {"forward": 20.0, "left": 16.0, "right": 16.0},
        yaw_err=2.0,
        limits=LIMITS,
    )
    assert name == "face"
    assert action[0] == 0.0
    assert action[1] == 0.0
    assert action[2] == 0.0
    assert action[3] > 0.0


def test_blocked_nose_shifts_without_yaw_when_goal_is_ahead():
    action, name = select_street_action(
        {"forward": 6.0, "left": 16.0, "right": 7.0},
        yaw_err=-0.4,
        limits=LIMITS,
    )
    assert name == "shift"
    assert action[0] > 0.0
    assert action[1] > 0.0
    assert action[2] == 0.0
    assert action[3] == 0.0


def test_short_cones_do_not_climb():
    action, name = select_street_action(
        {"forward": 3.0, "left": 3.0, "right": 4.0},
        yaw_err=0.0,
        limits=LIMITS,
    )
    assert name == "shift"
    assert action[2] == 0.0
    assert action[3] == 0.0
