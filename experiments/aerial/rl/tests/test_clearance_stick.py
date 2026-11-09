import numpy as np

from experiments.aerial.rl.clearance_stick import stick_target


def test_goal_ahead_stays_a_small_forward_stick():
    goal = np.array([11.3, 4.9, 0.0, 12.3], dtype=np.float32)
    stick = stick_target(14.5, 11.7, 22.8, goal)
    assert stick[0] > 0.5
    assert abs(float(stick[3])) < 0.1


def test_near_front_with_no_opening_stops_even_if_the_goal_is_ahead():
    goal = np.array([30.0, 8.0, 0.0, 31.0], dtype=np.float32)
    stick = stick_target(6.0, 7.0, 6.5, goal)
    assert stick[0] == 0.0
    assert abs(float(stick[3])) < 0.1


def test_side_goal_yaws_toward_the_open_side_and_stops():
    goal = np.array([17.3, -30.3, 0.0, 35.0], dtype=np.float32)
    stick = stick_target(11.6, 10.6, 36.1, goal)
    assert stick[0] == 0.0
    assert float(stick[3]) <= -0.15


def test_measured_clearances_uses_the_center_of_the_depth_map():
    from experiments.aerial.rl.clearance_stick import measured_clearances

    depth = np.full((40, 40), 28.0)
    depth[16:24, 16:24] = 5.0
    mean_m, left, right = measured_clearances(depth)
    assert mean_m < 8.0
    assert left > 20.0 and right > 20.0
