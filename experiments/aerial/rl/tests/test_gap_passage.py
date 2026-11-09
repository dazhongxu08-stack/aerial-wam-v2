"""The gap teacher holds west until the opening, then south, then the goal."""
import math

import numpy as np

from experiments.aerial.rl.gap_passage import (
    close_then_gap_action,
    gap_action,
    passage_phase,
    reset_close_passage,
)


def test_west_until_the_gap_then_south_then_goal():
    yaw_w, phase_w = passage_phase((-930.0, 80.0, 24.0), (-952.6, 15.9))
    assert phase_w == "west"
    assert abs(yaw_w - math.pi) < 1e-6

    yaw_s, phase_s = passage_phase((-970.0, 60.0, 24.0), (-952.6, 15.9))
    assert phase_s == "south"
    assert abs(yaw_s + math.pi / 2.0) < 1e-6

    # At the open latitude, cross to the south-street center, not the goal x.
    yaw_e, phase_e = passage_phase((-970.0, 22.0, 24.0), (-952.6, -14.1))
    assert phase_e == "east"
    assert abs(yaw_e) < 1e-6

    # Route 1 still has to go south after that crossing.
    _, phase_s2 = passage_phase((-945.0, 10.0, 24.0), (-952.6, -14.1))
    assert phase_s2 == "south"
    yaw_g, phase_g = passage_phase((-954.0, -12.0, 24.0), (-952.6, -14.1))
    assert phase_g == "goal"
    assert abs(yaw_g) < 1.2

    action, phase = gap_action((-930.0, 80.0, 20.0), math.pi, (-952.6, 15.9), 24.0)
    assert phase == "west"
    assert action[0] == 1.0
    assert action[2] > 0.0
    assert action[3] == 0.0


def test_close_presses_until_forward_is_inside_20m():
    reset_close_passage()
    goal = (-952.6, -14.1)
    pos = (-930.0, 71.0, 29.0)
    press = -2.094395160675049
    far = np.full((40, 64), 30.0)
    action, phase = close_then_gap_action(pos, press, goal, 29.0, far, press_yaw=press)
    assert phase == "close"
    assert action[0] == 1.0
    assert action[3] == 0.0

    near = np.full((40, 64), 12.0)
    action, phase = close_then_gap_action(pos, press, goal, 29.0, near, press_yaw=press)
    assert phase == "west"
    assert action[0] == 0.0
    assert action[3] < 0.0
    # Once inside 20 m, a later open view does not resume the press.
    action, phase = close_then_gap_action(pos, math.pi, goal, 29.0, far, press_yaw=press)
    assert phase == "west"
    assert action[0] == 1.0
