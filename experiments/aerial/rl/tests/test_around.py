"""Detour when the goal bearing is a wall and a side street is open."""
import math

import numpy as np

from experiments.aerial.rl.around import AroundDetour

LIM = np.array([1.0, 0.4, 0.4, math.pi / 10.0])


def _step_body(x, y, yaw, action):
    dx, dy, dz, dyaw = (float(v) for v in action)
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * dx - s * dy, y + s * dx + c * dy, yaw + dyaw


def _sense(x, y, yaw):
    """South is blocked until the vehicle has moved west of x=-958."""
    south = abs((yaw + math.pi / 2.0 + math.pi) % (2.0 * math.pi) - math.pi) < 0.35
    west = abs((yaw - math.pi + math.pi) % (2.0 * math.pi) - math.pi) < 0.35
    if south and x > -958.0:
        # After the first sidestep the other cone becomes longer. The detour
        # must keep the original side instead of turning back.
        if x < -952.0:
            return 3.0, 50.0, 10.0
        return 3.0, 10.0, 50.0
    if south:
        return 34.0, 20.0, 20.0
    if west:
        return 80.0, 20.0, 20.0
    return 40.0, 20.0, 20.0


def test_open_goal_goes_forward():
    detour = AroundDetour()
    action, phase = detour.act(
        [-950.0, 40.0], -math.pi / 2.0, [-952.0, 16.0], 34.0, 20.0, 20.0, LIM
    )
    assert phase == "goal"
    assert action[0] > 0.0
    assert abs(action[3]) < 1e-6


def test_blocked_goal_flies_to_the_open_side_then_south():
    detour = AroundDetour(open_m=15.0, block_m=6.0, recheck_m=8.0, cruise=1.0)
    x, y, yaw = -950.0, 48.7, -math.pi / 2.0
    phases = []
    min_x = x
    for _ in range(80):
        fwd, left, right = _sense(x, y, yaw)
        action, phase = detour.act([x, y], yaw, [-952.6, 15.9], fwd, left, right, LIM)
        phases.append(phase)
        x, y, yaw = _step_body(x, y, yaw, action)
        min_x = min(min_x, x)
        if "detour" in phases and phase == "goal":
            break
    assert min_x < -957.5
    assert phases[-1] == "goal"
