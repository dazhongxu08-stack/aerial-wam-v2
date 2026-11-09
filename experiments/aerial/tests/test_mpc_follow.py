import math

import numpy as np

from experiments.aerial.rl.mpc_consequence_eval import follow_fan, follow_pick, follow_side_choice, goal_offset

YAW = 0.3141592653589793


def test_goal_offset_straight_and_turned():
    fwd = np.tile([0.6, 0.0, 0.0, 0.0], (1, 10, 1))
    a, d = goal_offset(fwd, 0.0, 0.2)
    assert abs(a[0]) < 1e-9 and abs(d[0] - 1.2) < 1e-9
    a, _ = goal_offset(fwd, math.radians(30), 0.2)
    assert abs(math.degrees(a[0]) + 30) < 1e-6
    left = np.zeros((1, 10, 4))
    left[0, :4, 3], left[0, 4:, 0] = YAW, 0.6
    a, _ = goal_offset(left, 0.0, 0.2)
    assert a[0] > math.radians(10)
    strafe = np.zeros((1, 10, 4))
    strafe[0, :, 1] = 0.4
    a, _ = goal_offset(strafe, 0.0, 0.2)
    assert abs(math.degrees(a[0]) - 90) < 1e-6


def test_follow_fan_shapes():
    f = follow_fan(10, YAW)
    assert f.shape == (19, 10, 4)
    assert (f[..., 0] >= 0).all() and (f[..., 1] == 0).all() and (f[..., 2] == 0).all()
    spins = [i for i in range(len(f)) if f[i, :, 0].max() == 0]
    assert sorted(np.sign(f[spins, 0, 3]).tolist()) == [-1.0, 1.0]


def test_follow_pick_stays_on_side():
    alpha = np.radians([-40.0, -12.0, 10.0, 60.0, 0.0, -3.0])
    length = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    safe = np.ones(6, bool)
    assert follow_pick(alpha, length, safe, 1.0) == 5
    assert follow_pick(alpha, length, safe, -1.0) == 5
    safe[5] = False
    assert follow_pick(alpha, length, safe, 1.0) == 2
    assert follow_pick(alpha, length, safe, -1.0) == 2
    safe[1] = safe[2] = False
    assert follow_pick(alpha, length, safe, 1.0) == 3
    assert follow_pick(alpha, length, safe, -1.0) == 0
    assert follow_pick(alpha, length, np.zeros(6, bool), 1.0) == -1


def test_follow_side_choice():
    alpha = np.radians([-30.0, 20.0, 0.0])
    length = np.ones(3)
    clear = np.array([0.0, 0.0, 0.0])
    assert follow_side_choice(alpha, length, np.array([True, True, False]), clear, 0.0) == 1.0
    assert follow_side_choice(np.radians([-20.0, 20.0, 0.0]), length, np.array([True, True, False]),
                              np.array([2.0, 1.0, 0.0]), 0.0) == -1.0
    assert follow_side_choice(alpha, length, np.zeros(3, bool), clear, -0.5) == -1.0
