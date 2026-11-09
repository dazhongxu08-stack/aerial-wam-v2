from __future__ import annotations

import numpy as np

from experiments.aerial.rl.consequence_model import altitude_allowed


def _seq(dz_per_step: float) -> np.ndarray:
    a = np.zeros((10, 4))
    a[:, 0] = 0.5
    a[:, 2] = dz_per_step
    return a


def test_level_goal_blocks_net_climb_and_descent():
    c = np.stack([_seq(0.0), _seq(0.2), _seq(-0.2), _seq(0.04)])
    ok = altitude_allowed(c, np.array([0, 0, 24, 0, 0, 0, 0.0]), np.array([50, 0, 24.5]))
    assert ok.tolist() == [True, False, False, True]


def test_goal_above_allows_climb_only():
    c = np.stack([_seq(0.2), _seq(-0.2)])
    ok = altitude_allowed(c, np.array([0, 0, 24, 0, 0, 0, 0.0]), np.array([50, 0, 30.0]))
    assert ok.tolist() == [True, False]
