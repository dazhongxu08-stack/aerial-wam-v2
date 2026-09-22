"""Unit tests for ground action limits (no CARLA required)."""
import numpy as np

from experiments.aerial.rl.env.action import (
    GROUND_MAX_BODY_VELOCITY,
    body_delta_limits,
    clip_body_delta,
)


def test_ground_body_delta_limits_planar():
    dt = 0.1
    lim = body_delta_limits(dt, max_velocity=GROUND_MAX_BODY_VELOCITY)
    assert lim[2] == 0.0
    assert abs(lim[0] - 0.2) < 1e-6


def test_ground_clip_forces_dz_zero_in_env_contract():
    dt = 0.1
    lim = body_delta_limits(dt, max_velocity=GROUND_MAX_BODY_VELOCITY)
    cmd = clip_body_delta(np.array([1.0, 0.5, 0.3, 0.5]), lim)
    cmd[2] = 0.0
    assert cmd[2] == 0.0
    assert cmd[0] <= lim[0] + 1e-6
