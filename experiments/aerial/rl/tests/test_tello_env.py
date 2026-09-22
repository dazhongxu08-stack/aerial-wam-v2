"""Unit tests for Tello body-delta → rc mapping (no hardware)."""
from __future__ import annotations

import math

import numpy as np

from experiments.aerial.rl.env.tello_env import body_delta_to_rc


def test_body_delta_to_rc_forward_positive() -> None:
    dt = 0.2
    # 0.12 m forward in 0.2s → 0.6 m/s → full stick forward
    a, b, c, d = body_delta_to_rc(np.array([0.12, 0.0, 0.0, 0.0]), dt, max_v_mps=0.6)
    assert a == 0 and c == 0 and d == 0
    assert b == 100


def test_body_delta_to_rc_left_maps_negative_a() -> None:
    dt = 0.2
    a, b, c, d = body_delta_to_rc(np.array([0.0, 0.12, 0.0, 0.0]), dt, max_v_mps=0.6)
    assert a == -100 and b == 0


def test_body_delta_to_rc_yaw_ccw_maps_negative_d() -> None:
    dt = 0.2
    # +yaw CCW at max rate → Tello d is CW+, so negative
    dyaw = (math.pi / 4.0) * dt
    a, b, c, d = body_delta_to_rc(
        np.array([0.0, 0.0, 0.0, dyaw]), dt, max_yaw_rad_s=math.pi / 4.0
    )
    assert d == -100
