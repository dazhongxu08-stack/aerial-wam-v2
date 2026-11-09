"""Yaw shift of an image-depth column profile."""
import math

import numpy as np

from experiments.aerial.rl.cone_transition import (
    column_profile,
    cones_from_column_profile,
    stamp_column_profile,
    yaw_shift_profile,
)
from experiments.aerial.rl.depth_geometry import cone_clearances
from experiments.aerial.rl.env.obs import Observation


def test_positive_yaw_moves_the_right_wall_leftward():
    profile = np.full(90, 30.0)
    profile[80:] = 2.0
    shifted = yaw_shift_profile(profile, math.radians(45.0))
    assert shifted[35] == 2.0
    cones = cones_from_column_profile(shifted)
    assert cones[1] < 5.0


def test_column_profile_matches_cone_halves():
    depth = np.full((20, 40), 30.0)
    depth[:, 30:] = 2.0
    prof = column_profile(depth)
    cones = cones_from_column_profile(prof)
    raw = cone_clearances(depth)
    assert cones[1] == raw["left"]
    assert cones[2] == raw["right"]


def test_stamp_uses_the_depth_already_computed():
    depth = np.full((8, 10), 12.0)
    depth[:, :5] = 4.0

    class _Pred:
        last_depth = depth

    obs = Observation(rgb=np.zeros((8, 8, 3), dtype=np.uint8), state=np.zeros(7, dtype=np.float32))
    stamp_column_profile(obs, _Pred())
    assert obs.info["depth_profile_pred"].shape == (10,)
    assert obs.info["depth_profile_pred"][0] == 4.0


def test_zero_yaw_keeps_the_open_side():
    profile = np.full(90, 8.0)
    profile[:45] = 20.0
    same = yaw_shift_profile(profile, 0.0)
    cones = cones_from_column_profile(same)
    assert cones[1] > cones[2]
