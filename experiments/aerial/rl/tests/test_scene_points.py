"""A remembered point meets a step by distance to the body path. No grid."""
from __future__ import annotations

import torch

from experiments.aerial.rl.scene_points import nearest, path_distance


def test_a_point_beside_the_path_is_within_the_body():
    fwd = torch.tensor([0.4, 0.4])
    left = torch.tensor([0.3, 1.0])
    dist = path_distance(fwd, left, 0.8, 0.0)
    assert abs(float(dist[0]) - 0.3) < 1e-5
    assert float(dist[1]) > 0.4


def test_a_forward_step_meets_the_wall_it_walks_into():
    depth = torch.full((1, 56, 56), 20.0)
    depth[:, 26:30, 26:30] = 0.5
    state = torch.zeros(1, 7)
    state[0, 2] = 20.0
    gone, stay = nearest(depth, state, state, [0.8, 0.0, 0.0, 0.0])
    assert gone is None
    assert stay is not None and stay <= 0.4


def test_a_wall_left_of_the_path_is_not_entered_by_going_forward():
    depth = torch.full((1, 56, 56), 20.0)
    depth[:, 20:36, 0:4] = 3.0
    state = torch.zeros(1, 7)
    state[0, 2] = 20.0
    _gone, stay = nearest(depth, state, state, [0.8, 0.0, 0.0, 0.0])
    assert stay is None or stay > 0.4


def test_yaw_puts_the_old_wall_outside_and_off_a_forward_step():
    depth = torch.full((1, 56, 56), 20.0)
    depth[:, 24:32, 26:30] = 3.0
    state = torch.zeros(1, 7)
    state[0, 2] = 20.0
    here = state.clone()
    here[0, 6] = 1.2
    gone, stay = nearest(depth, state, here, [0.8, 0.0, 0.0, 0.0])
    assert gone is not None and gone > 0.4
    assert stay is None


def test_a_point_the_body_has_already_reached_does_not_count():
    depth = torch.full((1, 56, 56), 20.0)
    depth[:, 26:30, 26:30] = 3.0
    state = torch.zeros(1, 7)
    state[0, 2] = 20.0
    here = state.clone()
    here[0, 0] = 3.0
    gone, stay = nearest(depth, state, here, [0.8, 0.0, 0.0, 0.0])
    assert gone is None
    assert stay is None


def test_ground_under_the_camera_is_not_a_surface():
    depth = torch.full((1, 56, 56), 30.0)
    depth[:, 52:, :] = 5.0
    state = torch.zeros(1, 7)
    state[0, 2] = 20.0
    gone, stay = nearest(depth, state, state, [0.8, 0.0, 0.0, 0.0])
    assert gone is None and stay is None
