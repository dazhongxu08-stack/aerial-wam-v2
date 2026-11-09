"""A remembered surface occupies a 3D cell, indexed in forward, left, and up."""
from __future__ import annotations

import torch

from experiments.aerial.rl.scene_voxels import NZ, nearest_voxel, voxel_index


def test_up_is_its_own_index():
    fwd = torch.tensor([1.0, 1.0])
    left = torch.tensor([0.2, 0.2])
    dz = torch.tensor([-1.0, 0.2])
    ok, i, j, k = voxel_index(fwd, left, dz)
    assert bool(ok.all())
    assert int(i[0]) == int(i[1])
    assert int(j[0]) == int(j[1])
    assert int(k[0]) != int(k[1])
    assert NZ == 5


def test_a_point_above_the_band_has_no_cell():
    ok, *_ = voxel_index(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([3.0]))
    assert not bool(ok.any())


def test_a_forward_step_meets_the_wall_at_camera_height():
    depth = torch.full((1, 56, 56), 20.0)
    depth[:, 26:30, 26:30] = 0.5
    state = torch.zeros(1, 7)
    state[0, 2] = 20.0
    gone, stay = nearest_voxel(depth, state, state, [0.8, 0.0, 0.0, 0.0])
    assert gone is None
    assert stay is not None and stay[0] <= 0.4
    assert stay[1] < 0.0


def test_a_point_the_body_has_already_reached_is_not_written():
    depth = torch.full((1, 56, 56), 20.0)
    depth[:, 26:30, 26:30] = 3.0
    state = torch.zeros(1, 7)
    state[0, 2] = 20.0
    here = state.clone()
    here[0, 0] = 3.0
    gone, stay = nearest_voxel(depth, state, here, [0.8, 0.0, 0.0, 0.0])
    assert gone is None and stay is None


def test_a_wall_left_of_the_path_is_not_entered_by_going_forward():
    depth = torch.full((1, 56, 56), 20.0)
    depth[:, 20:36, 0:4] = 3.0
    state = torch.zeros(1, 7)
    state[0, 2] = 20.0
    _gone, stay = nearest_voxel(depth, state, state, [0.8, 0.0, 0.0, 0.0])
    assert stay is None or stay[0] > 0.4
