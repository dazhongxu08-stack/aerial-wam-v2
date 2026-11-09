"""Features of a seen view stay in the current body frame. Survival is not read."""
from __future__ import annotations

import numpy as np
import torch

from experiments.aerial.rl.consequence_model import ConsequenceModel
from experiments.aerial.rl.latent_map import (
    MapEncoder, block_body, block_rays, fuse_nearest, latent_map, outside_view, scene_loss,
    spatial_features,
)


def _state(yaw: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 20.0) -> np.ndarray:
    return np.array([x, y, z, 0.0, 0.0, 0.0, yaw], np.float64)


def _wall() -> np.ndarray:
    depth = np.full((56, 56), 30.0, np.float32)
    depth[24:32, 26:30] = 3.0
    return depth


def _feat(wall: np.ndarray) -> np.ndarray:
    """Two-channel map: channel 0 marks the center wall, channel 1 the rest."""
    feat = np.zeros((2, 56, 56), np.float32)
    center = np.zeros_like(wall, dtype=bool)
    center[24:32, 26:30] = True
    feat[0][center] = 1.0
    feat[1][~center] = 1.0
    return feat


def test_the_wall_feature_stays_ahead_and_is_not_a_range():
    depth = _wall()
    grid = latent_map([depth], [_feat(depth)], [_state()])
    rows, cols = np.nonzero(grid[0] > 0)
    assert rows.size
    assert abs(float(cols.mean()) - 16.0) < 2.0
    assert float(rows.mean()) > 16.0
    assert float(grid[0, rows, cols].min()) == 1.0
    assert float(grid[1, rows, cols].max()) == 0.0


def test_yaw_moves_that_same_feature_to_the_left():
    depth = _wall()
    far = np.full_like(depth, 30.0)
    grid = latent_map(
        [depth, far],
        [_feat(depth), np.zeros((2, 56, 56), np.float32)],
        [_state(0.0), _state(1.0)],
    )
    cols = np.nonzero(grid[0] > 0)[1]
    assert cols.size
    assert float(cols.mean()) > 18.0
    rows = np.nonzero(grid[0] > 0)[0]
    assert float(grid[0, rows, cols].min()) == 1.0


def test_ground_below_the_camera_is_not_kept():
    depth = np.full((56, 56), 30.0, np.float32)
    depth[54:, :] = 5.0
    feat = np.ones((2, 56, 56), np.float32)
    grid = latent_map([depth], [feat], [_state()])
    assert float(grid.sum()) == 0.0


def test_reading_features_does_not_change_survival():
    torch.manual_seed(0)
    model = ConsequenceModel(horizon=5, hidden=32).eval()
    depth = torch.rand(2, 56, 56)
    goal, prev = torch.randn(2, 4), torch.zeros(2, 4)
    actions = torch.zeros(2, 5, 4)
    with torch.no_grad():
        before = model(depth, goal, prev, actions)["hazard_logit"].clone()
    feat = spatial_features(model, depth)
    with torch.no_grad():
        after = model(depth, goal, prev, actions)["hazard_logit"]
    assert feat.shape == (2, 64, 14, 14)
    assert torch.allclose(before, after, atol=1e-6)


def test_nearer_feature_overwrites_and_keeps_its_gradient():
    feat = torch.zeros(1, 2, 1, 2, requires_grad=True)
    with torch.no_grad():
        feat[0, 0, 0, 0] = 4.0
        feat[0, 1, 0, 1] = 7.0
    cell = torch.tensor([[[5, 5]]])
    rng = torch.tensor([[[3.0, 1.0]]])
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    grid, dist = fuse_nearest(feat, cell, rng, valid)
    assert float(grid[0, 1, 0, 5].detach()) == 7.0
    assert float(grid[0, 0, 0, 5].detach()) == 0.0
    assert abs(float(dist[0, 0, 5].detach()) - 1.0) < 1e-5
    grid[0, 1, 0, 5].backward()
    assert float(feat.grad[0, 1, 0, 1]) != 0.0
    assert float(feat.grad[0, 0, 0, 0]) == 0.0


def test_a_turned_wall_stays_outside_the_camera_on_the_left():
    depth = torch.full((1, 56, 56), 20.0)
    depth[:, 20:36, 0:8] = 4.0
    state = torch.zeros(1, 7)
    state[0, 2] = 20.0
    here = state.clone()
    here[0, 6] = 1.0
    cell, _rng, valid, dist, fwd, left, _dz = block_body(depth, state, here)
    near = valid[0] & (dist[0] < 6.0)
    assert bool(near.any())
    assert bool(outside_view(fwd, left)[0][near].all())
    cols = cell[0][near] % 32
    assert int(cols.min()) > 18


def test_scene_loss_trains_the_depth_of_a_wall_that_left_the_view():
    torch.manual_seed(0)
    depth = torch.full((2, 56, 56), 20.0)
    depth[0, 20:36, 0:8] = 4.0
    state = torch.zeros(2, 7)
    state[:, 2] = 20.0
    state[1, 6] = 1.0
    depth_m = depth.unsqueeze(0)
    pose = state.unsqueeze(0)
    enc = MapEncoder(8)
    opt = torch.optim.AdamW(enc.parameters(), lr=3e-3)
    last = None
    for _ in range(80):
        loss, last = scene_loss(enc, depth_m, pose)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert last["out_n"] > 0
    assert last["out_mae"] is not None and last["out_mae"] < 1.0


def test_a_center_wall_lands_in_front_of_the_body():
    depth = torch.full((1, 56, 56), 30.0)
    depth[:, 28:32, 28:32] = 3.0
    state = torch.zeros(1, 7)
    state[0, 2] = 20.0
    cell, rng, valid, _ = block_rays(depth, state, state)
    assert bool(valid[0, 7, 7])
    assert int(cell[0, 7, 7]) // 32 == 22
    assert abs(int(cell[0, 7, 7]) % 32 - 16) <= 1
    assert abs(float(rng[0, 7, 7]) - 3.0) < 0.3
