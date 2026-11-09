"""Seen depth stays in the current body frame. Default survival checkpoints stay unchanged."""
from __future__ import annotations

import numpy as np
import torch

from experiments.aerial.rl.body_surround import body_from_world, depth_to_world, in_view
from experiments.aerial.rl.consequence_model import (
    ConsequenceModel,
    load_checkpoint,
    load_into_pose_scene,
    save_checkpoint,
)
from experiments.aerial.rl.pose_scene import scene_in_body


def _state(yaw: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 20.0) -> np.ndarray:
    return np.array([x, y, z, 0.0, 0.0, 0.0, yaw], np.float64)


def _wall() -> np.ndarray:
    depth = np.full((56, 56), 30.0, np.float32)
    depth[24:32, 26:30] = 3.0
    return depth


def test_yaw_moves_the_seen_wall_to_the_left_of_the_body():
    depth = _wall()
    st0 = _state(0.0)
    st1 = _state(1.0)
    grid = scene_in_body([depth, np.full_like(depth, 30.0)], [st0, st1], stride=1)
    pts = depth_to_world(depth, st0, max_range=16.0, stride=1)
    fwd, left, _ = body_from_world(st1, pts)
    assert float(np.median(left)) > 1.0
    assert not bool(in_view(np.array([float(np.median(fwd))]), np.array([float(np.median(left))]))[0])
    cols = np.nonzero(grid[0] > 0)[1]
    assert cols.size
    assert float(cols.mean()) > 18.0


def test_the_same_wall_stays_ahead_before_the_yaw():
    depth = _wall()
    st0 = _state(0.0)
    grid = scene_in_body([depth], [st0], stride=1)
    rows, cols = np.nonzero(grid[0] > 0)
    assert rows.size
    assert abs(float(cols.mean()) - 16.0) < 2.0
    assert float(rows.mean()) > 16.0


def test_ground_below_the_camera_is_not_the_scene():
    depth = np.full((56, 56), 30.0, np.float32)
    depth[54:, :] = 5.0
    grid = scene_in_body([depth], [_state()], stride=1)
    assert float(grid.sum()) == 0.0


def test_empty_grid_matches_the_frozen_head_and_a_wall_does_not_until_the_columns_move(tmp_path):
    torch.manual_seed(0)
    base = ConsequenceModel(horizon=5, hidden=32)
    assert base.pose_scene_dim == 0
    save_checkpoint(tmp_path / "base.pt", base)
    loaded = load_checkpoint(tmp_path / "base.pt")
    assert loaded.pose_scene_dim == 0
    torch.manual_seed(1)
    model = ConsequenceModel(horizon=5, hidden=32, pose_scene_dim=8)
    load_into_pose_scene(model, tmp_path / "base.pt")
    depth = torch.rand(4, 56, 56)
    goal, prev = torch.randn(4, 4), torch.zeros(4, 4)
    actions = torch.zeros(4, 5, 4)
    wall = torch.as_tensor(scene_in_body([_wall()], [_state()], stride=1)).unsqueeze(0).expand(4, -1, -1, -1)
    empty = torch.zeros(4, 2, 32, 32)
    with torch.no_grad():
        a = base(depth, goal, prev, actions)["hazard_logit"]
        b = model(depth, goal, prev, actions, scene=empty)["hazard_logit"]
        c = model(depth, goal, prev, actions, scene=wall)["hazard_logit"]
        feat = model.scene_enc(wall[:1])
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(b, c, atol=1e-5)
    assert float(feat.abs().sum()) > 0.0
    with torch.no_grad():
        model.ctx.weight[:, -8:].normal_(0.0, 0.02)
    with torch.no_grad():
        moved = model(depth, goal, prev, actions, scene=wall)["hazard_logit"]
        still = model(depth, goal, prev, actions, scene=empty)["hazard_logit"]
    assert float((moved - still).abs().sum()) > 0.0
    assert float(model.ctx.weight[:, -8:].detach().abs().sum()) > 0.0
