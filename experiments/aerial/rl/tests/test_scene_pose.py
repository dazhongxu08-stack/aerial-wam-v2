"""A stored scene keeps its image. Pose is the reference in the current body frame."""
from __future__ import annotations

import math

import torch

from experiments.aerial.rl.consequence_model import (
    ConsequenceModel,
    load_checkpoint,
    load_into_scene_pose,
    save_checkpoint,
)
from experiments.aerial.rl.scene_pose import relative_pose


def _state(x=0.0, y=0.0, z=20.0, yaw=0.0) -> torch.Tensor:
    s = torch.zeros(7)
    s[0], s[1], s[2], s[6] = x, y, z, yaw
    return s


def test_yaw_changes_the_pose_reference_and_keeps_the_image():
    torch.manual_seed(0)
    mem_dim, hidden = 8, 16
    from experiments.aerial.rl.scene_pose import ScenePoseMemory
    mem = ScenePoseMemory(mem_dim, hidden, length=8)
    image = torch.zeros(1, 1, 56, 56)
    image[:, :, 24:32, 26:30] = 1.0
    other = torch.zeros_like(image)
    other[:, :, 4:12, 4:12] = 1.0
    with torch.no_grad():
        a = mem.encode(image)
        b = mem.encode(image.clone())
        c = mem.encode(other)
    assert torch.allclose(a, b)
    assert float((a - c).abs().sum()) > 0.0

    seen = _state(x=3.0, yaw=0.0).view(1, 1, 7)
    now = _state(yaw=1.0).view(1, 7)
    rel = relative_pose(now, seen)
    forward, left, _, sy, cy = rel[0, 0].tolist()
    assert abs(forward - 3.0 * math.cos(1.0) / 8.0) < 1e-5
    assert abs(left - 3.0 * math.sin(1.0) / 8.0) < 1e-5
    assert abs(sy - math.sin(-1.0)) < 1e-5
    assert abs(cy - (math.cos(-1.0) - 1.0)) < 1e-5
    turned = relative_pose(_state(yaw=0.0).view(1, 7), seen)
    assert float((rel - turned).abs().sum()) > 0.0
    here = _state().view(1, 7)
    assert torch.allclose(relative_pose(here, here.view(1, 1, 7)), torch.zeros(1, 1, 5), atol=1e-6)


def test_empty_records_match_the_frozen_head(tmp_path):
    torch.manual_seed(0)
    base = ConsequenceModel(horizon=5, hidden=32)
    assert base.scene_pose_dim == 0
    save_checkpoint(tmp_path / "base.pt", base)
    assert load_checkpoint(tmp_path / "base.pt").scene_pose_dim == 0
    torch.manual_seed(1)
    model = ConsequenceModel(horizon=5, hidden=32, scene_pose_dim=8)
    load_into_scene_pose(model, tmp_path / "base.pt")
    depth = torch.rand(4, 56, 56)
    goal, prev = torch.randn(4, 4), torch.zeros(4, 4)
    actions = torch.zeros(4, 5, 4)
    image = torch.zeros(4, 1, 56, 56)
    image[:, :, 24:32, 26:30] = 1.0
    seen = _state(x=3.0).view(1, 1, 7).expand(4, 1, 7)
    now = _state(yaw=1.0).view(1, 7).expand(4, 7)
    with torch.no_grad():
        a = base(depth, goal, prev, actions)["hazard_logit"]
        b = model(depth, goal, prev, actions)["hazard_logit"]
        c = model(depth, goal, prev, actions, scene_depth=image, scene_state=seen, body_state=now)["hazard_logit"]
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(b, c, atol=1e-5)
    cones = model(depth, goal, prev, actions)["log_cones"]
    with torch.no_grad():
        model.scene_hazard.s.weight.normal_(0.0, 0.02)
        moved = model(depth, goal, prev, actions, scene_depth=image, scene_state=seen, body_state=now)
        still = model(depth, goal, prev, actions)
        repeat = depth[:, None]
        same_pose = now[:, None, :]
        held = model(depth, goal, prev, actions, scene_depth=repeat, scene_state=same_pose, body_state=now)
    assert float((moved["hazard_logit"] - still["hazard_logit"]).abs().sum()) > 0.0
    assert torch.allclose(moved["log_cones"], cones, atol=1e-5)
    assert torch.allclose(held["hazard_logit"], still["hazard_logit"], atol=1e-5)
    blank = torch.zeros(4, 1, dtype=torch.bool)
    with torch.no_grad():
        masked = model(depth, goal, prev, actions, scene_depth=image, scene_state=seen,
                       body_state=now, scene_mask=blank)["hazard_logit"]
    assert torch.allclose(masked, still["hazard_logit"], atol=1e-5)
