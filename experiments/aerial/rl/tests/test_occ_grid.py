from __future__ import annotations

import numpy as np

from experiments.aerial.rl.goal_features import advance_goal_rel_body
from experiments.aerial.rl.occ_grid import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    advance_points,
    class_balanced_ce_np,
    counted_same_sign,
    fuse_window,
    gate_passes,
    hits_from_depth,
    integrate_label,
    ray_length,
    score_gate_window,
)


def _center_depth(h: int, w: int, meters: float) -> np.ndarray:
    depth = np.full((h, w), np.nan, dtype=np.float32)
    depth[h // 2, w // 2] = meters
    return depth


def test_hit_marks_occupied_and_free_before_it():
    depth = _center_depth(32, 32, 10.0)
    pose = np.zeros((1, 7), dtype=np.float64)
    grid = fuse_window(depth[None], pose, 0, stride=1)
    hits = hits_from_depth(depth, stride=1)
    assert hits.shape == (1, 3)
    np.testing.assert_allclose(hits[0, 0], 10.0, atol=1e-6)
    occupied = np.argwhere(grid == OCCUPIED)
    free = np.argwhere(grid == FREE)
    assert occupied.shape[0] >= 1
    assert free.shape[0] >= 1
    assert int(free[:, 2].min()) < int(occupied[:, 2].max())


def test_far_depth_does_not_clear_cells():
    depth = _center_depth(32, 32, 100.0)
    pose = np.zeros((1, 7), dtype=np.float64)
    grid = fuse_window(depth[None], pose, 0, stride=1)
    assert not np.any(grid == FREE)
    assert not np.any(grid == OCCUPIED)


def test_later_hit_overwrites_a_cell_the_long_ray_called_free():
    far = _center_depth(32, 32, 12.0)
    near = _center_depth(32, 32, 10.0)
    pose = np.zeros((2, 7), dtype=np.float64)
    grid = fuse_window(np.stack([far, near]), pose, 1, stride=1)
    near_only = fuse_window(near[None], pose[:1], 0, stride=1)
    cell = tuple(int(v) for v in np.argwhere(near_only == OCCUPIED)[0])
    assert int(grid[cell]) == OCCUPIED


def test_action_integration_matches_goal_features_and_shifts_the_cell():
    point = np.array([[10.0, 0.0, 0.0]], dtype=np.float64)
    action = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float64)
    moved = advance_points(point, action)[0]
    goal = advance_goal_rel_body(np.array([10.0, 0.0, 0.0, 10.0]), action)
    np.testing.assert_allclose(moved, goal[:3], atol=1e-6)

    grid = np.full((4, 30, 60), UNKNOWN, dtype=np.uint8)
    grid[2, 15, 30] = OCCUPIED
    out = integrate_label(grid, action.reshape(1, 4))
    assert not np.any(out[:, :, 30] == OCCUPIED)
    assert np.any(out == OCCUPIED)


def test_copy_and_all_free_fail_the_gate():
    assert counted_same_sign(10.0, 12.0, 10.0, label_stop="occupied") is False
    assert counted_same_sign(8.0, 12.0, 10.0, label_stop="occupied") is False
    assert counted_same_sign(12.0, 12.0, 10.0, label_stop="occupied") is True
    assert counted_same_sign(12.0, 10.2, 10.0, label_stop="occupied") is None
    assert counted_same_sign(12.0, 14.0, 10.0, label_stop="unknown") is None
    assert gate_passes([None] * 80) is False
    assert gate_passes([False] * 50) is False
    assert gate_passes([True] * 40 + [False] * 10) is True


def test_balanced_ce_skips_an_empty_class():
    logits = np.zeros((4, 3), dtype=np.float64)
    target = np.full((4,), UNKNOWN, dtype=np.int64)
    got = class_balanced_ce_np(logits, target)
    assert abs(got - np.log(3.0)) < 1e-5


def _forward_wall() -> np.ndarray:
    grid = np.full((4, 30, 60), FREE, dtype=np.uint8)
    grid[:, :, 30] = OCCUPIED
    return grid


def _eight_forward(dyaw: float) -> np.ndarray:
    actions = np.zeros((8, 4), dtype=np.float64)
    actions[:, 0] = 0.6
    actions[0::2, 3] = dyaw
    return actions


def test_forward_into_wall_rejects_an_unchanged_grid():
    start = _forward_wall()
    label = integrate_label(start, _eight_forward(0.0))
    same = score_gate_window(start, label, label, 0.0)
    copy = score_gate_window(start, label, start, 0.0)
    assert same["forward"] is True
    assert copy["forward"] is False


def test_turn_toward_the_longer_side_is_the_opening_check():
    start = np.full((4, 30, 60), FREE, dtype=np.uint8)
    start[:, 15:, 50] = OCCUPIED
    start[:, :15, 35] = OCCUPIED
    actions = _eight_forward(0.05)
    label = integrate_label(start, actions)
    toward = score_gate_window(start, label, label, float(actions[:, 3].sum()))
    away = score_gate_window(start, label, label, -float(actions[:, 3].sum()))
    assert toward["open"] is True
    assert away["open"] is None


def test_forward_ray_stops_on_occupied():
    grid = np.full((4, 30, 60), FREE, dtype=np.uint8)
    grid[:, :, 30] = OCCUPIED
    meters, stop = ray_length(grid, 0.0)
    assert stop == "occupied"
    assert 9.0 < meters < 11.5
