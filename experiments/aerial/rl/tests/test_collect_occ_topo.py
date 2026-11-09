from __future__ import annotations

import numpy as np

from experiments.aerial.rl.collect_occ_topo import (
    DYAW,
    facade_ahead,
    holdout_index,
    imu6_from_airsim,
    overlaps_holdout,
    pack_proprio,
    review_episode,
    scripted_action,
)


def test_imu_ned_to_body_forward_left_up():
    got = imu6_from_airsim([0.1, 0.2, 0.3], [1.0, 2.0, 9.8])
    np.testing.assert_allclose(got, [0.1, -0.2, -0.3, 1.0, -2.0, -9.8])


def test_proprio_is_command_then_imu_not_a_world_pose():
    proprio = pack_proprio([0.6, 0.0, 0.0, DYAW], imu6_from_airsim([0.0, 0.0, -0.1], [0.0, 0.0, 9.8]))
    assert proprio.shape == (10,)
    np.testing.assert_allclose(proprio[:4], [0.6, 0.0, 0.0, DYAW])
    np.testing.assert_allclose(proprio[4:], [0.0, 0.0, 0.1, 0.0, 0.0, -9.8])


def test_scripted_yaw_alternates_every_eighth_step():
    first = scripted_action(0)
    later = scripted_action(1)
    eighth = scripted_action(8)
    flipped = scripted_action(0, yaw_sign=-1.0)
    np.testing.assert_allclose(first, [0.6, 0.0, 0.0, DYAW])
    np.testing.assert_allclose(later, [0.6, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(eighth, [0.6, 0.0, 0.0, -DYAW])
    np.testing.assert_allclose(flipped, [0.6, 0.0, 0.0, -DYAW])


def test_facade_uses_center_depth_inside_15m():
    depth = np.full((9, 9), 40.0, dtype=np.float32)
    assert facade_ahead(depth) is False
    depth[3:6, 3:6] = 8.0
    assert facade_ahead(depth) is True


def test_holdout_rejects_same_image_and_nearby_start():
    holdout = holdout_index(
        [{"image_path": "scene/a", "pos": [[0.0, 0.0, 12.0], [10.0, 0.0, 12.0]]}]
    )
    assert overlaps_holdout({"image_path": "scene/a", "pos": [[1000.0, 0.0, 12.0]]}, holdout)
    assert overlaps_holdout({"image_path": "", "pos": [[10.0, 0.0, 12.0]]}, holdout)
    assert overlaps_holdout({"image_path": "", "pos": [[80.0, 0.0, 12.0]]}, holdout) is False


def test_spawn_collision_and_short_episode_are_dropped():
    positions = np.zeros((5, 3), dtype=np.float64)
    collided = np.array([False, True, False, False, False])
    regions = {"urban_regions": [], "urban_interior": [], "water_exclusion": []}
    ok, reasons = review_episode(positions, collided, regions, holdout_overlap=False)
    assert ok is False
    assert "too_short" in reasons
    assert "spawn_collision" in reasons
