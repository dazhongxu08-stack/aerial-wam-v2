"""Near-view clearance. No weights and no flight."""
from __future__ import annotations

import math

import numpy as np

from experiments.aerial.rl.body_surround import BODY_RADIUS_M, CAM_DZ, world_from_body
from experiments.aerial.rl.near_clearance import NearClearance
from experiments.aerial.rl.scene_memory import project_pixel


def _state(yaw: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 20.0) -> np.ndarray:
    return np.array([x, y, z, 0.0, 0.0, 0.0, yaw], np.float64)


def _paint(state: np.ndarray, forward: float, left: float, z: float) -> np.ndarray:
    depth = np.full((56, 56), 30.0)
    world = world_from_body(state, forward, left, z)
    pix = project_pixel(world, state[:3], float(state[6]), depth.shape)
    assert pix is not None
    depth[pix] = forward
    return depth


def _wall_seen_earlier(forward: float, left: float):
    here = _state()
    wall_z = float(here[2] + CAM_DZ)
    wall = world_from_body(here, forward, left, wall_z)
    old = _state(x=float(wall[0] - 2.0), y=float(wall[1]), z=float(here[2]))
    mem = NearClearance(history_s=1.0)
    mem.add(_paint(old, 2.0, 0.0, wall_z), old, 0.0)
    mem.add(np.full((56, 56), 30.0), here, 0.2)
    return here, wall, mem


def test_shell_outside_the_current_picture_uses_the_earlier_frame():
    here, wall, mem = _wall_seen_earlier(0.25, 0.35)
    assert project_pixel(wall, here[:3], 0.0, (56, 56)) is None
    got = mem.step(here, np.array([0.8, 0.0, 0.0, 0.0]), 0.2)
    assert got.seen and got.age_s > 0.0
    assert abs(got.distance_m - 0.35) < 0.08
    assert got.distance_m < BODY_RADIUS_M


def test_wall_three_meters_off_the_step_stays_outside_the_body():
    here, _wall, mem = _wall_seen_earlier(1.0, 3.0)
    got = mem.step(here, np.array([0.8, 0.0, 0.0, 0.0]), 0.2)
    assert got.seen
    assert got.distance_m > 2.0


def test_open_depth_and_a_spin_are_not_inside_the_body():
    here = _state()
    mem = NearClearance(history_s=1.0)
    mem.add(np.full((56, 56), 30.0), here, 0.0)
    forward = mem.step(here, np.array([0.8, 0.0, 0.0, 0.0]), 0.0)
    spin = mem.step(here, np.array([0.0, 0.0, 0.0, 0.3]), 0.0)
    assert forward.seen and forward.distance_m > 8.0
    assert not spin.seen and math.isinf(spin.distance_m)


def test_a_frame_older_than_one_second_is_dropped():
    here = _state()
    wall_z = float(here[2] + CAM_DZ)
    wall = world_from_body(here, 0.25, 0.35, wall_z)
    old = _state(x=float(wall[0] - 2.0), y=float(wall[1]), z=float(here[2]))
    mem = NearClearance(history_s=1.0)
    mem.add(_paint(old, 2.0, 0.0, wall_z), old, 0.0)
    mem.add(np.full((56, 56), 30.0), here, 1.4)
    got = mem.step(here, np.array([0.8, 0.0, 0.0, 0.0]), 1.4)
    assert got.distance_m > 8.0
