"""Surround map geometry. No weights and no flight."""
from __future__ import annotations

import math

import numpy as np

from experiments.aerial.rl.body_surround import (
    BODY_RADIUS_M,
    BodySurround,
    body_from_world,
    depth_to_world,
    in_view,
    step_sweeps,
    world_from_body,
)
from experiments.aerial.rl.mpc_consequence_eval import depth_points


def _state(yaw: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 20.0) -> np.ndarray:
    return np.array([x, y, z, 0.0, 0.0, 0.0, yaw], np.float64)


def test_left_front_wall_leaves_the_view_and_stays_on_the_map():
    st = _state()
    wall = world_from_body(st, 3.0 * math.cos(math.radians(30)), 3.0 * math.sin(math.radians(30)), 20.0)
    mem = BodySurround()
    mem.add_world(wall[None, :], 0.0)
    turned = _state(yaw=math.radians(40))
    fwd, left, _ = body_from_world(turned, wall[None, :])
    assert not bool(in_view(fwd, left)[0])
    assert left[0] > 0.0
    assert mem.occupied(turned, 0.0)
    assert not step_sweeps(mem, turned, np.array([0.8, 0.0, 0.0, 0.0]), 0.0)


def test_forward_step_sweeps_a_wall_just_outside_the_view():
    st = _state()
    # Just outside the 45 deg field, and within the 0.4 m body of a forward step.
    mem = BodySurround()
    mem.add_body(st, 0.25, 0.35, 20.0, 0.0)
    fwd, left, _ = body_from_world(st, world_from_body(st, 0.25, 0.35, 20.0)[None, :])
    assert not bool(in_view(fwd, left)[0])
    assert math.hypot(0.25, 0.35) > 0.0
    assert 0.35 <= BODY_RADIUS_M + 0.05
    assert step_sweeps(mem, st, np.array([0.8, 0.0, 0.0, 0.0]), 0.0)


def test_wall_seen_four_seconds_ago_stays_while_looking_away():
    st = _state()
    mem = BodySurround()
    mem.add_body(st, 3.0 * math.cos(math.radians(30)), 3.0 * math.sin(math.radians(30)), 20.0, 0.0)
    away = _state(yaw=math.radians(40))
    mem.add_body(away, 6.0, -1.0, 20.0, 4.0)
    img = mem.render(away, 4.0)
    fwd, left, _ = body_from_world(away, world_from_body(st, 3.0 * math.cos(math.radians(30)), 3.0 * math.sin(math.radians(30)), 20.0)[None, :])
    i = int(np.floor((fwd[0] + 8.0) / 0.5))
    j = int(np.floor((left[0] + 8.0) / 0.5))
    assert img[0, i, j] < 1.0
    assert abs(float(img[1, i, j]) - 0.4) < 1e-6


def test_cells_farther_than_40m_are_dropped():
    st = _state()
    mem = BodySurround()
    mem.add_body(st, 3.0, 0.0, 20.0, 0.0)
    gone = mem.render(_state(x=45.0), 1.0)
    assert mem.cells == {}
    assert float(gone[0].min()) == 1.0


def test_image_left_matches_the_flight_projection():
    depth = np.full((56, 56), 100.0)
    depth[28, 8] = 5.0
    st = _state()
    ours = depth_to_world(depth, st, max_range=40.0)
    theirs = depth_points(depth, st, max_range=40.0)
    assert len(ours) == 1 and len(theirs) == 1
    assert np.allclose(np.round(ours / 0.2) * 0.2, theirs, atol=1e-6)
    fwd, left, _ = body_from_world(st, ours)
    assert left[0] > 0.0
    assert fwd[0] > 0.0
