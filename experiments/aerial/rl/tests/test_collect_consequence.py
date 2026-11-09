from __future__ import annotations

import math

import numpy as np

from experiments.aerial.rl.collect_consequence_dataset import (
    BACK_LIMIT,
    LIMITS,
    MACROS,
    goal_bearing_body,
    pool_min,
    sample_macro,
)


def test_macros_stay_in_box_and_never_reverse():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(400):
        name, seq = sample_macro(rng, 8, bearing=float(rng.uniform(-math.pi, math.pi)))
        seen.add(name)
        assert seq.shape == (8, 4)
        assert np.all(seq[:, 0] >= 0.0)
        assert np.all(np.abs(seq) <= LIMITS + 1e-9)
    assert seen == set(MACROS)


def test_extra_macros_hover_spin_back():
    rng = np.random.default_rng(0)
    _, hover = sample_macro(rng, 10, 0.3, name="hover")
    assert np.all(hover == 0.0)
    _, spin = sample_macro(rng, 10, 0.3, name="spin")
    assert np.all(spin[:, :3] == 0.0) and np.all(np.abs(spin[:, 3]) > 0.0)
    for _ in range(50):
        _, back = sample_macro(rng, 10, 0.3, name="back")
        assert back[0, 0] < 0.0 and np.all(back[:, 0] >= -BACK_LIMIT - 1e-9)
        assert np.all(back[:, 1:] == 0.0)
        moving = np.flatnonzero(back[:, 0] < 0.0)
        assert moving[0] == 0 and np.all(np.diff(moving) == 1) and len(moving) >= 3


def test_goal_turn_turns_toward_goal_then_goes_forward():
    rng = np.random.default_rng(1)
    for bearing in (0.9, -0.9):
        while True:
            name, seq = sample_macro(rng, 10, bearing=bearing)
            if name == "goal_turn_fwd":
                break
        turned = float(np.sum(seq[:, 3]))
        assert math.copysign(1.0, turned) == math.copysign(1.0, bearing)
        assert abs(turned - bearing) < 0.06
        assert seq[-1, 0] >= 0.4 and abs(seq[-1, 3]) < 1e-9


def test_goal_bearing_is_body_left_positive():
    pos = np.array([0.0, 0.0, 20.0])
    goal = np.array([0.0, 10.0, 20.0])
    assert goal_bearing_body(pos, 0.0, goal) == math.pi / 2
    assert abs(goal_bearing_body(pos, math.pi / 2, goal)) < 1e-9


def test_pool_min_keeps_nearest_pixel():
    d = np.full((224, 224), 50.0, dtype=np.float32)
    d[101, 7] = 2.5
    p = pool_min(d)
    assert p.shape == (56, 56)
    assert float(p[101 // 4, 7 // 4]) == 2.5
    assert float(p.max()) == 50.0
