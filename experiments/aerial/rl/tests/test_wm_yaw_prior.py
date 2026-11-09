import math

import numpy as np

from experiments.aerial.rl.wm_yaw_prior import (
    gate_pass,
    sample_route_xy,
    split_yaw,
    toward_open,
    wrap_angle,
)


def test_split_yaw_sums_and_respects_the_body_limit():
    total = math.radians(40.0)
    steps = split_yaw(total)
    assert len(steps) == 3
    assert abs(sum(steps) - total) < 1e-6
    assert max(abs(s) for s in steps) <= 0.3141592653589793 + 1e-9
    back = split_yaw(-total)
    assert all(s < 0 for s in back)
    assert abs(sum(back) + total) < 1e-6


def test_toward_open_follows_the_farther_side():
    # Positive yaw faces image right, so a farther left side takes a negative yaw.
    assert toward_open(30.0, 10.0, -0.4)
    assert not toward_open(30.0, 10.0, 0.4)
    assert toward_open(8.0, 20.0, 0.3)
    assert not toward_open(10.0, 11.0, 0.4)


def test_gate_requires_matching_sides_and_a_longer_street():
    row = {
        "spread_m": 2.0,
        "dest_clear": True,
        "dest_same_side": True,
        "gt_lengthens": True,
        "pred_lengthens": True,
        "open_longer": True,
    }
    assert gate_pass([dict(row) for _ in range(4)])
    missed = [dict(row) for _ in range(4)]
    missed[0]["pred_lengthens"] = False
    assert not gate_pass(missed)
    biased = [dict(row) for _ in range(4)]
    biased[0]["open_longer"] = False
    biased[1]["open_longer"] = False
    assert not gate_pass(biased)
    stuck = [dict(row, spread_m=0.2) for _ in range(4)]
    assert not gate_pass(stuck)
    assert not gate_pass([])


def test_sample_route_keeps_the_ends():
    xy = np.stack([np.linspace(0, 40, 9), np.zeros(9)], axis=1)
    sampled = sample_route_xy(xy, spacing=10.0)
    assert sampled.shape[1] == 2
    assert abs(sampled[0, 0]) < 1e-6
    assert abs(sampled[-1, 0] - 40.0) < 1.0


def test_wrap_angle_stays_in_range():
    assert abs(wrap_angle(math.radians(270)) - math.radians(-90)) < 1e-6


def test_stick_yaw_stays_inside_the_body_limit():
    import torch

    from experiments.aerial.rl.wm_yaw_prior import build_stick

    stick = build_stick("cpu")
    cones = torch.log1p(torch.tensor([[12.0, 30.0, 8.0], [40.0, 9.0, 22.0]]))
    goal = torch.tensor([[20.0, 4.0, 0.0, 30.0], [-5.0, -8.0, 0.0, 12.0]])
    dx, dyaw = stick(cones, goal)
    assert dx.shape == (2,)
    assert float(dx.min()) >= 0.0 and float(dx.max()) <= 1.0
    assert float(dyaw.abs().max()) <= 0.3141592653589793 + 1e-5
