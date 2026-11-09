"""Training-set rules for the street-level yaw depth finetune."""
from __future__ import annotations

import math

import pytest
import torch

from experiments.aerial.scripts.finetune_da3_yaw_render import (
    HOLDOUT_ROUTES,
    OPEN_M,
    SKY_M,
    sample_pose_indices,
    side_target_m,
    training_routes,
    _yaw_side_loss,
)


def test_holdout_corners_stay_out_of_training():
    trained = training_routes(range(13))
    for hold in HOLDOUT_ROUTES:
        assert hold not in trained
    assert trained == [0, 1, 3, 4, 5, 7, 8, 9, 11, 12]


def test_samples_cover_the_street_span():
    idx = sample_pose_indices(35, 12)
    assert idx[0] == 0
    assert idx[-1] == 34
    assert len(idx) == 12


def test_sky_median_targets_far_not_the_old_cap():
    assert side_target_m(3300.0) == SKY_M
    assert side_target_m(float("inf")) == SKY_M
    assert side_target_m(30.0) == 30.0
    assert SKY_M > 40.0
    assert OPEN_M < SKY_M


def test_wall_versus_sky_still_ranks():
    gt_l = torch.tensor([3300.0])
    gt_r = torch.tensor([10.0])
    wrong = _yaw_side_loss(torch.tensor([10.0]), torch.tensor([SKY_M]), gt_l, gt_r)
    fixed = _yaw_side_loss(torch.tensor([SKY_M]), torch.tensor([10.0]), gt_l, gt_r)
    overshot = _yaw_side_loss(torch.tensor([4000.0]), torch.tensor([10.0]), gt_l, gt_r)
    wall_blown = _yaw_side_loss(torch.tensor([SKY_M]), torch.tensor([4000.0]), gt_l, gt_r)
    assert float(fixed) == pytest.approx(0.0, abs=1e-5)
    assert float(wrong) > float(fixed)
    assert float(overshot) > float(fixed)
    assert float(wall_blown) > float(fixed)
    assert math.log(SKY_M) > math.log(40.0)


def test_both_far_does_not_rank_and_rejects_a_near_paint():
    gt_l = torch.tensor([3016.0])
    gt_r = torch.tensor([96.0])
    pinned = _yaw_side_loss(torch.tensor([SKY_M]), torch.tensor([SKY_M]), gt_l, gt_r)
    painted = _yaw_side_loss(torch.tensor([8.0]), torch.tensor([SKY_M]), gt_l, gt_r)
    assert float(pinned) == pytest.approx(0.0, abs=1e-5)
    assert float(painted) > float(pinned)

    gt_l = torch.tensor([40.0])
    gt_r = torch.tensor([55.0])
    flipped = _yaw_side_loss(torch.tensor([40.0]), torch.tensor([30.0]), gt_l, gt_r)
    near = _yaw_side_loss(torch.tensor([8.0]), torch.tensor([50.0]), gt_l, gt_r)
    assert float(flipped) == pytest.approx(0.0, abs=1e-5)
    assert float(near) > float(flipped)
