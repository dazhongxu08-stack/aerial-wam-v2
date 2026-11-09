"""A random yaw_fwd must not be driven when it turns the nose off the goal."""
from __future__ import annotations

import numpy as np

from aerial_finder.nav.consequence_model import (
    heading_yaw_allowed, sample_macro, select_candidate, select_open_candidate,
)


def _yaw_set(left: float, right: float) -> np.ndarray:
    cands = np.zeros((3, 10, 4))
    cands[0, :, 0] = 0.8
    cands[1, 0, 0] = 0.1
    cands[1, 0, 3] = left
    cands[2, 0, 0] = 0.1
    cands[2, 0, 3] = right
    return cands


def test_aligned_goal_rejects_both_yaw_signs():
    cands = _yaw_set(-0.20, 0.20)
    allowed = heading_yaw_allowed(cands, 0.0, 1)
    assert allowed.tolist() == [True, False, False]
    # The logged failure: the left yaw has the best progress and still clears survival.
    surv = np.array([0.98, 0.99, 0.99])
    prog = np.array([1.0, 5.0, 4.0])
    assert select_candidate(surv, prog, allowed, 0.95) == 0


def test_goal_on_the_right_rejects_a_left_yaw():
    cands = _yaw_set(-0.20, 0.20)
    allowed = heading_yaw_allowed(cands, 0.35, 1)
    assert allowed.tolist() == [True, False, True]
    surv = np.array([0.98, 0.99, 0.97])
    prog = np.array([1.0, 5.0, 3.0])
    assert select_candidate(surv, prog, allowed, 0.95) == 2


def test_yaw_that_overshoots_the_goal_is_rejected():
    cands = _yaw_set(-0.20, 0.314)
    allowed = heading_yaw_allowed(cands, 0.08, 1)
    assert allowed.tolist() == [True, False, False]


def test_open_pick_takes_the_clearer_survivor_and_keeps_a_tie_on_progress():
    surv = np.array([0.98, 0.97, 0.99])
    prog = np.array([3.0, 1.0, 2.5])
    allowed = np.array([True, True, True])
    # Straight has the most progress, but the right arc is clearly more open.
    assert select_open_candidate(surv, prog, np.array([2.0, 1.5, 3.2]), allowed, 0.95) == 2
    # Clearances within the margin stay with the higher progress.
    assert select_open_candidate(surv, prog, np.array([2.0, 1.9, 2.05]), allowed, 0.95) == 0


def test_goal_turn_and_forward_stay_allowed():
    rng = np.random.default_rng(0)
    bearing = 0.40
    _, fwd = sample_macro(rng, 10, bearing, "fwd")
    _, turn = sample_macro(rng, 10, bearing, "goal_turn_fwd")
    assert heading_yaw_allowed(np.stack([fwd, turn]), bearing, 1).all()
