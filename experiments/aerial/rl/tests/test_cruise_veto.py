"""Cruise veto: keep Phase-2 when it is safe, replace circles with a straight step."""

import math

import numpy as np

from experiments.aerial.rl.cruise_veto import (
    CruiseVeto,
    circle_step,
    decide,
    stalled,
    straight_action,
)


LIMITS = np.array([1.0, 0.4, 0.4, 0.3141592653589793])


def test_straight_faces_goal_then_advances():
    ahead = straight_action(0.0, LIMITS)
    assert ahead[0] == LIMITS[0] and ahead[3] == 0.0
    turning = straight_action(math.radians(90), LIMITS)
    assert turning[0] == 0.0 and turning[3] > 0.0


def test_safe_forward_cruise_is_kept_when_progress_is_close():
    cruise = np.array([1.0, 0.0, 0.0, 0.0])
    straight = straight_action(0.0, LIMITS)
    action, tag = decide(
        cruise, straight, 0.99, 1.0, 0.99, 1.1,
        bearing=0.0, limits=LIMITS, stall=False, d_goal=80.0,
        arrive_m=3.0, surv_min=0.95, fov_deg=35.0, progress_margin_m=0.15,
    )
    assert tag == "cruise"
    assert np.allclose(action, cruise)


def test_clearly_better_straight_replaces_cruise():
    cruise = np.array([0.4, 0.0, 0.0, 0.05])
    straight = straight_action(0.0, LIMITS)
    action, tag = decide(
        cruise, straight, 0.99, 0.2, 0.99, 1.0,
        bearing=0.0, limits=LIMITS, stall=False, d_goal=80.0,
        arrive_m=3.0, surv_min=0.95, fov_deg=35.0, progress_margin_m=0.15,
    )
    assert tag == "straight"
    assert action[0] > 0.5


def test_circle_with_worse_progress_is_replaced():
    cruise = np.array([0.0, 0.0, 0.0, 0.3])
    assert circle_step(cruise, 0.0, LIMITS)
    straight = straight_action(0.0, LIMITS)
    action, tag = decide(
        cruise, straight, 0.99, 0.0, 0.99, 1.0,
        bearing=0.0, limits=LIMITS, stall=False, d_goal=80.0,
        arrive_m=3.0, surv_min=0.95, fov_deg=35.0, progress_margin_m=0.15,
    )
    assert tag == "straight"
    assert action[0] > 0.5


def test_blocked_straight_keeps_the_detour():
    cruise = np.array([0.2, 0.3, 0.0, 0.2])
    straight = straight_action(0.1, LIMITS)
    action, tag = decide(
        cruise, straight, 0.99, 0.2, 0.2, 2.0,
        bearing=0.1, limits=LIMITS, stall=False, d_goal=40.0,
        arrive_m=3.0, surv_min=0.95, fov_deg=35.0, progress_margin_m=0.15,
    )
    assert tag == "cruise"
    assert np.allclose(action, cruise)


def test_stall_forces_straight_when_it_survives():
    dists = [20.0] * 9
    yaws = [i * math.radians(12) for i in range(9)]
    assert stalled(dists, yaws, 8, math.pi / 2, 1.0)
    cruise = np.array([0.2, 0.0, 0.0, 0.2])
    straight = straight_action(0.0, LIMITS)
    action, tag = decide(
        cruise, straight, 0.99, 0.4, 0.99, 0.1,
        bearing=0.0, limits=LIMITS, stall=True, d_goal=20.0,
        arrive_m=3.0, surv_min=0.95, fov_deg=35.0, progress_margin_m=0.15,
    )
    assert tag == "stall"
    assert action[0] == LIMITS[0]


def test_inside_arrival_keeps_a_safe_closing_step():
    cruise = np.array([1.0, 0.0, 0.0, 0.0])
    action, tag = decide(
        cruise, cruise, 0.99, 0.4, 0.99, 0.4,
        bearing=0.0, limits=LIMITS, stall=False, d_goal=2.0,
        arrive_m=3.0, surv_min=0.95, fov_deg=35.0, progress_margin_m=0.15,
    )
    assert tag == "cruise"
    assert np.allclose(action, cruise)


def test_inside_arrival_circle_becomes_straight():
    cruise = np.array([0.0, 0.0, 0.0, 0.3])
    straight = straight_action(0.0, LIMITS)
    action, tag = decide(
        cruise, straight, 0.99, 0.0, 0.99, 0.4,
        bearing=0.0, limits=LIMITS, stall=False, d_goal=2.0,
        arrive_m=3.0, surv_min=0.95, fov_deg=35.0, progress_margin_m=0.15,
    )
    assert tag == "straight"
    assert action[0] > 0.5


def test_equal_progress_circle_still_yields():
    cruise = np.array([0.0, 0.0, 0.0, 0.3])
    straight = straight_action(0.0, LIMITS)
    action, tag = decide(
        cruise, straight, 0.99, 0.4, 0.99, 0.4,
        bearing=0.0, limits=LIMITS, stall=False, d_goal=40.0,
        arrive_m=3.0, surv_min=0.95, fov_deg=35.0, progress_margin_m=0.15,
    )
    assert tag == "straight"
    assert action[0] > 0.5


def test_an_open_road_does_not_spin_in_place():
    veto = CruiseVeto(lambda *args: (0.99, 1.0))
    pos = np.array([0.0, 0.0, 20.0])
    goal = np.array([30.0, 0.0, 20.0])
    for i in range(3):
        action, tag = veto.apply(
            np.array([0.0, 0.0, 0.0, 0.3]),
            np.full((56, 56), 40.0, np.float32),
            pos,
            i * math.radians(6),
            goal,
            LIMITS,
        )
        assert tag == "cruise"
        assert abs(float(action[3])) <= 0.08 + 1e-6
        assert action[0] == 0.0


def test_closed_road_does_not_replace_cruise_with_straight():
    cruise = np.array([0.2, 0.0, 0.0, 0.2])
    straight = straight_action(0.0, LIMITS)
    action, tag = decide(
        cruise, straight, 0.99, 0.2, 0.99, 5.0,
        bearing=0.0, limits=LIMITS, stall=True, d_goal=200.0,
        arrive_m=3.0, surv_min=0.95, fov_deg=35.0, progress_margin_m=0.15,
        d_fwd=10.0, d_left=24.0, d_right=5.0,
    )
    assert tag == "cruise"
    assert np.allclose(action, cruise)


def _facade(center: float, left: float, right: float) -> np.ndarray:
    depth = np.full((56, 56), left, np.float32)
    depth[:, 31:] = right
    depth[:, 25:31] = center
    return depth


def test_a_short_road_centre_yaws_and_drops_forward():
    veto = CruiseVeto(lambda *args: (0.2, 0.0))
    action, tag = veto.apply(
        np.array([1.0, 0.2, 0.0, 0.0]),
        _facade(4.0, 20.0, 6.0),
        np.array([0.0, 0.0, 24.0]),
        0.0,
        np.array([200.0, 0.0, 24.0]),
        LIMITS,
    )
    assert tag == "escape"
    assert action[0] == 0.0 and action[3] > 0.0
    flipped, tag = veto.apply(
        np.array([1.0, 0.0, 0.0, -0.2]),
        _facade(4.0, 5.0, 24.0),
        np.array([0.0, 0.0, 24.0]),
        0.2,
        np.array([200.0, 0.0, 24.0]),
        LIMITS,
    )
    assert tag == "escape" and flipped[3] > 0.0


def test_open_forward_cone_still_keeps_cruise():
    cruise = np.array([1.0, 0.0, 0.0, 0.0])
    straight = straight_action(0.0, LIMITS)
    action, tag = decide(
        cruise, straight, 0.99, 1.0, 0.99, 1.05,
        bearing=0.0, limits=LIMITS, stall=False, d_goal=200.0,
        arrive_m=3.0, surv_min=0.95, fov_deg=35.0, progress_margin_m=0.15,
        d_fwd=30.0, d_left=12.0, d_right=12.0,
    )
    assert tag == "cruise"
    assert np.allclose(action, cruise)


def test_yaw_budget_stops_until_the_road_opens():
    veto = CruiseVeto(lambda *args: (0.2, 0.0))
    wall = _facade(4.0, 20.0, 5.0)
    pos = np.array([0.0, 0.0, 24.0])
    goal = np.array([200.0, 0.0, 24.0])
    cruise = np.array([1.0, 0.0, 0.0, 0.3])
    for _ in range(12):
        action, tag = veto.apply(cruise, wall, pos, 0.0, goal, LIMITS)
        assert tag == "escape" and action[0] == 0.0 and action[3] > 0.0
    action, tag = veto.apply(cruise, wall, pos, 0.4, goal, LIMITS)
    assert tag == "hover" and np.allclose(action, 0.0)
    action, tag = veto.apply(
        np.array([0.8, 0.2, 0.0, 0.3]),
        np.full((56, 56), 30.0, np.float32),
        pos,
        0.0,
        goal,
        LIMITS,
    )
    assert tag == "cruise" and action[0] == 0.8
    assert abs(float(action[3])) <= 0.08 + 1e-6
    assert veto.gave_up is False


def test_open_road_keeps_forward_and_clips_yaw():
    veto = CruiseVeto(lambda *args: (0.2, 5.0))
    cruise = np.array([0.8, 0.3, 0.04, -0.3])
    action, tag = veto.apply(
        cruise,
        np.full((56, 56), 30.0, np.float32),
        np.array([0.0, 0.0, 24.0]),
        0.0,
        np.array([200.0, 0.0, 24.0]),
        LIMITS,
        offer_escape=True,
    )
    assert tag == "cruise"
    assert action[0] == 0.8
    assert action[1] == 0.0
    assert action[2] == 0.04
    assert abs(float(action[3])) <= 0.08 + 1e-6


def test_a_low_aircraft_climbs_only_while_the_wall_turn_rewrites_the_step():
    veto = CruiseVeto(lambda *args: (0.2, 0.0))
    action, tag = veto.apply(
        np.array([1.0, 0.0, 0.0, 0.0]),
        _facade(4.0, 8.0, 20.0),
        np.array([0.0, 0.0, 20.0]),
        0.0,
        np.array([200.0, 0.0, 24.0]),
        LIMITS,
    )
    assert tag == "escape" and action[2] > 0.0
    action, tag = veto.apply(
        np.array([0.8, 0.0, 0.0, 0.0]),
        np.full((56, 56), 30.0, np.float32),
        np.array([0.0, 0.0, 20.0]),
        0.0,
        np.array([200.0, 0.0, 24.0]),
        LIMITS,
    )
    assert tag == "cruise" and action[2] == 0.0


def test_blocked_nose_inside_arrival_hovers():
    cruise = np.array([1.0, 0.0, 0.0, 0.0])
    straight = straight_action(0.0, LIMITS)
    action, tag = decide(
        cruise, straight, 0.99, 0.2, 0.99, 0.4,
        bearing=0.0, limits=LIMITS, stall=False, d_goal=2.0,
        arrive_m=3.0, surv_min=0.95, fov_deg=35.0, progress_margin_m=0.15,
        d_fwd=3.0, d_left=20.0, d_right=4.0,
    )
    assert tag == "hover"
    assert np.allclose(action, 0.0)
