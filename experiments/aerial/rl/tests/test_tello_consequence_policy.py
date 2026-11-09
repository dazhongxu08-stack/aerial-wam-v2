"""Indoor Tello closed-loop policy: look at the frame, no canned stick program."""
from __future__ import annotations

import math

import numpy as np

from experiments.aerial.rl.tello_consequence_policy import (
    COMMIT_FWD_STEPS,
    TelloConsequencePolicy,
    clearance_from_rgb,
    contact_from_imu,
    forbid_combined_and_strafe,
    remaining_bearing_rad,
)


def _open_room(h: int = 240, w: int = 320) -> np.ndarray:
    """Textured far scene + flat floor; nothing sitting on the lower FOV."""
    rng = np.random.default_rng(1)
    yy = np.linspace(80, 175, h, dtype=np.float32)[:, None]
    im = np.repeat(yy[:, :, None], 3, axis=2)
    im = im + rng.integers(0, 45, (h, w, 3)).astype(np.float32)
    im[int(h * 0.62) :] = 146
    im[int(h * 0.62) :] += rng.integers(-3, 4, (h - int(h * 0.62), w, 3)).astype(np.float32)
    return np.clip(im, 0, 255).astype(np.uint8)


def test_wall_fill_is_closer_than_textured_scene() -> None:
    wall = np.full((240, 320, 3), 180, np.uint8)
    rng = np.random.default_rng(0)
    clutter = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert float(clearance_from_rgb(wall)[0]) < float(clearance_from_rgb(clutter)[0])


def test_open_aligned_goal_creeps_forward_without_yaw() -> None:
    rgb = _open_room()
    pol = TelloConsequencePolicy()
    dec = pol.decide(
        rgb=rgb,
        height_m=1.1,
        yaw_rad=0.0,
        takeoff_yaw_rad=0.0,
        goal_bearing0_rad=0.0,
    )
    assert dec.rc[0] == 0 and dec.rc[1] > 0 and dec.rc[3] == 0
    assert dec.family == "fwd"


def test_large_bearing_yaws_in_place_no_forward() -> None:
    rgb = _open_room()
    pol = TelloConsequencePolicy()
    dec = pol.decide(
        rgb=rgb,
        height_m=1.1,
        yaw_rad=0.0,
        takeoff_yaw_rad=0.0,
        goal_bearing0_rad=math.radians(90.0),
    )
    assert dec.rc[1] == 0 and dec.rc[3] != 0
    assert dec.family.startswith("yaw")


def test_align_yaws_toward_goal() -> None:
    # Tello: rc d > 0 raises yaw (clockwise); goal at +90 deg is to the right.
    rgb = _open_room()
    kw = dict(rgb=rgb, height_m=1.1, yaw_rad=0.0, takeoff_yaw_rad=0.0)
    right = TelloConsequencePolicy().decide(goal_bearing0_rad=math.radians(90.0), **kw)
    left = TelloConsequencePolicy().decide(goal_bearing0_rad=math.radians(-90.0), **kw)
    assert right.family == "yaw_right" and right.rc[3] > 0
    assert left.family == "yaw_left" and left.rc[3] < 0


def test_near_wall_does_not_creep_forward() -> None:
    wall = np.full((240, 320, 3), 190, np.uint8)
    pol = TelloConsequencePolicy(near_m=3.0)
    dec = pol.decide(
        rgb=wall,
        height_m=1.1,
        yaw_rad=0.0,
        takeoff_yaw_rad=0.0,
        goal_bearing0_rad=0.0,
    )
    assert dec.rc[1] == 0
    assert dec.family in ("hover", "yaw_left", "yaw_right")


def test_near_with_both_sides_tight_yaws_in_place_one_way() -> None:
    pol = TelloConsequencePolicy()
    first = pol._reactive(0.8, 0.85, 0.8, 0.0)
    assert first[0] == "yaw_left" and first[1] == 0
    for left, right in ((0.8, 0.9), (0.9, 0.8), (0.7, 0.75)):
        fam, fwd, _, _ = pol._reactive(0.8, left, right, 0.0)
        assert fam == "yaw_left" and fwd == 0
    pol._reactive(2.0, 2.0, 2.0, 0.0)
    assert pol._reactive(0.8, 0.8, 0.85, 0.0)[0] == "yaw_right"


def test_avoid_turn_is_kept_then_short_leg_before_realign() -> None:
    pol = TelloConsequencePolicy()
    assert pol._reactive(1.0, 0.9, 2.5, math.radians(-120.0))[0] == "yaw_right"
    # Right opens less than left later: still keep turning right while near.
    assert pol._reactive(1.0, 2.0, 1.2, math.radians(-120.0))[0] == "yaw_right"
    legs = [pol._reactive(2.0, 2.0, 2.0, math.radians(-120.0)) for _ in range(COMMIT_FWD_STEPS)]
    assert all(fam == "fwd" and fwd > 0 for fam, fwd, _, _ in legs)
    assert pol._reactive(2.0, 2.0, 2.0, math.radians(-120.0))[0] == "yaw_left"


def test_near_miss_fires_once_on_rising_edge() -> None:
    wall = np.full((240, 320, 3), 190, np.uint8)
    pol = TelloConsequencePolicy(near_m=3.0)
    kwargs = dict(height_m=1.1, yaw_rad=0.0, takeoff_yaw_rad=0.0, goal_bearing0_rad=0.0)
    a = pol.decide(rgb=wall, **kwargs)
    b = pol.decide(rgb=wall, **kwargs)
    assert a.near_miss and not b.near_miss


def test_no_strafe_or_combined_rc() -> None:
    a, b, c, d = forbid_combined_and_strafe(40, 20, 0, 26)
    assert a == 0 and b == 0 and d == 26
    a, b, c, d = forbid_combined_and_strafe(0, -10, 0, 0)
    assert b == 0


def test_grounded_object_in_lower_fov_is_nearer_than_open_floor() -> None:
    open_im = _open_room()
    box = _open_room()
    box[140:, 220:, :] = (160, 120, 70)
    right_open = float(clearance_from_rgb(open_im)[2])
    right_box = float(clearance_from_rgb(box)[2])
    assert right_box < 1.15
    assert right_box < right_open


def test_imu_tap_is_collision_and_stops_forward() -> None:
    assert contact_from_imu(
        agx=-208, agy=-25, agz=-945, pitch_deg=-6, vgx_dms=0, rc_b=16, height_m=1.0
    ) == "collision"
    assert (
        contact_from_imu(
            agx=-28, agy=10, agz=-999, pitch_deg=-1, vgx_dms=0, rc_b=0, height_m=1.0
        )
        == ""
    )
    assert (
        contact_from_imu(
            agx=-43, agy=-15, agz=-1007, pitch_deg=0, vgx_dms=-2, rc_b=16, height_m=1.0
        )
        == ""
    )
    pol = TelloConsequencePolicy()
    dec = pol.decide(
        rgb=_open_room(),
        height_m=1.0,
        yaw_rad=0.0,
        takeoff_yaw_rad=0.0,
        goal_bearing0_rad=0.0,
        agx=-208,
        agy=-25,
        agz=-945,
        pitch_deg=-6,
        vgx_dms=0,
        rc_b=16,
    )
    assert dec.collision and dec.rc[1] == 0


def test_bearing_uses_yaw_only() -> None:
    b = remaining_bearing_rad(yaw_rad=math.radians(30), takeoff_yaw_rad=0.0, goal_bearing0_rad=math.radians(30))
    assert abs(b) < 1e-6
    b = remaining_bearing_rad(yaw_rad=0.0, takeoff_yaw_rad=0.0, goal_bearing0_rad=math.radians(30))
    assert abs(b - math.radians(30)) < 1e-6
