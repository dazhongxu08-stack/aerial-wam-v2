"""Camera graft: arc around a close center, and do not treat the floor as a wall."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("yhs_wire", Path(__file__).with_name("yhs_wire.py"))
wire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wire)


def _depth(left: float, center: float, right: float) -> np.ndarray:
    d = np.full((480, 640), 6.0, np.float32)
    h0, h1 = int(480 * 0.22), int(480 * 0.42)
    c0, c1 = 640 // 3, 2 * 640 // 3
    d[h0:h1, :c0] = left
    d[h0:h1, c0:c1] = center
    d[h0:h1, c1:] = right
    d[int(480 * 0.40):int(480 * 0.58)] = 1.15
    return d


def test_side_arc_is_dropped_when_the_center_is_more_open():
    action = np.array([0.8, 0.0, 0.0, 0.16])
    out, family = wire.hold_straight_unless_side_open(action, "open_right", 4.42, 5.15, 3.42)
    assert family == "fwd" and out[3] == 0.0 and out[0] == 0.8
    kept, family = wire.hold_straight_unless_side_open(action, "open_right", 2.0, 1.2, 3.4)
    assert family == "open_right" and kept[3] == 0.16


def test_box_sight_uses_the_box_center_and_its_depth():
    depth = np.full((480, 640), 6.0, np.float32)
    depth[140:180, 440:480] = 2.4
    sight = wire._box_sight(depth, (440, 140, 480, 180))
    assert sight is not None
    cx, rng = sight
    assert abs(cx - 460 / 640) < 1e-6
    assert abs(rng - 2.4) < 1e-6


def test_open_room_keeps_the_model():
    left, center, right = wire._band_thirds(_depth(4.2, 2.7, 4.8))
    assert center is not None and 2.5 < center < 2.9
    assert wire.depth_graft(left, center, right) is None


def test_cone_in_the_center_arcs_toward_the_open_side():
    assert wire.depth_graft(3.6, 0.9, 1.2)[:2] == (0.10, 0.14)
    assert wire.depth_graft(1.1, 0.9, 3.4)[1] < 0


def test_blocked_center_creeps_toward_the_farther_side():
    assert wire.depth_graft(4.0, 0.4, 1.0) == (0.06, 0.14, "depth_arc")
    assert wire.depth_graft(1.0, 0.9, 1.2) == (0.06, -0.14, "depth_arc")
    assert wire.depth_graft(1.1, 0.9, 1.1) == (0.06, 0.14, "depth_arc")


def _floor():
    """The measured open-floor profile: closer toward the bottom, no jump."""
    depth = np.full((480, 640), 8.0, np.float32)
    samples = (
        (0.36, 5.6), (0.44, 1.64), (0.52, 1.10), (0.60, 0.84),
        (0.68, 0.71), (0.76, 0.59), (0.84, 0.51),
    )
    for top, meters in samples:
        depth[int(480 * top):int(480 * (top + 0.08))] = meters
    return depth


def test_smooth_floor_is_not_a_face():
    assert wire._near_face(_floor()) is None


_FY, _CY, _SRC_H = 386.394, 243.503, 480


def _model_floor() -> np.ndarray:
    pred = np.full((224, 224), 6.0, np.float32)
    edges = wire.LOWER_BANDS
    values = (5.2, 4.3, 3.2, 2.4, 1.8, 1.4, 1.1)
    for (top, bot), value in zip(zip(edges, edges[1:]), values):
        r0 = wire._warp_row(_FY, _CY, _SRC_H, 224, top)
        r1 = wire._warp_row(_FY, _CY, _SRC_H, 224, bot)
        pred[r0:r1] = value
    return pred


def test_open_floor_is_not_a_lower_obstacle():
    assert wire.lower_obstacle(_model_floor(), _FY, _CY, _SRC_H) is None
    assert wire.low_clearance(_model_floor(), _FY, _CY, _SRC_H) == (None, None, None)
    assert wire.approach_cap(wire.nearest_ahead(_model_floor(), _FY, _CY, _SRC_H)) is None


def test_a_bumper_cone_below_the_floor_line_is_kept():
    # The open floor in this band is 1.8 m. A cone at 0.55 m is not that floor.
    pred = _model_floor()
    r0 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.42)
    r1 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.50)
    pred[r0:r1, 224 // 3:2 * 224 // 3] = 0.55
    left, center, right = wire.low_clearance(pred, _FY, _CY, _SRC_H)
    assert left is None and right is None
    assert center is not None and center < 0.7
    assert wire.nearest_ahead(pred, _FY, _CY, _SRC_H) < 0.7


def test_a_goal_beside_the_nose_turns_instead_of_driving_past():
    right = wire.tight_face(80)
    assert right[0] == 0.10 and right[1] < 0 and right[2] == "face"
    left = wire.tight_face(-40)
    assert left[0] == 0.10 and left[1] > 0
    assert wire.tight_face(10) is None
    assert wire.tight_face(None) is None


def test_a_near_obstacle_is_not_full_speed():
    assert wire.approach_cap(None) is None
    assert wire.approach_cap(5.0) is None
    assert wire.approach_cap(3.2) == 0.24
    assert wire.approach_cap(2.0) == 0.18
    assert wire.approach_cap(1.2) == 0.10
    assert wire.nearest_ahead(_model_floor(), _FY, _CY, _SRC_H) is None
    pred = _model_floor()
    r0 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.38)
    r1 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.42)
    pred[r0:r1] = 1.1
    assert wire.approach_cap(wire.nearest_ahead(pred, _FY, _CY, _SRC_H)) == 0.10
    # After an avoidance, the next metre and a half stays at the cruise cap.
    assert wire.forward_cap(None, 2.0, 1.0) == 0.24
    assert wire.forward_cap(None, 3.0, 1.0) is None


def test_a_front_face_is_kept_when_the_floor_gate_misses_it():
    assert wire.face_low(None, (None, None, None)) == (None, None, None)
    assert wire.face_low((1.0, 3.5, 3.4), (None, None, None)) == (None, None, None)
    assert wire.face_low((2.4, 1.20, 2.5), (None, None, None)) == (None, 1.20, None)
    assert wire.face_low((2.4, 1.20, 2.5), (None, 0.90, None)) == (None, 0.90, None)


def test_bin_closer_than_the_floor_arcs():
    pred = _model_floor()
    r0 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.50)
    r1 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.58)
    pred[r0:r1, 224 // 3:2 * 224 // 3] = 0.40
    block = wire.lower_obstacle(pred, _FY, _CY, _SRC_H)
    assert block is not None and block[1] < 0.5
    graft = wire.depth_graft(*block)
    assert graft is not None and graft[0] > 0 and graft[2] == "depth_arc"


def test_a_fast_command_that_does_not_move_the_car_stops():
    # The timer starts before the first roll. A fast command latches at 0.6 s.
    moved, since, held = wire.update_contact(False, None, 0.0, 0.48, 10.0, False, False)
    assert held is False and since == 10.0 and moved is False
    _moved, _since, held = wire.update_contact(False, 10.0, 0.0, 0.48, 10.7, False, False)
    assert held is True
    _moved, _since, held = wire.update_contact(True, 10.0, 0.0, 0.48, 10.7, False, False)
    assert held is True
    # A slow command that never rolls latches at 1.5 s, not before.
    moved, since, held = wire.update_contact(False, None, 0.0, 0.105, 20.0, False, False)
    assert held is False and since == 20.0
    _moved, since, held = wire.update_contact(False, 20.0, 0.0, 0.105, 21.4, False, False)
    assert held is False
    _moved, since, held = wire.update_contact(False, 20.0, 0.0, 0.105, 21.5, False, False)
    assert held is True
    # A slow arc may pause briefly. A creep that stays stuck is contact.
    _moved, since, held = wire.update_contact(True, None, 0.0, 0.08, 10.0, False, False)
    assert held is False and since == 10.0
    _moved, since, held = wire.update_contact(True, 10.0, 0.0, 0.08, 10.7, False, False)
    assert held is False
    _moved, since, held = wire.update_contact(True, 10.0, 0.0, 0.08, 11.6, False, False)
    assert held is True
    # A real creep moves about 1.4 cm per step and must keep going.
    moved, since, held = wire.update_contact(True, 10.0, 0.014, 0.08, 11.6, False, False)
    assert moved is True and since is None and held is False
    # Before the first roll, a held stick is not a collision.
    moved, since, held = wire.update_contact(False, None, 0.0, 0.48, 10.0, True, False)
    assert held is False and since is None
    # After the car has rolled, the stick must not keep a fast command pushing.
    _moved, _since, held = wire.update_contact(True, 10.0, 0.0, 0.48, 10.7, True, False)
    assert held is True
    # Rolling clears the timer.
    moved, since, held = wire.update_contact(True, 10.0, 0.05, 0.48, 10.2, False, False)
    assert moved is True and since is None and held is False


def test_a_wide_storage_box_is_not_a_trash_can():
    assert wire._upright_can((328, 148, 371, 165)) is False
    assert wire._upright_can((200, 80, 260, 200)) is True
    assert wire._full_can((0, 163, 47, 211), 640, 480) is False
    assert wire._full_can((577, 151, 631, 201), 640, 480) is False
    assert wire._full_can((300, 150, 355, 200), 640, 480) is False
    assert wire._full_can((200, 80, 260, 200), 640, 480) is True


def test_display_box_does_not_mark_a_colour_blob():
    crate = (0.50, 0.30, 0.58, 0.36)
    assert wire._display_box(crate, False) is None
    assert wire._display_box((0.63, 0.29, 0.65, 0.32), True) is None
    kept = wire._display_box((0.40, 0.30, 0.55, 0.50), True)
    assert kept == (0.40, 0.30, 0.55, 0.50)


def test_target_box_is_a_fraction_of_the_image():
    box = wire._norm_box((160, 120, 480, 360), 640, 480)
    assert box == (0.25, 0.25, 0.75, 0.75)


def test_aisle_nearer_in_the_middle_keeps_the_model():
    pred = _model_floor()
    width = pred.shape[1]
    for top, bot in zip(wire.LOWER_BANDS, wire.LOWER_BANDS[1:]):
        r0 = wire._warp_row(_FY, _CY, _SRC_H, 224, top)
        r1 = wire._warp_row(_FY, _CY, _SRC_H, 224, bot)
        pred[r0:r1, : width // 3] = 3.05
        pred[r0:r1, int(width * 0.42):int(width * 0.58)] = 2.73
        pred[r0:r1, 2 * width // 3:] = 3.61
    assert wire.lower_obstacle(pred, _FY, _CY, _SRC_H) is None
    assert wire.obstacle_command(3.05, 2.73, 3.61) is None


def test_a_low_box_on_one_side_turns_before_three_metres():
    pred = _model_floor()
    r0 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.30)
    r1 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.34)
    pred[r0:r1, 2 * 224 // 3:] = 2.7
    block = wire.lower_obstacle(pred, _FY, _CY, _SRC_H)
    assert block is not None and block[2] < 3.0
    vx, wz, tag = wire.obstacle_command(*block)
    assert vx <= 0.10 and wz > 0 and tag == "depth_clear"
    assert wire.obstacle_command(4.94, 5.44, 2.77)[:2] == (0.08, 0.08)
    # A gap: the center is the near part and one side is open. Drive through.
    assert wire.obstacle_command(2.46, 1.73, 4.02) is None


def _paint(pred, top, bot, x0, x1, meters):
    r0 = wire._warp_row(_FY, _CY, _SRC_H, pred.shape[0], top)
    r1 = wire._warp_row(_FY, _CY, _SRC_H, pred.shape[0], bot)
    pred[r0:r1, x0:x1] = meters


def test_a_wide_gap_beside_a_cone_is_taken():
    """Width is taken at the cone. A far wall behind a narrow slot is not a gap."""
    pred = _model_floor()
    # 72 columns at 1.6 m is about 1.0 m. The cone is narrow. The box fills the rest.
    _paint(pred, 0.30, 0.42, 0, 72, 4.5)
    _paint(pred, 0.30, 0.42, 72, 94, 1.6)
    _paint(pred, 0.30, 0.42, 94, 224, 2.6)
    cmd = wire.gap_command(pred, _FY, _CY, _SRC_H)
    assert cmd is not None and cmd != "through"
    assert cmd[0] == wire.FACE_VX and cmd[1] > 0
    tight = _model_floor()
    _paint(tight, 0.30, 0.42, 0, 40, 4.5)
    _paint(tight, 0.30, 0.42, 40, 62, 1.4)
    _paint(tight, 0.30, 0.42, 62, 224, 2.6)
    assert wire.gap_command(tight, _FY, _CY, _SRC_H) is None
    assert wire.gap_command(_model_floor(), _FY, _CY, _SRC_H) is None


def test_the_far_side_of_the_box_is_not_the_passage():
    """The empty floor past the box is wider on the image and used to win."""
    pred = _model_floor()
    _paint(pred, 0.30, 0.42, 0, 72, 4.5)
    _paint(pred, 0.30, 0.42, 72, 94, 1.6)
    _paint(pred, 0.30, 0.42, 94, 150, 2.6)
    _paint(pred, 0.30, 0.42, 150, 224, 6.0)
    cmd = wire.gap_command(pred, _FY, _CY, _SRC_H)
    assert cmd is not None and cmd != "through"
    assert cmd[1] > 0
    # Cone with no box, or a box across the whole view: leave the old command.
    cone = _model_floor()
    _paint(cone, 0.30, 0.42, 20, 36, 1.5)
    assert wire.gap_command(cone, _FY, _CY, _SRC_H) is None
    box = _model_floor()
    _paint(box, 0.30, 0.42, 0, 224, 2.8)
    assert wire.gap_command(box, _FY, _CY, _SRC_H) is None


def test_a_gap_already_in_front_is_driven_through():
    pred = _model_floor()
    _paint(pred, 0.30, 0.42, 40, 62, 2.0)
    _paint(pred, 0.30, 0.42, 62, 170, 4.5)
    _paint(pred, 0.30, 0.42, 170, 224, 2.5)
    assert wire.gap_command(pred, _FY, _CY, _SRC_H) == "through"


def test_a_storage_box_at_three_metres_is_turned_around():
    pred = _model_floor()
    for top, bot in ((0.30, 0.34), (0.34, 0.38)):
        r0 = wire._warp_row(_FY, _CY, _SRC_H, 224, top)
        r1 = wire._warp_row(_FY, _CY, _SRC_H, 224, bot)
        pred[r0:r1] = 3.1
    block = wire.lower_obstacle(pred, _FY, _CY, _SRC_H)
    assert block is not None and block[1] < 3.4
    vx, wz, tag = wire.obstacle_command(*block)
    assert vx <= 0.15 and vx > 0 and wz != 0 and tag == "depth_arc"
    # One open side of the same face is still an arc, not a cruise step.
    cmd = wire.obstacle_command(5.2, 3.1, 3.0)
    assert cmd is not None and cmd[0] == 0.08 and cmd[1] > 0 and cmd[2] == "depth_arc"
    spin, job = wire.face_turn(cmd, 0.0, None, 0.0)
    assert spin[0] == 0.0 and spin[1] > 0 and spin[2] == "depth_turn"
    # Depth drops before the nose has left the face: keep spinning.
    held, job = wire.face_turn(None, 0.3, job, 2.0)
    assert held[0] == 0.0 and held[2] == "depth_turn"
    still, job = wire.face_turn(cmd, 1.3, job, 3.0)
    assert still[0] == 0.0 and still[2] == "depth_turn"
    # Nose has turned clear and the face has stayed gone.
    assert wire.face_turn(None, 2.2, job, 4.0) == (None, None)


def test_flat_close_face_in_the_model_is_an_obstacle():
    pred = _model_floor()
    edges = wire.LOWER_BANDS
    # The box replaces two bands under a farther row and stays at one range.
    for top, bot in ((0.30, 0.34), (0.34, 0.38)):
        r0 = wire._warp_row(_FY, _CY, _SRC_H, 224, top)
        r1 = wire._warp_row(_FY, _CY, _SRC_H, 224, bot)
        pred[r0:r1] = 1.9
    block = wire.lower_obstacle(pred, _FY, _CY, _SRC_H)
    assert block is not None
    vx, wz, tag = wire.obstacle_command(*block)
    assert vx > 0 and wz != 0 and tag == "depth_arc"


def test_cone_on_one_side_keeps_the_turn_away():
    pred = _model_floor()
    r0 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.42)
    r1 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.50)
    pred[r0:r1, :224 // 3] = 1.05
    block = wire.lower_obstacle(pred, _FY, _CY, _SRC_H)
    assert block is not None and block[0] < 1.2
    graft = wire.depth_graft(*block)
    assert graft is not None and graft[0] > 0 and graft[1] < 0
    pred = _model_floor()
    r0 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.42)
    r1 = wire._warp_row(_FY, _CY, _SRC_H, 224, 0.50)
    pred[r0:r1, int(224 * 0.47):int(224 * 0.53)] = 1.05
    block = wire.lower_obstacle(pred, _FY, _CY, _SRC_H)
    assert block is not None and block[1] < 1.2
    graft = wire.depth_graft(*block)
    assert graft is not None and graft[0] > 0 and graft[1] != 0.0
    pred = np.full((224, 224), 4.0, np.float32)
    out, row = wire.hide_ground(pred, 386.394, 243.503, 480)
    assert 90 < row < 110
    assert out[row - 2, 10] == 4.0
    assert out[row + 2, 10] == wire.INVALID_M
    assert out[-1, -1] == wire.INVALID_M


def test_open_floor_slope_is_not_a_face():
    assert wire._low_face(_floor()) is None


def test_flat_face_in_the_lower_center_is_a_face():
    depth = _floor()
    depth[int(480 * 0.42):int(480 * 0.66), 640 // 3:2 * 640 // 3] = 0.85
    face = wire._low_face(depth)
    assert face is not None and face[1] < 1.0
    graft = wire.depth_graft(*face)
    assert graft is not None and graft[0] > 0 and graft[2] == "depth_arc"


def test_low_face_below_the_upper_band_is_an_arc():
    depth = _floor()
    depth[int(480 * 0.52):int(480 * 0.90)] = 0.40
    face = wire._near_face(depth)
    assert face is not None and face[1] < 0.5
    graft = wire.depth_graft(*face)
    assert graft is not None and graft[0] > 0 and graft[2] == "depth_arc"


def test_floor_band_is_not_the_one_we_read():
    depth = _depth(4.0, 5.0, 4.0)
    _left, center, _right = wire._band_thirds(depth, top=0.40, bot=0.58)
    assert center is not None and center < 1.3
    left, center, right = wire._band_thirds(depth)
    assert wire.depth_graft(left, center, right) is None
