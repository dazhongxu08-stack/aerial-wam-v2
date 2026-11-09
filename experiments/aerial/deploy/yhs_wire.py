#!/usr/bin/env python3
"""Wire the consequence stack to the YHS chassis on ms-thor.

Camera and odom are read on the robot. A model step is a body speed and a yaw
rate. The chassis takes that as speed control on /nav/cmd_vel: gear 6 (4T4D),
x linear in m/s, z angular in deg/s. Nothing is published unless AERIAL_ARM=1.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_DEPLOY = Path(__file__).resolve().parent
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))
from steer_center import SPIN_POSE_DEG, hold_for_center, next_center_state, wheel_angles

os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("AERIAL_DA3_FP16", "1")

COLOR_TOPIC = "/camera1/camera1/color/image_raw"
COLOR_INFO = "/camera1/camera1/color/camera_info"
DEPTH_TOPIC = "/camera1/camera1/depth/image_rect_raw"
DEPTH_INFO = "/camera1/camera1/depth/camera_info"
ODOM_TOPIC = "/odom"
CMD_TOPIC = "/nav/cmd_vel"
# Vendor ctrl_cmd: gear 6 is 4T4D. motion_api turns this Twist into that frame
# (angular.z rad/s, CCW, becomes deg/s on the bus). Publish at 50 Hz.
SPEED_GEAR = 6
WEIGHTS = Path(os.environ.get("RT_RAM", "/dev/shm/rtw")) / "weights"
OFFICE_DEPTH = Path(os.environ.get(
    "OFFICE_DEPTH", "/home/user/rt/office_wam/depth_office_cone_yaw.pt"))
OFFICE_CONS = Path(os.environ.get(
    "OFFICE_CONS", "/home/user/rt/office_wam/consequence_best.pt"))
STATUS = Path(os.environ.get("YHS_STATUS", "/home/user/rt/wire_status.json"))
SEARCH_GOAL = Path(os.environ.get("YHS_SEARCH_GOAL", "/home/user/rt/search_goal.json"))
SEARCH_RESULT = Path("/home/user/rt/search_result.json")
TRACK_DIR = Path(os.environ.get("YHS_TRACK_DIR", "/home/user/rt/tracks"))
# Rows of the 56-map used to fit the RGB depth against this camera's depth.
# The floor at the feet sits around rows 29:39, so the fit stays above that.
SCALE_ROWS = (8, 29)
# RealSense rows that hold a low obstacle. Rows 40–58% are the floor at about 1.1 m.
GRAFT_TOP = 0.22
GRAFT_BOT = 0.42
# The tilted camera puts the floor in the lower half. Rows above this fraction
# are the forward view the aircraft survival head was trained on.
GROUND_FRAC = 0.42
# Bands from just above the floor line down through the ground. A storage
# box sits on the floor and shows up here before the bumper reaches it.
LOWER_BANDS = (0.26, 0.30, 0.34, 0.38, 0.42, 0.50, 0.58, 0.66)
# Dog find: a colour patch is a hint and cannot finish the trip. A class detection
# confirms only after several close views, and a stated distance is driven first.
REACH_M = 1.5
MIN_VERIFIED = 3
# Dog search at the point: one in-place turn, measured from odometry, until a full circle.
SCAN_DYAW = 0.06


def face_low(block, low):
    """Keep a front box the floor gate missed, so the cruise cannot push it.

    A side-only return leaves the center alone. The floor is not a face and
    does not arrive here.
    """
    if low is None:
        low = (None, None, None)
    if block is None or block[1] is None or float(block[1]) >= 1.6:
        return low
    sides = [float(v) for v in (block[0], block[2]) if v is not None]
    if sides and float(block[1]) > min(sides) + 0.30:
        return low
    center = float(block[1]) if low[1] is None else min(float(low[1]), float(block[1]))
    return (low[0], center, low[2])


def arrival_scan(mode: str, goal_dist, finished: bool, blocked, holding: bool) -> bool:
    """Point mode inside the arrival radius turns in place and does not drive.

    A low obstacle used to skip this and leave the gap drive in command. The
    car then went through the point, and after the nose turned it drove away.
    ``blocked`` does not defer the hold.
    """
    del blocked
    if mode != "point" or finished or holding:
        return False
    if goal_dist is None:
        return False
    return float(goal_dist) <= REACH_M
INVALID_M = 30.0
STEP_HZ = 5.0
VX_MAX = 0.3


def _rgb(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding.lower()
    if enc not in ("rgb8", "bgr8"):
        raise RuntimeError(f"unsupported color encoding {msg.encoding}")
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = np.ndarray((h, w, 3), dtype=np.uint8, buffer=buf, strides=(int(msg.step), 3, 1))
    if enc == "bgr8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def _depth_m(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    if msg.encoding.lower() not in ("16uc1", "mono16"):
        raise RuntimeError(f"unsupported depth encoding {msg.encoding}")
    buf = np.frombuffer(msg.data, dtype=np.uint16)
    raw = np.ndarray((h, w), dtype=np.uint16, buffer=buf, strides=(int(msg.step), 2))
    depth = raw.astype(np.float32) * 0.001
    depth[raw == 0] = np.nan
    return depth


def _band_thirds(depth: np.ndarray, top: float = GRAFT_TOP, bot: float = GRAFT_BOT):
    """Left, center, right of the low-obstacle band. Percentile 15, meters."""
    h, w = depth.shape
    sl = depth[int(h * top):int(h * bot)]
    if sl.size == 0:
        return None, None, None
    c0, c1 = w // 3, 2 * w // 3

    def p15(part: np.ndarray):
        fin = part[np.isfinite(part) & (part > 0.05)]
        if fin.size < 20:
            return None
        return float(np.percentile(fin, 15))

    return p15(sl[:, :c0]), p15(sl[:, c0:c1]), p15(sl[:, c1:])


def _avoid_wz(left, right, near_m: float) -> float:
    """Yaw toward the farther side, slowly enough to come back onto a gap.

    Positive is left. Ties search left. 0.32 rad/s turned past openings.
    """
    left_m = -1.0 if left is None else left
    right_m = -1.0 if right is None else right
    sign = 1.0 if left_m >= right_m else -1.0
    rate = 0.14 if float(near_m) < 1.2 else 0.08
    return sign * rate


def hold_straight_unless_side_open(action, family, left, center, right, margin: float = 0.45):
    """Keep a side arc only when that side is farther than the center.

    The predicted cone can prefer a turn while the camera still shows the
    center as the open lane. That turn is the large yaw at the start.
    """
    if family not in ("open_left", "open_right") or center is None:
        return action, family
    side = left if family == "open_left" else right
    if side is not None and side > center + margin:
        return action, family
    straight = np.array(action, np.float64).copy()
    straight[3] = 0.0
    return straight, "fwd"


def _near_face(depth: np.ndarray):
    """Left, center, right of a low object the 22–42% band looks over.

    Toward the bottom of the image the floor gets closer smoothly. A band
    that is much closer than the one above it is a face, not that floor.
    """
    prev_c = None
    for top in (0.36, 0.44, 0.52, 0.60, 0.68, 0.76, 0.84):
        left, center, right = _band_thirds(depth, top, top + 0.08)
        if (
            prev_c is not None
            and center is not None
            and center < 0.85
            and center < prev_c * 0.55
        ):
            return left, center, right
        if center is not None:
            prev_c = center
    return None


def _low_face(depth: np.ndarray):
    """Thirds of a vertical face that the floor-slope test can see.

    Open floor gets closer by several tenths of a meter over each of these
    windows. A bin face stays at one range. The very bottom is always the
    near floor, so it is not part of the test.
    """
    height, width = depth.shape
    thirds = (0, width // 3, 2 * width // 3, width)
    windows = ((0.36, 0.48), (0.42, 0.54), (0.48, 0.60))

    def near(top: float, bot: float, x0: int, x1: int):
        sl = depth[int(height * top):int(height * bot), x0:x1]
        fin = sl[np.isfinite(sl) & (sl > 0.15) & (sl < 12.0)]
        if fin.size < 30:
            return None
        return float(np.percentile(fin, 15))

    found = []
    for x0, x1 in zip(thirds, thirds[1:]):
        hit = None
        for top, bot in windows:
            ahead = near(top, top + 0.06, x0, x1)
            below = near(bot, bot + 0.06, x0, x1)
            if ahead is None or below is None:
                continue
            if ahead < 1.8 and below - ahead > -0.18:
                hit = ahead
                break
        found.append(hit)
    if found[1] is None:
        return None
    return tuple(6.0 if value is None else value for value in found)


def sensor_side_face(depth: np.ndarray):
    """A low box on one side of the RealSense image, not the floor.

    Open floor gets closer toward the bottom of the image. A storage box
    stays at one range and is closer than the floor in the middle. The
    nearest pixels at the bottom of the frame are the floor, so they are
    not used. Returns the left face, the right face, and whether each side
    positively follows the floor.
    """
    height, width = depth.shape
    if height < 40 or width < 40:
        return None, None, False, False
    mid0, mid1 = width // 3, 2 * width // 3
    windows = ((0.36, 0.48), (0.42, 0.54), (0.48, 0.60))

    def near(top: float, bot: float, x0: int, x1: int):
        sl = depth[int(height * top):int(height * bot), x0:x1]
        fin = sl[np.isfinite(sl) & (sl > 0.15) & (sl < 8.0)]
        if fin.size < 30:
            return None
        return float(np.percentile(fin, 15))

    def classify(x0: int, x1: int):
        face = None
        floor = False
        for top, bot in windows:
            ahead = near(top, top + 0.06, x0, x1)
            below = near(bot, bot + 0.06, x0, x1)
            center = near(top, top + 0.06, mid0, mid1)
            if ahead is None or below is None:
                continue
            if ahead < 1.8 and below - ahead > -0.18 and center is not None and center > ahead + 0.35:
                return ahead, False
            if below < ahead - 0.25:
                floor = True
        return face, floor

    left_face, left_floor = classify(0, mid0)
    right_face, right_floor = classify(mid1, width)
    return left_face, right_face, left_floor, right_floor


def _region_follows_floor(depth: np.ndarray, x0: int, x1: int) -> bool:
    """True when this image band gets closer toward the bottom, like open floor."""
    height = depth.shape[0]
    windows = ((0.36, 0.48), (0.42, 0.54), (0.48, 0.60))

    def near(a, b):
        sl = depth[int(height * a):int(height * b), x0:x1]
        fin = sl[np.isfinite(sl) & (sl > 0.15) & (sl < 8.0)]
        if fin.size < 30:
            return None
        return float(np.percentile(fin, 15))

    for top, bot in windows:
        ahead = near(top, top + 0.06)
        below = near(bot, bot + 0.06)
        if ahead is not None and below is not None and below < ahead - 0.25:
            return True
    return False


def passage_view(cones, low, depth: np.ndarray):
    """Use the camera to tell the floor from a low box.

    Open floor is not an obstacle, even when the model reads it at about a
    metre. A vertical face is the box, and that distance stays.
    """
    if depth is None or low is None:
        return cones, low
    left_face, right_face, left_floor, right_floor = sensor_side_face(depth)
    out_c = [float(v) for v in np.atleast_1d(np.asarray(cones, np.float64)).reshape(-1)[:3]]
    out_l = list(low)
    if left_face is not None:
        out_l[0] = left_face if out_l[0] is None else min(float(out_l[0]), left_face)
    elif left_floor:
        if out_l[0] is not None and float(out_l[0]) < 1.0:
            out_l[0] = None
        if out_c[1] < 1.6:
            out_c[1] = 2.5
    if right_face is not None:
        out_l[2] = right_face if out_l[2] is None else min(float(out_l[2]), right_face)
    elif right_floor:
        if out_l[2] is not None and float(out_l[2]) < 1.0:
            out_l[2] = None
        if out_c[2] < 1.6:
            out_c[2] = 2.5
    width = depth.shape[1]
    center_low = out_l[1]
    if (
        _region_follows_floor(depth, width // 3, 2 * width // 3)
        and (center_low is None or float(center_low) >= 1.0)
        and out_c[0] < 1.6
    ):
        out_c[0] = 2.5
    return out_c, tuple(out_l)


def _warp_row(fy: float, cy: float, src_h: int, out_h: int, frac: float) -> int:
    """Row in the 90° model image that matches a fraction of the source image."""
    focal = 0.5 * out_h / math.tan(math.radians(45.0))
    angle = math.atan((frac * src_h - cy) / fy)
    row = out_h / 2.0 + focal * math.tan(angle)
    return int(np.clip(math.floor(row), 0, out_h - 1))


def hide_ground(pred: np.ndarray, fy: float, cy: float, src_h: int, frac: float = GROUND_FRAC):
    """Drop the floor from the depth the survival head scores.

    The aircraft camera does not look down at the ground under the vehicle.
    Pixels below the floor line are set far, so that plane is not a wall.
    """
    out = np.array(pred, np.float32, copy=True)
    row = _warp_row(fy, cy, src_h, out.shape[0], frac)
    out[row:, :] = INVALID_M
    return out, row


def _model_thirds(depth: np.ndarray, top: int, bot: int):
    """Left, center, right of one band in the model depth image.

    The center is the middle sixth, not a full third. A cone is narrower
    than a third, and the floor beside it was keeping the old reading far.
    """
    height, width = depth.shape
    r0 = max(0, min(height - 1, int(top)))
    r1 = max(r0 + 1, min(height, int(bot)))
    c0, c1 = width // 3, 2 * width // 3
    n0, n1 = int(width * 0.42), int(width * 0.58)
    band = depth[r0:r1]

    def near(x0: int, x1: int):
        part = band[:, x0:x1]
        fin = part[np.isfinite(part) & (part > 0.15) & (part < INVALID_M)]
        if fin.size < 20:
            return None
        return float(np.percentile(fin, 15))

    return near(0, c0), near(n0, n1), near(c1, width)


def _lower_bands(model: np.ndarray, fy: float, cy: float, src_h: int):
    height = model.shape[0]
    rows = [_warp_row(fy, cy, src_h, height, frac) for frac in LOWER_BANDS]
    return [_model_thirds(model, r0, r1) for r0, r1 in zip(rows, rows[1:])]


def lower_obstacle(model: np.ndarray, fy: float, cy: float, src_h: int):
    """Left, center, right of a low object in the model depth.

    None leaves the survival head's command alone. Open ground matches on
    the left and right and keeps getting closer toward the bottom. A box
    on the floor is a flat patch with the room still far above it.
    """
    height = model.shape[0]
    rows = [_warp_row(fy, cy, src_h, height, frac) for frac in LOWER_BANDS]
    bands = []
    for top, r0, r1 in zip(LOWER_BANDS, rows, rows[1:]):
        left, center, right = _model_thirds(model, r0, r1)
        bands.append((top, left, center, right))
    for _top, left, center, right in bands:
        if center is None or left is None or right is None:
            continue
        ground = min(left, right)
        if center < ground - 0.30 and center < ground * 0.80:
            return left, center, right
        close = min(left, right)
        if close < 1.6 and close < center - 0.30 and close < center * 0.80:
            return left, center, right
    # A low box on one side: that side is near, the row above still sees the room.
    # A wall is near in the row above as well, so it does not match.
    for i in range(1, len(bands)):
        top, left, center, right = bands[i]
        if top >= 0.42 or center is None:
            continue
        _pt, prev_l, _pc, prev_r = bands[i - 1]
        for side, above in ((left, prev_l), (right, prev_r)):
            if side is None or above is None:
                continue
            if side < 3.5 and side < center - 0.30 and side < center * 0.80 and above > side + 0.8:
                return left, center, right
    for i in range(1, len(bands) - 1):
        top, left, center, right = bands[i]
        if top >= 0.42 or center is None or left is None or right is None:
            continue
        prev_c, next_c = bands[i - 1][2], bands[i + 1][2]
        if prev_c is None or next_c is None:
            continue
        # A storage box is a flat face. Open floor still drops about a metre
        # over these rows, so two bands at one range are the box, out to 4 m.
        if abs(center - next_c) <= 0.35 and prev_c > center + 0.8 and center < 4.0:
            return left, center, right
    return None


CONTACT_VX = 0.20
CONTACT_MOVE_M = 0.008
CONTACT_S = 0.6
CONTACT_SLOW_S = 1.5
CONTACT_CREEP = 0.05
CRUISE_AFTER_M = 0.15
SLOW_AFTER_M = 1.5
SLOW_VX = 0.24
# Open-floor range from just above the floor line down through the ground.
# A lower reading is an object. The bottom bands are the floor getting
# closer, not a wall; only a reading well inside that range is kept.
_FLOOR_M = (5.2, 4.3, 3.2, 2.4, 1.8, 1.4, 1.1)


def low_clearance(model: np.ndarray, fy: float, cy: float, src_h: int):
    """Left, center, right of a low object. None stays floor, not a wall."""
    height = model.shape[0]
    rows = [_warp_row(fy, cy, src_h, height, frac) for frac in LOWER_BANDS]
    found = [None, None, None]
    band_i = 0
    for _top, r0, r1 in zip(LOWER_BANDS, rows, rows[1:]):
        expect = _FLOOR_M[band_i]
        band_i += 1
        for index, value in enumerate(_model_thirds(model, r0, r1)):
            if value is None:
                continue
            if value < expect - 0.8 and value < expect * 0.75:
                found[index] = value if found[index] is None else min(found[index], value)
    return tuple(found)


def nearest_ahead(model: np.ndarray, fy: float, cy: float, src_h: int):
    """Closest object above the floor line. A smooth floor matches the open room."""
    height = model.shape[0]
    rows = [_warp_row(fy, cy, src_h, height, frac) for frac in LOWER_BANDS]
    near = None
    band_i = 0
    for _top, r0, r1 in zip(LOWER_BANDS, rows, rows[1:]):
        expect = _FLOOR_M[band_i]
        band_i += 1
        for value in _model_thirds(model, r0, r1):
            if value is None:
                continue
            if value < expect - 0.8 and value < expect * 0.75:
                near = value if near is None else min(near, value)
    return near


def approach_cap(near_m):
    """Forward cap when something is close. None keeps the cruise speed."""
    if near_m is None:
        return None
    if near_m < 1.5:
        return 0.10
    if near_m < 2.5:
        return 0.18
    if near_m < 4.0:
        return 0.24
    return None


def tight_face(bearing_deg):
    """Creep and yaw when the goal is off the nose. A shallow arc drives past it.

    Positive bearing means the goal is to the right. Positive wz is left.
    """
    if bearing_deg is None:
        return None
    err = abs(float(bearing_deg))
    if err < 25.0:
        return None
    wz = -0.40 if float(bearing_deg) > 0.0 else 0.40
    if err < 50.0:
        return 0.10, wz * 0.65, "face"
    return 0.10, wz, "face"


def forward_cap(ahead_m, traveled, slow_from):
    """Cruise cap, including the slow stretch after an avoidance."""
    cap = approach_cap(ahead_m)
    if slow_from is not None and traveled - slow_from < SLOW_AFTER_M:
        cap = SLOW_VX if cap is None else min(cap, SLOW_VX)
    if traveled < CRUISE_AFTER_M:
        cap = 0.15 if cap is None else min(cap, 0.15)
    return cap


def update_contact(moved, since, progress_m, cmd_vx, now, joypad, held):
    """Stop a command that is not moving the chassis.

    A held joystick before the first roll is the operator, not a collision.
    An in-place spin has no forward speed and does not count. A forward
    command that never produces the first few millimetres still starts the
    timer, so the wheels cannot push all night. After a real stop the latch
    stays until the task changes.
    """
    if held:
        return moved, since, True
    if progress_m >= CONTACT_MOVE_M:
        return True, None, False
    if cmd_vx < CONTACT_CREEP or (joypad and not moved):
        return moved, None, False
    if since is None:
        return moved, now, False
    limit = CONTACT_S if cmd_vx >= CONTACT_VX else CONTACT_SLOW_S
    if now - since >= limit:
        return True, since, True
    return moved, since, False


def obstacle_command(left, center, right):
    """Arc for a low object. None leaves the survival command, which drives ahead.

    A wide storage box is flat across the view, so the side graft never
    starts. That face still has to be turned around before 2.2 m, or the
    cruise step drives into it. An aisle that is merely nearer in the
    middle keeps the model step.
    """
    graft = depth_graft(left, center, right)
    if graft is not None:
        return graft
    if center is None or center >= 4.0:
        return None
    sides = [v for v in (left, right) if v is not None]
    if not sides:
        return None
    # A flat face, or a face with one side open. A nearer center with one
    # far side is a gap and keeps the model step.
    wide = max(abs(v - center) for v in sides) <= 0.45
    face_left = (
        right is not None and left is not None
        and abs(center - right) <= 0.45 and left > center + 1.2
    )
    face_right = (
        left is not None and right is not None
        and abs(center - left) <= 0.45 and right > center + 1.2
    )
    if not wide and not face_left and not face_right:
        return None
    # Creep only. 0.15 still drives the bumper into the face while the yaw is small.
    return 0.08, _avoid_wz(left, right, center), "depth_arc"


def depth_graft(left, center, right):
    """Near-range camera override. None leaves the model command alone.

    At 1.2 m and farther the model picks the action. Inside that, the
    command yaws toward the farther side. The drive loop turns it in place.
    Positive wz is left.
    """
    if center is None:
        return None
    open_left = left is not None and left > 0.90 and left > center + 0.45
    open_right = right is not None and right > 0.90 and right > center + 0.45
    if center < 0.55:
        return 0.06, _avoid_wz(left, right, center), "depth_arc"
    if center < 1.20 and (open_left or open_right):
        vx = 0.06 if center < 0.75 else 0.10
        return vx, _avoid_wz(left, right, center), "depth_arc"
    if center < 1.20:
        return 0.06, _avoid_wz(left, right, center), "depth_arc"
    if right is not None and right < 1.6 and center > right + 0.30 and right < center * 0.80:
        return 0.08, _avoid_wz(left, right, right), "depth_clear"
    if left is not None and left < 1.6 and center > left + 0.30 and left < center * 0.80:
        return 0.08, _avoid_wz(left, right, left), "depth_clear"
    if right is not None and right < 3.2 and center > right + 0.8 and right < center * 0.70:
        return 0.08, _avoid_wz(left, right, right), "depth_clear"
    if left is not None and left < 3.2 and center > left + 0.8 and left < center * 0.70:
        return 0.08, _avoid_wz(left, right, left), "depth_clear"
    if right is not None and right < 0.70 and center > right + 0.4:
        return 0.08, _avoid_wz(left, right, right), "depth_clear"
    if left is not None and left < 0.70 and center > left + 0.4:
        return 0.08, _avoid_wz(left, right, left), "depth_clear"
    return None


def _yaw(q) -> float:
    sin_y = 2.0 * (q.w * q.z + q.x * q.y)
    cos_y = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(sin_y, cos_y)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


FACE_VX = 0.08
FACE_WZ = 0.18
# Cruise stays off until the nose has yawed this far away from a face.
FACE_CLEAR = 0.8
# Width that counts as enough for this chassis to drive through.
PASS_M = 0.80
# A cone is narrower than this at its own range. A storage box is wider.
CONE_M = 0.50
BOX_M = 0.70


def _span_m(x0: float, x1: float, dist: float, width: int) -> float:
    """Meters across columns x0..x1 of the 90° image at ``dist``."""
    focal = 0.5 * float(width) / math.tan(math.radians(45.0))
    return abs(float(dist) * (float(x1) - float(x0)) / focal)


def _column_range(sl: np.ndarray) -> np.ndarray:
    width = sl.shape[1] if sl.ndim == 2 else 0
    out = np.full(width, np.nan, np.float64)
    if sl.size == 0 or sl.ndim != 2:
        return out
    for x in range(width):
        col = sl[:, x]
        fin = col[np.isfinite(col) & (col > 0.15) & (col < INVALID_M)]
        if fin.size >= 3:
            out[x] = float(np.percentile(fin, 15))
    return out


def _occ_columns(here: np.ndarray, above: np.ndarray) -> np.ndarray:
    """Columns that hit an object. The room above a low object is still far."""
    occ = np.zeros(here.shape[0], dtype=bool)
    for x in range(here.shape[0]):
        if not np.isfinite(here[x]) or not np.isfinite(above[x]):
            continue
        if here[x] < 4.0 and above[x] > here[x] + 0.8:
            occ[x] = True
        elif here[x] < 3.2 and above[x] < here[x] + 0.8:
            occ[x] = True
    return occ


def _range_clusters(occ: np.ndarray, here: np.ndarray):
    found = []
    x = 0
    while x < occ.shape[0]:
        if not occ[x] or not np.isfinite(here[x]):
            x += 1
            continue
        x1 = x + 1
        while (
            x1 < occ.shape[0]
            and occ[x1]
            and np.isfinite(here[x1])
            and abs(float(here[x1]) - float(here[x1 - 1])) <= 0.6
        ):
            x1 += 1
        found.append({"x0": x, "x1": x1, "dist": float(np.median(here[x:x1]))})
        x = x1
    return found


def _cluster_kind(cluster, here, above, nxt, width: int) -> str:
    x0, x1 = cluster["x0"], cluster["x1"]
    if x1 - x0 < 4:
        return "other"
    span = _span_m(x0, x1 - 1, cluster["dist"], width)
    look = flat = count = 0
    for x in range(x0, x1):
        if not np.isfinite(here[x]) or not np.isfinite(above[x]):
            continue
        count += 1
        if above[x] > here[x] + 0.8:
            look += 1
        if np.isfinite(nxt[x]) and abs(float(here[x]) - float(nxt[x])) <= 0.35:
            flat += 1
    if count == 0:
        return "other"
    if span <= CONE_M and cluster["dist"] < 3.2:
        return "cone"
    if span >= BOX_M and look / count >= 0.6 and flat / count >= 0.6 and cluster["dist"] < 4.0:
        return "box"
    return "other"


def _open_runs(occ: np.ndarray, here: np.ndarray):
    runs = []
    x = 0
    while x < here.shape[0]:
        if occ[x] or not np.isfinite(here[x]) or here[x] < 2.2:
            x += 1
            continue
        x1 = x + 1
        while x1 < here.shape[0] and not occ[x1] and np.isfinite(here[x1]) and here[x1] >= 2.2:
            x1 += 1
        runs.append((x, x1))
        x = x1
    return runs


def _gap_by_cone(gx0: int, gx1: int, cone, boxes) -> bool:
    """The opening touches the cone, and the box is the other side of that pair."""
    if 0 <= gx0 - cone["x1"] <= 3:
        return any(box["x0"] >= gx1 or box["x1"] <= cone["x0"] for box in boxes)
    if 0 <= cone["x0"] - gx1 <= 3:
        return any(box["x1"] <= gx0 or box["x0"] >= cone["x1"] for box in boxes)
    return False


def gap_command(model: np.ndarray, fy: float, cy: float, src_h: int):
    """Yaw into a gap that is wide enough at the cone, with the box on the other side.

    The opening is measured at the cone's range. A far wall seen through a
    narrow slot no longer counts as a wide passage, and the empty floor
    beside the box is not a passage. ``"through"`` means that gap is already
    in front of the nose. None leaves the existing command.
    """
    height, width = model.shape[:2]
    rows = [_warp_row(fy, cy, src_h, height, frac) for frac in LOWER_BANDS]
    best = None
    for i in range(1, len(LOWER_BANDS) - 2):
        if LOWER_BANDS[i] >= 0.42:
            break
        here = _column_range(model[rows[i]:rows[i + 1]])
        above = _column_range(model[rows[i - 1]:rows[i]])
        nxt = _column_range(model[rows[i + 1]:rows[i + 2]])
        occ = _occ_columns(here, above)
        clusters = _range_clusters(occ, here)
        kinds = [_cluster_kind(c, here, above, nxt, width) for c in clusters]
        cones = [c for c, kind in zip(clusters, kinds) if kind == "cone"]
        boxes = [c for c, kind in zip(clusters, kinds) if kind == "box"]
        if not cones or not boxes:
            continue
        for gx0, gx1 in _open_runs(occ, here):
            spans = [
                _span_m(gx0, gx1 - 1, cone["dist"], width)
                for cone in cones
                if _gap_by_cone(gx0, gx1, cone, boxes)
            ]
            if not spans:
                continue
            span = min(spans)
            if span < PASS_M:
                continue
            mid = 0.5 * (gx0 + gx1 - 1)
            if best is None or span > best[0]:
                best = (span, mid)
    if best is None:
        return None
    _span, mid = best
    center = (float(width) - 1.0) * 0.5
    if abs(mid - center) <= float(width) * 0.08:
        return "through"
    wz = FACE_WZ if mid < center else -FACE_WZ
    return FACE_VX, wz, "depth_clear"


def face_turn(graft, yaw: float, job, now: float = 0.0):
    """Turn a box or a side obstacle in place. Forward speed stays zero.

    A short gap in the depth used to resume the 0.24 m/s cruise into the
    box. The same yaw continues until the nose has left the face and the
    depth has stayed clear.
    """
    face = graft is not None and graft[2] in ("depth_arc", "depth_clear")
    if face:
        sign = 1.0 if float(graft[1]) > 0.0 else -1.0
        if job is None:
            job = {
                "yaw0": float(yaw),
                "yaw_prev": float(yaw),
                "sign": sign,
                "turned": 0.0,
                "until": float(now),
            }
        job["until"] = float(now) + 0.4
    elif job is None:
        return None, None
    prev = float(job.get("yaw_prev", job["yaw0"]))
    step = job["sign"] * _wrap(float(yaw) - prev)
    job["yaw_prev"] = float(yaw)
    job["turned"] = float(job.get("turned", 0.0)) + max(0.0, step)
    if not face and job["turned"] >= FACE_CLEAR and float(now) >= job["until"]:
        return None, None
    rate = job["sign"] * FACE_WZ
    return (0.0, rate, "depth_turn"), job


def _k_from_msg(msg) -> Tuple[np.ndarray, np.ndarray]:
    k = msg.k
    matrix = np.array([[k[0], k[1], k[2]], [k[3], k[4], k[5]], [k[6], k[7], k[8]]], np.float64)
    return matrix, np.array(msg.d, np.float64)


class _SearchGoal:
    """The find task on disk. No file means the chassis stays still."""

    def __init__(self) -> None:
        self.task_id = ""
        self.goal: Optional[np.ndarray] = None
        self.target = ""
        self.distance_m: Optional[float] = None
        self.bearing_rad: Optional[float] = None
        self.mode = ""
        self.origin: Optional[Tuple[float, float, float]] = None
        self.heading = 0.0
        self.hits = 0
        self.verified_hits = 0
        self.sight: Optional[Tuple[float, float]] = None
        self.sight_verified = False
        self.mark_box: Optional[Tuple[float, float, float, float]] = None
        self.finished = False
        self.result = ""
        self.found_pose: Optional[dict] = None
        self.hold_since = 0.0
        self.spin_anchor: Optional[Tuple[float, float, float, float]] = None
        self.phase = ""
        self.scan_turned = 0.0
        self.scan_yaw: Optional[float] = None

    def _reset_look(self) -> None:
        self.hits = 0
        self.verified_hits = 0
        self.sight = None
        self.sight_verified = False
        self.mark_box = None
        self.finished = False
        self.result = ""
        self.found_pose = None
        self.hold_since = 0.0
        self.spin_anchor = None
        self.phase = ""
        self.scan_turned = 0.0
        self.scan_yaw = None

    def begin_scan(self, yaw: float) -> None:
        self.phase = "scan"
        self.scan_turned = 0.0
        self.scan_yaw = yaw

    def note_scan(self, yaw: float) -> bool:
        """Add the yaw moved since the previous step. True after one full turn."""
        if self.scan_yaw is not None:
            self.scan_turned += abs(_wrap(yaw - self.scan_yaw))
        self.scan_yaw = yaw
        return self.scan_turned >= 2.0 * math.pi

    def _place_along(self, x: float, y: float) -> None:
        ox, oy, oz = self.origin or (x, y, 0.0)
        dx, dy = x - ox, y - oy
        progress = dx * math.cos(self.heading) + dy * math.sin(self.heading)
        ahead = max(progress, 0.0) + 3.0
        self.goal = np.array(
            [ox + ahead * math.cos(self.heading), oy + ahead * math.sin(self.heading), oz],
            np.float64,
        )

    def update(self, x: float, y: float, z: float, yaw: float) -> Optional[np.ndarray]:
        try:
            task = json.loads(SEARCH_GOAL.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.task_id = ""
            self.goal = None
            self.mode = ""
            self._reset_look()
            return None
        task_id = str(task.get("id") or "")
        if task_id and task_id != self.task_id:
            raw = task.get("distance_m")
            try:
                distance = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                distance = None
            if distance is not None and not (1.0 <= distance <= 30.0):
                distance = None
            bearing = float(task.get("bearing_rad") or 0.0)
            origin = task.get("origin")
            if task.get("heading") is not None and isinstance(origin, (list, tuple)) and len(origin) >= 3:
                self.heading = float(task["heading"])
                self.origin = (float(origin[0]), float(origin[1]), float(origin[2]))
            else:
                self.heading = yaw + bearing
                self.origin = (x, y, z)
                task["origin"] = [x, y, z]
                task["heading"] = self.heading
                tmp = SEARCH_GOAL.with_suffix(".tmp")
                tmp.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, SEARCH_GOAL)
            self.task_id = task_id
            self.target = str(task.get("target") or "")
            self.distance_m = distance
            self.bearing_rad = bearing
            self.mode = "point" if distance is not None else "along"
            self._reset_look()
            if self.mode == "point":
                ox, oy, oz = self.origin
                self.goal = np.array(
                    [ox + distance * math.cos(self.heading), oy + distance * math.sin(self.heading), oz],
                    np.float64,
                )
        if self.mode == "along" and not self.finished and self.origin is not None:
            self._place_along(x, y)
        return self.goal

    def note_sight(self, cx: float, rng: float, verified: bool) -> None:
        """Colour is a hint. Only a class detection can accumulate a confirmation."""
        prev = self.sight
        self.sight = (cx, rng)
        self.sight_verified = verified
        if not verified:
            return
        if prev is not None and self.verified_hits > 0 and abs(cx - prev[0]) < 0.06 and abs(rng - prev[1]) < 0.30:
            self.verified_hits += 1
        else:
            self.verified_hits = 1

    def miss_sight(self, verified_seen: bool) -> None:
        if verified_seen:
            return
        self.verified_hits = max(0, self.verified_hits - 1)
        if self.verified_hits == 0:
            self.sight_verified = False

    def finish(self, result: str, x: float, y: float, yaw: float) -> None:
        self.finished = True
        self.result = result
        pose = {"robot_x": round(x, 2), "robot_y": round(y, 2), "robot_yaw": round(yaw, 2)}
        if result == "found" and self.sight is not None:
            cx, rng = self.sight
            ang = yaw + (cx - 0.5) * math.radians(79.2)
            pose["object_x"] = round(x + rng * math.cos(ang), 2)
            pose["object_y"] = round(y + rng * math.sin(ang), 2)
            pose["range_m"] = round(rng, 2)
        self.found_pose = pose
        SEARCH_RESULT.write_text(
            json.dumps({"result": result, "target": self.target, **pose}, ensure_ascii=False),
            encoding="utf-8",
        )


def _goal_already_finished() -> bool:
    """A restart must not drive a search that already wrote its result."""
    try:
        if SEARCH_RESULT.stat().st_mtime < SEARCH_GOAL.stat().st_mtime:
            return False
        done = json.loads(SEARCH_RESULT.read_text(encoding="utf-8")).get("result")
    except (OSError, json.JSONDecodeError):
        return False
    return done in ("found", "not_found")


class _TrackLog:
    """One jsonl per task: latch, then a pose and the command at every step."""

    def __init__(self) -> None:
        self.task_id = ""
        self.t0 = 0.0
        self.fp = None
        self.saw_result = ""

    def bind(self, search: _SearchGoal) -> None:
        if not search.task_id or search.task_id == self.task_id:
            return
        self.close()
        self.task_id = search.task_id
        self.t0 = time.time()
        self.saw_result = ""
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in search.task_id)[:80] or "task"
        path = TRACK_DIR / f"{safe}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = (not path.exists()) or path.stat().st_size == 0
        self.fp = path.open("a", encoding="utf-8")
        if fresh:
            self._write({
                "kind": "latch",
                "id": search.task_id,
                "target": search.target,
                "mode": search.mode,
                "distance_m": search.distance_m,
                "origin": None if search.origin is None else [round(float(v), 4) for v in search.origin],
                "heading": round(float(search.heading), 6),
                "goal": None if search.goal is None else [round(float(v), 3) for v in search.goal[:2]],
            })

    def sample(self, search: _SearchGoal, x: float, y: float, yaw: float, mode: str,
               wire: "YhsWire", nav_last: Optional[dict], goal_dist: Optional[float]) -> None:
        if self.fp is None:
            return
        if search.finished and self.saw_result == (search.result or mode):
            return
        with wire.lock:
            raw = wire.last_cmd
        try:
            cmd = json.loads(raw or "{}")
        except json.JSONDecodeError:
            cmd = {}
        nav_last = nav_last or {}
        self._write({
            "kind": "step",
            "t": round(time.time() - self.t0, 3),
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "yaw": round(float(yaw), 4),
            "mode": mode,
            "family": nav_last.get("family"),
            "guard": nav_last.get("guard"),
            "bearing_deg": nav_last.get("bearing_deg"),
            "clear": nav_last.get("clear_flr"),
            "low_m": nav_last.get("low_m"),
            "avoid_surv": nav_last.get("avoid_surv"),
            "lead": nav_last.get("lead_side"),
            "pass": nav_last.get("pass_lr"),
            "vx": cmd.get("vx"),
            "wz_deg_s": cmd.get("wz_deg_s"),
            "dist": None if goal_dist is None else round(float(goal_dist), 3),
            "result": search.result or "",
        })
        if search.finished:
            self.saw_result = search.result or mode

    def _write(self, row: dict) -> None:
        assert self.fp is not None
        self.fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fp.flush()

    def close(self) -> None:
        if self.fp is not None:
            self.fp.close()
            self.fp = None


_COLORS = {
    "红": (((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (180, 255, 255))),
    "蓝": (((95, 80, 50), (130, 255, 255)),),
    "黄": (((18, 90, 60), (35, 255, 255)),),
    "绿": (((40, 70, 40), (85, 255, 255)),),
}


def _color_sight(rgb: np.ndarray, depth: np.ndarray, target: str) -> Optional[Tuple[float, float]]:
    """Center fraction and range for one region whose color is named in the task."""
    ranges = None
    for word, spec in _COLORS.items():
        if word in target:
            ranges = spec
            break
    if ranges is None:
        return None
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = None
    for lo, hi in ranges:
        part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    height, width = mask.shape
    mask[: int(0.25 * height), :] = 0
    count, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        if area < 3500 or area > 60000 or bh < 40 or bw < 28:
            continue
        cx = (x + 0.5 * bw) / width
        cy = (y + 0.5 * bh) / height
        if not (0.20 <= cx <= 0.80 and 0.30 <= cy <= 0.85):
            continue
        if best is None or area > best[0]:
            best = (area, cx, x / width, y / height, (x + bw) / width, (y + bh) / height)
    if best is None:
        return None
    _area, cx, x0, y0, x1, y1 = best
    dh, dw = depth.shape
    patch = depth[int(y0 * dh):int(y1 * dh), int(x0 * dw):int(x1 * dw)]
    valid = patch[np.isfinite(patch) & (patch > 0.50) & (patch < 4.0)]
    if valid.size < 40:
        return None
    return cx, float(np.median(valid)), (x0, y0, x1, y1)


_PERSON_HELPER = None
_FRAME_PATH = Path("/tmp/person_frame.u8")
_SIGHT_PATH = Path("/tmp/person_sight.json")


def _ensure_person_helper() -> None:
    """The model process OpenCV has no cascade. The system one does."""
    global _PERSON_HELPER
    if _PERSON_HELPER is not None and _PERSON_HELPER.poll() is None:
        return
    import subprocess

    script = r"""
import json, os, time
import cv2
import numpy as np
root = "/usr/share/opencv4/haarcascades"
clfs = []
for name in ("haarcascade_upperbody.xml", "haarcascade_frontalface_default.xml"):
    clf = cv2.CascadeClassifier(os.path.join(root, name))
    if not clf.empty():
        clfs.append(clf)
req, out = "/tmp/person_frame.u8", "/tmp/person_sight.json"
seen = 0.0
while True:
    try:
        stamp = os.path.getmtime(req)
    except OSError:
        time.sleep(0.05)
        continue
    if stamp == seen:
        time.sleep(0.03)
        continue
    seen = stamp
    raw = open(req, "rb").read()
    hit = None
    if len(raw) == 320 * 240 and clfs:
        gray = np.frombuffer(raw, np.uint8).reshape(240, 320)
        best = None
        for clf in clfs:
            for x, y, bw, bh in clf.detectMultiScale(gray, 1.1, 4, minSize=(24, 32)):
                cx = (x + 0.5 * bw) / 320.0
                cy = (y + 0.5 * bh) / 240.0
                if not (0.20 <= cx <= 0.80 and 0.12 <= cy <= 0.85):
                    continue
                area = int(bw) * int(bh)
                if best is None or area > best[0]:
                    best = (area, cx, x / 320.0, y / 240.0, (x + bw) / 320.0, (y + bh) / 240.0)
        if best is not None:
            hit = {"cx": best[1], "x0": best[2], "y0": best[3], "x1": best[4], "y1": best[5]}
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"hit": hit, "stamp": stamp}, handle)
    os.replace(tmp, out)
"""
    _PERSON_HELPER = subprocess.Popen(
        ["/usr/bin/python3", "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _person_sight(rgb: np.ndarray, depth: np.ndarray) -> Optional[Tuple[float, float]]:
    try:
        import cv2

        _ensure_person_helper()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA)
        tmp = _FRAME_PATH.with_suffix(".u8.tmp")
        tmp.write_bytes(np.ascontiguousarray(small, dtype=np.uint8).tobytes())
        os.replace(tmp, _FRAME_PATH)
        if not _SIGHT_PATH.is_file():
            return None
        body = json.loads(_SIGHT_PATH.read_text(encoding="utf-8"))
        if abs(body.get("stamp", 0.0) - _FRAME_PATH.stat().st_mtime) > 1.0:
            return None
        hit = body.get("hit")
        if not hit:
            return None
        dh, dw = depth.shape
        patch = depth[int(hit["y0"] * dh):int(hit["y1"] * dh), int(hit["x0"] * dw):int(hit["x1"] * dw)]
        valid = patch[np.isfinite(patch) & (patch > 0.50) & (patch < 4.0)]
        if valid.size < 20:
            return None
        box = (float(hit["x0"]), float(hit["y0"]), float(hit["x1"]), float(hit["y1"]))
        return float(hit["cx"]), float(np.median(valid)), box
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None


_YOLO_PY = os.environ.get(
    "YHS_YOLO_PY", "/home/user/zikun_ws/illegal-parking-svc/.venv/bin/python")
_YOLO_HELPER = os.environ.get("YHS_YOLO_HELPER", "/home/user/rt/bin/yolo_world.pyc")
_YOLO_REQ = Path(os.environ.get("YHS_YOLO_REQ", "/tmp/yolo_req.json"))
_YOLO_FRAME = Path(os.environ.get("YHS_YOLO_FRAME", "/tmp/yolo_frame.rgb"))
_YOLO_SIGHT = Path(os.environ.get("YHS_YOLO_SIGHT", "/tmp/yolo_sight.json"))
_MARK_PATH = Path(os.environ.get("YHS_MARK", "/tmp/yhs_mark.json"))
_YOLO_PROC = None
_YOLO_DET = None
_YOLO_KEY = None


class _RemoteYolo:
    """Boxes from the YOLO-World process. ``set_prompts`` matches ``YoloWorld``."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self._n = 0

    def set_prompts(self, prompts) -> None:
        self.prompts = list(prompts)

    def __call__(self, rgb: np.ndarray):
        if not self.prompts or not _yolo_alive():
            return []
        height, width = rgb.shape[:2]
        self._n += 1
        stamp = self._n
        tmp = _YOLO_FRAME.with_suffix(".rgb.tmp")
        tmp.write_bytes(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        os.replace(tmp, _YOLO_FRAME)
        req = _YOLO_REQ.with_suffix(".json.tmp")
        req.write_text(json.dumps({
            "stamp": stamp, "w": width, "h": height, "prompts": self.prompts,
        }), encoding="utf-8")
        os.replace(req, _YOLO_REQ)
        deadline = time.time() + 0.35
        while time.time() < deadline:
            try:
                body = json.loads(_YOLO_SIGHT.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.01)
                continue
            if body.get("stamp") == stamp:
                return [((d["x0"], d["y0"], d["x1"], d["y1"]), d["conf"], d["prompt"]) for d in body.get("dets") or []]
            time.sleep(0.01)
        return []


def _yolo_alive() -> bool:
    global _YOLO_PROC
    if _YOLO_PROC is not None and _YOLO_PROC.poll() is None:
        return True
    helper = Path(_YOLO_HELPER)
    py = Path(_YOLO_PY)
    if not helper.is_file() or not py.is_file():
        return False
    import subprocess
    _YOLO_PROC = subprocess.Popen(
        [str(py), str(helper)],
        cwd="/home/user/rt",
        env={**os.environ, "PYTHONPATH": "/home/user/rt/app/pydeps"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return True


def _box_sight(depth: np.ndarray, bbox) -> Optional[Tuple[float, float]]:
    height, width = depth.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox)
    cx = ((x0 + x1) * 0.5) / max(width, 1)
    xi0, yi0 = max(0, int(x0)), max(0, int(y0))
    xi1, yi1 = min(width, max(xi0 + 1, int(round(x1)))), min(height, max(yi0 + 1, int(round(y1))))
    patch = depth[yi0:yi1, xi0:xi1]
    valid = patch[np.isfinite(patch) & (patch > 0.30) & (patch < 12.0)]
    if valid.size < 20:
        return None
    return cx, float(np.median(valid))


def _upright_can(bbox) -> bool:
    """A trash can stands up. The blue storage box is a wide, low slab."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    width, height = x1 - x0, y1 - y0
    if width <= 0.0 or height <= 0.0:
        return False
    return width <= height * 1.4


def _full_can(bbox, width: int, height: int) -> bool:
    """A whole trash can: taller than it is wide, and not cut by the frame.

    The blue storage box is a slab or a near-square patch on the image edge.
    Four pixels of margin still accepted that patch.
    """
    if not _upright_can(bbox):
        return False
    x0, y0, x1, y1 = (float(v) for v in bbox)
    box_w, box_h = x1 - x0, y1 - y0
    if box_w > box_h * 0.90:
        return False
    margin_x = 0.08 * float(width)
    margin_y = 0.08 * float(height)
    return (
        x0 > margin_x and y0 > margin_y
        and x1 < float(width) - margin_x and y1 < float(height) - margin_y
    )


def _yolo_verified(rgb: np.ndarray, depth: np.ndarray, target: str) -> Optional[Tuple[float, float, bool]]:
    """A YOLO-World box that matches the object and, when asked, its colour."""
    global _YOLO_DET, _YOLO_KEY
    try:
        from aerial_finder.mission.parser import ParseError, parse_instruction
        spec = parse_instruction(target)
    except (ImportError, ParseError, ValueError):
        return None
    if not spec.prompts or not _yolo_alive():
        return None
    key = (spec.label, tuple(spec.prompts), spec.color)
    if _YOLO_DET is None or key != _YOLO_KEY:
        from aerial_finder.perception.detector import TargetDetector
        _YOLO_DET = TargetDetector(_RemoteYolo(), min_color_frac=0.15, hints=None, min_conf=0.05)
        _YOLO_DET.configure(spec.label, spec.prompts, spec.color)
        _YOLO_KEY = key
    verified = [d for d in _YOLO_DET(rgb) if d.verified and d.source == "yolo"]
    if spec.label == "trash can":
        height, width = rgb.shape[:2]
        verified = [
            d for d in verified
            if d.prompt in ("trash can", "garbage can") and _full_can(d.bbox, width, height)
        ]
    if not verified:
        return None
    best = max(verified, key=lambda d: d.conf)
    sight = _box_sight(depth, best.bbox)
    if sight is None:
        return None
    height, width = rgb.shape[:2]
    return sight[0], sight[1], True, _norm_box(best.bbox, width, height)


def _norm_box(bbox, width: int, height: int) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return (x0 / max(width, 1), y0 / max(height, 1), x1 / max(width, 1), y1 / max(height, 1))


def _display_box(sight_box, verified: bool):
    """Box drawn on the first-person view. Colour alone is not the trash can."""
    if not verified or sight_box is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in sight_box)
    if (x1 - x0) < 0.04 or (y1 - y0) < 0.04:
        return None
    return (x0, y0, x1, y1)


def _write_mark(box, label: str, hold: bool) -> None:
    """Box fractions for the first-person view. A held box stays after the find."""
    body = {
        "t": time.time(),
        "label": label,
        "hold": hold,
        "box": None if box is None else [round(float(v), 4) for v in box],
    }
    try:
        tmp = _MARK_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, _MARK_PATH)
    except OSError:
        pass


def _target_sight(rgb: np.ndarray, depth: np.ndarray, target: str):
    """Center, range, class-detection flag, and the box as fractions of the image."""
    yolo_hit = _yolo_verified(rgb, depth, target)
    if yolo_hit is not None:
        return yolo_hit
    if any(word in target for word in _COLORS):
        hit = _color_sight(rgb, depth, target)
        return None if hit is None else (hit[0], hit[1], False, hit[2])
    if "人" in target:
        hit = _person_sight(rgb, depth)
        return None if hit is None else (hit[0], hit[1], True, hit[2])
    return None


def _body_step(prev, pose):
    """Odometry since the last step, in the model frame. None on the first step.

    dx is forward meters. dyaw is clockwise radians, the same sign the model uses.
    """
    if prev is None:
        return None
    x0, y0, yaw0 = prev
    x, y, yaw = pose
    dx = (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)
    dyaw = (yaw0 - yaw + math.pi) % (2.0 * math.pi) - math.pi
    return np.array([dx, 0.0, 0.0, dyaw], np.float32)


def _model_frame(x: float, y: float, z: float, yaw: float, goal: np.ndarray):
    """Model yaw is clockwise. Flip y with it so a point on the right stays on the right."""
    pos = np.array([x, -y, z], np.float64)
    aim = np.array([float(goal[0]), -float(goal[1]), float(goal[2])], np.float64)
    return pos, -float(yaw), aim


class YhsWire:
    def __init__(self, armed: bool = False) -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image

        self.armed = bool(armed)
        self.lock = threading.Lock()
        self.color = None
        self.depth = None
        self.color_k = None
        self.color_d = None
        self.depth_k = None
        self.yaw_tgt: Optional[float] = None
        self.pose: Optional[Tuple[float, float, float, float]] = None
        self.color_at = 0.0
        self.depth_at = 0.0
        self.odom_at = 0.0
        self.color_enc = ""
        self.color_hw = (0, 0)
        self.depth_hw = (0, 0)
        self.last_action = None
        self.last_twist = None
        self.inference_ok = False
        self.error = ""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        rclpy.init()
        self.node = Node("yhs_wire")
        self.node.create_subscription(Image, COLOR_TOPIC, self._on_color, qos)
        self.node.create_subscription(CameraInfo, COLOR_INFO, self._on_color_info, qos)
        self.node.create_subscription(Image, DEPTH_TOPIC, self._on_depth, qos)
        self.node.create_subscription(CameraInfo, DEPTH_INFO, self._on_depth_info, qos)
        self.node.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, qos)
        self.pub = self.node.create_publisher(Twist, CMD_TOPIC, 10)
        self.last_cmd = None
        self.model_step = np.zeros(4, np.float32)
        self.joypad = None
        try:
            from yhs_can_interfaces.msg import ChassisInfoFb
            self.node.create_subscription(ChassisInfoFb, "/chassis_info_fb", self._on_chassis, qos)
        except ImportError:
            pass
        self._spin = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin.start()
        self._hold = threading.Thread(target=self._republish, daemon=True)
        self._hold.start()

    def _on_color(self, msg) -> None:
        try:
            rgb = _rgb(msg)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            return
        with self.lock:
            self.color = rgb
            self.color_at = time.time()
            self.color_enc = msg.encoding
            self.color_hw = (int(msg.height), int(msg.width))

    def _on_color_info(self, msg) -> None:
        matrix, dist = _k_from_msg(msg)
        with self.lock:
            self.color_k = matrix
            self.color_d = dist

    def _on_depth_info(self, msg) -> None:
        matrix, _dist = _k_from_msg(msg)
        with self.lock:
            self.depth_k = matrix

    def _on_depth(self, msg) -> None:
        try:
            depth = _depth_m(msg)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            return
        with self.lock:
            self.depth = depth
            self.depth_at = time.time()
            self.depth_hw = (int(msg.height), int(msg.width))

    def _on_chassis(self, msg) -> None:
        angles = wheel_angles(msg)
        with self.lock:
            self.joypad = bool(msg.io_fb.io_fb_joypad_ctrl)
            if angles is not None:
                self.steer_deg = angles

    def joystick(self) -> Optional[bool]:
        with self.lock:
            return self.joypad

    def sent_speed(self) -> float:
        with self.lock:
            twist = self.last_twist
        if twist is None:
            return 0.0
        return float(twist.linear.x)

    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        yaw = _yaw(msg.pose.pose.orientation)
        with self.lock:
            self.pose = (float(p.x), float(p.y), float(p.z), yaw)
            self.odom_at = time.time()

    def snapshot(self):
        with self.lock:
            return (self.color, self.depth, self.pose, self.color_k, self.color_d, self.depth_k,
                    self.color_at, self.depth_at, self.odom_at)

    def _republish(self) -> None:
        while rcl_ok():
            twist = None
            with self.lock:
                if self.armed and self.last_twist is not None:
                    twist = self.last_twist
            if twist is not None:
                self.pub.publish(twist)
            time.sleep(0.02)

    def publish_action(self, action: np.ndarray, vx_cap: Optional[float] = None) -> None:
        """Model step to chassis speed. A pure yaw is sent with no forward speed."""
        from geometry_msgs.msg import Twist

        dx, _dy, _dz, dyaw = (float(v) for v in np.asarray(action, np.float64))
        now = time.time()
        with self.lock:
            prev = self.last_twist
            angles = getattr(self, "steer_deg", None)
            until = float(getattr(self, "_center_until", 0.0))
            gave_up = bool(getattr(self, "_center_gave_up", False))
        prev_vx = 0.0 if prev is None else float(prev.linear.x)
        prev_wz = 0.0 if prev is None else float(prev.angular.z)
        if dx <= 0.0:
            gave_up = False
        with self.lock:
            joypad = bool(getattr(self, "joypad", False))
        if joypad and angles and max(abs(float(a)) for a in angles) >= SPIN_POSE_DEG and dx > 0.0:
            gave_up = False
            until = 0.0
        holding = hold_for_center(dx, prev_vx, prev_wz, angles, now, until, gave_up)
        peak = 0.0 if not angles else max(abs(float(a)) for a in angles)
        until, gave_up = next_center_state(holding, peak, now, until, gave_up)
        with self.lock:
            self._center_until = until
            self._center_gave_up = gave_up
        if holding:
            twist = Twist()
            cmd = json.dumps(
                {"vx": 0.0, "wz_deg_s": 0.0, "gear": SPEED_GEAR, "steer": "center"},
                ensure_ascii=False,
            )
            with self.lock:
                self.last_action = [0.0, 0.0, 0.0, 0.0]
                self.last_twist = twist
                self.last_cmd = cmd
                self.model_step = np.zeros(4, np.float32)
                self.yaw_tgt = None
            if self.armed:
                self.pub.publish(twist)
            return
        twist = Twist()
        step = np.zeros(4, np.float32)
        # Score the training-sized step. This chassis cruises at the pre-speedup rate.
        dx *= 0.06
        dyaw *= 0.255
        if dx > 0.0 or abs(dyaw) > 1e-4:
            velocity = 0.0
            if dx > 0.0:
                limit = VX_MAX if vx_cap is None else min(VX_MAX, float(vx_cap))
                velocity = float(np.clip(dx * STEP_HZ, 0.0, limit))
                if velocity < 0.10:
                    velocity = 0.10
            # Model dyaw is clockwise. Twist angular.z is counterclockwise.
            yaw_rate = -dyaw * STEP_HZ
            if dx <= 0.0:
                yaw_rate = float(np.clip(math.copysign(max(abs(yaw_rate), 0.25), yaw_rate), -0.40, 0.40))
            twist.linear.x = velocity
            twist.angular.z = yaw_rate
            step = np.array([velocity / STEP_HZ, 0.0, 0.0, -yaw_rate / STEP_HZ], np.float32)
        cmd = json.dumps(
            {
                "vx": round(float(twist.linear.x), 3),
                "wz_deg_s": round(float(twist.angular.z) * 180.0 / math.pi, 2),
                "gear": SPEED_GEAR,
            },
            ensure_ascii=False,
        )
        with self.lock:
            self.last_action = [float(v) for v in np.asarray(action, np.float64)]
            self.last_twist = twist
            self.last_cmd = cmd
            self.model_step = step
            self.yaw_tgt = None
        if self.armed:
            self.pub.publish(twist)

    def publish_speed(self, vx: float, wz: float, tag: str) -> None:
        """Chassis speed already in m/s and rad/s. Positive wz is left."""
        from geometry_msgs.msg import Twist

        twist = Twist()
        twist.linear.x = float(vx)
        twist.angular.z = float(wz)
        cmd = json.dumps(
            {
                "vx": round(float(vx), 3),
                "wz_deg_s": round(float(wz) * 180.0 / math.pi, 2),
                "gear": SPEED_GEAR,
                "graft": tag,
            },
            ensure_ascii=False,
        )
        with self.lock:
            self.last_action = [float(vx), 0.0, 0.0, float(wz)]
            self.last_twist = twist
            self.last_cmd = cmd
            self.model_step = np.array([vx / STEP_HZ, 0.0, 0.0, -wz / STEP_HZ], np.float32)
            self.yaw_tgt = None
        if self.armed:
            self.pub.publish(twist)

    def publish_stop(self) -> None:
        from geometry_msgs.msg import Twist

        twist = Twist()
        cmd = json.dumps({"vx": 0.0, "wz_deg_s": 0.0, "gear": SPEED_GEAR}, ensure_ascii=False)
        with self.lock:
            self.last_action = [0.0, 0.0, 0.0, 0.0]
            self.last_twist = twist
            self.last_cmd = cmd
            self.model_step = np.zeros(4, np.float32)
            self.yaw_tgt = None
        if self.armed:
            self.pub.publish(twist)

    def write_status(self, extra: Optional[dict] = None) -> None:
        now = time.time()
        with self.lock:
            body = {
                "armed": self.armed,
                "cmd_topic": CMD_TOPIC,
                "color_topic": COLOR_TOPIC,
                "depth_topic": DEPTH_TOPIC,
                "odom_topic": ODOM_TOPIC,
                "color_age_s": None if self.color_at == 0 else round(now - self.color_at, 3),
                "depth_age_s": None if self.depth_at == 0 else round(now - self.depth_at, 3),
                "odom_age_s": None if self.odom_at == 0 else round(now - self.odom_at, 3),
                "color_encoding": self.color_enc,
                "color_hw": list(self.color_hw),
                "depth_hw": list(self.depth_hw),
                "pose": None if self.pose is None else [round(v, 3) for v in self.pose],
                "inference_ok": self.inference_ok,
                "last_action": self.last_action,
                "cmd": self.last_cmd,
                "error": self.error,
            }
        if extra:
            body.update(extra)
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, ensure_ascii=False))
        tmp.replace(STATUS)


def rcl_ok() -> bool:
    import rclpy
    return rclpy.ok()


def _resize_depth(depth: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    import cv2

    src = np.nan_to_num(np.asarray(depth, np.float32), nan=0.0)
    out = cv2.resize(src, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    out[out <= 0.0] = np.nan
    return out


def _depth_scale(pred: np.ndarray, seen: np.ndarray, sensor_depth: np.ndarray, depth_k: np.ndarray):
    """Median real/model ratio on the obstacle band. None when the overlap is too thin."""
    from aerial_finder.nav.consequence_model import pool_min
    from pinhole90 import warp_depth

    real = warp_depth(sensor_depth, depth_k)
    if real.shape != pred.shape[:2]:
        real = _resize_depth(real, pred.shape[:2])
    model = np.where(seen, pred, np.inf).astype(np.float32)
    real_f = np.where(np.isfinite(real), real, np.inf).astype(np.float32)
    if min(model.shape) < 56:
        return None, 0
    m56 = np.asarray(pool_min(model), np.float32)
    r56 = np.asarray(pool_min(real_f), np.float32)
    r0, r1 = SCALE_ROWS
    a, b = m56[r0:r1], r56[r0:r1]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0.2) & (a < 1e3) & (b > 0.4) & (b < 6.0)
    n = int(ok.sum())
    if n < 40:
        return None, n
    ratio = float(np.median(b[ok] / a[ok]))
    if not 0.05 < ratio < 2.0:
        return None, n
    return ratio, n


def _wait_sensors(wire: YhsWire, timeout_s: float = 20.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        color, depth, pose, color_k, _color_d, depth_k, *_ = wire.snapshot()
        if (
            color is not None and depth is not None and pose is not None
            and color_k is not None and depth_k is not None
        ):
            return
        wire.write_status()
        time.sleep(0.2)
    raise RuntimeError("camera or odom did not arrive")


def _load_office():
    import importlib.util

    here = Path(__file__).resolve().parent
    for cand in (here / "office_choice.py", here / "office_choice.pyc"):
        if not cand.exists():
            continue
        spec = importlib.util.spec_from_file_location("office_choice", cand)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    raise RuntimeError("office_choice is not next to yhs_wire")


def _office_candidates(horizon: int, bearing: float):
    from aerial_finder.nav.consequence_model import LIMITS

    limit = float(LIMITS[3])
    fwd = np.zeros((horizon, 4), np.float64)
    fwd[:, 0] = 0.8
    slow = np.zeros((horizon, 4), np.float64)
    slow[:, 0] = 0.35
    names = ["fwd", "slow", "spin_left", "spin_right"]
    spins = []
    for sign in (-1.0, 1.0):
        action = np.zeros((horizon, 4), np.float64)
        toward = float(bearing) * sign > 0.0
        mag = min(abs(float(bearing)), limit) if toward else limit
        left = mag
        for t in range(horizon):
            step = min(left, limit)
            action[t, 3] = sign * step
            left = max(0.0, left - step)
        spins.append(action)
    return names, np.stack([fwd, slow, *spins])


def _office_curves(nav, state, goal, cands, prev):
    import torch
    from aerial_finder.nav.consequence_model import goal_features, history_inputs, survival

    count = len(cands)
    goal_feat = torch.as_tensor(
        np.repeat(goal_features(state, goal)[None], count, 0), device=nav.dev)
    prev_action = torch.as_tensor(
        np.repeat(np.asarray(prev, np.float32)[None], count, 0), device=nav.dev)
    actions = torch.as_tensor(np.asarray(cands, np.float32), device=nav.dev)
    curves, progress = [], []
    depth_seq, action_seq = list(nav.dhist), list(nav.ahist)
    with torch.no_grad():
        for model in nav.models:
            depth_k, _hist = history_inputs(
                depth_seq, action_seq, model.frame_offsets, model.hist_actions)
            if len(model.frame_offsets) == 1:
                depth_k = depth_k[0]
            depth = torch.as_tensor(np.repeat(depth_k[None], count, 0), device=nav.dev)
            out = model(depth, goal_feat, prev_action, actions, None)
            curves.append(survival(out["hazard_logit"]).float().cpu().numpy())
            progress.append((-out["dgoal"][..., -1]).float().cpu().numpy())
    return np.mean(curves, 0), np.mean(progress, 0)


# Pooled rows kept clear of the floor line when reading the side cones.
SIDE_FLOOR_MARGIN = 3


def _office_cones(depth56: np.ndarray, floor_row: Optional[int] = None, src_h: Optional[int] = None) -> np.ndarray:
    """Forward, left, right. Side cones stay above the open floor.

    The forward cone keeps the whole middle band. The left and right cones
    stop a few pooled rows above the floor line, so the empty floor there
    is not a side obstacle.
    """
    depth = np.asarray(depth56, np.float32)
    height, width = depth.shape
    top, bot = height // 4, height - height // 4
    left_edge, right_edge = width // 4, width - width // 4

    def lowest(patch: np.ndarray) -> float:
        finite = patch[np.isfinite(patch) & (patch > 0.05)]
        return float(finite.min()) if finite.size else 30.0

    side_bot = bot
    if floor_row is not None and src_h:
        cell = max(1, int(src_h) // int(height))
        cut = int(floor_row) // cell
        kept = cut - SIDE_FLOOR_MARGIN
        if kept > top:
            side_bot = min(kept, bot)
    band = depth[top:bot]
    side = depth[top:side_bot]
    return np.array([
        lowest(band[:, left_edge:right_edge]),
        lowest(side[:, :left_edge]),
        lowest(side[:, right_edge:]),
    ], np.float32)


def _load_models():
    import cv2
    import torch
    from aerial_finder.nav.navigator import ConsequenceNavigator, NavConfig
    from aerial_finder.types import VehicleState
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation

    if not OFFICE_DEPTH.is_file() or not OFFICE_CONS.is_file():
        raise RuntimeError(f"office weights missing: {OFFICE_DEPTH} {OFFICE_CONS}")
    depth = DepthMinPredictor.from_checkpoint(str(OFFICE_DEPTH), device="cuda")
    nav = ConsequenceNavigator(
        [str(OFFICE_CONS)],
        (),
        device="cuda",
        cfg=NavConfig(
            surv_min=0.95,
            fwd_dx=0.8,
            fov_max_deg=35.0,
            seed=0,
            replan_every=1,
            near_goal_dist=0.0,
            goal_surv_min=0.0,
            brake_speed=0.0,
            spin_nopass=False,
            goal_arc_yaw=0.10,
            trim_yaw=0.04,
            trim_bearing_deg=25.0,
            align_bearing_deg=8.0,
            fwd_hold_m=0.0,
            move_dx=0.0,
            near_arc_yaw=0.0,
            families=("fwd", "goal_turn_fwd"),
            step_hz=STEP_HZ,
            guard_turn_m=None,
            guard_stop_m=None,
            lead_m=0.0,
            lead_pass_m=0.0,
            prefer_open=False,
            toward_goal_yaw=True,
            open_side_yaw=0.0,
        ),
    )
    return cv2, torch, depth, nav, VehicleState, Observation


def _env_vector(depth_model, frames, device: str):
    import torch

    model = depth_model._model
    if not hasattr(model, "encode_env"):
        return None
    stack = np.stack(frames, axis=0)
    tensor = torch.from_numpy(stack).unsqueeze(0).to(device)
    with torch.no_grad():
        env = model.encode_env(tensor)
    return env.squeeze(0).float().cpu().numpy()


def main() -> None:
    armed = os.environ.get("AERIAL_ARM", "0") == "1"
    _yolo_alive()
    wire = YhsWire(armed=armed)
    track = _TrackLog()
    try:
        _wait_sensors(wire)
        cv2, _torch, depth, nav, VehicleState, Observation = _load_models()
        from pinhole90 import warp_color

        period = 1.0 / STEP_HZ
        executed = None
        search = _SearchGoal()
        scale_hist: deque = deque(maxlen=7)
        arc_hold = None
        face_job = None
        slow_from = None
        mark_task = ""
        contact = False
        moved = False
        still_since = None
        sent_vx = 0.0
        last_xy = None
        last_yaw = None
        traveled = 0.0
        office_prev = np.zeros(4, np.float32)
        near_low = 0
        passage = None
        passage_clear = 0.0
        office = _load_office()
        while rcl_ok():
            t0 = time.time()
            color, sensor_depth, pose, color_k, color_d, depth_k, _color_at, depth_at, odom_at = wire.snapshot()
            now = time.time()
            fresh = depth_at > 0 and odom_at > 0 and now - depth_at < 0.4 and now - odom_at < 0.4
            if (
                color is None or pose is None or color_k is None or color_d is None
                or depth_k is None or sensor_depth is None or not fresh
            ):
                wire.publish_stop()
                wire.write_status()
                time.sleep(0.05)
                continue
            x, y, z, yaw = pose
            if _goal_already_finished():
                wire.publish_stop()
                wire.inference_ok = True
                wire.error = ""
                wire.write_status({"goal_mode": "idle", "publishing": False})
                time.sleep(period)
                continue
            goal = search.update(x, y, z, yaw)
            if search.task_id != mark_task:
                mark_task = search.task_id
                contact = False
                moved = False
                still_since = None
                sent_vx = 0.0
                last_xy = None
                last_yaw = None
                traveled = 0.0
                slow_from = None
                arc_hold = None
                face_job = None
                office_prev = np.zeros(4, np.float32)
                near_low = 0
                passage = None
                passage_clear = 0.0
                office.reset_passage()
                if search.result != "found":
                    search.mark_box = None
                    _write_mark(None, "", False)
            goal_dist = None if goal is None else float(math.hypot(goal[0] - x, goal[1] - y))
            if goal is not None:
                track.bind(search)
            if goal is None or search.finished:
                wire.inference_ok = True
                wire.error = ""
                wire.publish_stop()
                executed = None
                wire.write_status({
                    "goal_mode": "idle" if goal is None else search.result,
                    "search_target": "" if goal is None else search.target,
                    "search_result": "" if goal is None else search.result,
                    "found_pose": None if goal is None else search.found_pose,
                    "publishing": False,
                })
                time.sleep(period)
                continue
            progress = 0.0 if last_xy is None else math.hypot(x - last_xy[0], y - last_xy[1])
            yaw_step = 0.0 if last_yaw is None else abs(_wrap(yaw - last_yaw))
            if last_xy is not None:
                traveled += progress
            last_xy = (x, y)
            last_yaw = yaw
            moved, still_since, contact = update_contact(
                moved, still_since, progress, sent_vx, now, bool(wire.joystick()), contact,
            )
            if contact:
                wire.publish_stop()
                sent_vx = 0.0
                wire.inference_ok = True
                wire.error = ""
                track.sample(search, x, y, yaw, "contact", wire, nav.last, goal_dist)
                wire.write_status({
                    "goal_mode": "contact",
                    "goal_dist_m": None if goal_dist is None else round(goal_dist, 2),
                    "publishing": False,
                    "search_target": search.target,
                })
                time.sleep(period)
                continue
            rgb, seen = warp_color(color, color_k, color_d)
            pred = depth.predict_depth(Observation(rgb=rgb, state=np.zeros(7, np.float32), collided=False))
            if pred is None:
                wire.error = "depth model returned no map"
                wire.publish_stop()
                wire.write_status()
                time.sleep(period)
                continue
            pred = np.asarray(pred, np.float32)
            if seen.shape != pred.shape[:2]:
                seen = cv2.resize(seen.astype(np.uint8), (pred.shape[1], pred.shape[0]),
                                  interpolation=cv2.INTER_NEAREST) > 0
            ratio, scale_cells = _depth_scale(pred, seen, sensor_depth, depth_k)
            if ratio is not None:
                scale_hist.append(ratio)
            if not scale_hist:
                wire.error = "depth scale not fit"
                wire.publish_stop()
                wire.write_status({"metric_scale": None, "scale_cells": scale_cells})
                time.sleep(period)
                continue
            metric_scale = float(np.median(scale_hist))
            model_depth = np.where(seen, pred * metric_scale, INVALID_M).astype(np.float32)
            fy, cy = float(color_k[1, 1]), float(color_k[1, 2])
            block = lower_obstacle(model_depth, fy, cy, color.shape[0])
            low = face_low(block, low_clearance(model_depth, fy, cy, color.shape[0]))
            ahead_m = nearest_ahead(model_depth, fy, cy, color.shape[0])
            pred, horizon_row = hide_ground(model_depth, fy, cy, color.shape[0])
            frames = list(depth._hist)
            while len(frames) < depth.n_frames:
                frames.insert(0, frames[0])
            env = _env_vector(depth, frames[-depth.n_frames :], depth.device)
            nav.observe(pred, env, _body_step(executed, (x, y, yaw)))
            executed = (x, y, yaw)
            seen_now = _target_sight(color, sensor_depth, search.target)
            if seen_now is None:
                search.miss_sight(False)
                search.mark_box = None
            else:
                search.note_sight(seen_now[0], seen_now[1], seen_now[2])
                search.miss_sight(seen_now[2])
                search.mark_box = seen_now[3]
            shown = _display_box(search.mark_box, search.sight_verified)
            if shown is not None:
                _write_mark(shown, search.target, False)
            elif not search.finished:
                _write_mark(None, "", False)
            # A stated distance stays the goal. A colour hint must not pull the car off that point.
            arrived = search.mode == "point" and goal_dist is not None and goal_dist <= REACH_M
            person_close = (
                search.mode == "along" and search.sight_verified and search.sight is not None and search.sight[1] <= 2.0
            )
            centered = search.sight_verified and search.sight is not None and abs(search.sight[0] - 0.5) < 0.22
            if search.verified_hits >= MIN_VERIFIED and centered and (arrived or person_close):
                _write_mark(_display_box(search.mark_box, True), search.target, True)
                search.finish("found", x, y, yaw)
                wire.publish_stop()
                executed = None
                track.sample(search, x, y, yaw, "found", wire, nav.last, goal_dist)
                wire.write_status({
                    "goal_mode": "found",
                    "goal_dist_m": None if goal_dist is None else round(goal_dist, 2),
                    "search_target": search.target,
                    "search_result": "found",
                    "found_pose": search.found_pose,
                    "publishing": False,
                })
                time.sleep(period)
                continue
            holding = (arc_hold is not None and now < arc_hold[1]) or face_job is not None
            if arrival_scan(search.mode, goal_dist, search.finished, block, holding):
                if search.phase != "scan":
                    search.begin_scan(yaw)
                scan_done = search.note_scan(yaw)
                wire.inference_ok = True
                wire.error = ""
                wire.publish_action(np.array([0.0, 0.0, 0.0, SCAN_DYAW], np.float64))
                sent_vx = wire.sent_speed()
                if scan_done and search.verified_hits < MIN_VERIFIED:
                    _write_mark(None, "", False)
                    search.finish("not_found", x, y, yaw)
                    wire.publish_stop()
                    executed = None
                track.sample(search, x, y, yaw, "scan", wire, nav.last, goal_dist)
                wire.write_status({
                    "metric_scale": round(metric_scale, 3),
                    "scale_cells": scale_cells,
                    "publishing": wire.armed and not search.finished,
                    "goal_mode": search.result or "scan",
                    "goal_dist_m": round(goal_dist, 2),
                    "search_target": search.target,
                    "search_hits": search.verified_hits,
                    "search_result": search.result,
                    "scan_deg": round(math.degrees(search.scan_turned), 1),
                    "found_pose": search.found_pose,
                })
                time.sleep(max(0.0, period - (time.time() - t0)))
                continue
            pos_m, yaw_m, goal_m = _model_frame(x, y, z, yaw, goal)
            state = VehicleState(pos=pos_m, yaw=yaw_m).as_model_state()
            from aerial_finder.nav.consequence_model import goal_bearing_body

            bearing = goal_bearing_body(state[:3], float(state[6]), goal_m)
            names, cands = _office_candidates(nav.H, bearing)
            curve, prog = _office_curves(nav, state, goal_m, cands, office_prev)
            curve0, _prog0 = _office_curves(nav, state, goal_m, cands, np.zeros(4, np.float32))
            cones = _office_cones(
                np.asarray(nav.dhist[-1], np.float32), horizon_row, int(pred.shape[0]))
            surv = office.survival_choice(curve, curve0, float(cones[0]))
            _pick, action, family, n_pass = office.resolve(
                names, cands, surv, np.ones(len(names), dtype=bool), bearing, cones, prog,
                None, None, 0.95)
            passage_cones, low = passage_view(cones, low, sensor_depth)
            side, passage, passage_clear = office.commit_passage(
                passage_cones, low, passage, passage_clear, progress, yaw_step)
            if side is not None:
                action, family = side
            nose_close = family == "stop" and float(cones[0]) < office.NEAR_M
            if side is not None or nose_close:
                near_low += 1
            else:
                near_low = 0
            if near_low == 1:
                action = np.zeros(4, np.float64)
                family = "confirm"
            nav.last = {
                "family": family,
                "surv": None if family in ("stop", "confirm") else round(float(surv[max(_pick, 0)]), 3),
                "n_pass": int(n_pass),
                "bearing_deg": round(math.degrees(bearing), 1),
                "cones_m": [round(float(v), 2) for v in cones],
                "low_m": [None if v is None else round(float(v), 2) for v in low],
                "passage": passage,
                "passage_m": round(float(passage_clear), 2),
            }
            wire.inference_ok = True
            wire.error = ""
            if family in ("stop", "confirm"):
                wire.publish_stop()
                office_prev = np.zeros(4, np.float32)
            else:
                wire.publish_action(action)
                office_prev = np.asarray(action, np.float32)
            sent_vx = wire.sent_speed()
            real_l, real_c, real_r = float(cones[1]), float(cones[0]), float(cones[2])
            track.sample(search, x, y, yaw, family, wire, nav.last, goal_dist)
            finite = sensor_depth[np.isfinite(sensor_depth)]
            finite_model = pred[np.isfinite(pred) & (pred < INVALID_M)]
            wire.write_status({
                "metric_scale": round(metric_scale, 3),
                "scale_cells": scale_cells,
                "sensor_depth_min_m": None if finite.size == 0 else round(float(finite.min()), 3),
                "model_depth_min_m": None if finite_model.size == 0 else round(float(finite_model.min()), 3),
                "env_dim": None if env is None else int(env.shape[0]),
                "nav": nav.last,
                "real_clear_m": [
                    None if real_l is None else round(real_l, 2),
                    None if real_c is None else round(real_c, 2),
                    None if real_r is None else round(real_r, 2),
                ],
                "model_horizon_row": horizon_row,
                "depth_graft": "",
                "office_family": family,
                "publishing": wire.armed and not search.finished,
                "goal_mode": search.result or search.mode,
                "goal_x": round(float(goal[0]), 2),
                "goal_y": round(float(goal[1]), 2),
                "goal_dist_m": None if goal_dist is None else round(goal_dist, 2),
                "search_target": search.target,
                "search_hits": search.verified_hits,
                "search_result": search.result,
                "found_pose": search.found_pose,
            })
            time.sleep(max(0.0, period - (time.time() - t0)))
    finally:
        track.close()
        wire.write_status()
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
