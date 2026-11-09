#!/usr/bin/env python3
"""Office steering.

Go around toward the more open side when the nearer side is inside 2 m.
A side or the center inside 1 m has no forward speed, and the yaw never
points into that side. A moving detour is a nudge, not a full lock.
Spin in place while both sides and the nose stay inside 1.25 m. Stop that
spin after half a turn, and also stop turning further away once the goal
is 50° off the nose. If the goal side itself is inside 1 m, keep turning
the other way until that side clears, and stop at 100°. Turning back
toward the goal waits until that side is outside 1 m. A clear view drives and nudges toward the goal. It does
not spin in place to chase a goal. A new command waits three frames,
except a command that would drive or yaw into a side inside 1 m.
"""
from __future__ import annotations

import math

import numpy as np

HOLD_RAD = math.radians(25.0)
NEAR_M = 1.25
SLOW_M = 2.05
SIDE_SLOW_M = 1.6
OPEN_DA = 1.48
PROMOTE_TAIL = 0.80
SLOW_DX = 0.35
CRUISE_DX = 0.8
ARC_DYAW = 0.3141592653589793
NUDGE_DYAW = 0.10
SIDE_OPEN_M = 2.0
FORWARD_LIFT_M = 2.2
BODY_M = 1.0
ESCAPE_MARGIN_M = 0.12
ESCAPE_FLIP_FRAMES = 3
# Below this, a clear view drives straight. 4° over 15 m stays inside 1.5 m.
AIM_RAD = math.radians(4.0)
# Past this, more yaw away from the goal is what spun the heading around.
TURN_AWAY_RAD = math.radians(50.0)
# The goal side is on the body, so turning back would hit it. Keep turning
# toward the open side only until here, then stop.
CLEAR_AWAY_RAD = math.radians(100.0)
# After that side leaves the body, keep going straight this many frames
# before aiming at the goal again. Looking away is not the same as passing it.
SKIRT_FRAMES = 15
# A low band that also reads far inside this window is a flicker, not a box.
FLICKER_FRAMES = 8
FLICKER_FAR_M = 1.5
# In-place look in a closed corner. 40 frames is half a turn at 5 Hz.
BOXED_SPIN_FRAMES = 40
BOXED_SPIN_MIN_FRAMES = 5
BOXED_SPIN_RAD = math.pi
# Kept for the older closed-loop script. This rule does not use it.
ARC_HOLD_RAD = 0.8


def lift_false_floor(depth: np.ndarray) -> np.ndarray:
    """Lift a too-close forward floor when both sides are already open.

    The gap keeps its close side cones, so this does not move that reading.
    """
    d = np.array(depth, np.float32, copy=True)
    h, w = d.shape
    if h < 8 or w < 8:
        return d
    band = slice(h // 4, h - h // 4)
    c0, c1 = w // 4, w - w // 4
    if float(np.min(d[band, :c0])) < SIDE_OPEN_M or float(np.min(d[band, c1:])) < SIDE_OPEN_M:
        return d
    fwd = d[band, c0:c1]
    if float(np.min(fwd)) >= SIDE_OPEN_M:
        return d
    lifted = fwd.copy()
    short = lifted < SIDE_OPEN_M
    lifted[short] = np.maximum(lifted[short], FORWARD_LIFT_M)
    d[band, c0:c1] = lifted
    return d


def survival_choice(curve, curve0, da_f: float) -> np.ndarray:
    """Same open-floor score the simulator uses before it picks a step."""
    curve = np.asarray(curve, np.float64)
    if curve0 is None or float(da_f) < OPEN_DA or curve.shape[-1] < 6:
        return curve[..., -1]
    curve0 = np.asarray(curve0, np.float64)
    use0 = curve0[..., -1] >= curve[..., -1]
    best = np.where(use0[..., None], curve0, curve)
    surv = best[..., -1].copy()
    short = best[..., 5]
    promote = (surv < 0.95) & (short >= 0.95) & (
        (best[..., -1] >= PROMOTE_TAIL) | (float(da_f) >= SIDE_SLOW_M)
    )
    return np.where(promote, short, surv)


def _cones3(cones):
    vals = np.atleast_1d(np.asarray(cones, np.float64)).reshape(-1)
    forward = float(vals[0])
    left_mid = float(vals[1]) if vals.size > 1 else NEAR_M
    right_mid = float(vals[2]) if vals.size > 2 else NEAR_M
    return forward, left_mid, right_mid


def _drop_flicker(low, cones):
    """Drop a close low band that just read far, while that side's cone stays open.

    A box that stays close still counts. A cone inside 1 m still counts.
    One jump from 2 m to 0.6 m is not an obstacle.
    """
    unpacked = _unpack_low(low)
    if unpacked is None:
        return low
    _forward, left_cone, right_cone = _cones3(cones)
    hist = getattr(commit_passage, "_low_hist", None)
    if not isinstance(hist, list) or len(hist) != 3:
        hist = [[], [], []]
        commit_passage._low_hist = hist
    out = [None if v is None else float(v) for v in unpacked]
    cones_for = (left_cone, None, right_cone)
    for index in (0, 2):
        hist[index].append(out[index])
        del hist[index][:-FLICKER_FRAMES]
        cone = cones_for[index]
        if cone is not None and float(cone) < BODY_M:
            continue
        jumped = any(v is not None and v >= FLICKER_FAR_M for v in hist[index])
        if not jumped:
            continue
        if out[index] is None or out[index] >= BODY_M:
            continue
        out[index] = None
        side = "left" if index == 0 else "right"
        if getattr(commit_passage, "_skirt", None) == side:
            commit_passage._skirt = None
            commit_passage._skirt_n = 0
    return tuple(out)


def _unpack_low(low):
    if low is None:
        return None
    vals = list(low)
    if len(vals) != 3:
        return None
    return vals[0], vals[1], vals[2]


def _nearer(low, cone) -> float:
    """The side is as close as its nearer reading. A far cone does not open it."""
    if low is None:
        return float(cone)
    return min(float(low), float(cone))


def _scene(cones, low):
    forward, left_cone, right_cone = _cones3(cones)
    unpacked = _unpack_low(low)
    if unpacked is None:
        left_low = center = right_low = None
    else:
        left_low, center, right_low = unpacked
    left = _nearer(left_low, left_cone)
    right = _nearer(right_low, right_cone)
    center_close = center is not None and float(center) < BODY_M
    return forward, left, right, None if center is None else float(center), center_close


def _more_open(left: float, right: float):
    """The side with more room. A tie has no side to prefer."""
    if left >= right + ESCAPE_MARGIN_M:
        return "left"
    if right >= left + ESCAPE_MARGIN_M:
        return "right"
    return None


def _wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _emit(forward: float, side: str, in_place: bool, yaw: float = None):
    """In-place spin uses a full lock. A moving detour only nudges."""
    del forward
    sign = -1.0 if side == "left" else 1.0
    family = "arc_left" if side == "left" else "arc_right"
    action = np.zeros(4, np.float64)
    amount = ARC_DYAW if in_place else NUDGE_DYAW
    action[3] = sign * (amount if yaw is None else float(yaw))
    if not in_place:
        action[0] = SLOW_DX
    return action, family


def _larger(left: float, right: float) -> str:
    return "left" if left >= right else "right"


def _turning_away(side: str, bearing: float) -> bool:
    """True when this yaw would take the nose still further from the goal."""
    bearing = _wrap(bearing)
    if abs(bearing) < TURN_AWAY_RAD:
        return False
    return (bearing > 0.0 and side == "left") or (bearing < 0.0 and side == "right")


def _goal_side(bearing: float) -> str:
    return "right" if _wrap(bearing) > 0.0 else "left"


def _side_clear(side: str, left: float, right: float) -> float:
    return left if side == "left" else right


def _yaw_hits_body(yaw: float, left: float, right: float) -> bool:
    """Yaw toward a side that is already on the body."""
    if yaw < 0.0 and left < BODY_M and not (right < BODY_M and left >= right):
        return True
    if yaw > 0.0 and right < BODY_M and not (left < BODY_M and right >= left):
        return True
    return False


def _unsafe(action, left: float, right: float, center_close: bool) -> bool:
    """Forward speed, or a yaw into the body, while something is inside 1 m."""
    if float(action[0]) > 0.0 and (left < BODY_M or right < BODY_M or center_close):
        return True
    return _yaw_hits_body(float(action[3]), left, right)


def _drive_ahead(forward: float):
    """Keep the current heading. The detour has already turned far enough."""
    if forward < NEAR_M:
        return (np.zeros(4, np.float64), "stop", False, False)
    action = np.zeros(4, np.float64)
    action[0] = CRUISE_DX if forward >= SLOW_M else SLOW_DX
    return (action, "through", False, False)


def _face_blocked(forward: float, bearing: float):
    """Nose blocked, and the goal side is outside 1 m. Turn toward the goal."""
    bearing = _wrap(bearing)
    if abs(bearing) < AIM_RAD:
        return _drive_ahead(forward)
    side = _goal_side(bearing)
    in_place = forward < NEAR_M
    action, family = _emit(forward, side, in_place, ARC_DYAW if in_place else NUDGE_DYAW)
    if not in_place and forward >= SLOW_M:
        action[0] = CRUISE_DX
    return action, family, in_place, False


def _hold_recover(bearing, forward, left, right, center_close, latch: bool) -> bool:
    """Keep the turn back only while it does not point into the body."""
    if not latch or not bool(getattr(commit_passage, "_recover", False)):
        return False
    goal = _goal_side(bearing)
    if _side_clear(goal, left, right) < BODY_M or center_close:
        commit_passage._recover = False
        return False
    if (
        abs(_wrap(bearing)) < AIM_RAD
        and forward >= NEAR_M
        and left >= BODY_M
        and right >= BODY_M
    ):
        commit_passage._recover = False
        return False
    return True


def _remember_skirt(bearing, left, right, latch: bool) -> None:
    """The goal side is on the body. Do not treat a later open reading as clear."""
    if not latch or abs(_wrap(bearing)) < TURN_AWAY_RAD:
        return
    goal = _goal_side(bearing)
    if _side_clear(goal, left, right) < BODY_M:
        commit_passage._skirt = goal
        commit_passage._skirt_n = 0


def _skirting(bearing, forward, left, right, center_close, latch: bool):
    """Pass the goal-side obstacle before turning back toward it.

    A spin that looks away makes that side read open. Turning back then puts
    the same obstacle on the body again, and the car rocks in place.
    """
    side = getattr(commit_passage, "_skirt", None) if latch else None
    if side not in ("left", "right"):
        return None
    if (
        _side_clear(side, left, right) < BODY_M
        or center_close
        or min(left, right) < BODY_M
    ):
        commit_passage._skirt_n = 0
        return None
    if abs(_wrap(bearing)) < TURN_AWAY_RAD:
        commit_passage._skirt = None
        commit_passage._skirt_n = 0
        return None
    if forward < NEAR_M:
        return _stop_cmd()
    opened = int(getattr(commit_passage, "_skirt_n", 0)) + 1
    commit_passage._skirt_n = opened
    if opened >= SKIRT_FRAMES and _side_clear(side, left, right) >= 1.2:
        commit_passage._skirt = None
        commit_passage._skirt_n = 0
        return None
    action = np.zeros(4, np.float64)
    action[0] = SLOW_DX
    return action, "through", False, False


def _away_blocked(forward, side, bearing, left, right, latch: bool):
    """A blocked nose does not keep turning away, and does not turn into the body."""
    if not (_turning_away(side, bearing) and forward < NEAR_M):
        return None
    goal = _goal_side(bearing)
    if _side_clear(goal, left, right) < BODY_M:
        return _stop_cmd()
    if latch:
        commit_passage._recover = True
    return _face_blocked(forward, bearing)


def _toward_goal(forward: float, bearing: float):
    """The way is clear. Drive, and nudge toward the goal.

    Once the goal is beside or behind the nose, driving that heading is what
    pushed the box. Turn in place until it is back in front.
    """
    bearing = _wrap(bearing)
    if abs(bearing) >= math.radians(90.0):
        side = _goal_side(bearing)
        action, family = _emit(forward, side, True)
        return action, family, True, False
    if abs(bearing) < AIM_RAD:
        return _drive_ahead(forward)
    side = "right" if bearing > 0.0 else "left"
    action, family = _emit(forward, side, False, NUDGE_DYAW)
    if forward >= SLOW_M:
        action[0] = CRUISE_DX
    return action, family, False, False


def _choose(cones, low, boxed: bool, bearing: float = 0.0, latch: bool = False):
    """Return the action, its family, whether it is in place, and the boxed latch."""
    forward, left, right, center, center_close = _scene(cones, low)
    nearer = min(left, right)
    prefer = _more_open(left, right)
    if latch and nearer >= BODY_M:
        commit_passage._body_side = None
    both_near = left < NEAR_M and right < NEAR_M and forward < NEAR_M
    both_body = left < BODY_M and right < BODY_M and forward < NEAR_M
    if both_body or (boxed and both_near):
        side = prefer or _larger(left, right)
        if _turning_away(side, bearing):
            return _stop_cmd()
        action, family = _emit(forward, side, True)
        return action, family, True, True
    if left < BODY_M and right < BODY_M:
        # Both walls are on the body. Spin; do not drive between them.
        side = prefer or _larger(left, right)
        if _turning_away(side, bearing):
            return _stop_cmd()
        action, family = _emit(forward, side, True)
        return action, family, True, False
    if _hold_recover(bearing, forward, left, right, center_close, latch):
        return _face_blocked(forward, bearing)
    if nearer < BODY_M:
        _remember_skirt(bearing, left, right, latch)
        side = prefer or _larger(left, right)
        blocked = _away_blocked(forward, side, bearing, left, right, latch)
        if blocked is not None:
            return blocked
        if _turning_away(side, bearing):
            # Turning back would point into the goal side. Turn the other way
            # until that side leaves the body.
            goal = _goal_side(bearing)
            if (
                _side_clear(goal, left, right) < BODY_M
                and side != goal
                and abs(_wrap(bearing)) < CLEAR_AWAY_RAD
            ):
                action, family = _emit(forward, side, True)
                return action, family, True, False
            return _stop_cmd()
        # A side inside 1 m gets no forward speed. Driving here pushed the
        # blue obstacle. If the turn sweeps that obstacle onto the side we
        # are yawing toward, stop instead of reversing into a rock.
        if forward >= NEAR_M and not center_close and latch:
            saved = getattr(commit_passage, "_body_side", None)
            if saved not in (None, side) and _side_clear(saved, left, right) < BODY_M:
                return _stop_cmd()
            commit_passage._body_side = side
        action, family = _emit(forward, side, True)
        return action, family, True, False
    if center_close:
        goal = _goal_side(bearing)
        # The nose is on the box. Face the goal when that side is clear.
        if _side_clear(goal, left, right) >= BODY_M and abs(_wrap(bearing)) >= AIM_RAD:
            action, family = _emit(forward, goal, True)
            return action, family, True, False
        if prefer is not None and not _turning_away(prefer, bearing):
            action, family = _emit(forward, prefer, True)
            return action, family, True, False
        if (
            prefer is not None
            and _side_clear(goal, left, right) < BODY_M
            and abs(_wrap(bearing)) < CLEAR_AWAY_RAD
        ):
            action, family = _emit(forward, prefer, True)
            return action, family, True, False
        return _stop_cmd()
    # A box across the nose inside 1.6 m. The forward cone can look over it
    # and the cruise then pushes the box.
    if center is not None and center < SIDE_SLOW_M and center <= min(left, right):
        if prefer is not None and not _turning_away(prefer, bearing):
            action, family = _emit(forward, prefer, True)
            return action, family, True, False
        return _stop_cmd()
    skirted = _skirting(bearing, forward, left, right, center_close, latch)
    if skirted is not None:
        return skirted
    if nearer < SIDE_OPEN_M and prefer is not None:
        blocked = _away_blocked(forward, prefer, bearing, left, right, latch)
        if blocked is not None:
            return blocked
        if _turning_away(prefer, bearing):
            # The heading is already away from the goal and a side is inside 2 m.
            # Driving that heading is what pushed the obstacle.
            action, family = _emit(forward, _goal_side(bearing), True)
            return action, family, True, False
        action, family = _emit(forward, prefer, False)
        return action, family, False, False
    if forward < NEAR_M:
        # A head-on obstacle with an open side used to sit forever.
        if prefer is not None and not _turning_away(prefer, bearing):
            action, family = _emit(forward, prefer, True)
            return action, family, True, False
        return (np.zeros(4, np.float64), "stop", False, False)
    return _toward_goal(forward, bearing)


def _yaw_sign(family) -> int:
    if family == "arc_left":
        return -1
    if family == "arc_right":
        return 1
    return 0


def side_passage(cones, low):
    """One frame of the detour. The goal bearing is not part of it."""
    action, family, _in_place, _boxed = _choose(cones, low, False, 0.0)
    return action, family


def _stop_cmd():
    return np.zeros(4, np.float64), "stop", False, False


def _command_sig(action, family, in_place, boxed) -> tuple:
    return (
        family,
        round(float(action[0]), 3),
        round(float(action[3]), 3),
        bool(in_place),
        bool(boxed),
    )


def reset_passage() -> None:
    """Drop the yaw memory. A new task calls this; a straight frame does not."""
    commit_passage._bearing = 0.0
    commit_passage._boxed = False
    commit_passage._held = None
    commit_passage._pending = None
    commit_passage._pending_n = 0
    commit_passage._stuck = False
    commit_passage._recover = False
    commit_passage._body_side = None
    commit_passage._skirt = None
    commit_passage._skirt_n = 0
    commit_passage._low_hist = [[], [], []]
    commit_passage._spin = 0.0
    commit_passage._spin_frames = 0


def _remember(action, family, in_place, boxed_now):
    action = np.array(action, np.float64, copy=True)
    commit_passage._held = (
        _command_sig(action, family, in_place, boxed_now),
        action,
        family,
        bool(in_place),
        bool(boxed_now),
    )
    commit_passage._boxed = bool(boxed_now)
    kept = family if _yaw_sign(family) else None
    return (action.copy(), family), kept, 0.0


def _spin_exhausted(family, in_place, boxed_now, yaw_step: float) -> bool:
    if not (boxed_now and in_place and _yaw_sign(family)):
        return False
    frames = int(getattr(commit_passage, "_spin_frames", 0)) + 1
    spun = float(getattr(commit_passage, "_spin", 0.0)) + abs(float(yaw_step))
    if frames >= BOXED_SPIN_FRAMES:
        return True
    return frames >= BOXED_SPIN_MIN_FRAMES and spun >= BOXED_SPIN_RAD


def _release(action, family, in_place, boxed_now, yaw_step: float):
    """Send this command. A boxed spin that has already looked halfway becomes a stop."""
    if _spin_exhausted(family, in_place, boxed_now, yaw_step):
        commit_passage._stuck = True
        commit_passage._spin = 0.0
        commit_passage._spin_frames = 0
        commit_passage._pending = None
        commit_passage._pending_n = 0
        return _remember(*_stop_cmd())
    if boxed_now and in_place and _yaw_sign(family):
        commit_passage._spin = float(getattr(commit_passage, "_spin", 0.0)) + abs(float(yaw_step))
        commit_passage._spin_frames = int(getattr(commit_passage, "_spin_frames", 0)) + 1
    else:
        commit_passage._spin = 0.0
        commit_passage._spin_frames = 0
        if family != "stop":
            commit_passage._stuck = False
    return _remember(action, family, in_place, boxed_now)


def commit_passage(cones, low, passage, clear_m: float = 0.0, step_m: float = 0.0,
                   yaw_step: float = 0.0):
    """Keep the current command until a different one has held for three frames."""
    del clear_m, step_m
    bearing = float(getattr(commit_passage, "_bearing", 0.0))
    boxed = bool(getattr(commit_passage, "_boxed", False))
    stuck = bool(getattr(commit_passage, "_stuck", False))
    low = _drop_flicker(low, cones)
    forward, left, right, center, center_close = _scene(cones, low)
    blocked = left < NEAR_M and right < NEAR_M and forward < NEAR_M
    if stuck and blocked:
        action, family, in_place, boxed_now = _stop_cmd()
    else:
        action, family, in_place, boxed_now = _choose(cones, low, boxed, bearing, True)
    sig = _command_sig(action, family, in_place, boxed_now)
    held = getattr(commit_passage, "_held", None)
    if held is None and passage in ("arc_left", "arc_right") and passage != family:
        side = "left" if passage == "arc_left" else "right"
        seed, seed_family = _emit(forward, side, in_place)
        held = (
            _command_sig(seed, seed_family, in_place, False),
            seed,
            seed_family,
            in_place,
            False,
        )
        commit_passage._held = held
    held_yaw = 0 if held is None else _yaw_sign(held[2])
    chosen_yaw = _yaw_sign(family)
    reversing = held is not None and held_yaw != 0 and chosen_yaw != 0 and held_yaw != chosen_yaw
    dropping = held is not None and float(held[1][0]) > 0.0 and float(action[0]) == 0.0
    if held is not None and (_unsafe(held[1], left, right, center_close) or dropping) and not reversing:
        commit_passage._pending = None
        commit_passage._pending_n = 0
        return _release(action, family, in_place, boxed_now, yaw_step)
    if held is None or sig == held[0]:
        commit_passage._pending = None
        commit_passage._pending_n = 0
        return _release(action, family, in_place, boxed_now, yaw_step)
    pending = getattr(commit_passage, "_pending", None)
    if pending == sig:
        commit_passage._pending_n = int(getattr(commit_passage, "_pending_n", 0)) + 1
    else:
        commit_passage._pending = sig
        commit_passage._pending_n = 1
    if int(commit_passage._pending_n) >= ESCAPE_FLIP_FRAMES:
        commit_passage._pending = None
        commit_passage._pending_n = 0
        return _release(action, family, in_place, boxed_now, yaw_step)
    _sig_h, action_h, family_h, in_place_h, boxed_h = held
    if _unsafe(action_h, left, right, center_close):
        stopped, stop_family = _stop_cmd()[:2]
        return (stopped, stop_family), None, 0.0
    return _release(action_h, family_h, in_place_h, boxed_h, yaw_step)


def hold_arc(cones, low, hold, turned):
    """Older script hook. A clear pair of sides releases. A close side still yaws."""
    del hold, turned
    action, family = side_passage(cones, low)
    if family in ("arc_left", "arc_right"):
        return action, family
    return None


def resolve(names, cands, surv, allowed, bearing, da_f, prog, select_candidate, choose_align,
            surv_min: float = 0.95) -> tuple[int, np.ndarray, str, int]:
    """Record the goal bearing and return the same command, using the cones alone."""
    del cands, allowed, prog, select_candidate, choose_align
    bearing = float(bearing)
    commit_passage._bearing = bearing
    n_pass = int((np.asarray(surv, np.float64) >= float(surv_min)).sum())
    action, family, _in_place, _boxed = _choose(da_f, None, False, bearing)
    index = -1
    for i, name in enumerate(names):
        if name == family or (family == "through" and name == "fwd"):
            index = i
            break
    return index, action, family, n_pass
