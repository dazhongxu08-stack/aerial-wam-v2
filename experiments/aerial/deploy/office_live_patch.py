"""The restored office rules, with three changes.

A closed corner turns a full circle, then drives toward the more open side.
A close center reading is not its own rule. A close forward cone does not
stop the car by itself.
"""
from __future__ import annotations

import importlib.machinery
import math
import os

_BASE_PATH = os.environ.get(
    "OFFICE_LIVE_BASE",
    "/home/user/rt/app/experiments/aerial/deploy/office_choice.pyc.bak-before-turn-away",
)
_loader = importlib.machinery.SourcelessFileLoader("_office_live", _BASE_PATH)
_base = _loader.load_module()
_orig_release = _base._release
_orig_commit = _base.commit_passage

# A full turn at the in-place rate. 80 frames is 16 s, which is 360° at 0.40 rad/s.
_base.BOXED_SPIN_RAD = math.pi * 2.0
_base.BOXED_SPIN_FRAMES = 80


def _choose(cones, low, boxed, bearing=0.0):
    """Same side rules, without the center rule and without the forward stop."""
    forward, left, right, _center_close = _base._scene(cones, low)
    nearer = min(left, right)
    prefer = _base._more_open(left, right)
    both_near = left < _base.NEAR_M and right < _base.NEAR_M and forward < _base.NEAR_M
    both_body = left < _base.BODY_M and right < _base.BODY_M and forward < _base.NEAR_M
    if both_body or (boxed and both_near):
        side = prefer or _base._larger(left, right)
        action, family = _base._emit(forward, side, True)
        return action, family, True, True
    if left < _base.BODY_M and right < _base.BODY_M:
        side = prefer or _base._larger(left, right)
        action, family = _base._emit(forward, side, False)
        return action, family, False, False
    if nearer < _base.BODY_M:
        side = prefer or _base._larger(left, right)
        action, family = _base._emit(forward, side, True)
        return action, family, True, False
    if nearer < _base.SIDE_OPEN_M and prefer is not None:
        action, family = _base._emit(forward, prefer, False)
        return action, family, False, False
    return _base._toward_goal(forward, bearing)


def _drive_open(forward, left, right):
    """Forward toward the side that has more room."""
    side = _base._more_open(left, right) or _base._larger(left, right)
    return _base._emit(forward, side, False)


def _corner_blocked(view) -> bool:
    forward, left, right = view
    return left < _base.NEAR_M and right < _base.NEAR_M and forward < _base.NEAR_M


def _release(action, family, in_place, boxed_now, yaw_step):
    exhausted = _base._spin_exhausted(family, in_place, boxed_now, yaw_step)
    was_stuck = bool(getattr(_base.commit_passage, "_stuck", False))
    out = _orig_release(action, family, in_place, boxed_now, yaw_step)
    view = getattr(_base.commit_passage, "_view", None)
    if view is None or not _corner_blocked(view) or not (exhausted or was_stuck):
        return out
    # A finished spin must keep the latch, or the next frame starts another circle.
    _base.commit_passage._stuck = True
    if exhausted or family == "stop":
        drive, drive_family = _drive_open(*view)
        _base.commit_passage._spin = 0.0
        _base.commit_passage._spin_frames = 0
        return _base._remember(drive, drive_family, False, True)
    return out


def commit_passage(cones, low, passage, clear_m=0.0, step_m=0.0, yaw_step=0.0):
    forward, left, right, _center = _base._scene(cones, low)
    _base.commit_passage._view = (float(forward), float(left), float(right))
    return _orig_commit(cones, low, passage, clear_m, step_m, yaw_step)


_base._choose = _choose
_base._release = _release
_base.commit_passage._view = None
_commit = commit_passage

for _name in dir(_base):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_base, _name)
# The loop above rebinds this module's commit_passage to the original.
# The wire calls this module, so put the wrapper back. The original stays
# in _base, which is where reset_passage keeps the spin latch.
globals()["commit_passage"] = _commit
