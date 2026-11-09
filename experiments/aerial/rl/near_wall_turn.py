"""Frames where the aircraft is still east of the west gap and the nose is in a near wall.

Route 0's open-street frames and its final turn toward the goal are outside
this set. The actor update may change an action only inside the set.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from experiments.aerial.rl.gap_passage import gap_action


def near_wall_east(
    x: float,
    y: float,
    fwd_m: float,
    *,
    near_m: float = 20.0,
    gap_x: float = -960.0,
    north_of_m: float = 42.0,
) -> bool:
    """True only for a near wall east of the gap and north of the south crossing."""
    if not np.isfinite(fwd_m) or float(fwd_m) >= float(near_m):
        return False
    if float(x) <= float(gap_x):
        return False
    if float(y) <= float(north_of_m):
        return False
    return True


def west_turn_action(pos, yaw: float) -> np.ndarray:
    """Body delta that yaws west, then flies west. Altitude stays where it is."""
    goal = np.asarray(pos, dtype=np.float64).reshape(3).copy()
    goal[1] = float(goal[1]) - 80.0
    action, phase = gap_action(pos, float(yaw), goal, z_hold=float(pos[2]))
    if phase != "west":
        raise ValueError(f"west turn requested outside the west phase ({phase})")
    return np.asarray(action, dtype=np.float64)


class WallBitPassage:
    """Start the side passage only on image codes route 0 never produced.

    The actor weights stay frozen. Once a reserved bit is on, fly the measured
    west opening until the south-street handoff, then stay out of the way.
    """

    def __init__(self, bits: np.ndarray):
        self.bits = np.asarray(bits, dtype=np.int64).reshape(-1)
        self.reset()

    def reset(self) -> None:
        self.latched = False
        self.finished = False
        self.steps = 0
        self.phase = ""

    @classmethod
    def load(cls, path: str | Path) -> "WallBitPassage":
        bits = np.load(path)["bits"]
        return cls(bits)

    def apply(self, z, pos, yaw: float, goal):
        if self.finished or z is None:
            return None
        code = np.asarray(z, dtype=np.float64).reshape(-1) > 0.5
        if not self.latched:
            if self.bits.size == 0 or int(self.bits.max()) >= code.size:
                return None
            if not bool(code[self.bits].any()):
                return None
            self.latched = True
        self.steps += 1
        x = float(pos[0])
        y = float(pos[1])
        yaw_f = float(yaw)
        south_err = (yaw_f + math.pi / 2.0 + math.pi) % (2.0 * math.pi) - math.pi
        if x >= -948.0 and y <= 26.0 and abs(south_err) < 0.45:
            self.latched = False
            self.finished = True
            self.phase = "release"
            return None
        z_hold = max(float(np.asarray(pos).reshape(-1)[2]), 29.0)
        action, phase = gap_action(pos, yaw_f, goal, z_hold=z_hold)
        self.phase = phase
        return np.asarray(action, dtype=np.float64), phase
