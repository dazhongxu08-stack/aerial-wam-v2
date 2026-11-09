"""Teacher for the route-0/1 building: hold the west gap until level with the goal.

The measured opening is west of x=-960. Turning toward the goal while still
north of it cuts back into the slab. Stay on that line until the goal's
latitude, then fly across. This module only produces the demonstration.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Deployed body box at 5 Hz: (dx, dy, dz, dyaw).
LIMITS = np.array([1.0, 0.4, 0.4, 0.3141592653589793], dtype=np.float64)
GAP_X = -960.0
# Street center south of the first slab. The goal x sits in the next row.
STREET_X = -945.5
# Latitude where the eastward crossing past the first slab was open.
CROSS_Y = 22.0
LEVEL_M = 4.0


def _wrap(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def passage_phase(
    pos,
    goal,
    *,
    gap_x: float = GAP_X,
    street_x: float = STREET_X,
    cross_y: float = CROSS_Y,
    level_m: float = LEVEL_M,
) -> Tuple[float, str]:
    """Return ``(target_yaw, phase)`` for one pose. Yaw is world radians."""
    x = float(pos[0])
    y = float(pos[1])
    gx = float(goal[0])
    gy = float(goal[1])
    # Reach the west opening while still north of the slab.
    if y > cross_y + 20.0 and x > float(gap_x):
        return math.pi, "west"
    # South through the opening down to the latitude where east was open.
    if x <= float(gap_x) + 2.0 and y > float(cross_y):
        return -math.pi / 2.0, "south"
    # Turn east in place on that latitude. A few metres higher is still the slab.
    if (
        x < float(street_x) - 1.0
        and float(cross_y) - 3.0 <= y <= float(cross_y) + 2.0
    ):
        return 0.0, "east"
    if y > gy + float(level_m):
        return -math.pi / 2.0, "south"
    return math.atan2(gy - y, gx - x), "goal"


def gap_action(
    pos,
    yaw: float,
    goal,
    z_hold: float,
    *,
    gap_x: float = GAP_X,
    street_x: float = STREET_X,
    cross_y: float = CROSS_Y,
    level_m: float = LEVEL_M,
) -> Tuple[np.ndarray, str]:
    """Body delta that turns onto the passage heading, then flies it. Holds z."""
    target, phase = passage_phase(
        pos, goal, gap_x=gap_x, street_x=street_x, cross_y=cross_y, level_m=level_m,
    )
    err = _wrap(target - float(yaw))
    dyaw = float(np.clip(err, -LIMITS[3], LIMITS[3]))
    dx = 0.0 if abs(err) > 0.35 else float(LIMITS[0])
    z = float(pos[2])
    if z < float(z_hold) - 0.4:
        dz = float(LIMITS[2])
    elif z > float(z_hold) + 0.4:
        dz = -float(LIMITS[2])
    else:
        dz = 0.0
    return np.array([dx, 0.0, dz, dyaw], dtype=np.float64), phase


def _z_hold_dz(z: float, z_hold: float) -> float:
    if z < float(z_hold) - 0.4:
        return float(LIMITS[2])
    if z > float(z_hold) + 0.4:
        return -float(LIMITS[2])
    return 0.0


_close_latched = False


def reset_close_passage() -> None:
    """Clear the one-shot latch at the start of a recording."""
    global _close_latched
    _close_latched = False


def close_then_gap_action(
    pos,
    yaw: float,
    goal,
    z_hold: float,
    depth,
    *,
    close_m: float = 20.0,
    press_yaw: float = -2.094395160675049,
) -> Tuple[np.ndarray, str]:
    """Face the facade until forward clearance is inside ``close_m``, then the passage.

    The west-phase box is still north of the slab. Pressing the spawn heading
    there brings the facade inside 20 m before the nose turns down the open street.
    """
    from experiments.aerial.rl.depth_geometry import approach_clearances

    global _close_latched
    if _close_latched:
        return gap_action(pos, yaw, goal, z_hold)
    _target, phase = passage_phase(pos, goal)
    fwd = float("inf")
    if depth is not None and phase == "west":
        arr = np.asarray(depth, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2:
            fwd = float(approach_clearances(arr, center_frac=0.12)[0])
    if phase == "west" and fwd < float(close_m):
        _close_latched = True
        return gap_action(pos, yaw, goal, z_hold)
    if phase == "west" and fwd >= float(close_m):
        err = _wrap(float(press_yaw) - float(yaw))
        dyaw = float(np.clip(err, -LIMITS[3], LIMITS[3]))
        dx = 0.0 if abs(err) > 0.35 else float(LIMITS[0])
        dz = _z_hold_dz(float(pos[2]), z_hold)
        return np.array([dx, 0.0, dz, dyaw], dtype=np.float64), "close"
    return gap_action(pos, yaw, goal, z_hold)
