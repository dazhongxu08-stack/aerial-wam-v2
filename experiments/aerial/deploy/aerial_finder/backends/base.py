from __future__ import annotations

import math
from typing import Optional, Protocol, Tuple

import numpy as np

from ..geo import GeoRef
from ..types import Frame, VehicleState

# Per-step body displacement caps at 5 Hz (m, m, m, rad); same as the model's LIMITS.
STEP_LIMITS = np.array([1.0, 0.4, 0.4, 0.3141592653589793])


class Backend(Protocol):
    geo_ref: GeoRef
    step_hz: float

    def observe(self) -> Tuple[Frame, VehicleState]: ...

    def step(self, action: np.ndarray) -> None:
        """Execute a body delta (dx fwd, dy, dz up, dyaw) over 1 / step_hz."""

    def hover(self) -> None: ...

    def close(self) -> None: ...


def body_delta_to_world_velocity(delta: np.ndarray, yaw: float, dt: float) -> Tuple[float, float, float, float]:
    """(v_north, v_east, v_up, yaw_rate) for a body delta; identical rotation to the training env."""
    dx, dy, dz, dyaw = (float(v) for v in np.clip(np.asarray(delta, np.float64), -STEP_LIMITS, STEP_LIMITS))
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * dx - s * dy) / dt, (s * dx + c * dy) / dt, dz / dt, dyaw / dt


def maybe_float(v) -> Optional[float]:
    return None if v is None else float(v)
