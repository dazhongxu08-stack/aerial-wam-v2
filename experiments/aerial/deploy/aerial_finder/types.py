from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera; pixel (u, v) = (cx + fx * X/Z, cy + fy * Y/Z), camera x right, y down."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "Intrinsics":
        f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(f, f, 0.5 * width, 0.5 * height, int(width), int(height))

    def scaled(self, width: int, height: int) -> "Intrinsics":
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, int(width), int(height))


@dataclass
class VehicleState:
    """World frame x north, y east, z up; yaw from north toward east (rad)."""

    pos: np.ndarray
    yaw: float
    roll: float = 0.0
    pitch: float = 0.0
    collided: bool = False
    t: float = 0.0

    def as_model_state(self) -> np.ndarray:
        """The consequence model's 7-vector [x, y, z, vx, vy, vz, yaw]."""
        return np.array([*self.pos[:3], 0.0, 0.0, 0.0, self.yaw], dtype=np.float64)


@dataclass
class Frame:
    rgb: np.ndarray
    rgb_intr: Intrinsics
    depth: Optional[np.ndarray] = None
    depth_intr: Optional[Intrinsics] = None
    valid: Optional[np.ndarray] = None
    t: float = 0.0


@dataclass
class Detection:
    label: str
    prompt: str
    bbox: Tuple[float, float, float, float]
    conf: float
    verified: bool
    color_frac: float = 0.0
    source: str = "yolo"
    world: Optional[np.ndarray] = None
    range_m: Optional[float] = None
    extra: dict = field(default_factory=dict)
