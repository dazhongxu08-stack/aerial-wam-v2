"""Command -> body-velocity response of the Lumos LUD dog under the H12 bridge.

The dog only takes high-level ``(vx, vyaw)`` commands; its gait controller is a
black box. Each channel is modelled as: pure delay -> deadband -> static gain ->
first-order lag, integrated at ``tick_hz``. Defaults: yaw onset 0.1-0.2 s and a deadband near 0.2 rad/s are from the
2026-10-19 bridge runs. The 2026-10-20 tape measured forward at 0.57 of the
command (0.40 m in 2.8 s at 0.25 m/s). The same session's IMU measured yaw at
about half of a 0.40 rad/s command. ``DogDynamicsRanges.sample`` draws
per-episode parameters so a model trained on the data covers that uncertainty.

Model actions stay in the consequence model's box (``LIMITS`` of
``collect_consequence_dataset``): ``dx`` in [0, 1] maps to ``vx`` in
[0, ``vx_max``] and ``dyaw`` in [-0.314, 0.314] to ``vyaw`` in
[-``vyaw_max``, ``vyaw_max``]. Yaw is AirSim-positive (clockwise from above),
the opposite of the bridge, whose ``vyaw > 0`` turns left.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, fields
from typing import Deque, Dict, Tuple

import numpy as np

MODEL_LIMITS = np.array([1.0, 0.4, 0.4, 0.3141592653589793], dtype=np.float64)


@dataclass
class DogDynamicsParams:
    vx_max: float = 0.35          # bridge clamp (m/s)
    vyaw_max: float = 0.5         # H12 VYAW_MAX (rad/s)
    vx_delay_s: float = 0.15
    vx_deadband: float = 0.05
    vx_gain: float = 0.57
    vx_tau_s: float = 0.4
    yaw_delay_s: float = 0.15
    yaw_deadband: float = 0.2
    yaw_gain: float = 0.50
    yaw_tau_s: float = 0.3

    def to_dict(self) -> Dict[str, float]:
        return {k: round(float(v), 4) for k, v in asdict(self).items()}


@dataclass
class DogDynamicsRanges:
    """Uniform per-episode ranges; a field absent here keeps the ``DogDynamicsParams`` default."""

    vx_delay_s: Tuple[float, float] = (0.1, 0.3)
    vx_deadband: Tuple[float, float] = (0.0, 0.1)
    vx_gain: Tuple[float, float] = (0.4, 0.8)
    vx_tau_s: Tuple[float, float] = (0.2, 0.8)
    yaw_delay_s: Tuple[float, float] = (0.1, 0.25)
    yaw_deadband: Tuple[float, float] = (0.15, 0.25)
    yaw_gain: Tuple[float, float] = (0.2, 0.6)
    yaw_tau_s: Tuple[float, float] = (0.2, 0.6)

    def sample(self, rng: np.random.Generator, base: DogDynamicsParams | None = None) -> DogDynamicsParams:
        p = DogDynamicsParams(**asdict(base or DogDynamicsParams()))
        for f in fields(self):
            lo, hi = getattr(self, f.name)
            setattr(p, f.name, float(rng.uniform(lo, hi)))
        return p


def model_action_to_cmd(action: np.ndarray, p: DogDynamicsParams) -> Tuple[float, float]:
    """Model action ``(dx, dy, dz, dyaw)`` -> commanded ``(vx, vyaw)``; dy / dz are dropped."""
    a = np.asarray(action, dtype=np.float64).reshape(4)
    vx = float(np.clip(a[0] / MODEL_LIMITS[0], 0.0, 1.0)) * p.vx_max
    vyaw = float(np.clip(a[3] / MODEL_LIMITS[3], -1.0, 1.0)) * p.vyaw_max
    return vx, vyaw


def _deadband(u: float, db: float) -> float:
    return 0.0 if abs(u) <= db else u


class DogDynamics:
    """Stateful plant; ``advance(cmd, dt)`` returns the mean body ``(vx, yaw_rate)`` over ``dt``."""

    def __init__(self, params: DogDynamicsParams | None = None, tick_hz: float = 50.0) -> None:
        self.p = params or DogDynamicsParams()
        self.tick = 1.0 / float(tick_hz)
        self.reset()

    def reset(self) -> None:
        nvx = max(0, int(round(self.p.vx_delay_s / self.tick)))
        nyaw = max(0, int(round(self.p.yaw_delay_s / self.tick)))
        self._qvx: Deque[float] = deque([0.0] * nvx)
        self._qyaw: Deque[float] = deque([0.0] * nyaw)
        self.vx = 0.0
        self.yaw_rate = 0.0

    @staticmethod
    def _delayed(q: Deque[float], u: float) -> float:
        if not q:
            return u
        q.append(u)
        return q.popleft()

    def advance(self, cmd: Tuple[float, float], dt: float) -> Tuple[float, float]:
        p = self.p
        vx_cmd = float(np.clip(cmd[0], 0.0, p.vx_max))
        yaw_cmd = float(np.clip(cmd[1], -p.vyaw_max, p.vyaw_max))
        n = max(1, int(round(dt / self.tick)))
        sx = sy = 0.0
        ax = 1.0 - math.exp(-self.tick / max(p.vx_tau_s, 1e-3))
        ay = 1.0 - math.exp(-self.tick / max(p.yaw_tau_s, 1e-3))
        for _ in range(n):
            ux = p.vx_gain * _deadband(self._delayed(self._qvx, vx_cmd), p.vx_deadband)
            uy = p.yaw_gain * _deadband(self._delayed(self._qyaw, yaw_cmd), p.yaw_deadband)
            self.vx += ax * (ux - self.vx)
            self.yaw_rate += ay * (uy - self.yaw_rate)
            sx += self.vx
            sy += self.yaw_rate
        return sx / n, sy / n
