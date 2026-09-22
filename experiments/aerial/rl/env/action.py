"""Action space for the aerial RL env.

Action = 4-D body-frame command ``(dx, dy, dz, dyaw)`` — a per-step displacement
in the drone body frame (x forward, y left, z up; yaw CCW), matching the FastWAM
``action_dim = 4`` and ``openfly_actions.pos_yaw_to_body_delta``.

Two execution paths:

  * continuous (Plan-A physics step): ``body_delta_to_velocity_ned`` turns the
    displacement into a velocity + yaw-rate command for
    ``moveByVelocityAsync(vx, vy, vz, dt, yaw_mode=...)``. AirSim is NED, so world
    "up" (dz > 0) maps to ``vz < 0``.
  * discrete fallback: reuse ``delta_to_nearest_primitive`` /
    ``primitive_to_delta`` so the env can also be driven by the existing 10
    OpenFly macro-primitives (and by the current FastWAM primitive policy).

NOTE: ``clip_body_delta`` is defined here (not imported) — this branch's
``openfly_actions`` does not export it; other branches do, so keep this local
copy behaviourally compatible (per-axis symmetric magnitude clip).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# Re-exported so callers have a single action module to import from.
from experiments.aerial.openfly_actions import (  # noqa: F401
    OPENFLY_PRIMITIVES,
    delta_to_nearest_primitive,
    primitive_to_delta,
    wrap_angle,
)

ACTION_DIM = 4
DEFAULT_STEP_HZ = 30.0

# Physical continuous-control limits, expressed as body-frame VELOCITIES
# (fwd, lateral, vertical [m/s], yaw-rate [rad/s]). The per-step body-delta cap
# is velocity * dt (see ``body_delta_limits``), so it scales correctly with the
# control rate instead of hard-coding a displacement.
MAX_BODY_VELOCITY = np.array(
    [5.0, 2.0, 2.0, math.pi / 2.0], dtype=np.float64  # 5 m/s fwd, 90°/s yaw
)

# Ground UGV (G0): planar 3-DOF; dz channel unused (forced 0 in env).
GROUND_MAX_BODY_VELOCITY = np.array(
    [2.0, 1.0, 0.0, math.pi / 4.0], dtype=np.float64  # ~2 m/s fwd, 45°/s yaw
)

# Sparse OpenFly macro-primitive spans (fwd 9 m, strafe 3 m, climb 3 m, turn 30°).
# These are DISCRETE teleport magnitudes, NOT a 1/step_hz continuous increment:
# clipping a per-step command to these would sanction ~270 m/s at 30 Hz. Kept
# only to describe the discrete-primitive fallback's reach — a macro primitive
# must be executed as a sustained multi-step command, never crammed into one
# ``dt`` (see ``env.airsim_env`` / ``collector``), so it does NOT feed
# ``clip_body_delta``.
MACRO_PRIMITIVE_SPAN = np.array([9.0, 3.0, 3.0, math.pi / 6.0], dtype=np.float64)


def body_delta_limits(
    dt: float,
    max_velocity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-step body-delta cap = ``max_velocity * dt`` (a real displacement)."""
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    mv = np.asarray(
        MAX_BODY_VELOCITY if max_velocity is None else max_velocity,
        dtype=np.float64,
    ).reshape(ACTION_DIM)
    return mv * float(dt)


# Continuous per-step cap at the default control rate. At 30 Hz this is
# ~[0.167, 0.067, 0.067, 0.052] m|rad — a sane 33 ms increment, not a 9 m hop.
DEFAULT_BODY_DELTA_LIMITS = body_delta_limits(1.0 / DEFAULT_STEP_HZ)


def clip_body_delta(
    delta: np.ndarray,
    limits: Optional[np.ndarray] = None,
    *,
    forbid_backward: bool = False,
) -> np.ndarray:
    """Clamp a 4-D body delta to ±limits per axis. Rejects non-finite input.

    ``limits`` defaults to the continuous per-step cap at ``DEFAULT_STEP_HZ``.
    Callers that step at another rate should pass ``body_delta_limits(dt)`` so the
    displacement bound tracks ``dt``.

    When ``forbid_backward`` is True, body ``dx`` is clamped to ``≥ 0`` (no rear
    sensor → reverse flight is not a legal motion).
    """
    d = np.asarray(delta, dtype=np.float64).reshape(ACTION_DIM)
    if not np.isfinite(d).all():
        raise ValueError(f"action must be finite, got {d}")
    lim = np.asarray(
        DEFAULT_BODY_DELTA_LIMITS if limits is None else limits, dtype=np.float64
    ).reshape(ACTION_DIM)
    out = np.clip(d, -lim, lim)
    if forbid_backward:
        out = forbid_backward_dx(out)
    return out


def forbid_backward_dx(delta: np.ndarray) -> np.ndarray:
    """Clamp body-forward channel to ``dx ≥ 0`` (copy; no rear sensor)."""
    d = np.asarray(delta, dtype=np.float64).reshape(ACTION_DIM).copy()
    if d[0] < 0.0:
        d[0] = 0.0
    return d


def body_delta_to_velocity_ned(
    delta: np.ndarray,
    yaw: float,
    dt: float,
) -> Tuple[float, float, float, float]:
    """Body-frame displacement over ``dt`` -> AirSim NED velocity + yaw-rate.

    Returns ``(vx, vy, vz_ned, yaw_rate_deg)`` where ``vx, vy`` are world-frame
    horizontal velocities (m/s), ``vz_ned`` is NED vertical velocity (negative =
    climb), and ``yaw_rate_deg`` is degrees/s for ``YawMode(is_rate=True)``.

    The world (wx, wy) rotation matches ``run_closed_loop.apply_body_delta`` so
    the continuous path and the kinematic mock agree on heading conventions.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    dx, dy, dz, dyaw = (float(v) for v in np.asarray(delta, dtype=np.float64).reshape(ACTION_DIM))
    c, s = math.cos(yaw), math.sin(yaw)
    wx = c * dx - s * dy          # body x fwd, body y left -> world
    wy = s * dx + c * dy
    vx = wx / dt
    vy = wy / dt
    vz_ned = -dz / dt             # NED: up (dz>0) is negative z
    yaw_rate_deg = math.degrees(dyaw) / dt
    return vx, vy, vz_ned, yaw_rate_deg
