"""Detection box + depth + vehicle pose -> world point (x north, y east, z up)."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from ..types import Frame, Intrinsics, VehicleState


def _rot_frd_to_ned(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch),
                              math.cos(roll), math.sin(roll))
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


_CAM_TO_FRD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_NED_TO_NEU = np.diag([1.0, 1.0, -1.0])


def camera_rotation(state: VehicleState, cam_pitch: float = 0.0) -> np.ndarray:
    """3x3 map from camera axes (x right, y down, z forward) to world (north, east, up).

    ``cam_pitch`` is the camera's mount pitch relative to the body (rad, positive up).
    """
    cp, sp = math.cos(cam_pitch), math.sin(cam_pitch)
    mount = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return _NED_TO_NEU @ _rot_frd_to_ned(state.yaw, state.pitch, state.roll) @ mount @ _CAM_TO_FRD


def camera_to_world(p_cam: np.ndarray, state: VehicleState, cam_pitch: float = 0.0) -> np.ndarray:
    """Camera-frame point -> world offset (north, east, up) from the vehicle."""
    return camera_rotation(state, cam_pitch) @ np.asarray(p_cam, np.float64)


def world_to_pixel(p_world: np.ndarray, state: VehicleState, intr: Intrinsics, cam_pitch: float = 0.0,
                   cam_offset_frd: Sequence[float] = (0.0, 0.0, 0.0)) -> Optional[Tuple[float, float, float]]:
    """Inverse of ``camera_to_world`` for a world point; (u, v, Z) or None when behind the camera."""
    d = np.asarray(p_world, np.float64) - camera_origin(state, cam_offset_frd)
    x, y, z = camera_rotation(state, cam_pitch).T @ d
    if z <= 0.1:
        return None
    return intr.cx + intr.fx * x / z, intr.cy + intr.fy * y / z, z


def estimate_ground_z(depth: np.ndarray, intr: Intrinsics, state: VehicleState, cam_pitch: float = 0.0,
                      max_depth: float = 60.0, min_points: int = 20,
                      cam_offset_frd: Sequence[float] = (0.0, 0.0, 0.0)) -> Optional[float]:
    """Ground height from the lower-centre depth pixels (street seen below the horizon), or None."""
    h, w = depth.shape[:2]
    vs = np.arange(int(0.8 * h), h, max(1, h // 56))
    us = np.arange(w // 4, w - w // 4, max(1, w // 56))
    uu, vv = np.meshgrid(us, vs)
    z = depth[vv, uu].astype(np.float64)
    ok = np.isfinite(z) & (z > 0.3) & (z < max_depth)
    if ok.sum() < min_points:
        return None
    p = np.stack([(uu[ok] - intr.cx) / intr.fx * z[ok], (vv[ok] - intr.cy) / intr.fy * z[ok], z[ok]])
    z0 = float(camera_origin(state, cam_offset_frd)[2])
    wz = z0 + (camera_rotation(state, cam_pitch) @ p)[2]
    below = wz[wz < z0 - 0.5]
    return float(np.percentile(below, 20)) if below.size >= min_points else None


def depth_scale_from_ground(depth: np.ndarray, intr: Intrinsics, state: VehicleState, ground_z: float,
                            cam_pitch: float = 0.0, cam_offset_frd: Sequence[float] = (0.0, 0.0, 0.0),
                            range_m: Tuple[float, float] = (5.0, 60.0), min_points: int = 20) -> Optional[float]:
    """Metric scale for a relative depth map: plane-implied / predicted depth over lower-centre pixels."""
    h, w = depth.shape[:2]
    vs = np.arange(int(0.75 * h), h, max(1, h // 56))
    us = np.arange(w // 4, w - w // 4, max(1, w // 56))
    uu, vv = np.meshgrid(us, vs)
    rays = np.stack([(uu - intr.cx) / intr.fx, (vv - intr.cy) / intr.fy, np.ones_like(uu, np.float64)]).reshape(3, -1)
    rz = (camera_rotation(state, cam_pitch) @ rays)[2]
    dz = float(ground_z) - float(camera_origin(state, cam_offset_frd)[2])
    if dz >= -0.5:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        expect = np.where(rz < -1e-3, dz / rz, np.nan)
    pred = depth[vv, uu].astype(np.float64).ravel()
    ok = np.isfinite(expect) & (expect > range_m[0]) & (expect < range_m[1]) & np.isfinite(pred) & (pred > 0.3)
    if ok.sum() < min_points:
        return None
    r = expect[ok] / pred[ok]
    med = float(np.median(r))
    inl = np.abs(r / med - 1.0) < 0.2
    return float(np.median(r[inl])) if inl.sum() >= min_points else None


def box_depth(depth: np.ndarray, d_intr: Intrinsics, bbox: Sequence[float], rgb_intr: Intrinsics,
              pct: float = 30.0, max_depth: float = 150.0) -> Optional[float]:
    """Low percentile of planar depth inside the box (nearer than the background behind it)."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    xs = [d_intr.cx + d_intr.fx * (x - rgb_intr.cx) / rgb_intr.fx for x in (x0, x1)]
    ys = [d_intr.cy + d_intr.fy * (y - rgb_intr.cy) / rgb_intr.fy for y in (y0, y1)]
    cx, cy = 0.5 * sum(xs), 0.5 * sum(ys)
    hw, hh = max(0.5, 0.35 * (xs[1] - xs[0])), max(0.5, 0.35 * (ys[1] - ys[0]))
    h, w = depth.shape[:2]
    a0, a1 = max(0, int(math.floor(cx - hw))), min(w, int(math.ceil(cx + hw)) + 1)
    b0, b1 = max(0, int(math.floor(cy - hh))), min(h, int(math.ceil(cy + hh)) + 1)
    if a1 <= a0 or b1 <= b0:
        return None
    v = depth[b0:b1, a0:a1].astype(np.float64).ravel()
    v = v[np.isfinite(v) & (v > 0.3) & (v < max_depth)]
    return float(np.percentile(v, pct)) if v.size else None


def camera_origin(state: VehicleState, cam_offset_frd: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """World position of the camera mounted at ``cam_offset_frd`` (body forward, right, down; meters)."""
    ned = _rot_frd_to_ned(state.yaw, state.pitch, state.roll) @ np.asarray(cam_offset_frd, np.float64)
    return np.asarray(state.pos, np.float64) + _NED_TO_NEU @ ned


def locate(bbox: Sequence[float], frame: Frame, state: VehicleState, *, cam_pitch: float = 0.0,
           ground_z: Optional[float] = None, max_depth: float = 150.0,
           cam_offset_frd: Sequence[float] = (0.0, 0.0, 0.0),
           prefer_ground: bool = False) -> Optional[Tuple[np.ndarray, float, str]]:
    """World point of the box, its range and the method ("depth" or "ground"); None when unknown.

    Depth: the box centre ray scaled to the box's planar depth. Ground: the
    box bottom-centre ray intersected with the plane ``z = ground_z`` (an object
    standing on the ground), used when depth is missing or out of range, or first with ``prefer_ground``.
    """
    ri = frame.rgb_intr
    origin = camera_origin(state, cam_offset_frd)
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if not (prefer_ground and ground_z is not None) and frame.depth is not None and frame.depth_intr is not None:
        z = box_depth(frame.depth, frame.depth_intr, bbox, ri, max_depth=max_depth)
        if z is not None:
            u, v = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            p_cam = z * np.array([(u - ri.cx) / ri.fx, (v - ri.cy) / ri.fy, 1.0])
            off = camera_to_world(p_cam, state, cam_pitch)
            return origin + off, float(np.linalg.norm(off)), "depth"
    if ground_z is None:
        return None
    u, v = 0.5 * (x0 + x1), y1
    ray = camera_to_world(np.array([(u - ri.cx) / ri.fx, (v - ri.cy) / ri.fy, 1.0]), state, cam_pitch)
    dz = float(ground_z) - float(origin[2])
    if ray[2] >= -1e-3 or dz >= 0.0:
        return None
    t = dz / ray[2]
    off = t * ray
    if float(np.linalg.norm(off)) > max_depth:
        return None
    return origin + off, float(np.linalg.norm(off)), "ground"
