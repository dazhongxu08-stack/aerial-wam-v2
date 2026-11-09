#!/usr/bin/env python3
"""Stage 3: shield-off MPC flight with the consequence-model ensemble.

Every ``--replan-every`` steps: sample one candidate 10-step sequence per family
(same families and macro sampler as the Stage 2 probes), keep those that pass
the hold-altitude rule, score them with the ensemble (mean survival, mean
predicted goal progress) and pick with ``select_candidate`` (the rule fixed on
the training-region held-out set). Execute the first steps of the pick, replan.
No shield, no hand-written turns: the only control logic is the model's choice.

Per route: arrived (within --success-dist of the goal), severe collision (any
collision ends the episode), progress ratio = max(0, d0 - d_final) / route
length, stuck (no 1 m gain on the best distance for --stuck-s seconds).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("mpc_consequence")


def candidate_side(cands: np.ndarray) -> np.ndarray:
    """+1 / -1 for candidates ending turned (or displaced) to one side, 0 for straight or in-place.

    Yaw and lateral velocity share a sign convention (mirror_batch negates both)."""
    c = np.asarray(cands, np.float64)
    turn, lat = c[..., 3].sum(-1), c[..., 1].sum(-1)
    return np.where(np.abs(turn) > 0.15, np.sign(turn), np.where(np.abs(lat) > 0.2, np.sign(lat), 0.0))


def goal_offset(cands: np.ndarray, bearing: float, dt: float):
    """Angle of each candidate's net horizontal displacement from the goal direction (+ = the +yaw side), and its length.

    Body frame at plan time; +vy points to the +yaw side, the same sense as ``goal_bearing_body``."""
    c = np.asarray(cands, np.float64)
    th = np.cumsum(c[..., 3], -1) * dt
    dx = (c[..., 0] * np.cos(th) - c[..., 1] * np.sin(th)).sum(-1) * dt
    dy = (c[..., 0] * np.sin(th) + c[..., 1] * np.cos(th)).sum(-1) * dt
    return (np.arctan2(dy, dx) - bearing + np.pi) % (2 * np.pi) - np.pi, np.hypot(dx, dy)


def follow_fan(k: int, yaw_limit: float) -> np.ndarray:
    """Fixed yaw-then-forward fan both ways plus full-rate turns in place (yaw_fwd / yaw_only shapes)."""
    seqs = [np.tile([0.6, 0.0, 0.0, 0.0], (k, 1))]
    for sgn in (-1.0, 1.0):
        for frac in (0.5, 1.0):
            for m in (1, 2, 3, 4):
                a = np.zeros((k, 4))
                a[:m, 3], a[:m, 0], a[m:, 0] = sgn * frac * yaw_limit, 0.1, 0.6
                seqs.append(a)
        a = np.zeros((k, 4))
        a[:, 3] = sgn * yaw_limit
        seqs.append(a)
    return np.stack(seqs)


def follow_pick(alpha: np.ndarray, length: np.ndarray, safe: np.ndarray, sigma: float,
                min_len: float = 0.5, cross_tol: float = math.radians(15.0)) -> int:
    """Boundary following: the safe moving candidate heading closest to the goal, crossing to the other side by at
    most ``cross_tol``.

    -1 when no safe candidate moves at least ``min_len``."""
    a = sigma * np.asarray(alpha, np.float64)
    ok = np.asarray(safe, bool) & (np.asarray(length) >= min_len) & (a >= -cross_tol)
    return int(np.argmin(np.where(ok, np.abs(a), np.inf))) if ok.any() else -1


def near_pick(cands: np.ndarray, bearing: float, safe: np.ndarray, prog: np.ndarray, dt: float = 0.2,
              max_deg: float = 30.0, min_len: float = 0.3) -> int:
    """Most progressing safe candidate whose net displacement heads within ``max_deg`` of the goal; -1 if none."""
    alpha, length = goal_offset(cands, bearing, dt)
    ok = np.asarray(safe, bool) & (length >= min_len) & (np.abs(alpha) <= math.radians(max_deg))
    return int(np.argmax(np.where(ok, prog, -np.inf))) if ok.any() else -1


def nopass_pick(cands: np.ndarray, bearing: float, surv: np.ndarray, prog: np.ndarray, allowed: np.ndarray,
                margin: float, floor: float, min_prog: float, dt: float = 0.2, min_len: float = 0.5) -> int:
    """Most progressing moving candidate nearly as safe as the safest allowed one; -1 if none."""
    _, length = goal_offset(cands, bearing, dt)
    allowed = np.asarray(allowed, bool)
    s = np.asarray(surv, np.float64)
    bar = max(float(floor), float(s[allowed].max()) - float(margin))
    ok = allowed & (s >= bar) & (length >= min_len) & (np.asarray(prog) >= min_prog)
    return int(np.argmax(np.where(ok, prog, -np.inf))) if ok.any() else -1


def candidate_waypoints(cands: np.ndarray, state: np.ndarray, k: int) -> np.ndarray:
    """World (x, y, z-up) after each of the first ``k`` per-step body displacements of every candidate, (M, k, 3)."""
    c = np.asarray(cands, np.float64)[:, :int(k)]
    th = float(state[6]) + np.concatenate([np.zeros((len(c), 1)), np.cumsum(c[:, :-1, 3], -1)], -1)
    dx = c[..., 0] * np.cos(th) - c[..., 1] * np.sin(th)
    dy = c[..., 0] * np.sin(th) + c[..., 1] * np.cos(th)
    return (np.asarray(state[:3], np.float64)
            + np.stack([np.cumsum(dx, -1), np.cumsum(dy, -1), np.cumsum(c[..., 2], -1)], -1))


def depth_points(depth: np.ndarray, state: np.ndarray, max_range: float, cam_dz: float = -1.0,
                 hfov_deg: float = 90.0, vfov_deg: float = 58.7, voxel: float = 0.2) -> np.ndarray:
    """World points (x, y, z-up) of a pooled planar depth map closer than ``max_range``, voxel-deduplicated.

    The pooled map squeezes the 16:9 frame to a square, so the two axes get their own focal length."""
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    fx, fy = (w / 2) / math.tan(math.radians(hfov_deg / 2)), (h / 2) / math.tan(math.radians(vfov_deg / 2))
    v, u = np.mgrid[0:h, 0:w]
    ok = np.isfinite(d) & (d > 0.05) & (d < max_range)
    fwd = d[ok]
    right = (u[ok] + 0.5 - w / 2) / fx * fwd
    down = (v[ok] + 0.5 - h / 2) / fy * fwd
    yaw = float(state[6])
    pts = np.stack([state[0] + fwd * math.cos(yaw) - right * math.sin(yaw),
                    state[1] + fwd * math.sin(yaw) + right * math.cos(yaw),
                    state[2] + cam_dz - down], -1)
    return np.unique(np.round(pts / voxel), axis=0) * voxel if len(pts) else pts.reshape(0, 3)


def memory_clear(wps: np.ndarray, here: np.ndarray, pts: np.ndarray, clear_m: float, dz_m: float,
                 slack: float = 0.1) -> np.ndarray:
    """True for candidates that keep ``clear_m`` horizontally from remembered points within ``dz_m``, or, when the
    drone is already closer than that, do not get more than ``slack`` closer than it is now."""
    flat = wps.reshape(-1, 3)
    allp = np.vstack([flat, np.asarray(here, np.float64)[None, :3]])
    lo, hi = allp.min(0) - clear_m, allp.max(0) + clear_m
    pts = pts[np.all((pts[:, :2] >= lo[:2]) & (pts[:, :2] <= hi[:2]), -1)]
    if not len(pts):
        return np.ones(len(wps), bool)

    def clearance(q: np.ndarray) -> np.ndarray:
        dxy = np.linalg.norm(q[:, None, :2] - pts[None, :, :2], axis=-1)
        return np.where(np.abs(q[:, None, 2] - pts[None, :, 2]) < dz_m, dxy, np.inf).min(-1)

    bar = min(clear_m, float(clearance(allp[-1:])[0]) - slack)
    return clearance(flat).reshape(wps.shape[:2]).min(-1) >= bar


def _cam_axes(state: np.ndarray, pts: np.ndarray, cam_dz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward, image-right, and down of world points in the camera that ``depth_points`` uses."""
    st = np.asarray(state, np.float64)
    q = np.asarray(pts, np.float64)
    yaw = float(st[6])
    dx, dy = q[:, 0] - st[0], q[:, 1] - st[1]
    c, s = math.cos(yaw), math.sin(yaw)
    fwd = c * dx + s * dy
    right = s * dx - c * dy
    down = (st[2] + cam_dz) - q[:, 2]
    return fwd, right, down


def _seen_by_newer(state: np.ndarray, depth: np.ndarray, pts: np.ndarray, hfov_deg: float,
                   vfov_deg: float, cam_dz: float, max_range: float) -> np.ndarray:
    """True where ``pts`` fall on a valid pixel of a newer view, so that view replaces them."""
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    fwd, right, down = _cam_axes(state, pts, cam_dz)
    fx = (w / 2) / math.tan(math.radians(hfov_deg / 2))
    fy = (h / 2) / math.tan(math.radians(vfov_deg / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.rint(right / fwd * fx + w / 2 - 0.5).astype(int)
        v = np.rint(down / fwd * fy + h / 2 - 0.5).astype(int)
    inside = (fwd > 0.05) & (fwd < max_range) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    seen = np.zeros(len(pts), bool)
    if not inside.any():
        return seen
    vv, uu = v[inside], u[inside]
    pix = d[vv, uu]
    seen[inside] = np.isfinite(pix) & (pix > 0.05) & (pix < max_range)
    return seen


def _segment_gap(a: np.ndarray, b: np.ndarray, pts: np.ndarray) -> float:
    """Minimum horizontal distance from ``pts`` to the segment ``a``–``b``."""
    if len(pts) == 0:
        return math.inf
    a, b = np.asarray(a, np.float64)[:2], np.asarray(b, np.float64)[:2]
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return float(np.linalg.norm(pts[:, :2] - a, axis=1).min())
    t = np.clip(((pts[:, 0] - a[0]) * ab[0] + (pts[:, 1] - a[1]) * ab[1]) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(pts[:, :2] - proj, axis=1).min())


def remembered_surface(frames: List[tuple[np.ndarray, np.ndarray]], here: np.ndarray,
                       keep_m: float = 40.0, dz_m: float = 8.0, max_range: float = 40.0,
                       hfov_deg: float = 90.0, vfov_deg: float = 58.7, cam_dz: float = -1.0) -> np.ndarray:
    """World points still standing after the newest view of each direction replaces older ones.

    ``frames`` is oldest first, each ``(depth, state)``. A frame whose camera is more than
    ``keep_m`` from ``here`` is dropped. A point more than ``dz_m`` off the current altitude
    is dropped. An older point that still projects into a newer valid pixel is dropped, so a
    false near reading does not stick once the camera looks there again.
    """
    here = np.asarray(here, np.float64)
    kept: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for depth, st in frames:
        st = np.asarray(st, np.float64)
        if math.hypot(float(st[0] - here[0]), float(st[1] - here[1])) > keep_m:
            continue
        pts = depth_points(depth, st, max_range, cam_dz, hfov_deg, vfov_deg)
        if len(pts):
            pts = pts[np.abs(pts[:, 2] - here[2]) < dz_m]
        kept.append((st, np.asarray(depth, np.float64), pts))
    pieces: List[np.ndarray] = []
    for i, (_, _, pts) in enumerate(kept):
        if len(pts) == 0:
            continue
        survive = np.ones(len(pts), bool)
        for st2, depth2, _ in kept[i + 1:]:
            survive &= ~_seen_by_newer(st2, depth2, pts, hfov_deg, vfov_deg, cam_dz, max_range)
            if not survive.any():
                break
        if survive.any():
            pieces.append(pts[survive])
    return np.concatenate(pieces) if pieces else np.zeros((0, 3), np.float64)


def step_seen_gap(frames: List[tuple[np.ndarray, np.ndarray]], state: np.ndarray, cands: np.ndarray,
                  keep_m: float = 40.0, dz_m: float = 8.0, max_range: float = 40.0) -> np.ndarray:
    """Horizontal meters from each candidate's first step to the remembered surface.

    Only the step that will be flown is measured. No remembered surface yields infinity.
    """
    state = np.asarray(state, np.float64)
    pts = remembered_surface(frames, state[:3], keep_m, dz_m, max_range)
    ends = candidate_waypoints(cands, state, 1)[:, 0]
    return np.array([_segment_gap(state[:3], end, pts) for end in ends], np.float64)


def step_seen_clear(frames: List[tuple[np.ndarray, np.ndarray]], state: np.ndarray, cands: np.ndarray,
                    clear_m: float = 1.0, keep_m: float = 40.0, dz_m: float = 8.0,
                    max_range: float = 40.0) -> np.ndarray:
    """True when the first step stays at least ``clear_m`` from the remembered surface."""
    return step_seen_gap(frames, state, cands, keep_m, dz_m, max_range) >= float(clear_m)


def alt_hold_dz(state: np.ndarray, hold_z: Optional[float], hold_gain: float = 0.5,
                hold_max: float = 0.2) -> float:
    """Vertical body step toward ``hold_z``, or 0 when no hold is set."""
    if hold_z is None:
        return 0.0
    return float(np.clip(hold_gain * (float(hold_z) - float(state[2])), -hold_max, hold_max))


def brake_action(state: np.ndarray, step_hz: float, limits: np.ndarray, hold_z: Optional[float] = None,
                 hold_gain: float = 0.5, hold_max: float = 0.2) -> np.ndarray:
    """Per-step body displacement that commands the opposite of the current horizontal velocity, no turn.

    With ``hold_z``, also steer the height toward it: a hard stop pitches the drone back and lifts it ~0.15 m."""
    vx, vy = float(state[3]), float(state[4])
    yaw = float(state[6])
    a = np.zeros(4)
    a[0] = -(vx * math.cos(yaw) + vy * math.sin(yaw)) / step_hz
    a[1] = -(-vx * math.sin(yaw) + vy * math.cos(yaw)) / step_hz
    a[2] = alt_hold_dz(state, hold_z, hold_gain, hold_max)
    return np.clip(a, -limits, limits)


def carrot_yaw_hold(bearing: float, latch: float, move_ok: bool, yaw_cap: float,
                    reverse_rad: float = 0.35) -> tuple[Optional[float], float]:
    """Yaw toward the schematic carrot when forward and goal_arc miss survival.

    Returns ``(None, 0)`` when a forward action already clears the line, so the
    normal selector runs. Otherwise the yaw rate has no forward creep. The sign
    stays until the heading error has clearly crossed to the other side.
    """
    if move_ok:
        return None, 0.0
    cap = abs(float(yaw_cap))
    b = float(bearing)
    side = 0.0 if abs(b) < 0.05 else (1.0 if b > 0.0 else -1.0)
    held = float(latch)
    if held == 0.0:
        held = side
    elif side != 0.0 and side != held and abs(b) >= float(reverse_rad):
        held = side
    if held == 0.0:
        return 0.0, 0.0
    return float(np.clip(held * abs(b), -cap, cap)), held


def stall_yaw(bearing: float, yaw_cap: float) -> float:
    """Yaw toward the goal and stop on it.

    A no-pass stall that keeps turning after the nose is already on the goal
    leaves on that new heading. The command is the remaining bearing, limited
    to ``yaw_cap``, so an aligned nose does not yaw away."""
    cap = abs(float(yaw_cap))
    return float(np.clip(float(bearing), -cap, cap))


def leaves_point(pos: np.ndarray, yaw: float, point: np.ndarray, action: np.ndarray) -> bool:
    """True when the body step that will actually be flown increases distance to ``point``.

    The progress head scores the whole 10-step macro, but only this step is flown. A macro
    that turns back toward the point later can score as progress while this step walks away.
    """
    pos = np.asarray(pos, np.float64)[:2]
    point = np.asarray(point, np.float64)[:2]
    dx, dy = float(action[0]), float(action[1])
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    nxt = pos + np.array([c * dx - s * dy, s * dx + c * dy], np.float64)
    return float(np.linalg.norm(point - nxt)) > float(np.linalg.norm(point - pos)) + 1e-6


def corner_turn(depth: np.ndarray, clearance_m: float = 4.0, open_m: float = 8.0) -> Optional[float]:
    """+1 yaws toward image right when only the image-left side is inside ``clearance_m``.

    The center of the image can be an open gap while the corner stays beside the body.
    Survival then stays above 0.95 and the forward step is the one that gets flown.
    ``None`` when both sides are close, or both are clear.
    """
    d = np.asarray(depth, np.float64)
    if d.ndim != 2 or min(d.shape) < 8:
        return None
    r0, r1 = d.shape[0] // 7, d.shape[0] * 5 // 7
    side = d.shape[1] // 3
    left = d[r0:r1, :side]
    right = d[r0:r1, d.shape[1] - side:]
    if left.size == 0 or right.size == 0:
        return None
    left_m = float(np.nanmin(left))
    right_m = float(np.nanmin(right))
    if not (math.isfinite(left_m) and math.isfinite(right_m)):
        return None
    if left_m < float(clearance_m) and right_m >= float(open_m):
        return 1.0
    if right_m < float(clearance_m) and left_m >= float(open_m):
        return -1.0
    return None


def open_turn_sign(depth: np.ndarray, cap: float = 10.0) -> float:
    """+1 (yaw rate > 0 turns toward the image right) when the right half of the depth image is more open, else -1."""
    col = np.minimum(np.asarray(depth, np.float64).min(0), cap)
    h = col.shape[0] // 2
    return 1.0 if col[h:].mean() > col[:h].mean() else -1.0


def cruise_yaw_cap(bearing: float, yaw_cap: float, trim_yaw: float = 0.0, trim_rad: float = 0.0) -> float:
    """A small heading error turns at ``trim_yaw``. A large one uses ``yaw_cap``.

    ``yaw_cap`` on an 8° error finishes the turn in one or two steps and then flies the new
    heading, which is how a long straight leg walks out to the side and corners back late.
    The shallow cap spreads that same correction into an arc."""
    cap = abs(float(yaw_cap))
    if trim_yaw > 0 and trim_rad > 0 and abs(float(bearing)) < float(trim_rad):
        return min(cap, abs(float(trim_yaw)))
    return cap


def goal_arc(bearing: float, k: int, dx: float, yaw_cap: float) -> np.ndarray:
    """Cruise forward while yawing toward the goal at up to ``yaw_cap`` rad/step, then straight.

    The whole horizon is what repeated one-step replans of this same rule would fly, so the
    consequence model scores the arc it will actually execute. A pure forward step while the
    nose is off the goal swings the bearing further out; this candidate is the one that can
    close that error without stopping to turn."""
    a = np.zeros((int(k), 4), dtype=np.float64)
    left = float(bearing)
    cap = abs(float(yaw_cap))
    for t in range(int(k)):
        step = float(np.clip(left, -cap, cap))
        a[t, 0] = float(dx)
        a[t, 3] = step
        left -= step
    return a


def smooth_turn(seq: np.ndarray, move_dx: float, yaw_cap: float) -> np.ndarray:
    """Keep a turn candidate moving: yawing steps advance at least ``move_dx`` and yaw at most ``yaw_cap``.

    Replan-every-step only flies the first action. The sampled turn puts the yaw there and the
    forward motion later, so the flown step is a stop-and-spin. This is the same turn, scored
    as an arc."""
    a = np.array(seq, dtype=np.float64, copy=True)
    yawing = np.abs(a[:, 3]) > 1e-3
    if move_dx > 0:
        a[yawing, 0] = np.maximum(a[yawing, 0], float(move_dx))
    if yaw_cap > 0:
        a[yawing, 3] = np.clip(a[yawing, 3], -abs(float(yaw_cap)), abs(float(yaw_cap)))
    return a


def hold_straight(names: List[str], surv: np.ndarray, score: np.ndarray, allowed: np.ndarray,
                  chosen: int, surv_min: float, margin: float) -> int:
    """Stay on a passing straight candidate unless the alternative beats it by ``margin``.

    Ties and sub-margin gains are what turned a safe line into the graze on SM_NYCB_578."""
    if margin <= 0 or names[chosen] == "fwd":
        return int(chosen)
    fwd = [i for i, name in enumerate(names)
           if name == "fwd" and bool(allowed[i]) and float(surv[i]) >= float(surv_min)]
    if not fwd:
        return int(chosen)
    i = fwd[0]
    if float(score[chosen]) - float(score[i]) < float(margin):
        return i
    return int(chosen)


def fit_slide(dx: float, dy: float, max_deg: float) -> tuple:
    """Shrink the lateral step so the slide stays inside the camera view.

    A 45 deg slide is thrown out by the 35 deg visibility gate, so it can never
    be selected. The forward step is kept and the sideways step is reduced."""
    dx, dy = abs(float(dx)), abs(float(dy))
    if dx <= 1e-6 or max_deg <= 0 or max_deg >= 89:
        return dx, dy
    cap = dx * math.tan(math.radians(float(max_deg)))
    return dx, min(dy, cap)


def open_slide(side: float, k: int, dx: float, dy: float, yaw: float) -> np.ndarray:
    """Move toward the open image side while yawing that way.

    Body +y is left and a positive yaw turns toward image right, so the slide's
    lateral step is the opposite sign of ``side``. The consequence model scores
    this the same way as every other candidate."""
    a = np.zeros((int(k), 4), dtype=np.float64)
    a[:, 0] = float(dx)
    a[:, 1] = -float(side) * float(dy)
    a[:, 3] = float(side) * float(yaw)
    return a


def prefer_motion(names: List[str], cands: np.ndarray, surv: np.ndarray, allowed: np.ndarray,
                  chosen: int, surv_min: float, min_step: float = 0.3) -> int:
    """Replace a stop-turn with a passing step that actually moves.

    The progress head scores the whole 10-step macro, but only step 0 is flown.
    A yaw-in-place can win that score and the aircraft stops. A forward or slide
    step that already clears the survival threshold is the one to fly."""
    cands = np.asarray(cands, np.float64)
    step0 = np.linalg.norm(cands[:, 0, :2], axis=1)
    if float(step0[chosen]) >= float(min_step):
        return int(chosen)
    movers = [i for i in range(len(names))
              if bool(allowed[i]) and float(surv[i]) >= float(surv_min) and float(step0[i]) >= float(min_step)]
    if not movers:
        return int(chosen)
    return int(max(movers, key=lambda i: (float(surv[i]), float(step0[i]))))


def prefer_safer_turn(names: List[str], cands: np.ndarray, surv: np.ndarray, allowed: np.ndarray,
                      chosen: int, surv_min: float, sag: float, min_step: float = 0.3) -> int:
    """When straight flight still passes but its survival has sagged, take a safer moving turn.

    The progress head scores the whole horizon, so a turn that leaves the goal ray ends a meter
    or two farther away and loses to straight flight. In front of a facade that straight step is
    the one about to fail. A yaw_fwd or goal_turn_fwd that still clears the survival line and is
    moving replaces it. A yaw-in-place does not: its survival is high because it does not move."""
    if sag <= 0 or names[int(chosen)] not in ("fwd", "goal_arc"):
        return int(chosen)
    surv = np.asarray(surv, np.float64)
    cands = np.asarray(cands, np.float64)
    straight = float(surv[int(chosen)])
    if straight < float(surv_min) or straight >= float(sag):
        return int(chosen)
    step0 = np.linalg.norm(cands[:, 0, :2], axis=1)
    turns = [i for i, name in enumerate(names)
             if name in ("yaw_fwd", "goal_turn_fwd") and bool(allowed[i])
             and float(surv[i]) >= float(surv_min) and float(surv[i]) > straight
             and float(step0[i]) >= float(min_step)]
    if not turns:
        return int(chosen)
    return int(max(turns, key=lambda i: (float(surv[i]), float(step0[i]))))


def side_arc(sign: float, k: int, dx: float, yaw: float) -> np.ndarray:
    """Forward arc yawing to one image side for the whole horizon.

    Positive ``sign`` yaws toward image right. The consequence model scores this arc; only the
    first step is flown when the planner replans every step."""
    a = np.zeros((int(k), 4), dtype=np.float64)
    a[:, 0] = float(dx)
    a[:, 3] = math.copysign(abs(float(yaw)), float(sign))
    return a


def upper_halves(depth: np.ndarray):
    """Mean depth of the upper image-left and upper image-right, or ``(None, None)``."""
    d = np.asarray(depth, np.float64)
    if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 2:
        return None, None
    upper = d[: d.shape[0] // 2]
    mid = upper.shape[1] // 2
    right = float(np.nanmean(upper[:, mid:]))
    left = float(np.nanmean(upper[:, :mid]))
    if not (math.isfinite(right) and math.isfinite(left)):
        return None, None
    return left, right


def gap_turn_sign(depth: np.ndarray) -> float:
    """+1 when the upper image-right is farther than the upper image-left.

    Column-min openness calls a street canyon open, because its nearest pixel is far, and calls
    the sky beside a facade blocked, because the roofline is close. The upper half is the gap.
    A flat upper image defaults to image right."""
    left, right = upper_halves(depth)
    if left is None or abs(right - left) <= 1e-3:
        return 1.0
    return 1.0 if right > left else -1.0


def upper_gap_sign(depth: np.ndarray, min_ratio: float = 0.10):
    """+1 or -1 when one upper half is farther by ``min_ratio`` of the nearer mean.

    ``None`` means the upper image does not show a side, so a detour must not start."""
    left, right = upper_halves(depth)
    if left is None:
        return None
    near = min(left, right)
    if near <= 1e-6 or abs(right - left) / near < float(min_ratio):
        return None
    return 1.0 if right > left else -1.0


def choose_skirt_side(surv_left: float, surv_right: float, tie_sign: float, margin: float = 0.02) -> float:
    """Pick the side arc with higher survival. A gap under ``margin`` defers to ``tie_sign``."""
    if abs(float(surv_right) - float(surv_left)) < float(margin):
        return 1.0 if float(tie_sign) >= 0 else -1.0
    return 1.0 if float(surv_right) > float(surv_left) else -1.0


def skirt_tie(surv_left: float, surv_right: float, surv_min: float, margin: float = 0.03) -> bool:
    """True when both side arcs clear survival and neither leads by ``margin``.

    One arc under ``surv_min`` is not a tie: the billboard case stays with the consequence
    picker. A wider survival gap is a real preference and stays there too. The upper image
    chooses the side."""
    left, right = float(surv_left), float(surv_right)
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    return abs(right - left) < float(margin) and min(left, right) >= float(surv_min)


def skirt_hold(held: float, yawed: float, yaw_cap: float, proposal: float, flips: int,
               max_flips: int = 1) -> tuple:
    """Return ``(side, flips, gave_up)``.

    The first call takes ``proposal``. That side is held until the nose has yawed ``yaw_cap``.
    The model may then switch once. A second cap, or the same side still losing, gives up."""
    if float(held) == 0.0:
        return float(proposal), int(flips), False
    if abs(float(yawed)) < float(yaw_cap):
        return float(held), int(flips), False
    if int(flips) >= int(max_flips) or float(proposal) == float(held):
        return 0.0, int(flips), True
    return float(proposal), int(flips) + 1, False


def skirt_gate(goal_open: bool, far: bool, closed_n: int, open_n: int, held: float, gave_up: bool,
               release: int, start: bool = False) -> tuple:
    """Return ``(skirt, clear, closed_n, open_n)``.

    ``start`` is a near tie between two side arcs that both clear survival, broken by the
    upper image. A decisive side preference does not start a detour. Once started, that
    side is held while the goal direction stays closed. After ``release`` open steps the nose
    goes back to alignment. Giving up on the yaw cap stays down until the goal has been open,
    so one closed step cannot restart the cap.
    """
    if not far:
        return False, True, 0, 0
    if goal_open:
        closed_n = 0
        open_n = int(open_n) + 1
        if float(held) != 0.0 and not gave_up and open_n < int(release):
            return True, False, closed_n, open_n
        if gave_up and open_n < int(release):
            return False, False, closed_n, open_n
        return False, True, closed_n, 0
    closed_n = int(closed_n) + 1
    open_n = 0
    if gave_up:
        return False, False, closed_n, open_n
    if float(held) != 0.0 or start:
        return True, False, closed_n, open_n
    return False, False, closed_n, open_n


def choose_align(names: List[str], surv: np.ndarray, allowed: np.ndarray, bearing: float,
                 surv_min: float, align_rad: float, slack: float = 0.02) -> int:
    """Pull the nose back with goal_arc once the heading error exceeds ``align_rad``.

    Straight flight while the nose is off the goal swings the bearing further out. Take the
    arc when it survives and is about as safe as going straight. A clearly less safe arc is
    left to the normal pick."""
    if align_rad <= 0 or abs(float(bearing)) < float(align_rad):
        return -1
    arcs = [i for i, name in enumerate(names)
            if name == "goal_arc" and bool(allowed[i]) and float(surv[i]) >= float(surv_min)]
    if not arcs:
        return -1
    i = max(arcs, key=lambda k: float(surv[k]))
    fwds = [j for j, name in enumerate(names)
            if name == "fwd" and bool(allowed[j]) and float(surv[j]) >= float(surv_min)]
    if fwds and float(surv[i]) + float(slack) < float(surv[fwds[0]]):
        return -1
    return int(i)


def choose_near(names: List[str], cands: np.ndarray, surv: np.ndarray, allowed: np.ndarray,
                bearing: float, surv_min: float, min_dx: float = 0.3) -> int:
    """One moving step toward the goal. Only step 0 is flown, so the 10-step net heading is ignored."""
    cands = np.asarray(cands, np.float64)
    passing = [i for i in range(len(names)) if bool(allowed[i]) and float(surv[i]) >= float(surv_min)]
    if not passing:
        return -1
    bearing = float(bearing)

    def dx(i: int) -> float:
        return float(cands[i, 0, 0])

    def closure(i: int) -> float:
        return bearing * float(cands[i, 0, 3])

    for name in ("near_arc", "goal_arc"):
        preferred = [i for i in passing if names[i] == name and dx(i) >= min_dx]
        if preferred:
            return int(max(preferred, key=lambda i: (closure(i), float(surv[i]))))
    movers = [i for i in passing if dx(i) >= min_dx and closure(i) > 0.0]
    if not movers:
        return -1
    return int(max(movers, key=lambda i: (closure(i), dx(i))))


def follow_side_choice(alpha: np.ndarray, length: np.ndarray, safe: np.ndarray, clear: np.ndarray, bearing: float) -> float:
    """Side whose strictly-on-side best candidate heads closer to the goal; ties go to the larger predicted clearance."""
    best = {}
    for sd in (-1.0, 1.0):
        i = follow_pick(alpha, length, safe, sd, cross_tol=0.0)
        if i >= 0:
            best[sd] = (abs(float(alpha[i])), -float(clear[i]))
    if not best:
        return 1.0 if bearing >= 0 else -1.0
    return min(best, key=best.get)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation", default="experiments/aerial/phase3_unified/annotations/outdoor_complex_od_holdout.json")
    ap.add_argument("--routes", default="", help="comma list of route_id; empty = all")
    ap.add_argument("--ckpts", required=True, help="comma list of consequence checkpoints (ensemble)")
    ap.add_argument("--depth-ckpt", default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt")
    ap.add_argument("--config", default="configs/_tmp_feasible_side_h125.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--surv-min", type=float, default=0.7)
    ap.add_argument("--replan-every", type=int, default=3)
    ap.add_argument("--success-dist", type=float, default=3.0)
    ap.add_argument("--max-steps-per-m", type=float, default=4.0)
    ap.add_argument("--budget-ref", choices=("chord", "astar"), default="chord",
                    help="length the step budget scales with: annotation polyline, or complexity_meta astar_1p5_m")
    ap.add_argument("--stuck-s", type=float, default=60.0)
    ap.add_argument("--min-z-m", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="model", choices=("model", "goal_line"),
                    help="goal_line = baseline that always takes the goal_turn_fwd candidate (model ignored)")
    ap.add_argument("--input-depth", default="da3", choices=("da3", "gt"),
                    help="gt = simulator depth as model input; diagnostic only, must match the checkpoints")
    ap.add_argument("--alt-fix", action="store_true",
                    help="hold-altitude rule also on the executed prefix; add a descend candidate when >1 m above the goal")
    ap.add_argument("--alt-band", action="store_true",
                    help="executed prefix must keep altitude within 1 m of the [start, goal] altitude band")
    ap.add_argument("--goal-arc-yaw", type=float, default=0.0,
                    help="if >0, add a goal_arc candidate: fly at --fwd-dx while yawing toward the goal "
                         "at up to this many rad/step, then straight. Scored like every other candidate; "
                         "0 leaves the candidate set unchanged")
    ap.add_argument("--fwd-hold-m", type=float, default=0.0,
                    help="if >0 and straight forward still clears the survival threshold, keep it unless the "
                         "chosen candidate beats its score by at least this many meters")
    ap.add_argument("--move-dx", type=float, default=0.0,
                    help="if >0, yaw_fwd and goal_turn_fwd steps that yaw also advance at least this far, "
                         "so a one-step replan does not stop and spin")
    ap.add_argument("--move-yaw", type=float, default=0.0,
                    help="if >0 together with --move-dx, cap that yaw (rad/step) so the moving turn is an arc")
    ap.add_argument("--yield-sag", type=float, default=0.0,
                    help="if >0, when the chosen fwd or goal_arc still clears the survival line but its survival "
                         "is below this, fly a moving yaw_fwd or goal_turn_fwd that survives better. "
                         "0 keeps the progress pick")
    ap.add_argument("--skirt-min-d", type=float, default=0.0,
                    help="if >0, after --skirt-after-m of progress and while the goal is still at least this "
                         "far, fly a shallow side arc when forward and goal_arc both miss survival, both "
                         "side arcs clear survival, and they are within --skirt-tie. The upper image picks "
                         "the side. A one-sided or wider gap stays with the consequence picker. 0 leaves "
                         "the no-pass spin unchanged")
    ap.add_argument("--skirt-after-m", type=float, default=40.0,
                    help="distance already closed before --skirt-min-d may replace the no-pass spin")
    ap.add_argument("--skirt-yaw", type=float, default=0.10,
                    help="yaw of the skirt arc (rad/step), held to one side")
    ap.add_argument("--skirt-yaw-cap-deg", type=float, default=35.0,
                    help="hold the skirt side until the nose has yawed this far, then allow one flip")
    ap.add_argument("--skirt-release", type=int, default=1,
                    help="open steps required before alignment may take the nose back from a skirt")
    ap.add_argument("--skirt-tie", type=float, default=0.03,
                    help="both side arcs must clear survival and differ by less than this before a skirt starts")
    ap.add_argument("--skirt-gap", type=float, default=0.10,
                    help="the farther upper half must exceed the nearer by this fraction, or the skirt does not start")
    ap.add_argument("--align-bearing-deg", type=float, default=0.0,
                    help="if >0, once the nose is at least this far off the goal, fly goal_arc when it survives "
                         "and is within 0.02 of straight-forward survival")
    ap.add_argument("--trim-yaw", type=float, default=0.0,
                    help="if >0, goal_arc yaws at most this many rad/step while the heading error is inside "
                         "--trim-bearing-deg, so a small error becomes a shallow arc instead of a late corner")
    ap.add_argument("--trim-bearing-deg", type=float, default=25.0,
                    help="heading error below this uses --trim-yaw; at or above it, goal_arc uses --goal-arc-yaw")
    ap.add_argument("--align-slack", type=float, default=0.02,
                    help="goal_arc may be this much less safe than straight forward and still be forced")
    ap.add_argument("--slide-dx", type=float, default=0.0,
                    help="if >0, add an open_slide candidate: move this far forward while sliding and yawing "
                         "toward the more open image side. 0 leaves the candidate set unchanged")
    ap.add_argument("--slide-dy", type=float, default=0.35,
                    help="lateral step (m) of open_slide, toward the open side")
    ap.add_argument("--slide-yaw", type=float, default=0.08,
                    help="yaw (rad/step) of open_slide, toward the open side")
    ap.add_argument("--spin-yaw", type=float, default=0.0,
                    help="if >0, the in-place fallback yaws at most this many rad/step instead of the full limit")
    ap.add_argument("--near-tight-bearing-deg", type=float, default=0.0,
                    help="inside the near-goal rule, a heading error above this uses the tight arc; 0 keeps 0.25 rad")
    ap.add_argument("--near-tight-yaw", type=float, default=0.30,
                    help="yaw cap (rad/step) of the tight near arc")
    ap.add_argument("--near-tight-dx", type=float, default=0.40,
                    help="forward step (m) of the tight near arc")
    ap.add_argument("--near-arc-yaw", type=float, default=0.0,
                    help="if >0, inside the near-goal rule fly a moving arc toward the goal at up to this yaw "
                         "(rad/step) instead of a 10-step plan whose first step is only a turn")
    ap.add_argument("--fwd-dx", type=float, default=0.8, help="forward command of the fwd family")
    ap.add_argument("--dx-scale", type=float, default=1.0,
                    help="scale forward commands of the sampled candidates, capped at 1.0 (the training range)")
    ap.add_argument("--goal-surv-min", type=float, default=0.0,
                    help="if >0, survival threshold used for the pick within --goal-surv-dist of the goal")
    ap.add_argument("--goal-surv-dist", type=float, default=8.0)
    ap.add_argument("--extra-families", default="",
                    help="comma list from EXTRA_MACROS (hover, spin, back) added to the candidates")
    ap.add_argument("--fov-max-deg", type=float, default=0.0,
                    help="if >0, executed horizontal motion must be within this angle of the nose (camera sees +-45)")
    ap.add_argument("--escape-after-s", type=float, default=0.0,
                    help="if >0, after this long without 1 m of goal progress, spend --escape-s choosing the survivor "
                         "with the largest predicted forward clearance instead of the largest goal progress")
    ap.add_argument("--escape-s", type=float, default=4.0)
    ap.add_argument("--commit-after-s", type=float, default=0.0,
                    help="if >0, within --commit-dist of the goal and this long without 1 m of goal progress, lock one "
                         "side (the one whose surviving candidates have the larger predicted forward clearance) for "
                         "--commit-s; picks stay goal-progress among that side's candidates")
    ap.add_argument("--commit-s", type=float, default=4.0)
    ap.add_argument("--commit-dist", type=float, default=15.0)
    ap.add_argument("--follow-after-s", type=float, default=0.0,
                    help="if >0, within --follow-dist of the goal and this long without 1 m of goal progress, follow the "
                         "obstacle boundary on one side of the goal direction (follow_pick) until --follow-leave-m closer "
                         "than where it started with the goal direction clear; one side switch after --follow-max-s, a "
                         "270 deg turn or getting --follow-away-m farther than where that side started; the leave "
                         "distance is kept across the switch; after both sides, back to the default rule")
    ap.add_argument("--follow-dist", type=float, default=15.0)
    ap.add_argument("--follow-leave-m", type=float, default=1.5)
    ap.add_argument("--follow-leave-deg", type=float, default=20.0)
    ap.add_argument("--follow-max-s", type=float, default=8.0)
    ap.add_argument("--follow-away-m", type=float, default=4.0)
    ap.add_argument("--follow-mline-m", type=float, default=0.0,
                    help="if >0 (Bug2), leaving the boundary also requires being within this distance of the "
                         "start-goal line")
    ap.add_argument("--progress-ckpts", default="",
                    help="comma list of checkpoints whose dgoal head gives the goal progress (e.g. geodesic-label "
                         "models); survival and clearance still come from --ckpts")
    ap.add_argument("--nopass-margin", type=float, default=0.0,
                    help="if >0, when no allowed candidate reaches the survival threshold, pick the most progressing "
                         "moving candidate (>= 0.5 m net, predicted progress >= --nopass-min-prog) whose survival is "
                         "within this margin of the safest allowed one and >= --nopass-floor, instead of the safest")
    ap.add_argument("--park-before-reset", type=float, default=0.0,
                    help="if >0, hover this many seconds --park-alt-m above the start before the reset teleport; the "
                         "teleport keeps the velocity the previous episode ended with")
    ap.add_argument("--park-alt-m", type=float, default=0.0,
                    help="parking 20-40 m up made the eval drone sink ~3 m on its first step after the teleport down")
    ap.add_argument("--settle-after-reset", type=float, default=0.0,
                    help="if >0, hover after reset until the speed drops below --settle-speed or this many seconds "
                         "pass, then observe again; reset does not zero the velocity")
    ap.add_argument("--settle-speed", type=float, default=0.1)
    ap.add_argument("--brake-speed", type=float, default=0.0,
                    help="if >0, when no allowed candidate reaches the survival threshold (and no nopass pick) and the "
                         "horizontal speed exceeds this (m/s), command the opposite velocity for one step instead of "
                         "the safest candidate; zero-velocity picks only coast to a stop")
    ap.add_argument("--brake-hold-alt", action="store_true",
                    help="brake steps also steer the height toward the middle of the start/goal altitude band; braking "
                         "alone climbs ~0.15 m per stop and lifted od_hard_02 above SM_NYCB_579's rendered roof, whose "
                         "collision extends ~7 m higher")
    ap.add_argument("--spin-nopass", action="store_true",
                    help="when no allowed candidate reaches the survival threshold (and no nopass pick or brake), turn "
                         "in place at full yaw rate toward the more open image half, keeping that side until one passes; "
                         "the safest candidate otherwise still creeps forward (every family has dx >= 0)")
    ap.add_argument("--mem-clear-m", type=float, default=0.0,
                    help="if >0, disallow candidates whose first --mem-steps waypoints come within this horizontal "
                         "distance of points seen in the last --mem-frames model-input depth frames (walls that left "
                         "the view), or closer than the drone already is; ignored when it would disallow every "
                         "candidate")
    ap.add_argument("--mem-steps", type=int, default=3)
    ap.add_argument("--mem-frames", type=int, default=25)
    ap.add_argument("--mem-range", type=float, default=6.0)
    ap.add_argument("--mem-dz", type=float, default=1.0)
    ap.add_argument("--seen-clear-m", type=float, default=0.0,
                    help="if >0, drop a candidate whose first step comes within this many meters horizontally "
                         "of a surface seen on this flight. The newest view of a direction replaces older points. "
                         "A frame whose camera is farther than --seen-keep-m is dropped. Points more than "
                         "--seen-dz-m off the current altitude are ignored. An empty set brakes or spins; "
                         "it is not filled back in. 0 leaves the candidate set unchanged")
    ap.add_argument("--seen-keep-m", type=float, default=40.0,
                    help="drop a remembered view once the aircraft is farther than this from that camera")
    ap.add_argument("--seen-dz-m", type=float, default=8.0,
                    help="ignore remembered points farther than this above or below the aircraft")
    ap.add_argument("--no-corner-turn", action="store_true",
                    help="do not replace the chosen step with an in-place yaw toward the open side")
    ap.add_argument("--schematic", default="",
                    help="prior schematic JSON. A grid is occupancy; polylines are optional and ignored when a grid is set. Empty keeps the task goal")
    ap.add_argument("--scene-memory", default="",
                    help="seen-scene JSON. Empty leaves memory off. Loaded if present, written when --scene-mode records")
    ap.add_argument("--scene-mode", choices=("record", "use", "both"), default="both",
                    help="record writes depth views and does not steer; use places the subgoal from those "
                         "views and drops a schematic edge whose heading was blocked; both does both")
    ap.add_argument("--scene-horizon-m", type=float, default=8.0,
                    help="subgoal distance cap, the consequence horizon (drone ~8 m). Not a candidate veto")
    ap.add_argument("--scene-pose", choices=("metric", "rough"), default="metric",
                    help="metric keeps the live pose; rough applies a remembered offset at the live pose after a scene match")
    ap.add_argument("--use-far", action="store_true",
                    help="place the subgoal on this flight's far continuation. Off unless set. "
                         "Needs --scene-memory with --scene-mode use or both. Does not change survival or drop candidates")
    ap.add_argument("--nopass-floor", type=float, default=0.85)
    ap.add_argument("--nopass-min-prog", type=float, default=-1.0)
    ap.add_argument("--nopass-stall-prog", type=float, default=0.0,
                    help="if >0 (with --nopass-margin), when the best passing candidate predicts less progress than "
                         "this and the pick barely moves, take the most progressing passing candidate that moves")
    ap.add_argument("--near-goal-dist", type=float, default=0.0,
                    help="if >0, within this distance of the goal and --near-goal-after-s without 1 m of goal "
                         "progress, add slow goal-turn candidates (--near-goal-dx) and pick the most progressing "
                         "survivor heading within --near-goal-deg of the goal; survival threshold unchanged")
    ap.add_argument("--near-goal-after-s", type=float, default=3.0)
    ap.add_argument("--near-goal-dx", default="0.3,0.5")
    ap.add_argument("--near-goal-deg", type=float, default=30.0)
    ap.add_argument("--keep-frames", type=int, default=15,
                    help="frames kept in last_frames_*.npz; 0 keeps the whole flight")
    ap.add_argument("--use-spawn-z", action="store_true",
                    help="start and goal at the validated spawn_z_ok (validate_od_routes_airsim) instead of the annotation z")
    ap.add_argument("--hover-after-reset", action="store_true",
                    help="hold a hover from reset until the first action; the first plan takes ~2 s and an "
                         "uncommanded drone can sink several metres in that gap")
    ap.add_argument("--alt-band-return", type=float, default=0.0,
                    help="outside the band, required commanded return toward it per executed step (m)")
    ap.add_argument("--tune-on-train", action="store_true",
                    help="fly train routes instead, for choosing decision settings without touching held-out routes")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.follow_after_s > 0 and (args.commit_after_s > 0 or args.escape_after_s > 0):
        raise SystemExit("--follow-after-s excludes --commit-after-s and --escape-after-s")

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    import yaml

    from experiments.aerial.rl.collect_consequence_dataset import (
        EXTRA_MACROS, LIMITS, encode_jpeg, goal_bearing_body, pool_min, sample_macro,
    )
    from experiments.aerial.rl.consequence_model import (
        altitude_allowed, altitude_band_allowed, depth_input, history_inputs, goal_features, load_checkpoint, select_candidate, survival,
        visible_motion_allowed,
    )
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.precompute_env_consequence import env_pair
    from experiments.aerial.rl.scene_memory import open_guide
    from experiments.aerial.rl.probe_counterfactual_airsim import FAMILIES
    from experiments.aerial.rl.train_rl import _build_env

    extra = tuple(s for s in args.extra_families.split(",") if s)
    if any(s not in EXTRA_MACROS for s in extra):
        raise SystemExit(f"unknown extra families {extra}; choose from {EXTRA_MACROS}")
    fams = tuple(FAMILIES) + extra
    dev = torch.device(args.device)
    models = [load_checkpoint(root / p, device=args.device) for p in args.ckpts.split(",") if p]
    prog_models = [load_checkpoint(root / p, device=args.device) for p in args.progress_ckpts.split(",") if p]
    for m in models + prog_models:
        m.eval()
    hist_len = max([8] + [max(m.frame_offsets) + 1 for m in models + prog_models]
                   + [m.hist_actions + 1 for m in models + prog_models])
    H = models[0].horizon
    fan = follow_fan(H, float(LIMITS[3]))
    near_dx = [float(x) for x in args.near_goal_dx.split(",") if x]
    pred = DepthMinPredictor.from_checkpoint(root / args.depth_ckpt, device=args.device)

    routes = json.loads((root / args.annotation).read_text(encoding="utf-8"))["episodes"]
    want = "train" if args.tune_on_train else "holdout"
    if any(r.get("complexity_meta", {}).get("split") != want for r in routes):
        raise SystemExit(f"refusing: this mode flies {want} routes only")
    only = [r for r in args.routes.split(",") if r]
    if only:
        routes = [r for r in routes if r["route_id"] in only]

    cfg = yaml.safe_load((root / args.config).read_text()) or {}
    env_cfg = dict(cfg.get("env") or {})
    env_cfg.update({"backend": "airsim", "step_hz": 5.0, "grab_depth": True})
    env = _build_env(env_cfg)
    step_hz = 5.0

    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "MPC_EVAL.json"
    results: Dict[str, Any] = (json.loads(res_path.read_text()) if res_path.is_file()
                               else {"args": vars(args), "episodes": []})
    done_ids = {e["route_id"] for e in results["episodes"]}
    rng = np.random.default_rng(args.seed)
    guide = open_guide(args.schematic, args.scene_memory, args.scene_horizon_m, args.scene_pose, args.scene_mode,
                       args.use_far)

    def score(dseq: List[np.ndarray], aseq: List[np.ndarray], state: np.ndarray, goal: np.ndarray,
              prev: np.ndarray, cands: np.ndarray, rgb: np.ndarray, pose_seq=None):
        M = len(cands)
        g = torch.as_tensor(np.repeat(goal_features(state, goal)[None], M, 0), device=dev)
        pa = torch.as_tensor(np.repeat(prev[None].astype(np.float32), M, 0), device=dev)
        a = torch.as_tensor(cands.astype(np.float32), device=dev)
        env = None
        if any(getattr(m, "env_dim", 0) for m in models + prog_models):
            e, _ = env_pair(pred._model, [rgb])
            env = torch.as_tensor(np.repeat(e[:1].astype(np.float32), M, 0), device=dev)
        s_all, p_all, f_all, q_all = [], [], [], []
        remembered = {}
        scene = None
        if any(getattr(m, "pose_scene_dim", 0) for m in models):
            from experiments.aerial.rl.pose_scene import scene_in_body
            if pose_seq:
                grid = scene_in_body([d for d, _ in pose_seq], [st for _, st in pose_seq])
            else:
                grid = np.zeros((2, 32, 32), np.float32)
            scene = torch.as_tensor(grid, device=dev)
        scene_depth = scene_state = body = None
        if any(getattr(m, "scene_pose_dim", 0) for m in models):
            from experiments.aerial.rl.consequence_model import depth_input as _depth_input
            from experiments.aerial.rl.scene_pose import LENGTH
            past = list(pose_seq or [])[:-1][-LENGTH:]
            if past:
                scene_depth = torch.as_tensor(
                    np.stack([_depth_input(d) for d, _ in past])[None], device=dev)
                scene_state = torch.as_tensor(
                    np.stack([st for _, st in past])[None], device=dev)
                body = torch.as_tensor(np.asarray(state, np.float32)[None], device=dev)
        with torch.no_grad():
            for i, m in enumerate(models + prog_models):
                d_k, a_k = history_inputs(dseq, aseq, m.frame_offsets, m.hist_actions)
                d_k = d_k[0] if len(m.frame_offsets) == 1 else d_k
                d = torch.as_tensor(np.repeat(d_k[None], M, 0), device=dev)
                ha = torch.as_tensor(np.repeat(a_k[None], M, 0), device=dev) if m.hist_actions else None
                kwargs = {}
                if getattr(m, "env_dim", 0):
                    kwargs["env"] = env
                if i < len(models) and getattr(m, "view_memory_dim", 0):
                    if id(m) not in remembered:
                        frames = np.stack([depth_input(fr) for fr in dseq]).astype(np.float32)
                        remembered[id(m)] = m.remember(torch.as_tensor(frames, device=dev))
                    kwargs["memory"] = remembered[id(m)]
                if i < len(models) and getattr(m, "pose_scene_dim", 0):
                    kwargs["scene"] = scene
                if i < len(models) and getattr(m, "scene_pose_dim", 0) and scene_depth is not None:
                    kwargs["scene_depth"] = scene_depth
                    kwargs["scene_state"] = scene_state
                    kwargs["body_state"] = body
                o = m(d, g, pa, a, ha, **kwargs)
                if i >= len(models):
                    q_all.append((-o["dgoal"][..., -1]).cpu().numpy())
                    continue
                s_all.append(survival(o["hazard_logit"])[..., -1].cpu().numpy())
                p_all.append((-o["dgoal"][..., -1]).cpu().numpy())
                f_all.append(o["log_cones"][..., -1, 0].cpu().numpy())
        return np.mean(s_all, 0), np.mean(q_all or p_all, 0), np.mean(f_all, 0)

    for r in routes:
        if r["route_id"] in done_ids:
            continue
        pos = np.asarray(r["pos"], dtype=np.float64)
        start, goal = pos[0].copy(), pos[-1].copy()
        if args.use_spawn_z and r.get("complexity_meta", {}).get("spawn_z_ok") is not None:
            start[2] = goal[2] = float(r["complexity_meta"]["spawn_z_ok"])
        ref_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

        def mline_dist(xy: np.ndarray, a: np.ndarray = start[:2].copy(), b: np.ndarray = goal[:2].copy()) -> float:
            ab = b - a
            t = float(np.clip(np.dot(xy - a, ab) / max(1e-9, float(np.dot(ab, ab))), 0.0, 1.0))
            return float(np.linalg.norm(xy - (a + t * ab)))

        yaw0 = float(r["yaw"][0])
        if guide is not None:
            guide.bind(r["route_id"], goal)
        budget_len = ref_len
        if args.budget_ref == "astar":
            astar = r.get("complexity_meta", {}).get("astar_1p5_m")
            if astar is None:
                raise SystemExit(f"{r['route_id']}: --budget-ref astar needs complexity_meta.astar_1p5_m")
            budget_len = float(astar)
        max_steps = int(args.max_steps_per_m * budget_len)
        if args.park_before_reset > 0:
            c, airsim_ = env._connect(), env._airsim
            c.enableApiControl(True, **env._vk)
            c.armDisarm(True, **env._vk)
            c.simSetVehiclePose(airsim_.Pose(airsim_.Vector3r(float(start[0]), float(start[1]),
                                                              -(float(start[2]) + args.park_alt_m)),
                                             airsim_.to_quaternion(0.0, 0.0, 0.0)), True, **env._vk)
            c.hoverAsync(**env._vk)
            time.sleep(args.park_before_reset)
        try:
            obs = env.reset({"pos": [start.tolist(), goal.tolist()], "yaw": [yaw0, yaw0]})
        except RuntimeError as exc:
            logger.warning("%s reset failed: %s", r["route_id"], exc)
            results["episodes"].append({"route_id": r["route_id"], "end_reason": "reset_failed"})
            res_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
            continue
        if args.hover_after_reset or args.settle_after_reset > 0:
            env._client.hoverAsync(**env._vk)
        settle_s = 0.0
        if args.settle_after_reset > 0:
            t_settle = time.time()
            while time.time() - t_settle < args.settle_after_reset:
                v = env.observe_state()[3:6]
                if float(np.linalg.norm(v)) < args.settle_speed:
                    break
                time.sleep(0.1)
            settle_s = time.time() - t_settle
            obs = env.observe()
        pred.reset()
        get_contact = getattr(env, "collision_info", lambda: {})
        reset_contact = get_contact()
        z_ref = float(np.asarray(obs.position)[2])
        z_lo, z_hi = min(z_ref, float(goal[2])) - 1.0, max(z_ref, float(goal[2])) + 1.0
        d0 = float(np.linalg.norm(np.asarray(obs.position) - goal))
        best_d, best_step = d0, 0
        escape_until, escape_end = -1, 0
        commit_until, commit_end, commit_side = -1, 0, 0.0
        follow_side, follow_start, follow_end, follow_d, follow_turn, follow_switched = 0.0, 0, 0, 0.0, 0.0, False
        follow_side_d = 0.0
        spin_side = 0.0
        skirt_side, skirt_flips, skirt_yaw0, skirt_gave_up = 0.0, 0, 0.0, False
        skirt_closed, skirt_open = 0, 0
        prev = np.zeros(4, np.float32)
        plan: List[np.ndarray] = []
        steps: List[Dict[str, Any]] = []
        traj = [np.asarray(obs.position, dtype=np.float64).tolist()]
        end = "max_steps"
        t0 = time.time()
        depth = pred.predict_depth(obs)
        recent: deque = deque(maxlen=int(args.keep_frames) or None)

        def remember(o: Any, d_hat: np.ndarray) -> None:
            recent.append((encode_jpeg(np.asarray(o.rgb)), pool_min(np.asarray(d_hat, np.float32)),
                           pool_min(np.asarray(o.depth, np.float32))))

        remember(obs, depth)
        dhist: deque = deque(maxlen=hist_len)
        ahist: deque = deque(maxlen=hist_len)
        dhist.append(pool_min(np.asarray(obs.depth if args.input_depth == "gt" else depth, np.float32)))
        want_scene = any(getattr(m, "pose_scene_dim", 0) or getattr(m, "scene_pose_dim", 0) for m in models)
        pose_hist: List[tuple] = []

        def note_pose() -> None:
            if not want_scene:
                return
            pose_hist.append((np.array(dhist[-1], np.float32, copy=True),
                              np.asarray(obs.state, np.float64).copy()))

        note_pose()
        mem: deque = deque(maxlen=max(1, int(args.mem_frames)))
        seen_hist: List[tuple] = []

        def remember_seen() -> None:
            if args.seen_clear_m <= 0:
                return
            seen_hist.append((np.array(dhist[-1], np.float32, copy=True),
                              np.asarray(obs.state, np.float64).copy()))

        if args.mem_clear_m > 0:
            mem.append(depth_points(dhist[-1], np.asarray(obs.state, np.float64), args.mem_range))
        remember_seen()
        scene_kf, scene_surv, scene_yaw = None, 1.0, float(np.asarray(obs.state, np.float64)[6])
        for step in range(max_steps):
            if not plan or step % int(args.replan_every) == 0:
                state = np.asarray(obs.state, dtype=np.float64)
                aim = goal
                subgoal_source, kf_id = "goal", None
                placed_source = "goal"
                placed_aim = aim
                if guide is not None:
                    vector = None
                    if guide.wants_vector:
                        vector, _ = env_pair(pred._model, [np.asarray(obs.rgb)])
                        vector = np.asarray(vector[0], np.float32)
                    depth_view = np.array(dhist[-1], np.float32, copy=True) if guide.record else None
                    proposal = guide.step_view(state, goal, vector, depth_view)
                    aim, subgoal_source, kf_id = proposal.aim, proposal.source, proposal.kf_id
                    placed_source = subgoal_source
                    placed_aim = np.array(aim, np.float64, copy=True)
                schematic_aim = np.array(aim, np.float64, copy=True) if subgoal_source == "schematic" else None
                bearing = goal_bearing_body(state[:3], float(state[6]), aim)
                cands, names = [], []
                for fam in fams:
                    name, seq = sample_macro(rng, H, bearing, name=fam)
                    if fam == "fwd":
                        seq[:, 0] = args.fwd_dx
                    if args.move_dx > 0 and name in ("yaw_fwd", "goal_turn_fwd"):
                        seq = smooth_turn(seq, args.move_dx, args.move_yaw)
                    if args.dx_scale != 1.0:
                        seq[:, 0] = np.where(seq[:, 0] > 0, np.minimum(seq[:, 0] * args.dx_scale, 1.0), seq[:, 0])
                    cands.append(seq)
                    names.append(name)
                if args.alt_fix and state[2] > goal[2] + 1.0:
                    name, seq = sample_macro(rng, H, bearing, name="descend")
                    cands.append(seq)
                    names.append(name)
                if args.goal_arc_yaw > 0:
                    arc_dx = args.fwd_dx if args.dx_scale == 1.0 else min(args.fwd_dx * args.dx_scale, 1.0)
                    arc_cap = cruise_yaw_cap(bearing, args.goal_arc_yaw, args.trim_yaw,
                                             math.radians(args.trim_bearing_deg))
                    cands.append(goal_arc(bearing, H, arc_dx, arc_cap))
                    names.append("goal_arc")
                if args.slide_dx > 0:
                    side = open_turn_sign(dhist[-1])
                    view = args.fov_max_deg - 3.0 if args.fov_max_deg > 0 else 32.0
                    sdx, sdy = fit_slide(args.slide_dx, args.slide_dy, view)
                    cands.append(open_slide(side, H, sdx, sdy, args.slide_yaw))
                    names.append("open_slide")
                d_here = float(np.linalg.norm(state[:3] - goal))
                near_due = (args.near_goal_dist > 0 and d_here <= args.near_goal_dist
                            and (step - best_step) / step_hz >= args.near_goal_after_s)
                if near_due:
                    if args.near_arc_yaw > 0:
                        near_dx_cmd = args.fwd_dx if args.dx_scale == 1.0 else min(args.fwd_dx * args.dx_scale, 1.0)
                        # A cruise arc turns on about a 4 m radius and skims past the 3 m ball.
                        # Tighten while the nose is still well off the goal.
                        tight_rad = (math.radians(args.near_tight_bearing_deg)
                                     if args.near_tight_bearing_deg > 0 else 0.25)
                        if abs(bearing) > tight_rad:
                            cap, near_dx_cmd = args.near_tight_yaw, min(near_dx_cmd, args.near_tight_dx)
                        else:
                            cap = args.near_arc_yaw
                        cands.append(goal_arc(bearing, H, near_dx_cmd, cap))
                        names.append("near_arc")
                    for dx in near_dx:
                        _, seq = sample_macro(rng, H, bearing, name="goal_turn_fwd")
                        seq[seq[:, 3] == 0, 0] = dx
                        cands.append(seq)
                        names.append("near_creep")
                cands = np.stack(cands)
                follow_due = (args.follow_after_s > 0 and not follow_side and d_here <= args.follow_dist
                              and (step - max(best_step, follow_end)) / step_hz >= args.follow_after_s)
                if follow_side or follow_due:
                    cands = np.concatenate([cands, fan])
                    names += ["follow_fan"] * len(fan)
                allowed = altitude_allowed(cands, state, goal)
                if args.alt_fix:
                    allowed &= altitude_allowed(cands[:, :int(args.replan_every)], state, goal)
                if args.alt_band:
                    banded = allowed & altitude_band_allowed(
                        cands, state[2], z_lo, z_hi, int(args.replan_every), args.alt_band_return)
                    allowed = banded if banded.any() else allowed
                if args.fov_max_deg > 0:
                    seen = allowed & visible_motion_allowed(cands, int(args.replan_every), args.fov_max_deg)
                    allowed = seen if seen.any() else allowed
                mem_blocked = 0
                if args.mem_clear_m > 0:
                    clear = memory_clear(candidate_waypoints(cands, state, args.mem_steps), state[:3],
                                         np.concatenate(mem), args.mem_clear_m, args.mem_dz)
                    mem_blocked = int((allowed & ~clear).sum())
                    allowed = allowed & clear if (allowed & clear).any() else allowed
                seen_blocked = 0
                if args.seen_clear_m > 0 and seen_hist:
                    seen_ok = step_seen_clear(seen_hist, state, cands, args.seen_clear_m,
                                              args.seen_keep_m, args.seen_dz_m)
                    seen_blocked = int((allowed & ~seen_ok).sum())
                    allowed = allowed & seen_ok
                if subgoal_source == "dropped":
                    s = np.zeros(len(cands), np.float64)
                    p = np.zeros(len(cands), np.float64)
                    fwd_clear = np.zeros(len(cands), np.float64)
                else:
                    s, p, fwd_clear = score(list(dhist), list(ahist), state, aim, prev, cands, np.asarray(obs.rgb),
                                            pose_hist)
                    surv_for_aim = (args.goal_surv_min if args.goal_surv_min > 0 and d_here <= args.goal_surv_dist
                                    else args.surv_min)
                    if (guide is not None and subgoal_source not in ("goal", "seen")
                            and not (allowed & (s >= surv_for_aim)).any()):
                        if guide.blocks_heading(state[:3], goal):
                            subgoal_source = "dropped"
                            s = np.zeros(len(cands), np.float64)
                            p = np.zeros(len(cands), np.float64)
                            fwd_clear = np.zeros(len(cands), np.float64)
                        else:
                            s, p, fwd_clear = score(list(dhist), list(ahist), state, goal, prev, cands,
                                                    np.asarray(obs.rgb), pose_hist)
                            aim, subgoal_source = goal, "goal_fallback"
                if (args.escape_after_s > 0 and step >= escape_until
                        and (step - max(best_step, escape_end)) / step_hz >= args.escape_after_s):
                    escape_until = step + int(round(args.escape_s * step_hz))
                    escape_end = escape_until
                escaping = step < escape_until
                pick_allowed = allowed
                if args.commit_after_s > 0:
                    side = candidate_side(cands)
                    safe = allowed & (s >= args.surv_min)
                    if (step >= commit_until and np.linalg.norm(state[:3] - goal) <= args.commit_dist
                            and (step - max(best_step, commit_end)) / step_hz >= args.commit_after_s):
                        best = {sd: float(fwd_clear[safe & (side == sd)].max()) for sd in (-1.0, 1.0)
                                if (safe & (side == sd)).any()}
                        if best:
                            commit_side = max(best, key=best.get)
                            commit_until = step + int(round(args.commit_s * step_hz))
                            commit_end = commit_until
                    if step < commit_until and (safe & (side == commit_side)).any():
                        pick_allowed = allowed & (side == commit_side)
                committing = pick_allowed is not allowed
                follow_c, alpha, nopass_used, braking, spinning, holding = -1, None, False, False, False, False
                skirting, skirt_log, gap_sign, gap_m = False, None, None, None
                reorienting = False
                aim_yaw = 0.0
                if follow_side or follow_due:
                    alpha, dlen = goal_offset(cands, bearing, 1.0 / step_hz)
                    safe = allowed & (s >= args.surv_min)
                    if follow_due:
                        follow_side = follow_side_choice(alpha, dlen, safe, fwd_clear, bearing)
                        follow_start, follow_d, follow_turn, follow_switched = step, d_here, 0.0, False
                        follow_side_d = d_here
                    elif (d_here <= follow_d - args.follow_leave_m
                          and (args.follow_mline_m <= 0 or mline_dist(state[:2]) <= args.follow_mline_m)
                          and (safe & (dlen >= 0.5) & (np.abs(alpha) <= math.radians(args.follow_leave_deg))).any()):
                        follow_side, follow_end = 0.0, step
                    elif ((step - follow_start) / step_hz >= args.follow_max_s or abs(follow_turn) >= 1.5 * math.pi
                          or d_here > follow_side_d + args.follow_away_m):
                        if follow_switched:
                            follow_side, follow_end = 0.0, step
                        else:
                            follow_side, follow_start, follow_turn, follow_side_d = -follow_side, step, 0.0, d_here
                            follow_switched = True
                    if follow_side:
                        follow_c = follow_pick(alpha, dlen, safe, follow_side)
                        if follow_c < 0:
                            spin = [i for i in range(len(cands)) if names[i] == "follow_fan" and allowed[i]
                                    and dlen[i] < 0.5 and np.sign(cands[i, 0, 3]) == follow_side]
                            follow_c = spin[0] if spin else -1
                surv_pick = (args.goal_surv_min if args.goal_surv_min > 0 and d_here <= args.goal_surv_dist
                             else args.surv_min)
                if args.policy == "goal_line":
                    c = names.index("goal_turn_fwd")
                elif follow_c >= 0:
                    c = follow_c
                elif near_due and args.near_arc_yaw > 0 and (
                        near_c := choose_near(names, cands, s, pick_allowed, bearing, surv_pick)) >= 0:
                    c = near_c
                elif near_due and args.near_arc_yaw <= 0 and (
                        near_c := near_pick(cands, bearing, allowed & (s >= args.surv_min), p,
                                            1.0 / step_hz, args.near_goal_deg)) >= 0:
                    c = near_c
                else:
                    c = select_candidate(s, fwd_clear if escaping else p, pick_allowed, surv_pick)
                    passing = pick_allowed & (s >= surv_pick)
                    c_np = -1
                    if args.nopass_margin > 0 and not escaping and pick_allowed.any():
                        if not passing.any():
                            c_np = nopass_pick(cands, bearing, s, p, pick_allowed, args.nopass_margin,
                                               args.nopass_floor, args.nopass_min_prog, 1.0 / step_hz)
                        elif (args.nopass_stall_prog > 0 and float(p[passing].max()) < args.nopass_stall_prog
                              and goal_offset(cands[c:c + 1], bearing, 1.0 / step_hz)[1][0] < 0.5):
                            c_np = nopass_pick(cands, bearing, s, p, passing, 1.0, surv_pick,
                                               args.nopass_min_prog, 1.0 / step_hz)
                    if c_np >= 0:
                        c, nopass_used = c_np, True
                    elif (args.brake_speed > 0 and not passing.any()
                          and math.hypot(float(state[3]), float(state[4])) > args.brake_speed):
                        braking = True
                    elif args.spin_nopass and not passing.any() and subgoal_source == "seen":
                        if (args.brake_speed > 0
                                and math.hypot(float(state[3]), float(state[4])) > args.brake_speed):
                            braking = True
                        elif abs(float(bearing)) > math.radians(20.0):
                            cap = min(float(LIMITS[3]), args.spin_yaw) if args.spin_yaw > 0 else float(LIMITS[3])
                            reorienting, aim_yaw = True, stall_yaw(float(bearing), cap)
                        else:
                            holding = True
                    elif args.spin_nopass and not passing.any():
                        spinning = True
                        spin_side = spin_side or open_turn_sign(dhist[-1])
                    elif not pick_allowed.any():
                        spinning = True
                        spin_side = spin_side or open_turn_sign(dhist[-1])
                    if passing.any():
                        spin_side = 0.0
                    if args.fwd_hold_m > 0 and not braking and not spinning:
                        c = hold_straight(names, s, fwd_clear if escaping else p, pick_allowed, c,
                                          surv_pick, args.fwd_hold_m)
                    if args.slide_dx > 0 and args.move_dx > 0 and not braking and not spinning:
                        c = prefer_motion(names, cands, s, pick_allowed, c, surv_pick)
                    if args.align_bearing_deg > 0 and not braking and not spinning:
                        aligned = choose_align(names, s, pick_allowed, bearing, surv_pick,
                                               math.radians(args.align_bearing_deg), args.align_slack)
                        if aligned >= 0:
                            c = aligned
                    if args.yield_sag > 0 and not braking and not spinning:
                        c = prefer_safer_turn(names, cands, s, pick_allowed, c, surv_pick, args.yield_sag)
                    if args.skirt_min_d > 0:
                        goal_open = any(names[i] in ("fwd", "goal_arc") and bool(pick_allowed[i])
                                        and float(s[i]) >= float(surv_pick) for i in range(len(names)))
                        far = d_here >= args.skirt_min_d and (d0 - d_here) >= args.skirt_after_m
                        side_surv = None
                        if far and not goal_open and not skirt_gave_up:
                            yaw_step = args.skirt_yaw if args.skirt_yaw > 0 else 0.10
                            arcs = np.stack([
                                side_arc(-1.0, H, args.fwd_dx, yaw_step),
                                side_arc(1.0, H, args.fwd_dx, yaw_step),
                            ])
                            s_side, _, _ = score(list(dhist), list(ahist), state, aim, prev, arcs,
                                                 np.asarray(obs.rgb), pose_hist)
                            side_surv = (float(s_side[0]), float(s_side[1]))
                            skirt_log = side_surv
                            left_m, right_m = upper_halves(dhist[-1])
                            if left_m is not None:
                                gap_m = (left_m, right_m)
                                gap_sign = upper_gap_sign(dhist[-1], args.skirt_gap)
                        start = bool(side_surv is not None and gap_sign is not None
                                     and skirt_tie(side_surv[0], side_surv[1], surv_pick, args.skirt_tie))
                        do_skirt, clear, skirt_closed, skirt_open = skirt_gate(
                            goal_open, far, skirt_closed, skirt_open, skirt_side, skirt_gave_up,
                            max(1, int(args.skirt_release)), start)
                        if clear:
                            skirt_side, skirt_flips, skirt_gave_up = 0.0, 0, False
                        if do_skirt and skirt_side != 0.0 and goal_open:
                            skirting, braking, spinning = True, False, False
                        elif do_skirt:
                            proposal = float(gap_sign) if gap_sign is not None else float(
                                skirt_side if skirt_side != 0.0 else 1.0)
                            yawed = 0.0 if skirt_side == 0.0 else (
                                (float(state[6]) - skirt_yaw0 + math.pi) % (2.0 * math.pi) - math.pi)
                            side, skirt_flips, gave_up = skirt_hold(
                                skirt_side, yawed, math.radians(args.skirt_yaw_cap_deg), proposal, skirt_flips)
                            if gave_up or side == 0.0:
                                skirt_side, skirt_gave_up = 0.0, True
                            else:
                                if side != skirt_side:
                                    skirt_yaw0 = float(state[6])
                                skirt_side = float(side)
                                skirting, braking, spinning = True, False, False
                if subgoal_source == "dropped":
                    fast = args.brake_speed > 0 and math.hypot(float(state[3]), float(state[4])) > args.brake_speed
                    if fast:
                        braking, spinning, skirting = True, False, False
                    else:
                        braking, spinning, skirting = False, True, False
                        spin_side = spin_side or open_turn_sign(dhist[-1])
                elif (args.seen_clear_m <= 0 and not args.no_corner_turn
                      and not (guide is not None and guide.use_seen)
                      and not braking and not spinning and not skirting
                      and not reorienting
                      and (side_turn := corner_turn(dhist[-1])) is not None
                      and (abs(float(cands[c, 0, 0])) > 0.2 or abs(float(cands[c, 0, 1])) > 0.2
                           or float(cands[c, 0, 3]) * side_turn <= 0.0)):
                    # One side is the corner, the other side is open. Do not fly the step.
                    cap = min(float(LIMITS[3]), args.spin_yaw) if args.spin_yaw > 0 else float(LIMITS[3])
                    fast = args.brake_speed > 0 and math.hypot(float(state[3]), float(state[4])) > args.brake_speed
                    if fast:
                        braking, reorienting = True, True
                        aim_yaw = float(side_turn) * cap
                    else:
                        reorienting, aim_yaw = True, float(side_turn) * cap
                elif (not braking and not spinning and not skirting and schematic_aim is not None
                      and leaves_point(state[:2], float(state[6]), schematic_aim[:2], cands[c, 0])):
                    # Only this step is flown. If it walks away from the schematic point, yaw
                    # toward the point instead. The 10-step progress score can still look fine.
                    cap = min(float(LIMITS[3]), args.spin_yaw) if args.spin_yaw > 0 else float(LIMITS[3])
                    yaw_cmd = stall_yaw(
                        goal_bearing_body(state[:3], float(state[6]), schematic_aim), cap)
                    fast = args.brake_speed > 0 and math.hypot(float(state[3]), float(state[4])) > args.brake_speed
                    if fast:
                        braking, reorienting = True, True
                    else:
                        reorienting, aim_yaw = True, float(yaw_cmd)
                plan = [np.asarray(a, np.float64) for a in cands[c]]
                hold_z = 0.5 * (z_lo + z_hi) if args.brake_hold_alt else None
                if braking:
                    plan = [brake_action(state, step_hz, np.asarray(LIMITS, np.float64), hold_z)] + [np.zeros(4)] * (H - 1)
                elif reorienting:
                    plan = [np.array([0.0, 0.0, alt_hold_dz(state, hold_z), aim_yaw])] + [np.zeros(4)] * (H - 1)
                elif holding:
                    plan = [np.array([0.0, 0.0, alt_hold_dz(state, hold_z), 0.0])] + [np.zeros(4)] * (H - 1)
                elif spinning:
                    spin_yaw = min(float(LIMITS[3]), args.spin_yaw) if args.spin_yaw > 0 else float(LIMITS[3])
                    plan = [np.array([0.0, 0.0, alt_hold_dz(state, hold_z), spin_side * spin_yaw])] + [np.zeros(4)] * (H - 1)
                elif skirting:
                    yaw_step = args.skirt_yaw if args.skirt_yaw > 0 else 0.10
                    plan = [np.asarray(a, np.float64) for a in side_arc(skirt_side, H, args.fwd_dx, yaw_step)]
                family = ("brake" if braking else "hold" if holding else "yaw_aim" if reorienting
                          else "spin_nopass" if spinning else "skirt" if skirting else names[c])
                if guide is not None:
                    scene_kf = kf_id
                    scene_yaw = float(state[6])
                    # The in-place turn is not a miss. Recording survival 0 would block each
                    # heading on the way to the schematic point, including that point's own.
                    if reorienting or holding:
                        scene_surv = max(float(s[c]), float(args.surv_min))
                    else:
                        scene_surv = 0.0 if (braking or spinning) else float(s[c])
                steps.append({"step": step, "family": family,
                              "surv": round(float(skirt_log[0 if skirt_side < 0 else 1]), 3) if skirting and skirt_log else round(float(s[c]), 3),
                              "skirt": 0 if not skirting else int(skirt_side),
                              "skirt_surv": None if skirt_log is None else [round(skirt_log[0], 3), round(skirt_log[1], 3)],
                              "gap": None if gap_sign is None else int(gap_sign),
                              "gap_m": None if gap_m is None else [round(gap_m[0], 2), round(gap_m[1], 2)],
                              "pred_prog": round(float(p[c]), 2), "escape": bool(escaping),
                              "commit": int(commit_side) if committing else 0,
                              "follow": int(follow_side), "near": bool(near_due), "nopass": bool(nopass_used),
                              "alpha_deg": None if alpha is None else round(math.degrees(float(alpha[c])), 1),
                              "n_pass": int((allowed & (s >= args.surv_min)).sum()),
                              "d_goal": round(float(np.linalg.norm(state[:3] - goal)), 1),
                              "yaw": round(float(state[6]), 4),
                              "z_off": round(float(state[2] - goal[2]), 2),
                              "speed": round(math.hypot(float(state[3]), float(state[4])), 2),
                              "mem_blocked": mem_blocked, "seen_blocked": seen_blocked,
                              "cand_family": names, "cand_surv": np.round(s, 3).tolist(),
                              "cand_pred_prog": np.round(p, 2).tolist(), "cand_allowed": allowed.tolist()})
                if guide is not None:
                    steps[-1]["subgoal"] = subgoal_source
                    steps[-1]["aim_m"] = round(float(np.linalg.norm(np.asarray(aim)[:2] - state[:2])), 1)
                    steps[-1]["placed_m"] = round(float(np.linalg.norm(np.asarray(placed_aim)[:2] - state[:2])), 1)
                    if guide.use_far and guide.continuation.far_point is not None:
                        far = guide.continuation.far_point
                        steps[-1]["far_m"] = round(float(np.linalg.norm(far[:2] - state[:2])), 1)
                        steps[-1]["placed"] = placed_source
            a = plan.pop(0)
            if follow_side:
                follow_turn += float(a[3]) / step_hz
            obs, _ = env.step(a)
            prev = a.astype(np.float32)
            depth = pred.predict_depth(obs)
            remember(obs, depth)
            ahist.append(prev)
            dhist.append(pool_min(np.asarray(obs.depth if args.input_depth == "gt" else depth, np.float32)))
            note_pose()
            if args.mem_clear_m > 0:
                mem.append(depth_points(dhist[-1], np.asarray(obs.state, np.float64), args.mem_range))
            remember_seen()
            p_now = np.asarray(obs.position, dtype=np.float64)
            traj.append(p_now.tolist())
            d_now = float(np.linalg.norm(p_now - goal))
            if guide is not None and guide.record:
                guide.note(scene_kf, scene_yaw, scene_surv,
                           float(np.linalg.norm(np.asarray(traj[-2]) - goal)), d_now,
                           bool(obs.collided), float(args.surv_min), p_now)
            if bool(obs.collided):
                end = "collision"
                break
            if d_now <= float(args.success_dist):
                end = "arrived"
                break
            if p_now[2] < float(args.min_z_m):
                end = "altitude_low"
                break
            if d_now <= best_d - 1.0:
                best_d, best_step = d_now, step
            elif (step - best_step) / step_hz >= float(args.stuck_s):
                end = "stuck"
                break
        p_end = np.asarray(traj[-1])
        d_final = float(np.linalg.norm(p_end - goal))
        d_min = float(min(np.linalg.norm(np.asarray(traj) - goal, axis=1)))
        fam_count: Dict[str, int] = {}
        for s_ in steps:
            fam_count[s_["family"]] = fam_count.get(s_["family"], 0) + 1
        ep = {
            "route_id": r["route_id"],
            "route_len_m": round(ref_len, 1),
            "end_reason": end,
            "arrived": end == "arrived",
            "severe_collision": end == "collision",
            "stuck": end == "stuck",
            "progress_ratio": round(float(np.clip(max(0.0, d0 - d_final) / max(1e-3, ref_len), 0, 1)), 3),
            "collided_at_step": len(traj) - 1 if end == "collision" else None,
        "d_start_m": round(d0, 1),
            "d_final_m": round(d_final, 1),
            "d_min_m": round(d_min, 1),
            "steps": len(traj) - 1,
            "wall_s": round(time.time() - t0, 1),
            "picked_families": fam_count,
            "z_range": [round(float(np.min(np.asarray(traj)[:, 2])), 1), round(float(np.max(np.asarray(traj)[:, 2])), 1)],
            "reset_contact": reset_contact,
            "settle_s": round(settle_s, 2),
            "start_offset_m": round(float(np.linalg.norm(np.asarray(traj[0]) - start)), 2),
            "collision_contact": get_contact() if end == "collision" else None,
        }
        np.savez_compressed(out / f"traj_{r['route_id']}.npz", traj=np.asarray(traj, np.float32), goal=goal)
        jpgs = [x[0] for x in recent]
        np.savez_compressed(
            out / f"last_frames_{r['route_id']}.npz",
            jpeg=np.frombuffer(b"".join(jpgs), dtype=np.uint8),
            jpeg_offsets=np.cumsum([0] + [len(b) for b in jpgs]).astype(np.int64),
            da3_depth56=np.stack([x[1] for x in recent]).astype(np.float16),
            gt_depth56=np.stack([x[2] for x in recent]).astype(np.float16),
        )
        (out / f"steps_{r['route_id']}.json").write_text(json.dumps(steps), encoding="utf-8")
        results["episodes"].append(ep)
        eps = [e for e in results["episodes"] if "arrived" in e]
        results["summary"] = {
            "n": len(eps),
            "arrived": sum(e["arrived"] for e in eps),
            "severe_collision": sum(e["severe_collision"] for e in eps),
            "stuck": sum(e["stuck"] for e in eps),
            "mean_progress_ratio": round(float(np.mean([e["progress_ratio"] for e in eps])), 3),
        }
        res_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        logger.info("%s end=%s prog=%.2f d_final=%.1f steps=%d fam=%s", r["route_id"], end,
                    ep["progress_ratio"], d_final, ep["steps"], fam_count)
        if guide is not None and guide.record:
            guide.save()
    logger.info("SUMMARY %s", results.get("summary"))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
