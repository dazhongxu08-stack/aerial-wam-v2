"""Goal-reaching with the consequence-model MPC (the deployed krs6 + env setup).

Same per-step choice as ``mpc_consequence_eval --policy model``: one candidate
10-step sequence per family, hold-altitude and in-view motion filters, then the
most predicted goal progress among candidates whose ensemble survival clears
``surv_min`` (else the safest). Survival comes from ``surv_ckpts``, progress
from ``prog_ckpts`` (the env-feature heads read the DA3 scene vector).

Optional rules from the frozen f35 recipe (all off unless set): within
``near_goal_dist`` prefer a creep aimed at the goal; when nothing clears
survival, brake if the horizontal speed exceeds ``brake_speed``, otherwise
spin toward the more open half of the depth image (``spin_nopass``).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..types import VehicleState
from .consequence_model import (
    DOG_FAMILIES, FAMILIES, LIMITS, altitude_allowed, altitude_band_allowed, goal_bearing_body, goal_features,
    heading_yaw_allowed, history_inputs, load_checkpoint, pool_min, sample_dog_macro, sample_macro,
    select_candidate, select_open_candidate, survival, visible_motion_allowed, wrap,
)


def realize_dog_commands(commands: np.ndarray, *, vx_max: float, vyaw_max: float, vx_gain: float,
                          yaw_gain: float, dt: float) -> np.ndarray:
    """Dog command sequence -> displacement the body actually travels, in the model box's units.

    ``commands`` stay in the consequence-model box: ``dx = 1`` is full forward, ``dyaw = ±LIMITS[3]``
    is full yaw. The returned array is meters and radians over ``dt`` after the measured gains.
    Lateral and vertical channels stay zero. Yaw keeps the model sign (clockwise positive).
    """
    c = np.asarray(commands, np.float64)
    out = np.zeros_like(c)
    lim = LIMITS.astype(np.float64)
    vx = np.clip(c[..., 0] / lim[0], 0.0, 1.0) * float(vx_max) * float(vx_gain)
    vyaw = np.clip(c[..., 3] / lim[3], -1.0, 1.0) * float(vyaw_max) * float(yaw_gain)
    out[..., 0] = vx * float(dt)
    out[..., 3] = vyaw * float(dt)
    return out


# Keep turning toward the latched side until the aisle is ahead. Past this, stop.
_BYPASS_MAX_ALIGN = 2.6
_BYPASS_REJOIN_RAD = math.radians(25.0)


def blocked_ahead(front: float, left: float, right: float, near_m: float) -> bool:
    """True when there is no aisle to walk into.

    One side inside ``near_m`` is the obstacle a bypass is already passing. That
    does not block while the front and the other side stay beyond ``near_m``.
    The front inside ``near_m``, or both sides inside it, leaves no way through.
    """
    near = float(near_m)
    return float(front) < near or (float(left) < near and float(right) < near)


def bypass_lateral(pos_xy: np.ndarray, entry_xy: np.ndarray, entry_yaw: float, side: float) -> float:
    """Meters moved toward the latched side, relative to the heading at entry.

    World yaw is clockwise from +x. ``side`` +1 is a right turn. Spinning in place stays at 0.
    """
    d = np.asarray(pos_xy, np.float64) - np.asarray(entry_xy, np.float64)
    right = d[0] * (-math.sin(entry_yaw)) + d[1] * math.cos(entry_yaw)
    return float(side) * float(right)


def bypass_command(
    state: Optional[dict], front: float, left: float, right: float,
    pos_xy: np.ndarray, yaw: float, goal_xy: np.ndarray, *,
    turn_m: float, stop_m: float, clear_m: float, pass_m: float, yaw_limit: float,
) -> Tuple[Optional[np.ndarray], Optional[dict], str]:
    """Latch a side and hold it until the body has passed the obstacle.

    Returns ``(command, state, phase)``. ``command`` is None when bypass is not in charge.
    The command is in the model box: full forward is ``dx = 1``, full yaw is ``±yaw_limit``
    (clockwise positive). The side is fixed at entry so goal-seeking cannot yaw back into the obstacle.
    """
    def yaw_to(sign: float) -> np.ndarray:
        a = np.zeros(4, np.float64)
        a[3] = float(sign) * float(yaw_limit)
        return a

    def forward() -> np.ndarray:
        a = np.zeros(4, np.float64)
        a[0] = 1.0
        return a

    if right >= left:
        open_m, closed_m, open_side = right, left, 1.0
    else:
        open_m, closed_m, open_side = left, right, -1.0
    blocked = front < turn_m and open_m >= front + 0.4 and open_m >= closed_m + 0.8 and open_m >= 1.2
    if state is None:
        if not blocked:
            return None, None, ""
        state = {"side": open_side, "x": float(pos_xy[0]), "y": float(pos_xy[1]),
                 "yaw": float(yaw), "phase": "align", "need": float(pass_m)}

    side = float(state["side"])
    entry = np.array([state["x"], state["y"]], np.float64)
    lat = bypass_lateral(pos_xy, entry, float(state["yaw"]), side)
    turned = side * wrap(float(yaw) - float(state["yaw"]))
    bearing = goal_bearing_body(np.asarray(pos_xy, np.float64), float(yaw), np.asarray(goal_xy, np.float64))
    boxed = front < stop_m and max(left, right) < stop_m + 0.4
    if boxed:
        state["phase"] = "stop"
        return np.zeros(4, np.float64), state, "bypass_stop"

    if state["phase"] == "align":
        if front < clear_m:
            if turned > _BYPASS_MAX_ALIGN:
                state["phase"] = "stop"
                return np.zeros(4, np.float64), state, "bypass_stop"
            return yaw_to(side), state, "bypass_align"
        state["phase"] = "pass"

    if state["phase"] == "pass":
        if front < turn_m:
            if turned > _BYPASS_MAX_ALIGN:
                state["phase"] = "stop"
                return np.zeros(4, np.float64), state, "bypass_stop"
            state["phase"] = "align"
            return yaw_to(side), state, "bypass_align"
        if lat < float(state["need"]) or front < clear_m:
            return forward(), state, "bypass_pass"
        state["phase"] = "rejoin"

    if front < turn_m:
        # Turning back toward the goal put the obstacle ahead again. Yaw back to the aisle
        # and demand another metre of side shift before the next attempt.
        state["phase"] = "align"
        state["need"] = max(float(state["need"]), lat + 1.0)
        return yaw_to(side), state, "bypass_align"
    if abs(bearing) <= _BYPASS_REJOIN_RAD and front >= clear_m:
        return forward(), None, "bypass_done"
    return yaw_to(math.copysign(1.0, bearing if abs(bearing) > 1e-6 else side)), state, "bypass_rejoin"


def goal_offset(cands: np.ndarray, bearing: float, dt: float):
    """Same geometry as ``mpc_consequence_eval.goal_offset`` (angle from the goal, net length)."""
    c = np.asarray(cands, np.float64)
    th = np.cumsum(c[..., 3], -1) * dt
    dx = (c[..., 0] * np.cos(th) - c[..., 1] * np.sin(th)).sum(-1) * dt
    dy = (c[..., 0] * np.sin(th) + c[..., 1] * np.cos(th)).sum(-1) * dt
    return (np.arctan2(dy, dx) - bearing + np.pi) % (2 * np.pi) - np.pi, np.hypot(dx, dy)


def near_pick(cands: np.ndarray, bearing: float, safe: np.ndarray, prog: np.ndarray, dt: float = 0.2,
              max_deg: float = 30.0, min_len: float = 0.3) -> int:
    alpha, length = goal_offset(cands, bearing, dt)
    ok = np.asarray(safe, bool) & (length >= min_len) & (np.abs(alpha) <= math.radians(max_deg))
    return int(np.argmax(np.where(ok, prog, -np.inf))) if ok.any() else -1


def cruise_yaw_cap(bearing: float, yaw_cap: float, trim_yaw: float = 0.0, trim_rad: float = 0.0) -> float:
    """A small heading error turns at ``trim_yaw``. A large one uses ``yaw_cap``.

    Same rule as the drone eval. A full cap on a few degrees finishes the turn in one step
    and the next straight steps then walk off the goal ray.
    """
    cap = abs(float(yaw_cap))
    if trim_yaw > 0 and trim_rad > 0 and abs(float(bearing)) < float(trim_rad):
        return min(cap, abs(float(trim_yaw)))
    return cap


def _band_median(sl: np.ndarray, lo: float, hi: float, min_count: int) -> Optional[float]:
    sl = np.asarray(sl, np.float64)
    fin = sl[np.isfinite(sl) & (sl >= lo) & (sl < hi)]
    if int(fin.size) < int(min_count):
        return None
    return float(np.median(fin))


def corridor_sides(depth: np.ndarray, rows: Sequence[int], body_cols: int,
                   self_m: float = 0.95) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Center, left, and right medians on the obstacle band.

    Columns ``[0, body_cols)`` are the vehicle. The remaining width is split in
    three. The center median keeps near obstacles. Side medians drop pixels
    closer than ``self_m`` so a sliver of the shell is not a side.
    """
    d = np.asarray(depth, np.float64)
    if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 8:
        return None, None, None
    r0 = max(0, min(int(rows[0]), d.shape[0] - 1))
    r1 = max(r0 + 1, min(int(rows[1]), d.shape[0]))
    body = max(0, min(int(body_cols), d.shape[1] - 6))
    band = d[r0:r1, body:]
    if band.shape[1] < 6:
        return None, None, None
    a, b = band.shape[1] // 3, 2 * band.shape[1] // 3
    center = _band_median(band[:, a:b], 0.3, 29.0, 6)
    left = _band_median(band[:, :a], float(self_m), 29.0, 6)
    right = _band_median(band[:, b:], float(self_m), 29.0, 6)
    return center, left, right


def lead_open(depth: np.ndarray, rows: Sequence[int], body_cols: int, margin_m: float,
              self_m: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Center range on the obstacle band, and which visible side is open.

    The sign is clockwise-positive, so +1 yaws toward image right. A side that
    was not measured does not start a turn.
    """
    center, left, right = corridor_sides(depth, rows, body_cols, self_m)
    side: Optional[float] = None
    if left is not None and right is not None and abs(right - left) >= float(margin_m):
        side = 1.0 if right > left else -1.0
    return center, side


def _open_fraction(sl: np.ndarray, pass_m: float, hi: float = 29.0, min_count: int = 6) -> Optional[float]:
    """Share of a side that is far enough to enter. Near pixels count as closed."""
    sl = np.asarray(sl, np.float64)
    fin = sl[np.isfinite(sl) & (sl >= 0.3) & (sl < hi)]
    if int(fin.size) < int(min_count):
        return None
    return float(np.mean(fin >= float(pass_m)))


def corridor_passage(depth: np.ndarray, rows: Sequence[int], body_cols: int, pass_m: float
                     ) -> Tuple[Optional[float], Optional[float]]:
    """Fraction of each side, after the vehicle columns, that clears ``pass_m``.

    Close pixels stay in the fraction. Dropping them would turn the vehicle's own
    arm into an open side whenever a gap behind it is far.
    """
    d = np.asarray(depth, np.float64)
    if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 8:
        return None, None
    r0 = max(0, min(int(rows[0]), d.shape[0] - 1))
    r1 = max(r0 + 1, min(int(rows[1]), d.shape[0]))
    body = max(0, min(int(body_cols), d.shape[1] - 6))
    band = d[r0:r1, body:]
    if band.shape[1] < 6:
        return None, None
    a, b = band.shape[1] // 3, 2 * band.shape[1] // 3
    return _open_fraction(band[:, :a], pass_m), _open_fraction(band[:, b:], pass_m)


def passage_side(left: Optional[float], right: Optional[float], pass_m: float, margin_m: float,
                 held: float = 0.0) -> Optional[float]:
    """Which close-range opening the chassis can still enter.

    ``+1`` is image right, ``-1`` is image left, ``0`` means both sides are open
    enough to drive ahead, and ``None`` means neither side was measured as open.
    A side already being followed stays while it still clears ``pass_m``, until
    the other side is farther by ``margin_m``. A missing side is not a vote.
    """
    left_ok = left is not None and float(left) >= float(pass_m)
    right_ok = right is not None and float(right) >= float(pass_m)
    margin = float(margin_m)
    if held > 0 and right_ok and not (left_ok and float(left) >= float(right) + margin):
        return 1.0
    if held < 0 and left_ok and not (right_ok and float(right) >= float(left) + margin):
        return -1.0
    if left_ok and right_ok:
        if abs(float(right) - float(left)) >= margin:
            return 1.0 if float(right) > float(left) else -1.0
        return 0.0
    if right_ok:
        return 1.0
    if left_ok:
        return -1.0
    return None


def lead_guard(action: np.ndarray, front: float, stop_m: Optional[float], turn_m: Optional[float],
               lead_side: float, lead_yaw: float, fwd_dx: float, through: bool = False
               ) -> Tuple[np.ndarray, Optional[str]]:
    """Keep moving through a gap that still fits the chassis.

    A side that already cleared the passage test keeps a short forward arc, even
    when the nearest forward pixel is the vehicle. Both sides open keeps a short
    straight step only while the nose is outside the stop distance. No open side
    stops, rather than spinning toward a nearer strip.
    """
    a = np.asarray(action, np.float64).copy()
    in_stop = stop_m is not None and front < float(stop_m)
    in_turn = turn_m is not None and front < float(turn_m)
    if not in_stop and not in_turn:
        return a, None
    if abs(float(lead_side)) < 1e-6:
        # Both sides clear: creep ahead only while the nose itself is outside the stop distance.
        if through and not in_stop:
            out = np.zeros_like(a)
            out[0] = 0.5 * abs(float(fwd_dx))
            return out, "through"
        return np.zeros_like(a), "stop"
    out = np.zeros_like(a)
    out[0] = 0.5 * abs(float(fwd_dx))
    out[3] = math.copysign(abs(float(lead_yaw)), float(lead_side))
    return out, "lead"


def goal_arc(bearing: float, k: int, dx: float, yaw_cap: float) -> np.ndarray:
    """Cruise forward while yawing toward the goal, then straight.

    The horizon is what repeated one-step replans of this rule fly, so progress scores the
    arc that will actually be driven. Model dyaw is clockwise-positive, matching bearing.
    """
    a = np.zeros((int(k), 4), dtype=np.float64)
    left = float(bearing)
    cap = abs(float(yaw_cap))
    for t in range(int(k)):
        step = float(np.clip(left, -cap, cap))
        a[t, 0] = float(dx)
        a[t, 3] = step
        left -= step
    return a


def open_side_arcs(k: int, dx: float, yaw: float) -> list[tuple[str, np.ndarray]]:
    """One steady left arc and one steady right arc, both moving forward.

    Model dyaw is clockwise, so the positive arc turns right. The same two
    sequences are scored every step. A new random yaw each replan flips the
    chassis from side to side.
    """
    yaw = min(abs(float(yaw)), float(LIMITS[3]))
    arcs = []
    for name, sign in (("open_left", -1.0), ("open_right", 1.0)):
        a = np.zeros((int(k), 4), dtype=np.float64)
        a[:, 0] = float(dx)
        a[:, 3] = sign * yaw
        arcs.append((name, a))
    return arcs


def smooth_turn(seq: np.ndarray, move_dx: float, yaw_cap: float) -> np.ndarray:
    """A sampled turn's yawing steps also advance, and the yaw stays inside ``yaw_cap``.

    Replan-every-step flies only the first action. The raw macro puts the yaw there and the
    forward motion later, so the driven step is a stop-and-spin.
    """
    a = np.array(seq, dtype=np.float64, copy=True)
    yawing = np.abs(a[:, 3]) > 1e-3
    if move_dx > 0:
        a[yawing, 0] = np.maximum(a[yawing, 0], float(move_dx))
    if yaw_cap > 0:
        a[yawing, 3] = np.clip(a[yawing, 3], -abs(float(yaw_cap)), abs(float(yaw_cap)))
    return a


def hold_straight(names: List[str], surv: np.ndarray, score: np.ndarray, allowed: np.ndarray,
                  chosen: int, surv_min: float, margin: float) -> int:
    """Stay on a passing straight candidate unless the alternative beats it by ``margin``."""
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


def allow_early_open(names: Sequence[str], heading_ok: np.ndarray, prefer_open: bool) -> np.ndarray:
    """Let a scored open arc leave the goal line. A random yaw still cannot."""
    ok = np.asarray(heading_ok, bool).copy()
    if not prefer_open:
        return ok
    for i, name in enumerate(names):
        if name in ("open_left", "open_right"):
            ok[i] = True
    return ok


def align_may_replace(name: str, prefer_open: bool) -> bool:
    """Heading alignment must not steer back onto an obstacle once the open side won."""
    return not (prefer_open and name in ("open_left", "open_right"))


def choose_align(names: List[str], surv: np.ndarray, allowed: np.ndarray, bearing: float,
                 surv_min: float, align_rad: float, slack: float = 0.02) -> int:
    """Take goal_arc once the heading error exceeds ``align_rad`` and the arc is about as safe."""
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


def open_turn_sign(depth: np.ndarray, cap: float = 10.0) -> float:
    """+1 when the right half of the depth image is more open (positive yaw turns that way)."""
    col = np.minimum(np.asarray(depth, np.float64).min(0), cap)
    h = col.shape[0] // 2
    return 1.0 if col[h:].mean() > col[:h].mean() else -1.0


def brake_action(state: np.ndarray, step_hz: float, hold_z: Optional[float] = None,
                 hold_gain: float = 0.5, hold_max: float = 0.2) -> np.ndarray:
    """Body step opposite the horizontal velocity. ``hold_z`` counters the pitch-up of a hard stop."""
    vx, vy = float(state[3]), float(state[4])
    yaw = float(state[6])
    a = np.zeros(4)
    a[0] = -(vx * math.cos(yaw) + vy * math.sin(yaw)) / step_hz
    a[1] = -(-vx * math.sin(yaw) + vy * math.cos(yaw)) / step_hz
    if hold_z is not None:
        a[2] = float(np.clip(hold_gain * (hold_z - float(state[2])), -hold_max, hold_max))
    return np.clip(a, -LIMITS, LIMITS)


@dataclass
class NavConfig:
    surv_min: float = 0.95
    fwd_dx: float = 0.8
    fov_max_deg: float = 35.0
    alt_band_m: float = 1.0
    alt_band_return: float = 0.0
    replan_every: int = 1
    seed: int = 0
    # Candidate families (None: all of FAMILIES). A ground robot drops strafe / climb / smooth_random.
    families: Optional[Sequence[str]] = None
    # Drop a candidate whose executed yaw increases the bearing error. yaw_fwd's sign is a random
    # draw, and predicted progress will otherwise pick that step while the goal is still ahead.
    toward_goal_yaw: bool = False
    # Drone cruise rules, from the current mpc eval. 0 leaves that rule off.
    # goal_arc flies forward while yawing toward the remaining bearing. A heading error inside
    # trim_bearing_deg uses the shallower trim_yaw. align_bearing_deg forces that arc when it
    # survives about as well as straight. fwd_hold_m keeps straight unless another survivor
    # beats its progress by that many meters. move_dx / move_yaw turn a sampled yaw into an arc
    # so the executed first step is not a stop-and-spin. near_arc_yaw replaces the near-goal
    # creep with a moving arc.
    goal_arc_yaw: float = 0.0
    trim_yaw: float = 0.0
    trim_bearing_deg: float = 25.0
    align_bearing_deg: float = 0.0
    align_slack: float = 0.02
    fwd_hold_m: float = 0.0
    move_dx: float = 0.0
    move_yaw: float = 0.0
    near_arc_yaw: float = 0.0
    near_tight_yaw: float = 0.20
    near_tight_dx: float = 0.50
    near_tight_bearing_deg: float = 0.0
    # Per-axis factor on every candidate (dx, dy, dz, dyaw) so the model scores what the vehicle executes,
    # e.g. a slow robot whose full step is far below the drone's.
    action_scale: Optional[Sequence[float]] = None
    # Depth guard on the pooled 56x56 map (None: off). Front clearance below guard_turn_m drops forward
    # motion and turns by guard_yaw (model units, CW+). The sign is the survival model's safer
    # in-place yaw. Inside guard_stop_m it still turns when one side is open, and holds still
    # only when both sides are closed.
    guard_turn_m: Optional[float] = None
    guard_stop_m: Optional[float] = None
    guard_yaw: float = 0.06
    # Rows of the 56-high pooled depth used for the guard. None keeps 14:42. A level ground camera
    # should stop above the horizon so the floor (about 1 m ahead) is not treated as a wall.
    guard_rows: Optional[Sequence[int]] = None
    # "dog": candidates from the dog-command macros. With realize_motion, krs6 scores the measured
    # body displacement and the backend still receives the command. rad_per_step is that displacement
    # for one full yaw command (vyaw_max * yaw_gain / step_hz).
    macro_set: str = "drone"
    rad_per_step: float = 0.06
    realize_motion: bool = False
    dog_vx_max: float = 0.35
    dog_vyaw_max: float = 0.5
    dog_vx_gain: float = 0.57
    dog_yaw_gain: float = 0.50
    step_hz: float = 5.0
    # Early corridor bias. 0 leaves it off. When the obstacle-band center median is inside
    # lead_m and one visible side is farther by lead_margin_m, fly a forward arc that way
    # while the nose is still near the goal. Columns [0, lead_body_cols) are the vehicle.
    # A side that was not measured does not start a turn.
    lead_m: float = 0.0
    lead_yaw: float = 0.08
    lead_rows: tuple = (8, 18)
    lead_body_cols: int = 0
    lead_margin_m: float = 0.5
    lead_bearing_deg: float = 25.0
    # Close-range passage. 0 leaves the old stop-when-no-side rule. A side clears it when
    # at least lead_pass_frac of that side, vehicle columns removed, is beyond lead_pass_m.
    # Near pixels count, so the vehicle's own arm is not an opening.
    lead_pass_m: float = 0.0
    lead_pass_frac: float = 0.6
    # Columns of the near guard band occupied by the vehicle. They stay out of the front trigger.
    guard_body_cols: int = 0
    # Dog bypass. While the front is blocked and one side is clearly open, ignore goal-seeking
    # and hold that side: yaw until the aisle is ahead, walk until the body has shifted
    # bypass_pass_m toward that side, then yaw back only while the front stays clear.
    bypass: bool = False
    bypass_clear_m: float = 2.0
    bypass_pass_m: float = 2.0
    # Frozen f35 rules. 0 / False leaves the previous pick unchanged.
    near_goal_dist: float = 0.0
    near_goal_after_s: float = 1.0
    near_goal_dx: Sequence[float] = field(default_factory=lambda: (0.3, 0.5))
    near_goal_deg: float = 30.0
    goal_surv_min: float = 0.0
    # Far range. Among actions that survive, take the one with the largest
    # predicted forward clearance. A near obstacle is handled by the caller.
    # open_side_yaw adds a fixed left arc and a fixed right arc. 0 leaves them off.
    prefer_open: bool = False
    open_side_yaw: float = 0.0
    goal_surv_dist: float = 8.0
    brake_speed: float = 0.0
    brake_hold_alt: bool = False
    spin_nopass: bool = False
    # When every allowed action is under surv_min, command a zero step.
    # Off by default so an older run still takes the least-bad action.
    stop_nopass: bool = False


class ConsequenceNavigator:
    def __init__(self, surv_ckpts: Sequence[str], prog_ckpts: Sequence[str] = (), *, device: str = "cuda",
                 cfg: Optional[NavConfig] = None) -> None:
        self.cfg = cfg or NavConfig()
        self.dev = torch.device(device)
        self.models = [load_checkpoint(p, device) for p in surv_ckpts]
        self.prog_models = [load_checkpoint(p, device) for p in prog_ckpts]
        if not self.models:
            raise ValueError("need at least one survival checkpoint")
        allm = self.models + self.prog_models
        self.hist_len = max([8] + [max(m.frame_offsets) + 1 for m in allm] + [m.hist_actions + 1 for m in allm])
        self.H = self.models[0].horizon
        self.uses_env = any(m.env_dim for m in allm)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.guide = None
        self.reset()

    def reset(self) -> None:
        self.dhist: Deque[np.ndarray] = deque(maxlen=self.hist_len)
        self.ahist: Deque[np.ndarray] = deque(maxlen=self.hist_len)
        self.prev = np.zeros(4, np.float32)
        self.env: Optional[np.ndarray] = None
        self.plan: List[np.ndarray] = []
        self.step_i = 0
        self.band: Optional[tuple] = None
        self.best_d: Optional[float] = None
        self.best_step = 0
        self.spin_side = 0.0
        self.lead_side = 0.0
        self.pass_side = 0.0
        self.bypass: Optional[dict] = None
        self.last: Dict[str, Any] = {}
        self._scene_pending: Optional[tuple] = None

    def set_band(self, z_now: float, goal_z: float) -> None:
        b = self.cfg.alt_band_m
        self.band = (min(z_now, goal_z) - b, max(z_now, goal_z) + b)
        self.plan = []

    def observe(self, depth: np.ndarray, env: Optional[np.ndarray], executed: Optional[np.ndarray]) -> None:
        """Push the current depth map and the action executed since the previous observe."""
        if executed is not None and self.dhist:
            a = np.asarray(executed, np.float32)
            self.ahist.append(a)
            self.prev = a
        self.dhist.append(pool_min(np.asarray(depth, np.float32)))
        self.env = None if env is None else np.asarray(env, np.float32)

    def _score(self, state: np.ndarray, goal: np.ndarray, cands: np.ndarray):
        M = len(cands)
        g = torch.as_tensor(np.repeat(goal_features(state, goal)[None], M, 0), device=self.dev)
        pa = torch.as_tensor(np.repeat(self.prev[None], M, 0), device=self.dev)
        a = torch.as_tensor(cands.astype(np.float32), device=self.dev)
        env = None
        if self.uses_env and self.env is not None:
            env = torch.as_tensor(np.repeat(self.env[None], M, 0), device=self.dev)
        s_all, p_all, q_all, f_all = [], [], [], []
        dseq, aseq = list(self.dhist), list(self.ahist)
        with torch.no_grad():
            for i, m in enumerate(self.models + self.prog_models):
                d_k, a_k = history_inputs(dseq, aseq, m.frame_offsets, m.hist_actions)
                d_k = d_k[0] if len(m.frame_offsets) == 1 else d_k
                d = torch.as_tensor(np.repeat(d_k[None], M, 0), device=self.dev)
                ha = torch.as_tensor(np.repeat(a_k[None], M, 0), device=self.dev) if m.hist_actions else None
                o = m(d, g, pa, a, ha, env=env) if m.env_dim else m(d, g, pa, a, ha)
                prog = (-o["dgoal"][..., -1]).cpu().numpy()
                if i >= len(self.models):
                    q_all.append(prog)
                    continue
                s_all.append(survival(o["hazard_logit"])[..., -1].cpu().numpy())
                p_all.append(prog)
                f_all.append(torch.exp(o["log_cones"][..., -1, 0]).cpu().numpy())
        self._fwd_clear = np.mean(f_all, 0) if f_all else np.zeros(M, np.float64)
        return np.mean(s_all, 0), np.mean(q_all or p_all, 0)

    def act(self, vs: VehicleState, goal: np.ndarray) -> np.ndarray:
        if not self.dhist:
            raise RuntimeError("observe() before act()")
        cfg = self.cfg
        state = vs.as_model_state()
        goal = np.asarray(goal, np.float64)
        if self.guide is not None and self._scene_pending is not None and self.guide.record:
            kf, yaw, surv, d0, thr = self._scene_pending
            self.guide.note(kf, yaw, surv, d0, float(np.linalg.norm(state[:3] - goal)), False, thr, state[:3])
            self._scene_pending = None
        if self.band is None:
            self.set_band(float(state[2]), float(goal[2]))
        if cfg.bypass:
            cmd = self._bypass(vs, goal)
            if cmd is not None:
                self.step_i += 1
                return cmd
        k = int(cfg.replan_every)
        if not self.plan or self.step_i % k == 0:
            aim = goal
            subgoal_source, kf_id = "goal", None
            if self.guide is not None:
                vector = None if self.env is None else np.asarray(self.env, np.float32)
                # The stored vector is only a retrieval key. _score keeps the live env.
                proposal = self.guide.step_view(state, goal, vector, None)
                aim, subgoal_source, kf_id = proposal.aim, proposal.source, proposal.kf_id
            bearing = goal_bearing_body(state[:3], float(state[6]), aim)
            cands, names = [], []
            dog = cfg.macro_set == "dog"
            fams = (DOG_FAMILIES if dog else FAMILIES) if cfg.families is None else tuple(cfg.families)
            for fam in fams:
                if dog:
                    name, seq = sample_dog_macro(self.rng, self.H, bearing, fam, cfg.rad_per_step)
                else:
                    name, seq = sample_macro(self.rng, self.H, bearing, name=fam)
                if fam == "fwd":
                    seq[:, 0] = cfg.fwd_dx
                if not dog and cfg.move_dx > 0 and name in ("yaw_fwd", "goal_turn_fwd"):
                    seq = smooth_turn(seq, cfg.move_dx, cfg.move_yaw)
                cands.append(seq)
                names.append(name)
            if not dog and cfg.open_side_yaw > 0:
                for name, seq in open_side_arcs(self.H, cfg.fwd_dx, cfg.open_side_yaw):
                    cands.append(seq)
                    names.append(name)
            if not dog and cfg.goal_arc_yaw > 0:
                arc_cap = cruise_yaw_cap(
                    bearing, cfg.goal_arc_yaw, cfg.trim_yaw, math.radians(cfg.trim_bearing_deg))
                cands.append(goal_arc(bearing, self.H, cfg.fwd_dx, arc_cap))
                names.append("goal_arc")
            self.lead_side = 0.0
            if not dog and cfg.lead_m > 0 and self.dhist:
                lead_center, lead_side = lead_open(
                    self.dhist[-1], cfg.lead_rows, cfg.lead_body_cols, cfg.lead_margin_m)
                due = (
                    lead_center is not None
                    and lead_center < float(cfg.lead_m)
                    and lead_side is not None
                    and abs(bearing) <= math.radians(cfg.lead_bearing_deg)
                )
                if due:
                    self.lead_side = float(lead_side)
                    cands.append(goal_arc(
                        self.lead_side * float(cfg.lead_yaw), self.H, cfg.fwd_dx, float(cfg.lead_yaw)))
                    names.append("lead_arc")
            if state[2] > goal[2] + 1.0 and (cfg.families is None or "descend" in fams):
                name, seq = sample_macro(self.rng, self.H, bearing, name="descend")
                cands.append(seq)
                names.append(name)
            d_here = float(np.linalg.norm(state[:3] - goal))
            if self.best_d is None or d_here < self.best_d - 1e-3:
                self.best_d, self.best_step = d_here, self.step_i
            dt = 1.0 / cfg.step_hz
            near_due = (cfg.near_goal_dist > 0 and d_here <= cfg.near_goal_dist
                        and (self.step_i - self.best_step) / cfg.step_hz >= cfg.near_goal_after_s)
            if near_due and not dog:
                if cfg.near_arc_yaw > 0:
                    tight_rad = (math.radians(cfg.near_tight_bearing_deg)
                                 if cfg.near_tight_bearing_deg > 0 else 0.25)
                    if abs(bearing) > tight_rad:
                        cap, near_dx = cfg.near_tight_yaw, min(cfg.fwd_dx, cfg.near_tight_dx)
                    else:
                        cap, near_dx = cfg.near_arc_yaw, cfg.fwd_dx
                    cands.append(goal_arc(bearing, self.H, near_dx, cap))
                    names.append("near_arc")
                for dx in cfg.near_goal_dx:
                    _, seq = sample_macro(self.rng, self.H, bearing, name="goal_turn_fwd")
                    seq[seq[:, 3] == 0, 0] = float(dx)
                    cands.append(seq)
                    names.append("near_creep")
            cands = np.stack(cands)
            scale = np.asarray(cfg.action_scale, np.float64) if cfg.action_scale is not None else None
            if scale is not None and not cfg.realize_motion:
                cands = cands * scale
            if cfg.realize_motion:
                scored = realize_dog_commands(
                    cands, vx_max=cfg.dog_vx_max, vyaw_max=cfg.dog_vyaw_max,
                    vx_gain=cfg.dog_vx_gain, yaw_gain=cfg.dog_yaw_gain, dt=1.0 / cfg.step_hz)
            else:
                scored = cands
            allowed = altitude_allowed(scored, state, goal) & altitude_allowed(scored[:, :k], state, goal)
            banded = allowed & altitude_band_allowed(scored, state[2], self.band[0], self.band[1], k,
                                                     cfg.alt_band_return)
            allowed = banded if banded.any() else allowed
            if cfg.fov_max_deg > 0:
                seen = allowed & visible_motion_allowed(scored, k, cfg.fov_max_deg)
                allowed = seen if seen.any() else allowed
            if cfg.toward_goal_yaw:
                allowed = allow_early_open(
                    names, allowed & heading_yaw_allowed(cands, bearing, k), cfg.prefer_open)
            s, p = self._score(state, aim, scored)
            surv_pick = cfg.goal_surv_min if cfg.goal_surv_min > 0 and d_here <= cfg.goal_surv_dist else cfg.surv_min
            if self.guide is not None and subgoal_source != "goal" and not (allowed & (s >= surv_pick)).any():
                s, p = self._score(state, goal, scored)
                aim, subgoal_source = goal, "goal_fallback"
            braking = spinning = stopping = False
            if near_due and cfg.near_arc_yaw > 0 and (
                    nc := choose_near(names, cands, s, allowed, bearing, surv_pick)) >= 0:
                c = nc
            elif near_due and cfg.near_arc_yaw <= 0 and (
                    nc := near_pick(cands, bearing, allowed & (s >= surv_pick), p, dt, cfg.near_goal_deg)) >= 0:
                c = nc
            else:
                if cfg.prefer_open:
                    c = select_open_candidate(s, p, self._fwd_clear, allowed, surv_pick)
                else:
                    c = select_candidate(s, p, allowed, surv_pick)
                passing = allowed & (s >= surv_pick)
                speed = math.hypot(float(state[3]), float(state[4]))
                if cfg.brake_speed > 0 and not passing.any() and speed > cfg.brake_speed:
                    braking = True
                elif cfg.spin_nopass and cfg.lead_m <= 0 and not passing.any():
                    spinning = True
                    self.spin_side = self.spin_side or open_turn_sign(self.dhist[-1])
                elif cfg.stop_nopass and not passing.any():
                    stopping = True
                if passing.any():
                    self.spin_side = 0.0
                if not braking and not spinning and not stopping and cfg.fwd_hold_m > 0:
                    c = hold_straight(names, s, p, allowed, c, surv_pick, cfg.fwd_hold_m)
                if (
                    not braking and not spinning and not stopping and cfg.align_bearing_deg > 0
                    and align_may_replace(names[int(c)], cfg.prefer_open)
                ):
                    aligned = choose_align(
                        names, s, allowed, bearing, surv_pick,
                        math.radians(cfg.align_bearing_deg), cfg.align_slack)
                    if aligned >= 0:
                        c = aligned
            if not braking and not spinning and not stopping and "lead_arc" in names:
                # The side came from the corridor medians. Survival stays a score
                # of the arc, not a second vote that can flip the side.
                c = names.index("lead_arc")
            if braking:
                hold_z = 0.5 * (self.band[0] + self.band[1]) if cfg.brake_hold_alt else None
                a0 = brake_action(state, cfg.step_hz, hold_z)
                if scale is not None:
                    a0 = a0 * scale
                self.plan = [a0] + [np.zeros(4)] * (self.H - 1)
                family = "brake"
            elif spinning:
                a0 = np.array([0.0, 0.0, 0.0, self.spin_side * float(LIMITS[3])])
                if scale is not None:
                    a0 = a0 * scale
                self.plan = [a0] + [np.zeros(4)] * (self.H - 1)
                family = "spin_nopass"
            elif stopping:
                a0 = np.zeros(4, np.float64)
                self.plan = [a0] + [np.zeros(4)] * (self.H - 1)
                family = "stop"
            else:
                self.plan = [np.asarray(a, np.float64) for a in cands[c]]
                family = names[c]
            self.last = {"family": family, "surv": float(s[c]), "pred_prog": float(p[c]),
                         "n_pass": int((allowed & (s >= surv_pick)).sum()), "near": bool(near_due),
                         "bearing_deg": round(math.degrees(bearing), 1),
                         "lead_side": self.lead_side}
            if self.guide is not None:
                self.last["subgoal"] = subgoal_source
                if self.guide.record:
                    taken = 0.0 if (braking or spinning or stopping) else float(s[c])
                    self._scene_pending = (kf_id, float(state[6]), taken, d_here, float(surv_pick))
        self.step_i += 1
        return self._guard(self.plan.pop(0), state, goal)

    def _guard_band(self) -> np.ndarray:
        cfg = self.cfg
        d56 = np.asarray(self.dhist[-1], np.float32)
        r0, r1 = (14, 42) if cfg.guard_rows is None else (int(cfg.guard_rows[0]), int(cfg.guard_rows[1]))
        return d56[r0:r1]

    def _front_sides(self) -> Tuple[float, float, float]:
        d = self._guard_band()
        return float(d[:, 14:42].min()), float(d[:, :14].min()), float(d[:, 42:].min())

    def _ahead_min(self, body_cols: int) -> float:
        """Front clearance with the vehicle's own columns removed."""
        d = self._guard_band()
        c0 = max(0, min(int(body_cols), d.shape[1] - 1))
        return float(d[:, c0:42].min()) if c0 < 42 else float("inf")

    def _bypass(self, vs: VehicleState, goal: np.ndarray) -> Optional[np.ndarray]:
        cfg = self.cfg
        front, left, right = self._front_sides()
        turn_m = 1.8 if cfg.guard_turn_m is None else float(cfg.guard_turn_m)
        stop_m = 0.8 if cfg.guard_stop_m is None else float(cfg.guard_stop_m)
        cmd, self.bypass, phase = bypass_command(
            self.bypass, front, left, right, vs.pos[:2], float(vs.yaw), goal[:2],
            turn_m=turn_m, stop_m=stop_m, clear_m=float(cfg.bypass_clear_m),
            pass_m=float(cfg.bypass_pass_m), yaw_limit=float(LIMITS[3]))
        if not phase:
            return None
        self.plan = []
        lat = None
        if self.bypass is not None:
            lat = round(bypass_lateral(
                vs.pos[:2], np.array([self.bypass["x"], self.bypass["y"]]),
                float(self.bypass["yaw"]), float(self.bypass["side"])), 2)
        self.last = {"family": phase, "guard": phase, "surv": None,
                     "clear_flr": (round(front, 2), round(left, 2), round(right, 2)),
                     "bypass_lat": lat}
        return cmd

    def _safer_yaw(self, state: np.ndarray, goal: np.ndarray) -> float:
        """Sign of model dyaw, clockwise positive, for the safer in-place turn.

        Both candidates yaw at the training limit for the whole horizon, so survival
        compares direction. A single near pixel in a side strip does not cast the vote.
        """
        yaw = float(LIMITS[3])
        cands = np.zeros((2, self.H, 4), np.float64)
        cands[0, :, 3] = -yaw
        cands[1, :, 3] = yaw
        surv, _prog = self._score(state, goal, cands)
        self.last["avoid_surv"] = [round(float(surv[0]), 3), round(float(surv[1]), 3)]
        return -1.0 if float(surv[0]) >= float(surv[1]) else 1.0

    def _guard(self, a: np.ndarray, state: Optional[np.ndarray] = None, goal: Optional[np.ndarray] = None) -> np.ndarray:
        cfg = self.cfg
        if cfg.guard_turn_m is None and cfg.guard_stop_m is None:
            return a
        front, left, right = self._front_sides()
        self.last["clear_flr"] = (round(front, 2), round(left, 2), round(right, 2))
        ahead = front
        if int(cfg.guard_body_cols) > 0:
            ahead = self._ahead_min(int(cfg.guard_body_cols))
        if cfg.lead_m > 0:
            side = self.lead_side
            through = False
            turn_m = None if cfg.guard_turn_m is None else float(cfg.guard_turn_m)
            if cfg.lead_pass_m > 0 and self.dhist and turn_m is not None and ahead < turn_m:
                left, right = corridor_passage(
                    self.dhist[-1], cfg.lead_rows, cfg.lead_body_cols, cfg.lead_pass_m)
                choice = passage_side(
                    left, right, cfg.lead_pass_frac, 0.25, self.pass_side)
                self.last["pass_lr"] = (
                    None if left is None else round(float(left), 2),
                    None if right is None else round(float(right), 2))
                if choice is None:
                    side, through, self.pass_side = 0.0, False, 0.0
                elif abs(choice) < 1e-6:
                    side, through, self.pass_side = 0.0, True, 0.0
                else:
                    side, through, self.pass_side = float(choice), False, float(choice)
                self.lead_side = self.pass_side
                self.last["lead_side"] = self.lead_side
            elif turn_m is not None and ahead >= turn_m:
                self.pass_side = 0.0
            stepped, phase = lead_guard(
                a, ahead, cfg.guard_stop_m, cfg.guard_turn_m,
                side, cfg.lead_yaw, cfg.fwd_dx, through)
            if phase:
                self.plan, self.last["guard"] = [], phase
                return stepped
            return a

        def turn_of(magnitude: float) -> np.ndarray:
            if state is not None and goal is not None:
                sign = self._safer_yaw(state, goal)
            else:
                sign = -1.0 if left >= right else 1.0
            out = np.zeros_like(a)
            out[3] = sign * magnitude
            return out

        if cfg.guard_stop_m is not None and front < cfg.guard_stop_m:
            open_m = max(left, right)
            if open_m >= cfg.guard_stop_m + 0.4:
                self.plan, self.last["guard"] = [], "turn"
                return turn_of(max(cfg.guard_yaw, 0.25))
            self.plan, self.last["guard"] = [], "stop"
            return np.zeros_like(a)
        if cfg.guard_turn_m is not None and front < cfg.guard_turn_m:
            self.plan, self.last["guard"] = [], "turn"
            return turn_of(cfg.guard_yaw)
        return a
