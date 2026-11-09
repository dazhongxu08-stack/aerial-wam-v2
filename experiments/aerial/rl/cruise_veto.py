"""Keep the world-model step on an open road. Turn only when the road centre closes.

The road centre is the middle 12% of the depth image, read as a 10th
percentile and left in the same metres as the shield. It is not scaled.

While that centre is at least 12 m, the world model's forward step stays.
Yaw is only a small correction toward the goal, so an open road cannot become
a spin. While the centre is shorter than 12 m, forward speed is removed and
the vehicle yaws about 0.10 rad toward the clearer side, up to about 70
degrees. The turn stops when the centre opens again. The shield still runs
after this and can still brake.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ScoreFn = Callable[[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray], Tuple[float, float]]

# Same gates the imagination planner already uses: open road, and the band
# where a forward step is about to be braked.
ROAD_OPEN_M = 12.0
BRAKE_M = 8.0
ROAD_STRIP_FRAC = 0.12
HOLD_YAW_RAD = 0.08
YAW_STEP_RAD = 0.10
YAW_BUDGET_RAD = 1.2

KRS6_CKPTS = (
    "experiments/aerial/rl/artifacts/shorew03_krs6_20261013/c_s218/consequence_best.pt",
    "experiments/aerial/rl/artifacts/shorew03_krs6_20261013/c_s219/consequence_best.pt",
    "experiments/aerial/rl/artifacts/shorew03_krs6_20261013/c_s220/consequence_best.pt",
)


def wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def goal_bearing(pos: np.ndarray, yaw: float, goal: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return wrap(math.atan2(float(goal[1] - pos[1]), float(goal[0] - pos[0])) - float(yaw))


def goal_dist_xy(pos: np.ndarray, goal: np.ndarray) -> float:
    d = np.asarray(goal, dtype=np.float64)[:2] - np.asarray(pos, dtype=np.float64)[:2]
    return float(np.hypot(d[0], d[1]))


def straight_action(bearing: float, limits: np.ndarray) -> np.ndarray:
    """Face the goal, and go forward once the nose is inside the camera."""
    lim = np.asarray(limits, dtype=np.float64).reshape(4)
    bearing = float(bearing)
    dyaw = float(np.clip(bearing, -lim[3], lim[3]))
    if abs(bearing) <= math.radians(20.0):
        return np.array([lim[0], 0.0, 0.0, 0.0], dtype=np.float64)
    fwd = float(lim[0]) if abs(bearing) <= math.radians(35.0) else 0.0
    return np.array([fwd, 0.0, 0.0, dyaw], dtype=np.float64)


def in_view(action: np.ndarray, max_deg: float, min_speed: float = 0.1) -> bool:
    dx, dy = float(action[0]), float(action[1])
    if math.hypot(dx, dy) < float(min_speed):
        return True
    if dx <= 0.0:
        return False
    return math.degrees(math.atan2(abs(dy), dx)) <= float(max_deg)


def circle_step(action: np.ndarray, bearing: float, limits: np.ndarray, fov_deg: float = 35.0) -> bool:
    """Yaw-in-place, or a sideways step while the goal is still ahead."""
    a = np.asarray(action, dtype=np.float64).reshape(4)
    lim = np.maximum(np.asarray(limits, dtype=np.float64).reshape(4), 1e-6)
    dx, dy, dyaw = float(a[0]), float(a[1]), float(a[3])
    if abs(dyaw) >= 0.5 * lim[3] and dx <= 0.3 * lim[0]:
        return True
    if math.hypot(dx, dy) >= 0.1 and abs(bearing) < math.radians(45.0):
        off = 180.0 if dx <= 0.0 else math.degrees(math.atan2(abs(dy), dx))
        if off > float(fov_deg):
            return True
    if abs(bearing) > math.radians(float(fov_deg)) and dx <= 0.3 * lim[0]:
        return True
    return False


def yaw_span(yaws: Sequence[float]) -> float:
    total = 0.0
    for a, b in zip(yaws, list(yaws)[1:]):
        total += abs(wrap(float(b) - float(a)))
    return total


def _metres_below(distance: Optional[float], limit_m: float) -> bool:
    return distance is not None and math.isfinite(float(distance)) and float(distance) < float(limit_m)


def _nose_is_opening(
    d_fwd: Optional[float], d_left: Optional[float], d_right: Optional[float]
) -> bool:
    """The forward cone is at least as open as either side."""
    if d_fwd is None or d_left is None or d_right is None:
        return False
    vals = (float(d_fwd), float(d_left), float(d_right))
    if not all(math.isfinite(v) for v in vals):
        return False
    return vals[0] + 0.5 >= max(vals[1], vals[2])


def hold_altitude(
    action: np.ndarray, pos: np.ndarray, goal: np.ndarray, limits: np.ndarray
) -> np.ndarray:
    """A zero vertical step sinks in this sim. Climb back toward the goal height."""
    out = np.asarray(action, dtype=np.float64).reshape(4).copy()
    if float(pos[2]) < float(goal[2]) - 0.4:
        lim = np.asarray(limits, dtype=np.float64).reshape(4)
        out[2] = min(0.15, float(lim[2]))
    return out


def stalled(
    dists: Sequence[float],
    yaws: Sequence[float],
    stall_steps: int,
    stall_yaw_rad: float,
    min_close_m: float,
) -> bool:
    n = int(stall_steps)
    if n <= 0 or len(dists) < n + 1 or len(yaws) < n + 1:
        return False
    closed = float(dists[-1 - n]) - float(dists[-1])
    return closed < float(min_close_m) and yaw_span(yaws[-1 - n :]) >= float(stall_yaw_rad)


def decide(
    cruise: np.ndarray,
    straight: np.ndarray,
    surv_c: float,
    prog_c: float,
    surv_s: float,
    prog_s: float,
    *,
    bearing: float,
    limits: np.ndarray,
    stall: bool,
    d_goal: float,
    arrive_m: float,
    surv_min: float,
    fov_deg: float,
    progress_margin_m: float,
    d_fwd: Optional[float] = None,
    d_left: Optional[float] = None,
    d_right: Optional[float] = None,
    block_m: float = 10.0,
    hard_m: float = 4.0,
    side_margin_m: float = 1.0,
    street_m: float = 15.0,
    fly_m: float = 9.0,
) -> Tuple[np.ndarray, str]:
    cruise = np.asarray(cruise, dtype=np.float64).reshape(4)
    straight = np.asarray(straight, dtype=np.float64).reshape(4)
    # A nose already inside the brake band, and the goal is in the arrival
    # ball: stop. Do not keep yawing in place on top of the goal.
    if _metres_below(d_fwd, BRAKE_M) and float(d_goal) <= float(arrive_m):
        return np.zeros(4, dtype=np.float64), "hover"
    cruise_ok = float(surv_c) >= float(surv_min) and in_view(cruise, fov_deg)
    straight_ok = float(surv_s) >= float(surv_min) and in_view(straight, fov_deg)
    # Straight replaces cruise only while the road ahead is open. A facade is
    # the planner's escape, not a new goal heading from this selector.
    if _metres_below(d_fwd, ROAD_OPEN_M):
        straight_ok = False
    circling = circle_step(cruise, bearing, limits, fov_deg)
    off_goal = abs(float(bearing)) > math.radians(float(fov_deg))
    straight_better = float(prog_s) > float(prog_c) + float(progress_margin_m)
    straight_not_worse = float(prog_s) + 1e-6 >= float(prog_c)
    if not straight_ok:
        if float(d_goal) <= float(arrive_m) and not cruise_ok:
            return np.zeros(4, dtype=np.float64), "hover"
        return cruise.copy(), "cruise"
    if float(d_goal) <= float(arrive_m):
        if cruise_ok and not circling:
            return cruise.copy(), "cruise"
        return straight.copy(), "straight"
    # Both models are scored every step. Keep the world-model step when it is
    # safe, facing the goal, and not clearly worse. Hand the step to the
    # goal-aligned action when the cruise step is unsafe, circling, off-goal,
    # stalled, or behind on predicted progress.
    replace = (
        (not cruise_ok)
        or bool(stall)
        or straight_better
        or ((circling or off_goal) and straight_not_worse)
    )
    if replace:
        return straight.copy(), "stall" if stall else "straight"
    return cruise.copy(), "cruise"


def depth56_from_map(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[0]
    h, w = int(d.shape[0]), int(d.shape[1])
    if h >= 56 and w >= 56 and h % 56 == 0 and w % 56 == 0:
        from experiments.aerial.rl.collect_consequence_dataset import pool_min

        return np.asarray(pool_min(d, 56), dtype=np.float32)
    h2, w2 = h - (h % 56), w - (w % 56)
    if h2 >= 56 and w2 >= 56:
        from experiments.aerial.rl.collect_consequence_dataset import pool_min

        y0, x0 = (h - h2) // 2, (w - w2) // 2
        return np.asarray(pool_min(d[y0 : y0 + h2, x0 : x0 + w2], 56), dtype=np.float32)
    ys = np.linspace(0, max(h - 1, 0), 56).astype(np.int64)
    xs = np.linspace(0, max(w - 1, 0), 56).astype(np.int64)
    return d[ys][:, xs].astype(np.float32)


class CruiseVeto:
    def __init__(
        self,
        score_fn: ScoreFn,
        *,
        surv_min: float = 0.95,
        fov_deg: float = 35.0,
        stall_steps: int = 8,
        stall_yaw_rad: float = math.pi / 2.0,
        min_close_m: float = 1.0,
        arrive_m: float = 3.0,
        progress_margin_m: float = 0.15,
        block_m: float = 11.0,
        hard_m: float = 4.0,
        side_margin_m: float = 1.0,
        street_m: float = 15.0,
        fly_m: float = 9.0,
        peel_commit: int = 8,
    ) -> None:
        self.score_fn = score_fn
        self.surv_min = float(surv_min)
        self.fov_deg = float(fov_deg)
        self.stall_steps = int(stall_steps)
        self.stall_yaw_rad = float(stall_yaw_rad)
        self.min_close_m = float(min_close_m)
        self.arrive_m = float(arrive_m)
        self.progress_margin_m = float(progress_margin_m)
        self.block_m = float(block_m)
        self.hard_m = float(hard_m)
        self.side_margin_m = float(side_margin_m)
        self.street_m = float(street_m)
        self.fly_m = float(fly_m)
        self.peel_commit = int(peel_commit)
        self.peel_sign = 0.0
        self.turn_yaw = 0.0
        self.gave_up = False
        self.dists: List[float] = []
        self.yaws: List[float] = []
        self.last: Dict[str, float] = {}

    def reset(self) -> None:
        self.dists.clear()
        self.yaws.clear()
        self.last = {}
        self.peel_sign = 0.0
        self.turn_yaw = 0.0
        self.gave_up = False

    def _release_turn(self) -> None:
        self.peel_sign = 0.0
        self.turn_yaw = 0.0
        self.gave_up = False

    def _hold_open_road(
        self, cruise: np.ndarray, bearing: float, limits: np.ndarray
    ) -> Tuple[np.ndarray, str, bool]:
        """Keep the world-model forward step. Yaw only a little toward the goal."""
        out = np.asarray(cruise, dtype=np.float64).reshape(4).copy()
        cap = min(HOLD_YAW_RAD, float(limits[3]))
        out[1] = 0.0
        out[3] = float(np.clip(float(bearing), -cap, cap))
        if float(out[0]) < 0.0:
            out[0] = 0.0
        self._release_turn()
        return out, "cruise", False

    def _turn_off_wall(
        self,
        cruise: np.ndarray,
        d_left: Optional[float],
        d_right: Optional[float],
        limits: np.ndarray,
        dist: float,
    ) -> Tuple[np.ndarray, str, bool]:
        """Road centre is a wall. Drop forward speed and yaw toward the open side."""
        if float(dist) <= self.arrive_m:
            self._release_turn()
            return np.zeros(4, dtype=np.float64), "hover", True
        if self.gave_up or self.turn_yaw >= YAW_BUDGET_RAD - 1e-6:
            self.gave_up = True
            return np.zeros(4, dtype=np.float64), "hover", True
        if self.peel_sign == 0.0:
            if (
                d_left is not None
                and d_right is not None
                and math.isfinite(float(d_left))
                and math.isfinite(float(d_right))
            ):
                self.peel_sign = 1.0 if float(d_left) >= float(d_right) else -1.0
            else:
                self.peel_sign = 1.0
        yaw = self.peel_sign * min(YAW_STEP_RAD, float(limits[3]))
        self.turn_yaw += abs(yaw)
        out = np.asarray(cruise, dtype=np.float64).reshape(4).copy()
        out[0] = 0.0
        out[1] = 0.0
        out[3] = yaw
        return out, "escape", True

    def apply(
        self,
        cruise: np.ndarray,
        depth_map: Optional[np.ndarray],
        pos: np.ndarray,
        yaw: float,
        goal: np.ndarray,
        limits: np.ndarray,
        offer_escape: bool = False,
    ) -> Tuple[np.ndarray, str]:
        pos = np.asarray(pos, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        limits = np.asarray(limits, dtype=np.float64).reshape(4)
        dist = goal_dist_xy(pos, goal)
        self.dists.append(dist)
        self.yaws.append(float(yaw))
        bearing = goal_bearing(pos, yaw, goal)
        straight = straight_action(bearing, limits)
        d_fwd = d_left = d_right = None
        if depth_map is not None:
            try:
                # Kept in this file. The 125 tree's depth_geometry does not
                # export approach_clearances, and a missing import was leaving
                # every step on the raw planner.
                d_fwd, d_left, d_right = _road_strip(depth_map, ROAD_STRIP_FRAC)
            except Exception:
                d_fwd = d_left = d_right = None
        if depth_map is None or d_fwd is None:
            self.last = {"d_goal": dist}
            return np.asarray(cruise, dtype=np.float64).reshape(4).copy(), "nodepth"
        surv_c = surv_s = prog_c = prog_s = float("nan")
        try:
            cruise_a = np.asarray(cruise, dtype=np.float64)
            surv_c, prog_c = self.score_fn(cruise_a, depth_map, pos, float(yaw), goal)
            surv_s, prog_s = self.score_fn(straight, depth_map, pos, float(yaw), goal)
        except Exception:
            pass
        stall = stalled(
            self.dists, self.yaws, self.stall_steps, self.stall_yaw_rad, self.min_close_m
        )
        # A short road centre is turned here. The survival of a forward step
        # into that wall does not cancel the turn. offer_escape stays in the
        # log so the planner's own mark is still visible.
        if float(d_fwd) >= ROAD_OPEN_M:
            action, tag, changed = self._hold_open_road(cruise, bearing, limits)
        else:
            action, tag, changed = self._turn_off_wall(
                cruise, d_left, d_right, limits, dist
            )
        if changed:
            action = hold_altitude(action, pos, goal, limits)
        self.last = {
            "d_goal": dist,
            "bearing": bearing,
            "surv_c": float(surv_c),
            "surv_s": float(surv_s),
            "prog_c": float(prog_c),
            "prog_s": float(prog_s),
            "stall": float(stall),
            "offer_escape": float(bool(offer_escape)),
            "d_fwd": float(d_fwd) if d_fwd is not None else float("nan"),
            "d_left": float(d_left) if d_left is not None else float("nan"),
            "d_right": float(d_right) if d_right is not None else float("nan"),
        }
        return np.clip(action, -limits, limits), tag


def _road_strip(depth_map: np.ndarray, center_frac: float) -> Tuple[float, float, float]:
    """Centre 12% at the 10th percentile; the two outside bands at the median.

    Metres are the depth map's own metres. Nothing is multiplied by two.
    """
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"road strip expects 2-D depth, got {depth.shape}")
    h, w = depth.shape
    cf = float(np.clip(center_frac, 0.05, 1.0))
    dh, dw = max(1, int(h * cf)), max(1, int(w * cf))
    r0, c0 = (h - dh) // 2, (w - dw) // 2
    crop = depth[r0 : r0 + dh, c0 : c0 + dw]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    d_fwd = float("inf") if valid.size == 0 else float(np.percentile(valid, 10.0))
    band = depth[r0 : r0 + dh, :]
    left = band[:, :c0] if c0 > 0 else band[:, : max(1, w // 4)]
    right_start = c0 + dw
    right = band[:, right_start:] if right_start < w else band[:, max(0, 3 * w // 4) :]

    def _median(region: np.ndarray) -> float:
        ok = region[np.isfinite(region) & (region > 0)]
        if ok.size == 0:
            return float("inf")
        return float(np.percentile(ok, 50.0))

    return d_fwd, _median(left), _median(right)


def ensemble_score_fn(ckpts: Sequence[str], device: str) -> ScoreFn:
    from experiments.aerial.rl.consequence_model import (
        LIMITS,
        goal_features,
        history_inputs,
        load_checkpoint,
        survival,
    )

    models = [load_checkpoint(Path(p), device) for p in ckpts]
    horizon = int(models[0].horizon)
    dev = device

    def score(
        action: np.ndarray,
        depth_map: np.ndarray,
        pos: np.ndarray,
        yaw: float,
        goal: np.ndarray,
    ) -> Tuple[float, float]:
        import torch

        depth = depth56_from_map(depth_map)
        act = np.clip(np.asarray(action, dtype=np.float32), -LIMITS, LIMITS)
        seq = np.repeat(act.reshape(1, 4), horizon, axis=0)
        state = np.zeros(7, dtype=np.float64)
        state[:3] = np.asarray(pos, dtype=np.float64)[:3]
        state[6] = float(yaw)
        g = goal_features(state, np.asarray(goal, dtype=np.float64)[:3])
        d_k, a_k = history_inputs([depth], [], models[0].frame_offsets, models[0].hist_actions)
        if len(models[0].frame_offsets) == 1:
            d_k = d_k[0]
        survs, progs = [], []
        with torch.no_grad():
            d = torch.as_tensor(d_k, device=dev).unsqueeze(0)
            g_t = torch.as_tensor(g, device=dev).unsqueeze(0)
            prev = torch.zeros(1, 4, device=dev)
            a = torch.as_tensor(seq, device=dev).unsqueeze(0)
            ha = (
                torch.as_tensor(a_k, device=dev).unsqueeze(0)
                if models[0].hist_actions
                else None
            )
            for model in models:
                out = model(d, g_t, prev, a, ha)
                survs.append(float(survival(out["hazard_logit"])[..., -1].cpu()))
                progs.append(float((-out["dgoal"][..., -1]).cpu()))
        return float(np.mean(survs)), float(np.mean(progs))

    return score
