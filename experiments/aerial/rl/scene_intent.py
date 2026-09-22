"""Non-polyline intent / subgoal outer loop (Phase-2 goal+scene nav).

Modes:
  * toward_g  — clip toward G at Phase-1 scale (E0 main)
  * direct_g  — raw far G into goal_rel (ablation A)
  * scene     — yaw-fan candidates + depth-aware score (E1)

No GT polyline. Stop / rem_dist semantics are Euclidean to G.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.aerial.rl.goal_features import goal_rel_body


def clip_toward_goal(
    curr_pos: np.ndarray,
    goal: np.ndarray,
    r_m: float,
) -> np.ndarray:
    """World-frame c*: at most ``r_m`` along the vector to G (or G if closer)."""
    p = np.asarray(curr_pos, dtype=np.float64).reshape(3)
    g = np.asarray(goal, dtype=np.float64).reshape(3)
    delta = g - p
    dist = float(np.linalg.norm(delta))
    r = float(max(1e-3, r_m))
    if dist <= r or dist < 1e-9:
        return g.copy()
    return p + delta * (r / dist)


def _safe_speed_from_depth(
    *,
    cruise_speed: float,
    min_creep_speed: float,
    d_danger: float,
    d_clear: float,
    d_fwd_hat: Optional[float],
    d_to_g: float,
    terminal_creep_rem_m: float = 8.0,
) -> float:
    v_safe = float(cruise_speed)
    if d_fwd_hat is not None and np.isfinite(float(d_fwd_hat)):
        d_fwd = float(d_fwd_hat)
        if d_fwd <= d_danger:
            v_safe = float(min_creep_speed)
        elif d_fwd < d_clear:
            t = (d_fwd - d_danger) / max(1e-6, d_clear - d_danger)
            v_safe = min_creep_speed + t * (cruise_speed - min_creep_speed)
    if d_to_g <= float(terminal_creep_rem_m):
        t_term = float(np.clip(d_to_g / max(1e-6, terminal_creep_rem_m), 0.0, 1.0))
        v_term = min_creep_speed + t_term * (cruise_speed - min_creep_speed)
        v_safe = float(min(v_safe, v_term))
    return float(v_safe)


@dataclass
class TowardGoalIntent:
    """E0 outer stub: no GT polyline; modes ``toward_g`` | ``direct_g``."""

    r_m: float = 25.0
    mode: str = "toward_g"
    cruise_speed: float = 10.0
    d_danger: float = 3.0
    d_clear: float = 22.0
    min_creep_speed: float = 1.0

    def reset(self) -> None:
        return None

    def compute(
        self,
        curr_pos: np.ndarray,
        curr_yaw: float,
        goal: np.ndarray,
        d_fwd_hat: Optional[float] = None,
        **_kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        mode = str(self.mode)
        if mode not in ("toward_g", "direct_g"):
            raise ValueError(f"unknown TowardGoalIntent.mode={mode!r}")
        p = np.asarray(curr_pos, dtype=np.float64).reshape(3)
        g = np.asarray(goal, dtype=np.float64).reshape(3)
        if mode == "direct_g":
            target = g.copy()
        else:
            r = float(self.r_m)
            if d_fwd_hat is not None and np.isfinite(float(d_fwd_hat)):
                alpha = float(
                    np.clip(
                        (float(d_fwd_hat) - self.d_danger)
                        / max(1e-6, self.d_clear - self.d_danger),
                        0.4,
                        1.0,
                    )
                )
                r = max(12.0, r * alpha)
            target = clip_toward_goal(p, g, r_m=r)

        g_rel = goal_rel_body(p, float(curr_yaw), target)
        d_to_g = float(np.linalg.norm(g - p))
        v_safe = _safe_speed_from_depth(
            cruise_speed=self.cruise_speed,
            min_creep_speed=self.min_creep_speed,
            d_danger=self.d_danger,
            d_clear=self.d_clear,
            d_fwd_hat=d_fwd_hat,
            d_to_g=d_to_g,
        )
        # Body yaw-to-carrot (rad): + = carrot left of nose. Needed by F15
        # heading cost and by the H=1 planner face-goal rule.
        yaw_err = 0.0
        if float(np.hypot(g_rel[0], g_rel[1])) > 1e-6:
            yaw_err = float(np.arctan2(float(g_rel[1]), float(g_rel[0])))
        info: Dict[str, Any] = {
            "subgoal_source": mode,
            "target_world": target.tolist(),
            "rem_dist": d_to_g,
            "s_progress": 0.0,
            "safe_speed_limit": v_safe,
            "cte_m": None,
            "r_lookahead": float(np.linalg.norm(target - p)),
            "seg_idx": 0,
            "yaw_err_rad": yaw_err,
        }
        return g_rel, info


@dataclass
class SceneIntentPlanner:
    """E1 outer loop: yaw-fan candidates + depth-aware hold/replan."""

    r_m: float = 25.0
    cruise_speed: float = 10.0
    # Fan must include lateral candidates for escape when forward is imminent (<d_danger).
    yaw_offsets_deg: Sequence[float] = field(
        default_factory=lambda: (
            0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0, -60.0, 60.0, -75.0, 75.0
        )
    )
    replan_period_s: float = 2.0
    step_hz: float = 5.0
    w_g: float = 1.0
    w_jump: float = 0.05
    # Forward-depth soft-penalty multiplier.  At 0.5 (old default) max penalty
    # was ~12.5 m < toward_g progress ~25 m, so offaxis was never chosen.
    # At 2.0 max penalty ~50 m > 25 m, so the fan genuinely steers away from
    # obstacles once d_fwd drops well below d_clear.
    w_fwd: float = 2.0
    d_danger: float = 3.0
    d_clear: float = 40.0
    min_creep_speed: float = 1.0
    stall_eps_m: float = 0.05
    # Altitude-hold: while horizontal distance to goal exceeds this radius, hold
    # the *current* altitude (do not blend candidate z toward goal z) so the fan
    # clears rooftops instead of creeping along a building face at street level.
    # 0 disables (legacy behaviour: candidate z always blends 30% toward goal z).
    descent_radius_m: float = 0.0
    # Stuck-escape (long-horizon, independent of the short replan hold above).
    # Diagnosed hard134 route 0, 2026-09-18: even after fixing the shield's
    # zero-yaw retreat, the ±75-105° fan + progress-weighted scoring settles
    # into a STABLE equilibrium a few metres off the direct line (a corner /
    # recess) and never escapes — the toward-goal progress term keeps pulling
    # it back before it clears the obstacle, round after round, regardless of
    # replan cadence or retreat-turn strength. A human pilot who's been
    # fighting the same spot for ~10s+ backs off further and tries a direction
    # they hadn't considered (even >90° off the goal bearing, or briefly
    # increasing distance to goal) instead of nibbling at the same blocked
    # approach forever. 0 disables (legacy: never widen/override on stall).
    stuck_escape_after_s: float = 12.0
    stuck_escape_hold_s: float = 5.0
    stuck_escape_progress_eps_m: float = 3.0
    stuck_escape_extra_yaw_offsets_deg: Sequence[float] = field(
        default_factory=lambda: (120.0, -120.0, 150.0, -150.0, 180.0)
    )

    _c_prev: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _steps_since_replan: int = field(default=10**9, init=False, repr=False)
    _last_d_to_g: Optional[float] = field(default=None, init=False, repr=False)
    _stall_steps: int = field(default=0, init=False, repr=False)
    # E1 observability: candidate 0 *is* the toward_g ray, so a scene arm with
    # offaxis_count == 0 is behaviourally identical to E0 main. Counted, not inferred.
    replan_count: int = field(default=0, init=False, repr=False)
    offaxis_count: int = field(default=0, init=False, repr=False)
    _last_choice_idx: int = field(default=0, init=False, repr=False)
    _last_n_feasible: int = field(default=0, init=False, repr=False)
    n_fan_starved: int = field(default=0, init=False, repr=False)
    _best_d_to_g: Optional[float] = field(default=None, init=False, repr=False)
    _no_improve_steps: int = field(default=0, init=False, repr=False)
    _escape_hold_left: int = field(default=0, init=False, repr=False)
    escape_count: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._c_prev = None
        self._steps_since_replan = 10**9
        self._last_d_to_g = None
        self._stall_steps = 0
        self.replan_count = 0
        self.offaxis_count = 0
        self._last_choice_idx = 0
        self._last_n_feasible = 0
        self.n_fan_starved = 0
        self._best_d_to_g = None
        self._no_improve_steps = 0
        self._escape_hold_left = 0
        self.escape_count = 0

    def _candidates(
        self,
        p: np.ndarray,
        yaw: float,
        goal: np.ndarray,
        *,
        extra_yaw_offsets_deg: Sequence[float] = (),
    ) -> List[np.ndarray]:
        out: List[np.ndarray] = [clip_toward_goal(p, goal, self.r_m)]
        r = float(self.r_m)
        for deg in tuple(self.yaw_offsets_deg) + tuple(extra_yaw_offsets_deg):
            psi = float(yaw) + np.deg2rad(float(deg))
            c = p + np.array(
                [r * np.cos(psi), r * np.sin(psi), 0.0], dtype=np.float64
            )
            out.append(c)
        d_horiz_to_g = float(np.hypot(goal[0] - p[0], goal[1] - p[1]))
        if self.descent_radius_m > 0.0 and d_horiz_to_g > float(self.descent_radius_m):
            z_target = float(p[2])  # hold current altitude — clear rooftops, don't hug walls
        else:
            z_target = float(p[2] + 0.3 * (goal[2] - p[2]))
        for c in out:
            c[2] = z_target
        return out

    def _should_replan(self, d_to_g: float, d_fwd_hat: Optional[float]) -> bool:
        period_steps = max(1, int(round(float(self.replan_period_s) * float(self.step_hz))))
        if self._c_prev is None:
            return True
        if self._steps_since_replan >= period_steps:
            return True
        # Emergency: imminent collision → pick escape candidate immediately.
        # Do NOT trigger on d_fwd < d_clear: that fired every step in forest
        # (d_clear=22 m is almost always exceeded), destroying the hold period.
        if d_fwd_hat is not None and np.isfinite(float(d_fwd_hat)):
            if float(d_fwd_hat) < float(self.d_danger):
                return True
        if self._last_d_to_g is not None:
            progressed = float(self._last_d_to_g) - float(d_to_g)
            if progressed < float(self.stall_eps_m):
                self._stall_steps += 1
            else:
                self._stall_steps = 0
            if self._stall_steps >= period_steps:
                return True
        return False

    def _nose_alignment(self, p: np.ndarray, cand: np.ndarray, yaw: float) -> float:
        """cos of the angle between c*-p and the body forward axis (horizontal)."""
        delta = cand - p
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        fwd = float(cos_y * delta[0] + sin_y * delta[1])
        left = float(-sin_y * delta[0] + cos_y * delta[1])
        horiz = max(1e-6, float(np.hypot(fwd, left)))
        return fwd / horiz

    def _body_bearing_deg(self, p: np.ndarray, cand: np.ndarray, yaw: float) -> float:
        """Body-frame bearing (degrees) from p to cand; 0=forward, +90=left, -90=right."""
        delta = cand - p
        world_angle = float(np.arctan2(delta[1], delta[0]))
        bearing = float(np.rad2deg(world_angle - float(yaw)))
        # normalise to (-180, 180]
        while bearing > 180.0:
            bearing -= 360.0
        while bearing <= -180.0:
            bearing += 360.0
        return bearing

    def _cone_depth(
        self,
        bearing_deg: float,
        depth_cones: Optional[Dict[str, Optional[float]]],
    ) -> Optional[float]:
        """Map a candidate's body-frame bearing to the appropriate cone depth.

        Uses the 5-cone dict from DepthMinPredictor (forward/left/right/up/down).
        Candidates pointing mostly forward use the forward cone; candidates pointing
        more than ±30° use the appropriate side cone (positive bearing = left in ENU).
        Falls back to the forward cone if the side cone is unavailable.
        """
        if depth_cones is None:
            return None
        a = abs(float(bearing_deg))
        if a <= 30.0:
            key = "forward"
        elif float(bearing_deg) > 0:
            key = "left"
        else:
            key = "right"
        v = depth_cones.get(key)
        if v is None or not np.isfinite(float(v)):
            v = depth_cones.get("forward")
        return float(v) if v is not None and np.isfinite(float(v)) else None

    def _blocked(
        self,
        p: np.ndarray,
        cand: np.ndarray,
        yaw: float,
        d_fwd_hat: Optional[float],
    ) -> bool:
        """Hard feasibility filter: only discard when imminent collision risk.

        Previous design blocked within a nose-centric cone whenever d_fwd < d_clear
        (22 m), which is almost always true in forest. When the drone faces G, the
        toward_g candidate (idx 0) aligns with the nose → gets blocked every step →
        only lateral candidates survive → drone drifts perpendicular to G (E1 bug).

        Fix: hard-block only when d_fwd < d_danger (< 3 m, truly imminent). The
        [d_danger, d_clear] range is handled as a soft cost in _score() instead.
        """
        if d_fwd_hat is None or not np.isfinite(float(d_fwd_hat)):
            return False
        d_fwd = float(d_fwd_hat)
        # Only hard-block in the imminent-danger zone (< d_danger).
        if d_fwd >= float(self.d_danger):
            return False
        # Within d_danger: block all candidates pointing more than ±90° away from
        # lateral (i.e. any candidate still pointing forward into the wall).
        return self._nose_alignment(p, cand, yaw) > 0.0

    def _score(
        self,
        p: np.ndarray,
        goal: np.ndarray,
        cand: np.ndarray,
        d_fwd_hat: Optional[float],
        yaw: float,
        depth_cones: Optional[Dict[str, Optional[float]]] = None,
    ) -> float:
        d_before = float(np.linalg.norm(goal - p))
        d_after = float(np.linalg.norm(goal - cand))
        progress = d_before - d_after
        jump = 0.0
        if self._c_prev is not None:
            jump = float(np.linalg.norm(cand - self._c_prev))
        # Per-candidate depth penalty: use the cone matching the candidate's body-frame
        # bearing so that off-axis candidates get penalised by their own direction's depth
        # (not just the drone's forward reading).  Falls back to d_fwd_hat when no cones.
        bearing = self._body_bearing_deg(p, cand, yaw)
        d_cand = self._cone_depth(bearing, depth_cones)
        if d_cand is None:
            d_cand = d_fwd_hat
        fwd_penalty = 0.0
        if d_cand is not None and np.isfinite(float(d_cand)):
            d_c = float(d_cand)
            if float(self.d_danger) <= d_c < float(self.d_clear):
                tight = float(
                    np.clip(
                        (float(self.d_clear) - d_c)
                        / max(1e-6, float(self.d_clear) - float(self.d_danger)),
                        0.0,
                        1.0,
                    )
                )
                # Penalise all candidates proportionally to how aligned they are with
                # their own danger direction (alignment now uses candidate bearing, not
                # nose — a candidate pointing 60° left is penalised by left-cone depth).
                alignment = max(0.0, self._nose_alignment(p, cand, yaw))
                fwd_penalty = tight * alignment * float(self.r_m) * float(self.w_fwd)
        return float(-float(self.w_g) * progress + float(self.w_jump) * jump + fwd_penalty)

    def _azimuth_depth(
        self,
        bearing_deg: float,
        depth_azimuth: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        """Nearest-bin clearance from a multi-sector azimuth scan (finer than
        the 2-bucket left/right in :meth:`_cone_depth`). Used only by the
        stuck-escape branch, where distinguishing e.g. 45° from 150° matters."""
        if not isinstance(depth_azimuth, dict):
            return None
        bearings = depth_azimuth.get("bearings_deg")
        clears = depth_azimuth.get("clearances_m")
        if not bearings or not clears or len(bearings) != len(clears):
            return None
        diffs = [abs(float(b) - float(bearing_deg)) for b in bearings]
        i = int(np.argmin(diffs))
        v = clears[i]
        return float(v) if v is not None and np.isfinite(float(v)) else None

    def compute(
        self,
        curr_pos: np.ndarray,
        curr_yaw: float,
        goal: np.ndarray,
        d_fwd_hat: Optional[float] = None,
        depth_cones: Optional[Dict[str, Optional[float]]] = None,
        depth_azimuth: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        p = np.asarray(curr_pos, dtype=np.float64).reshape(3)
        g = np.asarray(goal, dtype=np.float64).reshape(3)
        yaw = float(curr_yaw)
        d_to_g = float(np.linalg.norm(g - p))

        # Stuck-escape bookkeeping (long-horizon, independent of the short
        # replan hold below): has the drone made real net progress toward G
        # over the last ``stuck_escape_after_s`` seconds?
        if self._best_d_to_g is None or d_to_g < self._best_d_to_g - float(
            self.stuck_escape_progress_eps_m
        ):
            self._best_d_to_g = d_to_g
            self._no_improve_steps = 0
        else:
            self._no_improve_steps += 1

        escape_active = False
        replan = False
        if self.stuck_escape_after_s > 0.0 and self._escape_hold_left > 0:
            # Committed to an escape heading; hold it regardless of the raw
            # toward-goal pull (that pull is exactly what re-traps it).
            escape_active = True
            assert self._c_prev is not None
            target = self._c_prev.copy()
            cands = [target]
            self._escape_hold_left -= 1
            if self._escape_hold_left == 0:
                # Release: give it a fresh no-progress window before this can
                # trigger again, and force a normal replan next step.
                self._best_d_to_g = d_to_g
                self._no_improve_steps = 0
                self._steps_since_replan = 10**9
        elif self.stuck_escape_after_s > 0.0 and self._no_improve_steps >= max(
            1, int(round(float(self.stuck_escape_after_s) * float(self.step_hz)))
        ):
            # No real progress for a long while: widen the fan (including
            # near-reverse headings) and pick by CLEARANCE ALONE — ignore the
            # goal-progress term entirely, since chasing progress is what
            # produced the stall. Like a human backing fully out of a recess
            # before trying a direction they hadn't considered.
            escape_active = True
            cands = self._candidates(
                p, yaw, g, extra_yaw_offsets_deg=self.stuck_escape_extra_yaw_offsets_deg
            )
            best_idx, best, best_clear = 0, cands[0], -1.0
            for i, c in enumerate(cands):
                bearing = self._body_bearing_deg(p, c, yaw)
                # Prefer the finer multi-sector azimuth scan (distinguishes
                # e.g. 45° from 150°); fall back to the 2-bucket cone dict,
                # which cannot tell those apart (diagnosed hard134 route 0:
                # escape kept re-picking ~the same heading without it).
                dc = self._azimuth_depth(bearing, depth_azimuth)
                if dc is None:
                    dc = self._cone_depth(bearing, depth_cones)
                if dc is None:
                    dc = (
                        float(d_fwd_hat)
                        if d_fwd_hat is not None and np.isfinite(float(d_fwd_hat))
                        else 0.0
                    )
                if float(dc) > best_clear:
                    best_clear = float(dc)
                    best_idx = i
                    best = c
            target = best.copy()
            self._c_prev = target.copy()
            hold_steps = max(1, int(round(float(self.stuck_escape_hold_s) * float(self.step_hz))))
            self._escape_hold_left = hold_steps - 1
            self._steps_since_replan = 0
            self._stall_steps = 0
            self.replan_count += 1
            self.offaxis_count += 1
            self.escape_count += 1
            self._last_choice_idx = best_idx
            self._last_n_feasible = len(cands)

        if not escape_active:
            replan = self._should_replan(d_to_g, d_fwd_hat)
            cands = self._candidates(p, yaw, g)

            if replan:
                feasible = [
                    (i, c)
                    for i, c in enumerate(cands)
                    if not self._blocked(p, c, yaw, d_fwd_hat)
                ]
                n_feasible = len(feasible)
                if not feasible:
                    # Whole fan inside the danger cone: take the most lateral option
                    # and let creep speed + shield handle it. Never return nothing.
                    feasible = [
                        min(
                            enumerate(cands),
                            key=lambda ic: self._nose_alignment(p, ic[1], yaw),
                        )
                    ]
                best_idx, best = feasible[0]
                best_j = self._score(p, g, best, d_fwd_hat, yaw, depth_cones=depth_cones)
                for i, c in feasible[1:]:
                    j = self._score(p, g, c, d_fwd_hat, yaw, depth_cones=depth_cones)
                    if j < best_j:
                        best_j = j
                        best = c
                        best_idx = i
                target = best.copy()
                self._c_prev = target.copy()
                # Candidate 0 is clip_toward_goal — identical to TowardGoalIntent.
                # Skip the hold period so it tracks position every step, matching
                # the E0 baseline.  Offaxis candidates hold for stability.
                self._steps_since_replan = 10**9 if best_idx == 0 else 0
                self._stall_steps = 0
                self.replan_count += 1
                if best_idx != 0:
                    self.offaxis_count += 1
                if n_feasible == 0:
                    self.n_fan_starved += 1
                self._last_choice_idx = best_idx
                self._last_n_feasible = n_feasible
            else:
                assert self._c_prev is not None
                target = self._c_prev.copy()
                self._steps_since_replan += 1

        n_cands = len(cands)
        self._last_d_to_g = d_to_g
        # Horizontal peel of c* off the direct-to-G ray (deg); 0 = pure toward_g.
        v_c = (target - p)[:2]
        v_g = (g - p)[:2]
        n_c, n_g = float(np.linalg.norm(v_c)), float(np.linalg.norm(v_g))
        if n_c < 1e-6 or n_g < 1e-6:
            dev_deg = 0.0
        else:
            cos_dev = float(np.clip(np.dot(v_c, v_g) / (n_c * n_g), -1.0, 1.0))
            dev_deg = float(np.rad2deg(np.arccos(cos_dev)))
        g_rel = goal_rel_body(p, yaw, target)
        v_safe = _safe_speed_from_depth(
            cruise_speed=self.cruise_speed,
            min_creep_speed=self.min_creep_speed,
            d_danger=self.d_danger,
            d_clear=self.d_clear,
            d_fwd_hat=d_fwd_hat,
            d_to_g=d_to_g,
        )
        info: Dict[str, Any] = {
            "subgoal_source": "scene",
            "target_world": target.tolist(),
            "rem_dist": d_to_g,
            "s_progress": 0.0,
            "safe_speed_limit": v_safe,
            "cte_m": None,
            "r_lookahead": float(np.linalg.norm(target - p)),
            "seg_idx": 0,
            "n_candidates": int(n_cands),
            "replan": bool(replan),
            "n_feasible": int(self._last_n_feasible),
            "chosen_idx": int(self._last_choice_idx),
            "dev_deg": dev_deg,
            "replan_count": int(self.replan_count),
            "offaxis_count": int(self.offaxis_count),
            "n_fan_starved": int(self.n_fan_starved),
            "in_escape": bool(escape_active),
            "escape_count": int(self.escape_count),
        }
        return g_rel, info
