"""Hard safety shield (spec §2#6, §4.5) — interface + null stub.

The shield sits ABOVE the learned policy: if inflated predicted depth ``D̂``,
time-to-contact ``τ``, or world-model collision probability ``p_coll`` breaches a
threshold, it overrides the policy's action with a conservative one (brake /
hover / retreat). It is a *hard* override, not a learned behaviour — so it lives
outside the RL graph.

Only the contract is fixed here. ``NullSafetyShield`` never overrides (V0/V1
default). A real ``DepthTauShield`` is deferred until the perception heads that
produce ``D̂`` / ``τ`` exist (V2+); ``ThresholdSafetyShield`` shows the intended
trigger wiring against fields that may not be populated yet.

**Three-zone deploy (2026-08-23)**: ``ThreeZoneSpeedShield`` replaces the single
3 m depth latch with a graduated speed governor (8/5/1.5 m @ 2/1/0.2 m/s).
τ emergencies still latch + retreat; ``p_coll`` latch is vetoed when forward
clearance exceeds L1 (false WM collision while corridor is open).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from experiments.aerial.rl.env.action import MAX_BODY_VELOCITY, clip_body_delta, wrap_angle
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.three_zone import ThreeZoneSpec
from experiments.aerial.rl.tau_predictor import (
    DEFAULT_MIN_CLOSING_M_S,
    closing_speed_m_s,
)


@runtime_checkable
class SafetyShield(Protocol):
    def should_override(self, obs: Observation, wm_out: Optional[Any] = None) -> bool: ...

    def override_action(self, obs: Observation) -> np.ndarray: ...


class NullSafetyShield:
    """No-op shield: never intervenes. Default until D̂/τ heads exist."""

    def reset(self) -> None:
        return None

    def should_override(self, obs: Observation, wm_out: Optional[Any] = None) -> bool:
        return False

    def override_action(self, obs: Observation) -> np.ndarray:
        return np.zeros(4, dtype=np.float64)

    def apply_action(
        self,
        action: np.ndarray,
        obs: Observation,
        wm_out: Optional[Any] = None,
        limits: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, bool]:
        lim = limits
        return clip_body_delta(action, lim), False


@dataclass
class ThresholdSafetyShield:
    """Trigger contract for D̂ ∪ τ ∪ p_coll (fields wired at V2+).

    **Standoff semantics (v5, 2026-08-22)** — ``min_depth_m`` is the boundary of
    the *stable-hover zone*, not the distance at which braking *starts*. The
    vehicle must bleed closing speed **before** crossing the standoff so it is
    near-stationary **inside** ``min_depth_m``. Kinematic engage when::

        D̂ < min_depth_m + v_fwd * min_tau_s

    (same ``min_tau_s`` reaction budget as the τ leg; thresholds unchanged).

    Override uses **graduated body −x** scaled to ``v_fwd`` (capped by
    ``retreat_step_m``), including while still **outside** the standoff but
    inside the braking envelope — not a step function at 3 m.

    Prior latch + bounded retreat history (晚¹⁰–¹²) remains; v5 fixes the
    high-speed “coast into 3 m then panic” failure mode.

    **Legacy** — superseded for deploy by :class:`ThreeZoneSpeedShield`.
    """

    # Reaction standoff outer boundary — must be stable/hovering inside, not enter at cruise.
    min_depth_m: float = 3.0
    min_tau_s: float = 1.0            # τ breach + kinematic depth braking horizon (s)
    max_p_coll: float = 0.5           # brake if WM collision prob > this
    min_closing_m_s: float = DEFAULT_MIN_CLOSING_M_S
    brake_gain: float = 1.0           # retreat dx ≈ v_fwd * brake_gain per step (then clipped)
    retreat_step_m: float = 3.0       # max |body −x| per override step
    _engaged: bool = field(default=False, init=False, repr=False)

    def reset(self) -> None:
        """Clear the per-episode latch (the shield instance is reused across episodes)."""
        self._engaged = False

    def _kinematic_standoff_limit_m(self, v_fwd: float) -> float:
        """Outer engage surface: standoff + distance closed in ``min_tau_s`` at ``v_fwd``."""
        return float(self.min_depth_m) + max(float(v_fwd), 0.0) * float(self.min_tau_s)

    def _depth_channel_breach(self, d_hat: float, v_fwd: float) -> bool:
        d = float(d_hat)
        if d < float(self.min_depth_m):
            return True
        if v_fwd > float(self.min_closing_m_s):
            return d < self._kinematic_standoff_limit_m(v_fwd)
        return False

    def _needs_speed_bleed(self, obs: Observation) -> bool:
        d_hat = obs.info.get("depth_min_pred")
        if d_hat is None:
            return False
        v = closing_speed_m_s(obs)
        return self._depth_channel_breach(float(d_hat), v)

    def _breached(self, obs: Observation, wm_out: Optional[Any] = None) -> bool:
        d_hat = obs.info.get("depth_min_pred")
        tau = obs.info.get("tau_pred")
        p_coll = None
        if wm_out is not None:
            p_coll = getattr(wm_out, "p_coll", None)
        if d_hat is not None and self._depth_channel_breach(
            float(d_hat), closing_speed_m_s(obs)
        ):
            return True
        if tau is not None and float(tau) < self.min_tau_s:
            return True
        if p_coll is not None and float(p_coll) > self.max_p_coll:
            return True
        return False

    def should_override(self, obs: Observation, wm_out: Optional[Any] = None) -> bool:
        if self._engaged:
            return True
        if self._breached(obs, wm_out):
            self._engaged = True
            return True
        return False

    def override_action(self, obs: Observation) -> np.ndarray:
        if not self._needs_speed_bleed(obs):
            return np.zeros(4, dtype=np.float64)
        v = closing_speed_m_s(obs)
        # Graduated −x: bleed closing speed before/at standoff; collector clips to rate cap.
        mag = min(
            float(self.retreat_step_m),
            max(float(v), float(self.min_closing_m_s)) * float(self.brake_gain),
        )
        return np.array([-abs(mag), 0.0, 0.0, 0.0], dtype=np.float64)

    def apply_action(
        self,
        action: np.ndarray,
        obs: Observation,
        wm_out: Optional[Any] = None,
        limits: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, bool]:
        if self.should_override(obs, wm_out):
            return clip_body_delta(self.override_action(obs), limits), True
        return clip_body_delta(action, limits), False


@dataclass
class DepthTauShield(ThresholdSafetyShield):
    """τ/D̂ dual-channel hard shield (frozen spec V1).

    Same trigger/override contract as :class:`ThresholdSafetyShield`, but records
    which independent channel(s) breached for V1-③ diagnostics. Writes
    ``obs.info['shield_channels']`` on the step the latch engages.

    **Legacy** — deploy uses :class:`ThreeZoneSpeedShield`.
    """

    _last_channels: tuple[str, ...] = field(default=(), init=False, repr=False)

    @property
    def last_channels(self) -> tuple[str, ...]:
        return self._last_channels

    def reset(self) -> None:
        super().reset()
        self._last_channels = ()

    def _channels_breached(self, obs: Observation, wm_out: Optional[Any] = None) -> tuple[str, ...]:
        out: list[str] = []
        d_hat = obs.info.get("depth_min_pred")
        tau = obs.info.get("tau_pred")
        p_coll = None
        if wm_out is not None:
            p_coll = getattr(wm_out, "p_coll", None)
        if d_hat is not None and self._depth_channel_breach(
            float(d_hat), closing_speed_m_s(obs)
        ):
            out.append("depth")
        if tau is not None and float(tau) < self.min_tau_s:
            out.append("tau")
        if p_coll is not None and float(p_coll) > self.max_p_coll:
            out.append("p_coll")
        return tuple(out)

    def should_override(self, obs: Observation, wm_out: Optional[Any] = None) -> bool:
        if self._engaged:
            return True
        channels = self._channels_breached(obs, wm_out)
        if channels:
            self._engaged = True
            self._last_channels = channels
            obs.info["shield_channels"] = list(channels)
            return True
        return False


@dataclass
class ThreeZoneSpeedShield:
    """TTI speed governor: single-line + 3 m hard exclusion + τ/p_coll latch.

    Forward cap: v_fwd ≤ d_fwd / tti_coeff  (enforces TTI ≥ tti_coeff seconds).
    Lateral cap: v_lat ≤ d_lat / tti_coeff  per axis (same TTI budget).
    Hard exclusion: d_fwd ≤ exclusion_m → per-step −x retreat (no episode latch).
    τ / p_coll: latch + graduated −x retreat (unchanged from prior design).

    Default tti_coeff=4.0 → at 25 m/s trigger at 100 m; at 10 m/s trigger at 40 m.
    """

    zone: ThreeZoneSpec = field(default_factory=ThreeZoneSpec)
    min_tau_s: float = 1.0
    max_p_coll: float = 0.5
    min_closing_m_s: float = DEFAULT_MIN_CLOSING_M_S
    brake_gain: float = 1.0
    retreat_step_m: float = 3.0
    #: TTI budget (seconds). Trigger distance = tti_coeff × v_eff.
    tti_coeff: float = 4.0
    #: Once forward TTI cap engages, hold until d_fwd ≥ trigger × (1 + frac).
    #: Reduces bang-bang stutter when d_fwd hovers near the trigger surface.
    tti_hysteresis_release_frac: float = 0.0
    #: Hard exclusion zone (metres). Below this → retreat, no forward motion.
    exclusion_m: float = 3.0
    #: When True, exclusion uses forward cone only — ignore full-FOV ``depth_min_pred``
    #: (peripheral close objects that cannot hit the small airframe).
    exclusion_forward_only: bool = False
    #: Near-goal soft exclusion: if rem ≤ this and nose aligned, creep instead of
    #: full hard-brake (fixes terminal grind when d_fwd hovers near exclusion_m).
    #: 0 disables. Requires ``obs.info["d_to_g"]`` (or rem_dist) + yaw_err_rad.
    terminal_soft_exclusion_rem_m: float = 20.0
    #: |yaw_err_rad| below this → eligible for terminal soft exclusion (≈25°).
    terminal_soft_exclusion_yaw_rad: float = float(np.deg2rad(25.0))
    #: Below this forward clearance, always hard-brake (crash-imminent).
    terminal_soft_exclusion_min_fwd_m: float = 1.5
    #: Max body +dx (m/step) allowed under terminal soft exclusion.
    terminal_soft_exclusion_dx: float = 0.20
    #: When True (default), use max(v_now, v_cmd) as effective speed reference.
    dynamic_v_ref: bool = True
    #: Ignore WM ``p_coll`` emergency when forward clearance exceeds this (metres).
    #: ``None`` → use zone.l1_m (8 m). Set <=0 to disable.
    p_coll_clearance_veto_m: Optional[float] = None
    #: Shield contract: **speed governor only** — never choose heading.
    #: Emergency/exclusion may brake on body −x; dyaw must stay 0 unless an
    #: explicit ablation sets ``retreat_max_dyaw_rad > 0`` (discouraged: that
    #: path decided left/right from cones and caused reverse-flight spins).
    retreat_max_dyaw_rad: float = 0.0
    #: Only used when ``retreat_max_dyaw_rad > 0`` (ablation). Cap |Σ retreat
    #: dyaw| while continuously retreating. 0 disables the cum gate.
    retreat_max_cum_yaw_rad: float = 0.70
    #: Only used when ``retreat_max_dyaw_rad > 0`` (ablation). Goal-facing gate.
    retreat_max_head_vs_goal_rad: float = 1.047
    _emergency_engaged: bool = field(default=False, init=False, repr=False)
    _last_channels: tuple[str, ...] = field(default=(), init=False, repr=False)
    _clear_danger_steps: int = field(default=0, init=False, repr=False)
    _forward_cap_latched: bool = field(default=False, init=False, repr=False)
    _retreat_yaw_accum: float = field(default=0.0, init=False, repr=False)

    @property
    def last_channels(self) -> tuple[str, ...]:
        return self._last_channels

    def reset(self) -> None:
        self._emergency_engaged = False
        self._last_channels = ()
        self._clear_danger_steps = 0
        self._forward_cap_latched = False
        self._retreat_yaw_accum = 0.0

    def _effective_tti(self, obs: Observation) -> float:
        override = obs.info.get("shield_tti_coeff")
        if override is not None and np.isfinite(float(override)):
            return float(override)
        return float(self.tti_coeff)

    def _p_coll_clearance_veto_m(self) -> Optional[float]:
        if self.p_coll_clearance_veto_m is None:
            return float(self.zone.l1_m)
        v = float(self.p_coll_clearance_veto_m)
        return v if v > 0.0 else None

    def _emergency_channels(self, obs: Observation, wm_out: Optional[Any] = None) -> tuple[str, ...]:
        out: list[str] = []
        tau = obs.info.get("tau_pred")
        p_coll = None
        if wm_out is not None:
            p_coll = getattr(wm_out, "p_coll", None)
        if tau is not None and float(tau) < self.min_tau_s:
            out.append("tau")
        if p_coll is not None and float(p_coll) > self.max_p_coll:
            veto = self._p_coll_clearance_veto_m()
            d_fwd = self._forward_d_hat(obs)
            if (
                veto is not None
                and d_fwd is not None
                and np.isfinite(float(d_fwd))
                and float(d_fwd) > float(veto)
            ):
                obs.info["shield_p_coll_vetoed"] = True
                obs.info["shield_p_coll_veto_d_fwd_m"] = round(float(d_fwd), 4)
            else:
                out.append("p_coll")
        return tuple(out)

    def _signed_bearing_to_goal(self, obs: Observation) -> Optional[float]:
        """Signed yaw error to ``obs.info['goal']`` (+ = goal is left of heading)."""
        raw = obs.info.get("goal")
        if raw is None:
            return None
        try:
            g = np.asarray(raw, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if g.size < 2 or not np.all(np.isfinite(g[:2])):
            return None
        dx = float(g[0] - obs.position[0])
        dy = float(g[1] - obs.position[1])
        if abs(dx) + abs(dy) < 1e-6:
            return 0.0
        bearing = float(np.arctan2(dy, dx))
        return float(wrap_angle(bearing - float(obs.yaw)))

    def _clearer_cone_dyaw(self, obs: Observation, max_dyaw: float) -> float:
        cones = self._cones(obs)
        if not cones:
            return 0.0
        left = cones.get("left")
        right = cones.get("right")
        left_f = float(left) if left is not None and np.isfinite(float(left)) else None
        right_f = float(right) if right is not None and np.isfinite(float(right)) else None
        if left_f is None and right_f is None:
            return 0.0
        if right_f is None or (left_f is not None and left_f > right_f):
            return float(max_dyaw)
        if left_f is None or right_f > left_f:
            return -float(max_dyaw)
        return 0.0

    def _retreat_dyaw(self, obs: Observation) -> float:
        """Heading command during retreat — default **0** (no direction decision).

        Shield is a speed rule, not a pilot: it must not choose left/right from
        cones. Opt-in ``retreat_max_dyaw_rad > 0`` keeps the old clearer-cone
        ablation behind cum/goal gates for A/B only.
        """
        if float(self.retreat_max_dyaw_rad) <= 0.0:
            return 0.0
        max_step = float(self.retreat_max_dyaw_rad)
        max_cum = float(self.retreat_max_cum_yaw_rad)
        if max_cum > 0.0 and abs(float(self._retreat_yaw_accum)) >= max_cum:
            return 0.0

        raw = self._clearer_cone_dyaw(obs, max_step)
        bearing_err = self._signed_bearing_to_goal(obs)
        face_lim = float(self.retreat_max_head_vs_goal_rad)
        if bearing_err is not None and face_lim > 0.0:
            toward = float(np.sign(bearing_err)) * max_step if abs(bearing_err) > 1e-6 else 0.0
            if abs(bearing_err) >= face_lim:
                if raw == 0.0:
                    raw = toward
                elif toward != 0.0 and np.sign(raw) != np.sign(toward):
                    raw = toward
            elif raw != 0.0 and toward != 0.0 and np.sign(raw) != np.sign(toward):
                room = face_lim - abs(bearing_err)
                if room <= 1e-6:
                    raw = 0.0
                else:
                    raw = float(np.sign(raw)) * min(abs(raw), room)
            elif raw != 0.0 and toward == 0.0:
                raw = float(np.sign(raw)) * min(abs(raw), face_lim)

        if max_cum > 0.0:
            remain = max_cum - abs(float(self._retreat_yaw_accum))
            if remain <= 1e-6:
                return 0.0
            if abs(raw) > remain:
                raw = float(np.sign(raw)) * remain if abs(raw) > 1e-12 else 0.0
        return float(raw)

    def _speed_brake(self, obs: Observation, action: np.ndarray) -> np.ndarray:
        """Hard proximity: constrain forward speed only; keep policy lateral/yaw.

        Contract: shield is a speed rule, not a pilot. Forward axis is clipped to
        ``[-brake_mag, 0]`` (no forward into danger); dy/dz/dyaw stay as proposed
        unless an explicit ``retreat_max_dyaw_rad > 0`` ablation overwrites yaw.
        """
        out = np.asarray(action, dtype=np.float64).reshape(4).copy()
        v = closing_speed_m_s(obs)
        mag = min(
            float(self.retreat_step_m),
            max(float(v), float(self.min_closing_m_s)) * float(self.brake_gain),
        )
        out[0] = float(np.clip(float(out[0]), -abs(mag), 0.0))
        dyaw = self._retreat_dyaw(obs)
        if abs(float(dyaw)) > 1e-12:
            out[3] = float(dyaw)
            self._retreat_yaw_accum = float(self._retreat_yaw_accum) + float(dyaw)
            obs.info["shield_retreat_dyaw"] = round(float(dyaw), 4)
            obs.info["shield_retreat_yaw_accum"] = round(float(self._retreat_yaw_accum), 4)
        return out

    def _dt_from_limits(self, limits: Optional[np.ndarray]) -> float:
        if limits is not None and float(limits[0]) > 0:
            return float(limits[0]) / float(MAX_BODY_VELOCITY[0])
        return float(self.zone.dt_s)

    def _cones(self, obs: Observation) -> Optional[dict]:
        raw = obs.info.get("depth_cones_pred")
        return raw if isinstance(raw, dict) else None

    def _forward_d_hat(self, obs: Observation) -> Optional[float]:
        """Forward clearance: prioritize dedicated forward cone.
        Only fall back to full-min if cones are unavailable.
        """
        cones = self._cones(obs)
        if cones is not None:
            fwd = cones.get("forward")
            if fwd is not None and np.isfinite(float(fwd)):
                return float(fwd)
        full = obs.info.get("depth_min_pred")
        if full is not None and np.isfinite(float(full)):
            return float(full)
        return None

    def _cap_forward(self, action: np.ndarray, obs: Observation, limits: Optional[np.ndarray]) -> tuple[np.ndarray, bool]:
        """TTI forward cap: v_fwd ≤ d_fwd / tti_coeff. Exclusion zone handled by _exclusion_brake."""
        d_hat = self._forward_d_hat(obs)
        if d_hat is None:
            return action, False
        d = float(d_hat)
        if d <= float(self.exclusion_m):
            return action, False  # handled by _exclusion_brake
        dt = self._dt_from_limits(limits)
        capped = np.asarray(action, dtype=np.float64).reshape(4).copy()
        v_now = float(closing_speed_m_s(obs))
        v_cmd = max(0.0, float(capped[0]) / max(dt, 1e-6))
        v_ref = max(v_now, v_cmd) if bool(self.dynamic_v_ref) else float(self.zone.v_cruise_m_s)
        tti = self._effective_tti(obs)
        trigger = float(tti) * v_ref
        hyst = float(self.tti_hysteresis_release_frac)
        release = trigger * (1.0 + hyst) if hyst > 0.0 else trigger
        if v_ref < 1e-6:
            return action, False
        if hyst > 0.0 and self._forward_cap_latched:
            if d >= release:
                self._forward_cap_latched = False
                if d >= trigger:
                    return action, False
        elif d >= trigger:
            return action, False
        elif hyst > 0.0:
            self._forward_cap_latched = True
        v_cap = d / float(tti)
        obs.info["tii_speed_cap_m_s"] = round(v_cap, 4)
        obs.info["tii_d_hat_fwd_m"] = round(d, 4)
        max_dx = v_cap * dt
        if capped[0] > max_dx + 1e-6:
            capped[0] = max_dx
            return capped, True
        return action, False

    def _cap_lateral(
        self, action: np.ndarray, obs: Observation, limits: Optional[np.ndarray]
    ) -> tuple[np.ndarray, bool]:
        """TTI lateral cap: v_lat ≤ d_lat / tti_coeff per axis (body +y=left, +z=up)."""
        cones = self._cones(obs)
        if cones is None:
            return action, False
        dt = self._dt_from_limits(limits)
        capped = np.asarray(action, dtype=np.float64).reshape(4).copy()
        hit: list[str] = []

        def _finite(key: str) -> Optional[float]:
            v = cones.get(key)
            if v is None:
                return None
            f = float(v)
            return f if np.isfinite(f) else None

        def _tti_clamp(d_obs: Optional[float], delta: float, moving_toward: bool) -> tuple[float, bool]:
            if not moving_toward or d_obs is None:
                return delta, False
            v_abs = abs(delta) / max(dt, 1e-6)
            if v_abs < 1e-6:
                return delta, False
            trigger = float(self._effective_tti(obs)) * v_abs
            if float(d_obs) >= trigger:
                return delta, False
            max_delta = float(d_obs) / float(self._effective_tti(obs)) * dt
            if abs(delta) <= max_delta + 1e-6:
                return delta, False
            return float(np.sign(delta)) * max_delta, True

        left = _finite("left")
        right = _finite("right")
        up = _finite("up")
        down = _finite("down")

        new_y, hl = _tti_clamp(left, capped[1], capped[1] > 1e-6)
        if hl:
            capped[1] = new_y
            hit.append("left")
        new_y2, hr = _tti_clamp(right, capped[1], capped[1] < -1e-6)
        if hr:
            capped[1] = new_y2
            hit.append("right")
        new_z, hu = _tti_clamp(up, capped[2], capped[2] > 1e-6)
        if hu:
            capped[2] = new_z
            hit.append("up")
        new_z2, hd = _tti_clamp(down, capped[2], capped[2] < -1e-6)
        if hd:
            capped[2] = new_z2
            hit.append("down")

        if not hit:
            return capped, False
        ch = list(obs.info.get("shield_channels") or [])
        if "tii_lat" not in ch:
            ch.append("tii_lat")
        obs.info["shield_channels"] = ch
        obs.info["tii_lat_axes"] = hit
        return capped, True

    def _exclusion_brake(
        self, obs: Observation, action: np.ndarray, limits: Optional[np.ndarray]
    ) -> Optional[tuple[np.ndarray, bool]]:
        """Hard exclusion: d_fwd ≤ exclusion_m → forward-speed brake (keep yaw).

        Near-goal exception: when rem is small, yaw is aligned, and forward is
        not crash-imminent, creep forward under a soft cap instead of dx≤0
        (R4 terminal grind). Still reports intervened=True via governor path.
        """
        d_hat = self._forward_d_hat(obs)
        d_crit = d_hat
        if not bool(self.exclusion_forward_only):
            full = obs.info.get("depth_min_pred")
            full_f = float(full) if full is not None and np.isfinite(float(full)) else None
            if full_f is not None and (d_crit is None or full_f < d_crit):
                d_crit = full_f
        if d_crit is None or float(d_crit) > float(self.exclusion_m):
            return None

        # Terminal soft exclusion (aligned approach, not crash-imminent).
        rem = obs.info.get("d_to_g")
        if rem is None:
            rem = obs.info.get("rem_dist")
        yaw_err = obs.info.get("yaw_err_rad")
        rem_lim = float(self.terminal_soft_exclusion_rem_m)
        if (
            rem_lim > 0.0
            and rem is not None
            and yaw_err is not None
            and float(rem) <= rem_lim
            and abs(float(yaw_err)) <= float(self.terminal_soft_exclusion_yaw_rad)
            and float(d_crit) >= float(self.terminal_soft_exclusion_min_fwd_m)
        ):
            out = np.asarray(action, dtype=np.float64).reshape(4).copy()
            creep = float(self.terminal_soft_exclusion_dx)
            out[0] = float(np.clip(float(out[0]), 0.0, creep))
            ch = list(obs.info.get("shield_channels") or [])
            if "tii_exclusion_soft_terminal" not in ch:
                ch.append("tii_exclusion_soft_terminal")
            obs.info["shield_channels"] = ch
            obs.info["shield_governor_cap"] = True
            obs.info["shield_hard_brake"] = False
            obs.info["tii_d_hat_fwd_m"] = round(float(d_crit), 4)
            obs.info["shield_terminal_soft_exclusion"] = True
            self._last_channels = tuple(ch)
            return clip_body_delta(out, limits), True

        ch = list(obs.info.get("shield_channels") or [])
        if "tii_exclusion" not in ch:
            ch.append("tii_exclusion")
        obs.info["shield_channels"] = ch
        obs.info["tii_speed_cap_m_s"] = 0.0
        obs.info["tii_d_hat_fwd_m"] = round(float(d_crit), 4)
        obs.info["shield_hard_brake"] = True
        self._last_channels = tuple(ch)
        return clip_body_delta(self._speed_brake(obs, action), limits), True

    def apply_action(
        self,
        action: np.ndarray,
        obs: Observation,
        wm_out: Optional[Any] = None,
        limits: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, bool]:
        action = np.asarray(action, dtype=np.float64).reshape(4)

        # Per-step flags: collector may carry last-step values onto this obs.
        # Clear first so a clear/no-cap frame never keeps a sticky hard_brake
        # (which would falsely charge w_intervention on open sky).
        if isinstance(obs.info, dict):
            obs.info["shield_hard_brake"] = False
            obs.info["shield_governor_cap"] = False
            obs.info["shield_emergency_override"] = False

        # 1. τ / p_coll emergency latch — forward-speed brake, keep heading cmds
        channels = self._emergency_channels(obs, wm_out)
        if self._emergency_engaged:
            if not channels:
                self._clear_danger_steps += 1
                if self._clear_danger_steps >= 3:
                    self._emergency_engaged = False
                    self._clear_danger_steps = 0
                    self._retreat_yaw_accum = 0.0
            else:
                self._clear_danger_steps = 0
            if self._emergency_engaged:
                obs.info["shield_emergency_override"] = True
                obs.info["shield_hard_brake"] = True
                return clip_body_delta(self._speed_brake(obs, action), limits), True
        if channels:
            self._emergency_engaged = True
            self._last_channels = channels
            obs.info["shield_channels"] = list(channels)
            obs.info["shield_emergency_override"] = True
            obs.info["shield_hard_brake"] = True
            return clip_body_delta(self._speed_brake(obs, action), limits), True

        # 2. Hard exclusion zone: d_fwd ≤ exclusion_m → forward-speed brake
        braked = self._exclusion_brake(obs, action, limits)
        if braked is not None:
            out, _ = braked
            # Exclusion is a hard speed brake, not the τ/p_coll emergency latch.
            # Do not set shield_emergency_override (that flag is latch-only).
            return clip_body_delta(out, limits), True

        # Not in hard brake: drop cumulative yaw budget for optional ablation.
        self._retreat_yaw_accum = 0.0

        # 3. TTI forward + lateral cap (speed governor — not a heading decision)
        capped, fwd_ch = self._cap_forward(action, obs, limits)
        lat, lat_ch = self._cap_lateral(capped, obs, limits)
        if fwd_ch or lat_ch:
            obs.info["shield_governor_cap"] = True
            self._last_channels = tuple(obs.info.get("shield_channels") or [])
        return clip_body_delta(lat, limits), bool(fwd_ch or lat_ch)

    def should_override(self, obs: Observation, wm_out: Optional[Any] = None) -> bool:
        """Backward-compat: true only for τ/p_coll emergency latch."""
        if self._emergency_engaged:
            return True
        return bool(self._emergency_channels(obs, wm_out))

    def override_action(self, obs: Observation) -> np.ndarray:
        # Legacy path (no proposed action): cancel forward only.
        return self._speed_brake(obs, np.zeros(4, dtype=np.float64))
