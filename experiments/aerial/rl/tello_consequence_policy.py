"""Indoor Tello closed-loop collector policy (consequence post-train).

Looks at the current frame (DA3 depth when available, else an RGB wall-fill
proxy) and Tello yaw/height. Actions are hover, yaw-in-place, or a small
forward creep — never reverse, never strafe, never forward+yaw together.

Goal bearing is tracked from Tello yaw vs the takeoff-relative bearing in
``meta.json``. Range is *not* integrated from vg*.

Optional consequence checkpoints act as a collision shield: they score the
same four families; the pick is the survivor with the most forward clearance
(not predicted dgoal, which needs metric xy we do not have).

Optional ``avoid`` (``SemanticKeepout``) recognises named scene objects (e.g.
the red stool), writes them into the depth map as inflated pillars that both
the reactive cones and the consequence shield see, and aims goal alignment at
the pillar's tangent edge on one committed side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.aerial.rl.consequence_model import cones3_from_depth56
from experiments.aerial.rl.env.tello_env import body_delta_to_rc
from experiments.aerial.rl.semantic_keepout import KeepOut, SemanticKeepout, apply_keepouts

NEAR_M = 1.15
SLOW_M = 2.4
YAW_ALIGN_RAD = math.radians(25.0)
DETOUR_ALIGN_RAD = math.radians(8.0)
FOV_MAX_DEG = 35.0
HOVER_Z_LO_M = 0.90
HOVER_Z_HI_M = 1.60
TRIM_Z_M = 1.20
MAX_V_MPS = 0.35
MAX_YAW_RAD_S = math.pi / 6.0
FWD_STICK = 30
YAW_STICK = 26
COMMIT_FWD_STEPS = 8
TELLO_FAMILIES = ("hover", "yaw_left", "yaw_right", "fwd")
# Tello ag* is milli-g (hover agz ≈ -1000). Cruise |agx| stays ~50; a furniture
# tap on run_20261012_180635 hit |agx| 176–208 while still commanding forward.
IMU_NEAR_XY = 85.0
IMU_HIT_XY = 120.0
MIN_AIR_M = 0.45


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clearance_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Fwd / left / right meters.

    Blank walls still read close via low texture. Textured furniture that
    sits on the ground and touches the lower FOV also reads close — texture
    alone treated cardboard as 3 m+ and the policy crept into it.
    """
    g = np.asarray(rgb)
    if g.ndim == 3:
        g = g.mean(axis=2)
    g = g.astype(np.float32)
    h, w = g.shape
    c0, c1 = w // 4, 3 * w // 4
    mid = g[h // 4 : 3 * h // 4]
    low = g[int(h * 0.55) :, :]
    floor_med = float(np.median(g[int(h * 0.60) : int(h * 0.92), w // 3 : 2 * w // 3]))

    def tex_m(patch: np.ndarray) -> float:
        gy, gx = np.gradient(patch)
        mag = float(np.hypot(gx, gy).mean())
        score = 0.5 * float(patch.std()) + 0.5 * mag
        return float(np.clip(0.35 + score / 8.0, 0.35, 8.0))

    def occ_m(patch: np.ndarray) -> float:
        if patch.size == 0:
            return 8.0
        occ = float(np.mean(np.abs(patch - floor_med) > 18.0))
        return float(np.clip(0.40 + (1.0 - occ) * 5.2, 0.40, 8.0))

    def meters(mid_patch: np.ndarray, low_patch: np.ndarray) -> float:
        return float(min(tex_m(mid_patch), occ_m(low_patch)))

    return np.array(
        [
            meters(mid[:, c0:c1], low[:, c0:c1]),
            meters(mid[:, :c0], low[:, :c0]),
            meters(mid[:, c1:], low[:, c1:]),
        ],
        dtype=np.float32,
    )


def contact_from_imu(
    *,
    agx: Optional[float],
    agy: Optional[float],
    agz: Optional[float],
    pitch_deg: Optional[float],
    vgx_dms: Optional[float],
    rc_b: int,
    height_m: Optional[float],
) -> str:
    """'' / 'near_miss' / 'collision' from Tello IMU while airborne.

    Ground / land spikes are ignored via height. ``agz`` unused except as a
    future hook — vertical 1 g lives there in hover.
    """
    del agz
    if height_m is None or height_m < MIN_AIR_M:
        return ""
    mag = math.hypot(float(agx or 0.0), float(agy or 0.0))
    b = int(rc_b)
    vgx = 0.0 if vgx_dms is None else float(vgx_dms)
    pit = 0.0 if pitch_deg is None else float(pitch_deg)
    if mag >= IMU_HIT_XY:
        return "collision"
    # Tello vg* is dm/s integer; slow creep often reports vgx=-1/-2 without a hit.
    if b >= 8 and vgx <= -4.0 and mag >= IMU_NEAR_XY:
        return "collision"
    if b >= 8 and pit <= -8.0 and mag >= IMU_HIT_XY:
        return "collision"
    if mag >= IMU_NEAR_XY:
        return "near_miss"
    return ""


def clearance_from_depth56(depth56: np.ndarray) -> np.ndarray:
    c = cones3_from_depth56(np.asarray(depth56, np.float32))
    return np.clip(np.nan_to_num(c[:3], nan=8.0, posinf=8.0), 0.35, 8.0).astype(np.float32)


def remaining_bearing_rad(*, yaw_rad: float, takeoff_yaw_rad: float, goal_bearing0_rad: float) -> float:
    """Goal azimuth in body frame using yaw only (no vg integration)."""
    world = wrap(takeoff_yaw_rad + goal_bearing0_rad)
    return wrap(world - yaw_rad)


def altitude_rc(height_m: Optional[float], c_cmd: int = 0) -> int:
    if c_cmd != 0:
        return int(np.clip(c_cmd, -100, 100))
    if height_m is None:
        return 0
    if height_m < HOVER_Z_LO_M:
        return 18
    if height_m > HOVER_Z_HI_M:
        return -12
    if height_m < TRIM_Z_M - 0.12:
        return 10
    if height_m > TRIM_Z_M + 0.20:
        return -8
    return 0


def forbid_combined_and_strafe(a: int, b: int, c: int, d: int) -> Tuple[int, int, int, int]:
    """Doc action box: a=0, b≥0, never yaw while creeping forward."""
    a = 0
    b = max(0, int(b))
    if abs(int(d)) >= 8:
        b = 0
    elif b > 0:
        d = 0
    return a, b, int(c), int(d)


@dataclass
class PolicyDecision:
    rc: Tuple[int, int, int, int]
    family: str
    fwd_m: float
    left_m: float
    right_m: float
    bearing_rad: float
    near_miss: bool
    collision: bool
    depth_source: str
    note: str
    keepout: str = ""


class TelloConsequencePolicy:
    """Reactive indoor policy; optional consequence ensemble as a survival shield."""

    def __init__(
        self,
        *,
        models: Optional[Sequence[Any]] = None,
        device: str = "cpu",
        surv_min: float = 0.7,
        near_m: float = NEAR_M,
        slow_m: float = SLOW_M,
        dt: float = 0.2,
        avoid: Optional[SemanticKeepout] = None,
        side_near_m: Optional[float] = None,
        shield_mode: str = "argmax",
    ) -> None:
        if shield_mode not in ("argmax", "veto"):
            raise ValueError(f"shield_mode {shield_mode!r}")
        # argmax: take the safest family. veto: keep the reactive family unless its survival < surv_min.
        self.shield_mode = shield_mode
        self.models = list(models or [])
        self.avoid = avoid
        self._keepouts: List[KeepOut] = []
        self.device = device
        self.surv_min = float(surv_min)
        self.near_m = float(near_m)
        # Left/right cones may pass closer than straight ahead (gaps); None keeps one threshold.
        self.side_near_m = float(near_m if side_near_m is None else side_near_m)
        self.slow_m = float(slow_m)
        self.dt = float(dt)
        self._was_near = False
        self._tight_yaw = 0
        self._commit_fwd = 0
        self._hit = False
        self._prev_action = np.zeros(4, np.float32)

    def clearances(self, rgb: np.ndarray, depth56: Optional[np.ndarray]) -> Tuple[np.ndarray, str]:
        if depth56 is not None:
            return clearance_from_depth56(depth56), "da3"
        return clearance_from_rgb(rgb), "rgb_proxy"

    def decide(
        self,
        *,
        rgb: np.ndarray,
        height_m: Optional[float],
        yaw_rad: float,
        takeoff_yaw_rad: float,
        goal_bearing0_rad: float,
        depth56: Optional[np.ndarray] = None,
        agx: Optional[float] = None,
        agy: Optional[float] = None,
        agz: Optional[float] = None,
        pitch_deg: Optional[float] = None,
        vgx_dms: Optional[float] = None,
        rc_b: int = 0,
    ) -> PolicyDecision:
        self._keepouts = []
        raw_d56 = depth56
        if self.avoid is not None:
            self._keepouts = self.avoid.observe(rgb, depth56, height_m, yaw_rad)
            if depth56 is not None and self._keepouts:
                depth56 = apply_keepouts(depth56, self._keepouts)
        cones, src = self.clearances(rgb, depth56)
        raw_cones = self.clearances(rgb, raw_d56)[0] if raw_d56 is not None and self._keepouts else cones
        if depth56 is None and self._keepouts:
            cones = np.minimum(cones, clearance_from_depth56(apply_keepouts(np.full((56, 56), 8.0), self._keepouts)))
        fwd_m, left_m, right_m = (float(x) for x in cones)
        bearing = remaining_bearing_rad(
            yaw_rad=yaw_rad,
            takeoff_yaw_rad=takeoff_yaw_rad,
            goal_bearing0_rad=goal_bearing0_rad,
        )
        detour = False
        if self.avoid is not None and self.avoid.tracker.active:
            bearing, detour = self.avoid.tracker.steer(yaw_rad, bearing)
        imu = contact_from_imu(
            agx=agx,
            agy=agy,
            agz=agz,
            pitch_deg=pitch_deg,
            vgx_dms=vgx_dms,
            rc_b=rc_b,
            height_m=height_m,
        )
        vis_near = fwd_m <= self.near_m or min(left_m, right_m) <= self.side_near_m
        near = vis_near or imu in ("near_miss", "collision")
        hit = imu == "collision" or self._hit
        if hit:
            self._hit = True
        rising = bool(near and not self._was_near)
        self._was_near = near

        if hit:
            family, b, d, note = "hover", 0, 0, f"collision imu={imu or 'latched'}"
        else:
            side = self.avoid.tracker.side if self.avoid is not None else 0
            if near and side and self._tight_yaw == 0:
                self._tight_yaw = side
            family, b, d, note = self._reactive(
                fwd_m, left_m, right_m, bearing, force_near=near,
                align_rad=DETOUR_ALIGN_RAD if detour else YAW_ALIGN_RAD,
            )
            if detour:
                note = f"{note}; detour {'left' if side < 0 else 'right'} of keep-out"
            if self.models:
                family, b, d, note = self._shielded(
                    rgb, raw_d56, raw_cones, cones, bearing, height_m, family, b, d, note
                )
        c = altitude_rc(height_m)
        a, b, c, d = forbid_combined_and_strafe(0, b, c, d)
        self._prev_action = np.array(
            [(b / 100.0) * MAX_V_MPS * self.dt, 0.0, (c / 100.0) * MAX_V_MPS * self.dt,
             -(d / 100.0) * MAX_YAW_RAD_S * self.dt],
            dtype=np.float32,
        )
        return PolicyDecision(
            rc=(a, b, c, d),
            family=family,
            fwd_m=fwd_m,
            left_m=left_m,
            right_m=right_m,
            bearing_rad=bearing,
            near_miss=rising and not hit,
            collision=hit,
            depth_source=src,
            note=note,
            keepout="; ".join(
                f"{k.label} {math.degrees(k.lo_rad):+.0f}..{math.degrees(k.hi_rad):+.0f}deg "
                f"r={k.range_m:.2f}m({k.range_src}) pillar={k.pillar_m:.2f}m"
                for k in self._keepouts
            ),
        )

    def _reactive(
        self,
        fwd_m: float,
        left_m: float,
        right_m: float,
        bearing: float,
        force_near: bool = False,
        align_rad: float = YAW_ALIGN_RAD,
    ) -> Tuple[str, int, int, str]:
        if force_near or fwd_m <= self.near_m:
            # Commit to one avoidance direction until clear, then fly a short leg before re-aligning,
            # so avoidance and goal alignment do not undo each other.
            self._commit_fwd = COMMIT_FWD_STEPS
            if self._tight_yaw == 0:
                if abs(left_m - right_m) > 0.25:
                    self._tight_yaw = -1 if left_m > right_m else 1
                else:
                    self._tight_yaw = -1 if left_m >= right_m else 1
            if self._tight_yaw < 0:
                return "yaw_left", 0, -YAW_STICK, "near: committed yaw left in place"
            return "yaw_right", 0, YAW_STICK, "near: committed yaw right in place"
        self._tight_yaw = 0
        if self._commit_fwd > 0:
            self._commit_fwd -= 1
            return "fwd", max(8, FWD_STICK // 2), 0, "post-avoid leg before re-align"
        if abs(bearing) > align_rad:
            # Tello rc d > 0 raises yaw, and bearing = goal - yaw.
            stick = YAW_STICK if bearing > 0 else -YAW_STICK
            fam = "yaw_right" if stick > 0 else "yaw_left"
            return fam, 0, stick, "align goal yaw"
        b = FWD_STICK if fwd_m >= self.slow_m else max(8, FWD_STICK // 2)
        return "fwd", b, 0, "creep toward goal"

    def _shielded(
        self,
        rgb: np.ndarray,
        depth56: Optional[np.ndarray],
        raw_cones: np.ndarray,
        cones: np.ndarray,
        bearing: float,
        height_m: Optional[float],
        family: str,
        b: int,
        d: int,
        note: str,
    ) -> Tuple[str, int, int, str]:
        try:
            surv = self._survival_four(depth56, raw_cones, bearing, height_m)
        except Exception as exc:  # noqa: BLE001
            return family, b, d, f"{note}; shield-off ({exc})"
        if surv is None:
            return family, b, d, note
        names = list(TELLO_FAMILIES)
        if self.shield_mode == "veto" and family in names:
            own = float(surv[names.index(family)])
            if own >= self.surv_min:
                return family, b, d, f"{note}; shield ok surv={own:.2f}"
            note = f"veto {family} surv={own:.2f}"
        best = int(np.argmax(surv))
        if float(surv[best]) < self.surv_min:
            return "hover", 0, 0, f"shield hover surv={surv[best]:.2f}"
        pick = names[best]
        if pick == "hover":
            return "hover", 0, 0, f"shield hover surv={surv[best]:.2f}"
        if pick == "yaw_left":
            return "yaw_left", 0, -YAW_STICK, f"shield yaw L surv={surv[best]:.2f}"
        if pick == "yaw_right":
            return "yaw_right", 0, YAW_STICK, f"shield yaw R surv={surv[best]:.2f}"
        return "fwd", FWD_STICK if cones[0] >= self.slow_m else max(8, FWD_STICK // 2), 0, (
            f"shield fwd surv={surv[best]:.2f}"
        )

    def _survival_four(
        self,
        depth56: Optional[np.ndarray],
        cones: np.ndarray,
        bearing: float,
        height_m: Optional[float],
    ) -> Optional[np.ndarray]:
        if not self.models:
            return None
        import torch

        from experiments.aerial.rl.collect_consequence_dataset import pool_min
        from experiments.aerial.rl.consequence_model import (
            depth_input,
            goal_features,
            survival,
        )

        if depth56 is None:
            # Shield needs a 56 map; synthesize a flat plane from RGB cones.
            d56 = np.full((56, 56), float(cones[0]), np.float32)
            d56[:, :14] = float(cones[1])
            d56[:, 42:] = float(cones[2])
        else:
            d56 = np.asarray(depth56, np.float32)
            if d56.shape != (56, 56):
                d56 = np.asarray(pool_min(d56), np.float32)
        # A semantic model reads the object mask itself; a depth-only model gets the pillar.
        d56_pillar = apply_keepouts(d56, self._keepouts) if self._keepouts else d56
        z = float(height_m if height_m is not None else TRIM_Z_M)
        state = np.array([0.0, 0.0, z, 0.0, 0.0, 0.0, 0.0], np.float64)
        dist = 4.0
        goal = np.array([dist * math.cos(bearing), dist * math.sin(bearing), z], np.float64)
        g = goal_features(state, goal)
        dt = self.dt
        cands = []
        for name in TELLO_FAMILIES:
            seq = np.zeros((int(self.models[0].horizon), 4), np.float32)
            if name == "fwd":
                seq[:, 0] = MAX_V_MPS * dt
            elif name == "yaw_left":
                seq[:, 3] = -MAX_YAW_RAD_S * dt
            elif name == "yaw_right":
                seq[:, 3] = MAX_YAW_RAD_S * dt
            cands.append(seq)
        cands_a = np.stack(cands)
        M = len(cands_a)
        dev = torch.device(self.device)
        d_plain = torch.as_tensor(np.repeat(depth_input(d56)[None], M, 0), device=dev)
        d_pillar = torch.as_tensor(np.repeat(depth_input(d56_pillar)[None], M, 0), device=dev)
        gt = torch.as_tensor(np.repeat(g[None], M, 0), device=dev)
        pa = torch.as_tensor(np.repeat(tello_to_model_delta(self._prev_action)[None], M, 0), device=dev)
        a = torch.as_tensor(cands_a, device=dev)
        sem = kr = None
        if self.avoid is not None:
            sem = torch.as_tensor(np.repeat(self.avoid.mask56()[None], M, 0), device=dev)
            kr = torch.full((M,), float(self.avoid.radius_m), device=dev)
        s_all: List[np.ndarray] = []
        with torch.no_grad():
            for m in self.models:
                if tuple(getattr(m, "frame_offsets", (0,))) not in ((0,), (0, 0)):
                    continue
                ha = None
                if int(getattr(m, "hist_actions", 0) or 0) > 0:
                    ha = torch.zeros((M, int(m.hist_actions), 4), device=dev)
                if int(getattr(m, "sem_channels", 0) or 0) > 0:
                    o = m(d_plain, gt, pa, a, ha, sem, kr)
                    s = survival(o["hazard_logit"], o["keepout_logit"])
                else:
                    o = m(d_pillar, gt, pa, a, ha)
                    s = survival(o["hazard_logit"])
                s_all.append(s[..., -1].cpu().numpy())
        if not s_all:
            return None
        return np.mean(s_all, axis=0)


def tello_to_model_delta(delta: np.ndarray) -> np.ndarray:
    """Tello body delta (y left, yaw CCW+) -> the consequence model's AirSim NED convention (y right, yaw CW+)."""
    return (np.asarray(delta, np.float32) * np.array([1.0, -1.0, 1.0, -1.0], np.float32)).astype(np.float32)


def first_action_to_rc(delta: np.ndarray, dt: float) -> Tuple[int, int, int, int]:
    raw = body_delta_to_rc(delta, dt, max_v_mps=MAX_V_MPS, max_yaw_rad_s=MAX_YAW_RAD_S)
    return forbid_combined_and_strafe(*raw)


def decision_log(dec: PolicyDecision) -> Dict[str, Any]:
    return {
        "family": dec.family,
        "rc": list(dec.rc),
        "fwd_m": round(dec.fwd_m, 3),
        "left_m": round(dec.left_m, 3),
        "right_m": round(dec.right_m, 3),
        "bearing_deg": round(math.degrees(dec.bearing_rad), 1),
        "near_miss": dec.near_miss,
        "collision": dec.collision,
        "depth_source": dec.depth_source,
        "note": dec.note,
        "keepout": dec.keepout,
    }
