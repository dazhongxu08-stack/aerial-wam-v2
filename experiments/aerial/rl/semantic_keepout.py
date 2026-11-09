"""Scene objects the consequence model must fly around, not just not hit.

The consequence model only sees a 56x56 depth map, so it treats a low stool
like any floor clutter: at 1 m cruise it sits below the middle row band and
reads as clear. Here a detector finds a named object class in the RGB frame,
the object becomes a full-height virtual pillar inflated by a keep-out radius,
and that pillar is written into the depth map. The unchanged consequence model
and the reactive policy then score / steer around it like any near obstacle.

``KeepoutTracker`` remembers the pillar in the world yaw frame so goal
alignment aims at the pillar's tangent edge (one committed side) instead of
swinging straight back into the object once it leaves the frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Tello: 82.6 deg diagonal FOV on the 4:3 stream.
TELLO_HFOV_RAD = 2.0 * math.atan(math.tan(math.radians(82.6) / 2.0) * 0.8)
TELLO_VFOV_RAD = 2.0 * math.atan(math.tan(math.radians(82.6) / 2.0) * 0.6)
# ``cones3_from_depth56`` forward cone = middle half of the columns.
FWD_CONE_HALF_RAD = math.atan(0.5 * math.tan(TELLO_HFOV_RAD / 2.0))
DET_WIDTH = 160
PILLAR_MIN_M = 0.30
RANGE_MAX_M = 8.0


@dataclass(frozen=True)
class HsvBand:
    """OpenCV HSV (H 0-180). ``h_lo > h_hi`` wraps through red."""

    h_lo: int
    h_hi: int
    s_lo: int
    s_hi: int
    v_lo: int
    v_hi: int = 255

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        hue = (h >= self.h_lo) | (h <= self.h_hi) if self.h_lo > self.h_hi else (h >= self.h_lo) & (h <= self.h_hi)
        return hue & (s >= self.s_lo) & (s <= self.s_hi) & (v >= self.v_lo) & (v <= self.v_hi)


@dataclass(frozen=True)
class ColorObjectSpec:
    """An object found by a distinctive ``core`` colour plus ``attach`` parts below it.

    Only ``core`` blobs seed a detection, so things that share just the
    ``attach`` colour (a deep-red cloth next to a red stool's legs) are ignored.
    """

    label: str
    core: Tuple[HsvBand, ...]
    attach: Tuple[HsvBand, ...] = ()
    min_area_frac: float = 0.002
    core_s_median: Tuple[int, int] = (0, 255)
    core_v_median_min: int = 0
    height_m: float = 0.45


# Plastic stool in the test room: pale pink seat (blob median S 50-68, V 206-234),
# saturated red legs (S~200 V~130). Rejected on 10-14/10-15 Tello frames: the
# red velvet cloth (S~250 V~100, no pink so it never seeds), the desk coaster
# (S 72-98), pinkish floor / bare soil (S 27-45 or V <= 200).
RED_STOOL = ColorObjectSpec(
    label="red_stool",
    core=(HsvBand(158, 3, 40, 110, 190),),
    attach=(HsvBand(165, 10, 110, 255, 60),),
    min_area_frac=0.004,
    core_s_median=(45, 80),
    core_v_median_min=205,
)

SPECS: Dict[str, ColorObjectSpec] = {RED_STOOL.label: RED_STOOL}


@dataclass
class Detection:
    label: str
    mask: np.ndarray  # bool, detector resolution
    bbox: Tuple[float, float, float, float]  # u0, v0, u1, v1 in [0, 1]
    area_frac: float


@dataclass
class KeepOut:
    """Body-frame sector to stay out of; bearings are + right (Tello yaw sense)."""

    label: str
    lo_rad: float
    hi_rad: float
    range_m: float
    pillar_m: float
    range_src: str


def _any_band(hsv: np.ndarray, bands: Sequence[HsvBand]) -> np.ndarray:
    m = np.zeros(hsv.shape[:2], bool)
    for b in bands:
        m |= b.mask(hsv)
    return m


def detect_color_objects(rgb: np.ndarray, spec: ColorObjectSpec) -> List[Detection]:
    img = np.asarray(rgb, np.uint8)
    h0, w0 = img.shape[:2]
    w = DET_WIDTH
    h = max(1, int(round(h0 * w / w0)))
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.int16)
    k = np.ones((3, 3), np.uint8)
    core = cv2.morphologyEx(_any_band(hsv, spec.core).astype(np.uint8), cv2.MORPH_OPEN, k)
    attach = _any_band(hsv, spec.attach) if spec.attach else np.zeros_like(core, bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    min_px = spec.min_area_frac * w * h
    out: List[Detection] = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < min_px:
            continue
        m = lab == i
        s_med = float(np.median(hsv[..., 1][m]))
        if not spec.core_s_median[0] <= s_med <= spec.core_s_median[1]:
            continue
        if float(np.median(hsv[..., 2][m])) < spec.core_v_median_min:
            continue
        if attach.any():
            pad = max(1, bw // 10)
            band = np.zeros_like(m)
            band[y:, max(0, x - pad):min(w, x + bw + pad)] = True
            grown = cv2.dilate((m | (attach & band)).astype(np.uint8), k)
            nn, glab = cv2.connectedComponents(grown, connectivity=8)
            keep = np.unique(glab[m])
            m = np.isin(glab, keep[keep > 0]) & (m | (attach & band))
        ys, xs = np.nonzero(m)
        out.append(Detection(
            label=spec.label,
            mask=m,
            bbox=(xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h),
            area_frac=float(m.mean()),
        ))
    return out


def col_bearing(u: np.ndarray | float, hfov: float = TELLO_HFOV_RAD) -> np.ndarray | float:
    """Image column in [0, 1] -> body bearing (+ right)."""
    return np.arctan((np.asarray(u) - 0.5) * 2.0 * math.tan(hfov / 2.0))


def bearing_col(b: np.ndarray | float, hfov: float = TELLO_HFOV_RAD) -> np.ndarray | float:
    return 0.5 + np.tan(np.asarray(b)) / (2.0 * math.tan(hfov / 2.0))


def _row_angle_down(v: float, vfov: float = TELLO_VFOV_RAD) -> float:
    return math.atan((v - 0.5) * 2.0 * math.tan(vfov / 2.0))


def estimate_range(
    det: Detection,
    depth56: Optional[np.ndarray],
    height_m: Optional[float],
    obj_height_m: float,
) -> Tuple[float, str]:
    """Metres to the object: depth-head median inside the mask, else flat-floor geometry."""
    if depth56 is not None:
        d = np.asarray(depth56, np.float32)
        m56 = cv2.resize(det.mask.astype(np.float32), (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA) > 0.3
        v = d[m56 & np.isfinite(d) & (d > 0.05)]
        if v.size:
            return float(np.clip(np.median(v), PILLAR_MIN_M, RANGE_MAX_M)), "depth"
    z = float(height_m) if height_m is not None and height_m > 0.2 else 1.1
    u0, v0, u1, v1 = det.bbox
    if v1 < 0.97:
        a = _row_angle_down(v1)
        if a > 0.02:
            return float(np.clip(z / math.tan(a), PILLAR_MIN_M, RANGE_MAX_M)), "floor_contact"
    a = _row_angle_down(v0)
    if a > 0.02 and z > obj_height_m:
        return float(np.clip((z - obj_height_m) / math.tan(a), PILLAR_MIN_M, RANGE_MAX_M)), "top_edge"
    return PILLAR_MIN_M, "top_edge"


def keepout_from_detection(det: Detection, range_m: float, radius_m: float, src: str = "") -> KeepOut:
    """Object columns widened by ``radius_m`` on each side at ``range_m``; pillar sits radius closer."""
    u0, _, u1, _ = det.bbox
    r = max(float(range_m), 0.05)
    x0 = r * math.tan(float(col_bearing(u0))) - radius_m
    x1 = r * math.tan(float(col_bearing(u1))) + radius_m
    return KeepOut(
        label=det.label,
        lo_rad=math.atan2(x0, r),
        hi_rad=math.atan2(x1, r),
        range_m=r,
        pillar_m=max(PILLAR_MIN_M, r - radius_m),
        range_src=src,
    )


def apply_keepouts(depth56: np.ndarray, keepouts: Sequence[KeepOut]) -> np.ndarray:
    """Write each keep-out as a full-height pillar (min with what is there)."""
    d = np.array(depth56, np.float32, copy=True)
    w = d.shape[1]
    centers = col_bearing((np.arange(w) + 0.5) / w)
    for ko in keepouts:
        cols = (centers >= ko.lo_rad) & (centers <= ko.hi_rad)
        if cols.any():
            d[:, cols] = np.minimum(np.nan_to_num(d[:, cols], nan=RANGE_MAX_M), ko.pillar_m)
    return d


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class _Sector:
    lo: float  # world yaw
    hi: float
    age: int = 0


@dataclass
class KeepoutTracker:
    """Keep-out sectors in the world yaw frame, plus one committed pass side.

    ``side`` -1 passes the object on its left (yaw left), +1 on its right.
    The aim point sits ``margin_rad`` past the sector edge so the forward cone
    no longer overlaps the pillar.
    """

    ttl_steps: int = 24
    margin_rad: float = FWD_CONE_HALF_RAD + math.radians(2.0)
    sectors: List[_Sector] = field(default_factory=list)
    side: int = 0

    def update(self, yaw_rad: float, keepouts: Sequence[KeepOut]) -> None:
        if keepouts:
            self.sectors = [_Sector(wrap(yaw_rad + k.lo_rad), wrap(yaw_rad + k.hi_rad)) for k in keepouts]
            return
        for s in self.sectors:
            s.age += 1
        self.sectors = [s for s in self.sectors if s.age <= self.ttl_steps]
        if not self.sectors:
            self.side = 0

    @property
    def active(self) -> bool:
        return bool(self.sectors)

    def steer(self, yaw_rad: float, goal_bearing_rad: float) -> Tuple[float, bool]:
        """Body bearing to aim at: the goal, or the committed tangent edge when the goal is blocked."""
        g = wrap(yaw_rad + goal_bearing_rad)
        for s in self.sectors:
            lo, hi = s.lo - self.margin_rad, s.hi + self.margin_rad
            if wrap(g - lo) >= 0.0 and wrap(hi - g) >= 0.0:
                if self.side == 0:
                    self.side = -1 if abs(wrap(g - lo)) <= abs(wrap(hi - g)) else 1
                edge = lo if self.side < 0 else hi
                return wrap(edge - yaw_rad), True
        return goal_bearing_rad, False


class SemanticKeepout:
    """Detector + range + pillar injection + tracker for one or more object classes."""

    def __init__(self, specs: Sequence[ColorObjectSpec], radius_m: float = 0.5, ttl_steps: int = 24,
                 confirm_frames: int = 2) -> None:
        self.specs = list(specs)
        self.radius_m = float(radius_m)
        self.tracker = KeepoutTracker(ttl_steps=int(ttl_steps))
        self.confirm_frames = max(1, int(confirm_frames))
        self._seen_run = 0
        self.last: List[KeepOut] = []
        self.last_dets: List[Detection] = []

    @classmethod
    def from_names(cls, names: str, radius_m: float = 0.5) -> "SemanticKeepout":
        specs = []
        for n in names.split(","):
            n = n.strip()
            if not n:
                continue
            if n not in SPECS:
                raise ValueError(f"unknown avoid class {n!r}; known: {sorted(SPECS)}")
            specs.append(SPECS[n])
        return cls(specs, radius_m=radius_m)

    def observe(self, rgb: np.ndarray, depth56: Optional[np.ndarray], height_m: Optional[float],
                yaw_rad: float) -> List[KeepOut]:
        kos: List[KeepOut] = []
        dets: List[Detection] = []
        for spec in self.specs:
            for det in detect_color_objects(rgb, spec):
                r, src = estimate_range(det, depth56, height_m, spec.height_m)
                kos.append(keepout_from_detection(det, r, self.radius_m, src))
                dets.append(det)
        self._seen_run = self._seen_run + 1 if kos else 0
        if self._seen_run < self.confirm_frames and not self.tracker.active:
            kos, dets = [], []
        self.tracker.update(yaw_rad, kos)
        self.last = kos
        self.last_dets = dets
        return kos

    def mask56(self, shape: Tuple[int, int] = (56, 56)) -> np.ndarray:
        """Confirmed detections of the last frame as the consequence model's ``sem`` input."""
        m = np.zeros(shape, np.float32)
        for det in self.last_dets:
            r = cv2.resize(det.mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            m = np.maximum(m, (r > 0.3).astype(np.float32))
        return m
