"""Office camera bands: forward is above the horizon, not the floor.

The sim camera sits 60 cm up and 30° down. After the 90° warp,
``hide`` already sets the lower source rows far, but ``cones3`` still mins a
mid band that includes several 56-rows of ground below the horizon. This
module keeps that hide, and reads forward clearance only from rows above the
horizon. Side walls use the same upper band. Short crates are read from the
strip between the horizon and the hide row — not from the car ``low_band``
rows that start above the horizon.
"""
from __future__ import annotations

import math

import numpy as np

FAR = 30.0
CAM_Z_M = 0.60
PITCH_DEG = -30.0
# A crate sits well inside the ground ray. Floor in hz→hide is ~2.4–5 m,
# so GT < 1.5 without this test is mostly carpet, not a box.
GROUND_FRAC = 0.70
# Min over the upper band treats a few DA smear pixels as a wall (gap 4.3 m
# GT, 1.14 m DA min, 1.94 m DA p20). A wall that fills the cone still sits
# near p20.
FORWARD_PCT = 20
SHORT_PCT = 15


def horizon_row_224(out_n: int = 224, pitch_deg: float = PITCH_DEG) -> int:
    """Row of the 90° square where world-horizontal rays land."""
    focal = 0.5 * out_n / math.tan(math.radians(45.0))
    return int(np.clip(round(out_n / 2.0 - focal * math.tan(math.radians(abs(pitch_deg)))), 0, out_n - 1))


def horizon_row_56(out_n: int = 224, out: int = 56, pitch_deg: float = PITCH_DEG) -> int:
    return int(np.clip(horizon_row_224(out_n, pitch_deg) * out // out_n, 1, out - 1))


def floor_range_map(
    h: int,
    w: int,
    *,
    cam_z: float = CAM_Z_M,
    pitch_deg: float = PITCH_DEG,
    far: float = FAR,
) -> np.ndarray:
    """Range to a flat floor at each pixel, 90° warp, rows increase downward."""
    focal = 0.5 * h / math.tan(math.radians(45.0))
    hz = horizon_row_224(h, pitch_deg) if h >= 200 else horizon_row_56(out=h, pitch_deg=pitch_deg)
    below = np.maximum(np.arange(h) - hz, 0).astype(np.float32)
    ang = np.arctan(below / float(focal))
    rng = np.where(below > 0, float(cam_z) / np.clip(np.sin(ang), 1e-3, 1.0), far)
    rng = np.clip(rng, 0.05, far)
    return np.repeat(rng[:, None], w, axis=1)


def _pct(x: np.ndarray, pct: float, far: float) -> float:
    v = x[np.isfinite(x) & (x > 0.05) & (x < far - 0.5)]
    if v.size < 8:
        return far
    return float(np.percentile(v, pct))


def cones3_forward(
    depth56: np.ndarray,
    *,
    far: float = FAR,
    pitch_deg: float = PITCH_DEG,
    pct: float = FORWARD_PCT,
) -> np.ndarray:
    """Forward / left / right. Upper rows only. Percentile, not min."""
    d = np.asarray(depth56, np.float32)
    h, w = d.shape
    hi = min(horizon_row_56(out=h, pitch_deg=pitch_deg), h - 1)
    lo = max(h // 8, 0)
    if lo >= hi:
        lo = max(0, hi - 4)
    band = d[lo:hi]
    c0, c1 = w // 4, w - w // 4
    return np.array(
        [
            _pct(band[:, c0:c1], pct, far),
            _pct(band[:, : w // 3], pct, far),
            _pct(band[:, 2 * w // 3 :], pct, far),
        ],
        np.float32,
    )


def short_clearance(
    depth224: np.ndarray,
    *,
    horizon: int | None = None,
    hide: int | None = None,
    far: float = FAR,
    pct: float = SHORT_PCT,
) -> tuple:
    """Left / center / right short-object clearance, horizon→hide.

    Below ``hide`` is floor (~1.2 m) on this camera. Above the horizon is the
    forward cone. Crates in the failed side pose sit in this strip on GT
    (0.88 m) and are missing on DA (~3 m).
    """
    d = np.asarray(depth224, np.float32)
    h, w = d.shape
    if horizon is None:
        horizon = horizon_row_224(h) if h >= 200 else horizon_row_56(out=h)
    if hide is None:
        hide = min(h, horizon + 16)
    r0 = int(np.clip(horizon, 0, h - 2))
    r1 = int(np.clip(hide, r0 + 1, h))
    band = d[r0:r1]
    fl = floor_range_map(h, w)[r0:r1]
    def pct_obj(x, floor, span_max: float):
        v = x[np.isfinite(x) & (x > 0.05) & (x < far - 0.5) & (x < GROUND_FRAC * floor)]
        if v.size < 8:
            return far
        p50 = float(np.percentile(v, 50))
        p80 = float(np.percentile(v, 80))
        # Gap-entrance smear is a 1.7 m blob with span 0.46. The side crate
        # at the same range has span 0.33. Center allows the barrel face
        # (span 0.36–0.61).
        if p50 > 2.2 or (p80 - p50) > span_max:
            return far
        return p50

    vals = (
        pct_obj(band[:, : w // 3], fl[:, : w // 3], 0.42),
        pct_obj(band[:, w // 4 : 3 * w // 4], fl[:, w // 4 : 3 * w // 4], 0.65),
        pct_obj(band[:, 2 * w // 3 :], fl[:, 2 * w // 3 :], 0.42),
    )
    out = []
    for v in vals:
        out.append(None if v >= far - 1 else float(v))
    return tuple(out)
