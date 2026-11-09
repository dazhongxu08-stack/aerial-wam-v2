"""Low-band clearance used by the office closed loop. Same test as the car."""
from __future__ import annotations

import math

import numpy as np

INVALID_M = 30.0
LOWER_BANDS = (0.26, 0.30, 0.34, 0.38, 0.42, 0.50, 0.58, 0.66)
_FLOOR_M = (5.2, 4.3, 3.2, 2.4)


def _warp_row(fy: float, cy: float, src_h: int, out_h: int, frac: float) -> int:
    focal = 0.5 * out_h / math.tan(math.radians(45.0))
    angle = math.atan((frac * src_h - cy) / fy)
    row = out_h / 2.0 + focal * math.tan(angle)
    return int(np.clip(math.floor(row), 0, out_h - 1))


def _model_thirds(depth: np.ndarray, top: int, bot: int):
    height, width = depth.shape
    r0 = max(0, min(height - 1, int(top)))
    r1 = max(r0 + 1, min(height, int(bot)))
    c0, c1 = width // 3, 2 * width // 3
    n0, n1 = int(width * 0.42), int(width * 0.58)
    band = depth[r0:r1]

    def near(x0: int, x1: int):
        part = band[:, x0:x1]
        fin = part[np.isfinite(part) & (part > 0.15) & (part < INVALID_M)]
        if fin.size < 20:
            return None
        return float(np.percentile(fin, 15))

    return near(0, c0), near(n0, n1), near(c1, width)


def low_clearance(model: np.ndarray, fy: float, cy: float, src_h: int):
    height = model.shape[0]
    rows = [_warp_row(fy, cy, src_h, height, frac) for frac in LOWER_BANDS]
    found = [None, None, None]
    band_i = 0
    for top, r0, r1 in zip(LOWER_BANDS, rows, rows[1:]):
        if top >= 0.42:
            break
        expect = _FLOOR_M[band_i]
        band_i += 1
        for index, value in enumerate(_model_thirds(model, r0, r1)):
            if value is None:
                continue
            if value < expect - 0.8 and value < expect * 0.75:
                found[index] = value if found[index] is None else min(found[index], value)
    return tuple(found)
