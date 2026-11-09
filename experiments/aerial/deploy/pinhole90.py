"""Put a pinhole camera into the 90° square the depth model was trained on.

Pixels the lens does not cover stay empty. The image is not stretched.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

HFOV_DEG = 90.0
OUT = 224


def virtual_k(out: int = OUT) -> np.ndarray:
    f = 0.5 * out / math.tan(math.radians(HFOV_DEG) / 2.0)
    return np.array([[f, 0.0, out / 2.0], [0.0, f, out / 2.0], [0.0, 0.0, 1.0]], np.float64)


def _maps(k: np.ndarray, d: np.ndarray, out: int):
    import cv2

    return cv2.initUndistortRectifyMap(
        np.asarray(k, np.float64).reshape(3, 3),
        np.asarray(d, np.float64).ravel(),
        np.eye(3),
        virtual_k(out),
        (out, out),
        cv2.CV_16SC2,
    )


def warp_color(rgb: np.ndarray, k: np.ndarray, d: np.ndarray, out: int = OUT) -> Tuple[np.ndarray, np.ndarray]:
    """Return the 90° RGB image and a mask of pixels the camera actually saw."""
    import cv2

    m1, m2 = _maps(k, d, out)
    image = cv2.remap(rgb, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    seen = cv2.remap(np.ones(rgb.shape[:2], np.uint8), m1, m2, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, seen > 0


def warp_depth(depth_m: np.ndarray, k: np.ndarray, out: int = OUT) -> np.ndarray:
    """Nearest-neighbor remap of a metric depth map into the same 90° frame."""
    import cv2

    src = np.nan_to_num(np.asarray(depth_m, np.float32), nan=0.0)
    warped = cv2.remap(src, *_maps(k, np.zeros(5), out), cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped[warped <= 0.0] = np.nan
    return warped


def mask_unseen(depth: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Drop model depth on pixels the lens did not cover, so empty borders cannot look near."""
    import cv2

    out = np.array(depth, np.float32, copy=True)
    valid = cv2.resize(seen.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    out[~valid] = np.inf
    return out


def ground_guard_rows(fy: float, cy: float, height: int, out: int = OUT, n: int = 56,
                      top: float = 0.55, bot: float = 0.80) -> Tuple[int, int]:
    """Rows of the 56-map that see low obstacles and not the floor at the robot's feet.

    ``top`` and ``bot`` are fractions of the source image from the top. On this
    RealSense they sit below the horizon and above the nearest ground strip.
    """
    f = float(virtual_k(out)[0, 0])

    def y_out(frac: float) -> float:
        angle = math.atan((frac * height - cy) / fy)
        return out / 2.0 + f * math.tan(angle)

    r0 = int(np.clip(math.floor(y_out(top) / out * n), 0, n - 2))
    r1 = int(np.clip(math.ceil(y_out(bot) / out * n), r0 + 1, n))
    return r0, r1


def band_mins(depth56: np.ndarray, r0: int, r1: int) -> Tuple[float, float, float]:
    """Front, left, right minima on the same columns the navigator uses."""
    d = np.asarray(depth56, np.float32)[r0:r1]

    def mn(sl: np.ndarray) -> float:
        fin = sl[np.isfinite(sl) & (sl > 0.05) & (sl < 1e3)]
        return float(fin.min()) if fin.size else float("inf")

    return mn(d[:, 14:42]), mn(d[:, :14]), mn(d[:, 42:])


def brake(twist, real: Tuple[float, float, float], yaw_rate: float):
    """Stop using the depth camera when the model distance is still long.

    ``real`` is front, left, right in meters. A close front cancels forward
    speed. If both sides are also close, cancel the turn too.
    """
    front, left, right = real
    if front < 0.50:
        twist.linear.x = 0.0
        if min(left, right) < 0.45:
            twist.angular.z = 0.0
        else:
            twist.angular.z = yaw_rate if left >= right else -yaw_rate
    elif min(left, right) < 0.35 and abs(twist.angular.z) > 0.0:
        toward_near = (twist.angular.z > 0 and left <= right) or (twist.angular.z < 0 and right < left)
        if toward_near:
            twist.angular.z = 0.0
    return twist


def k_from_msg(msg) -> np.ndarray:
    k = msg.k
    return np.array([[k[0], k[1], k[2]], [k[3], k[4], k[5]], [k[6], k[7], k[8]]], np.float64)


def load_scale(path: str) -> Optional[float]:
    import json
    from pathlib import Path

    file = Path(path)
    if not file.is_file():
        return None
    raw = json.loads(file.read_text())
    scale = float(raw.get("scale", 0.0))
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return scale
