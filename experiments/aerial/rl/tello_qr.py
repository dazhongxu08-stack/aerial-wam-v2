"""Detect the single scene QR in Tello JPEGs.

The flight target is one printed QR (not a payment-code filter). The closest
view is the detection with the largest quad in the image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

# Close enough: QR occupies this fraction of the native frame.
DEFAULT_ARRIVE_AREA_FRAC = 0.02


@dataclass(frozen=True)
class QrHit:
    data: str
    area_frac: float
    quad: np.ndarray  # [4, 2]


def _quad_area(pts: np.ndarray) -> float:
    poly = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        return 0.0
    import cv2  # type: ignore

    return abs(float(cv2.contourArea(poly)))


def _hits_from_decode(decoded: Any, quads: Any, h: int, w: int) -> List[QrHit]:
    texts: List[str]
    if decoded is None:
        return []
    if isinstance(decoded, (bytes, str)):
        texts = [decoded.decode("utf-8", "replace") if isinstance(decoded, bytes) else decoded]
        quad_list = [quads] if quads is not None else []
    else:
        texts = [
            t.decode("utf-8", "replace") if isinstance(t, bytes) else str(t) for t in decoded
        ]
        quad_list = list(quads) if quads is not None else []
    area = max(1, int(h) * int(w))
    hits: List[QrHit] = []
    for i, text in enumerate(texts):
        payload = str(text).strip()
        if not payload:
            continue
        quad = np.asarray(quad_list[i] if i < len(quad_list) else np.zeros((4, 2)), dtype=np.float32)
        if quad.ndim == 3:
            quad = quad.reshape(-1, 2)
        hits.append(
            QrHit(
                data=payload,
                area_frac=float(_quad_area(quad) / area),
                quad=quad.reshape(-1, 2),
            )
        )
    return hits


def detect_qr(image: np.ndarray) -> List[QrHit]:
    """Return every decoded QR in a BGR/gray/RGB uint8 image."""
    import cv2  # type: ignore

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        # RGB vs BGR does not matter for QR geometry; gray is the decode input.
    elif arr.ndim == 2:
        gray = arr
    else:
        raise ValueError(f"expected HxW or HxWx3 image, got {arr.shape}")
    h, w = gray.shape[:2]
    hits: List[QrHit] = []
    det = cv2.QRCodeDetector()
    ok, decoded, quads, _ = det.detectAndDecodeMulti(gray)
    if ok:
        hits.extend(_hits_from_decode(decoded, quads, h, w))
    if not hits:
        data, pts, _ = det.detectAndDecode(gray)
        hits.extend(_hits_from_decode(data, pts, h, w))
    if not hits:
        aruco = cv2.QRCodeDetectorAruco()
        data, pts, _ = aruco.detectAndDecode(gray)
        hits.extend(_hits_from_decode(data, pts, h, w))
    hits.sort(key=lambda h: h.area_frac, reverse=True)
    return hits


def detect_qr_jpeg(path: Path) -> List[QrHit]:
    import cv2  # type: ignore

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    return detect_qr(bgr)


def best_hit(hits: Sequence[Optional[Sequence[QrHit]]]) -> Optional[tuple[int, QrHit]]:
    best_i = -1
    best: Optional[QrHit] = None
    for i, frame_hits in enumerate(hits):
        if not frame_hits:
            continue
        hit = max(frame_hits, key=lambda h: h.area_frac)
        if best is None or hit.area_frac > best.area_frac:
            best = hit
            best_i = i
    if best is None:
        return None
    return best_i, best


def arrived_from_hit(hit: Optional[QrHit], *, collided: bool, min_area_frac: float = DEFAULT_ARRIVE_AREA_FRAC) -> bool:
    if collided or hit is None:
        return False
    return float(hit.area_frac) >= float(min_area_frac)


def render_qr_bgr(payload: str, *, scale: int = 8, pad: int = 40) -> np.ndarray:
    """Synthetic BGR QR for tests (any payload)."""
    import cv2  # type: ignore

    img = cv2.QRCodeEncoder.create().encode(str(payload))
    if img is None or img.size == 0:
        raise RuntimeError("QR encode failed")
    big = cv2.resize(
        img,
        (int(img.shape[1] * scale), int(img.shape[0] * scale)),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.full((big.shape[0] + 2 * pad, big.shape[1] + 2 * pad), 255, dtype=np.uint8)
    canvas[pad : pad + big.shape[0], pad : pad + big.shape[1]] = big
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
