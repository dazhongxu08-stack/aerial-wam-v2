"""Single-scene QR detect + arrival (no payment filter)."""
from __future__ import annotations

import numpy as np

from experiments.aerial.rl.tello_qr import (
    arrived_from_hit,
    detect_qr,
    render_qr_bgr,
)


def test_detect_any_payload_not_just_payment() -> None:
    bgr = render_qr_bgr("aerial-wam-exit-qr")
    hits = detect_qr(bgr)
    assert hits
    assert hits[0].data == "aerial-wam-exit-qr"
    assert hits[0].area_frac > 0.05


def test_arrived_requires_large_quad() -> None:
    bgr = render_qr_bgr("exit")
    hit = detect_qr(bgr)[0]
    assert arrived_from_hit(hit, collided=False, min_area_frac=0.02)
    assert not arrived_from_hit(hit, collided=True, min_area_frac=0.02)
    assert not arrived_from_hit(hit, collided=False, min_area_frac=0.99)


def test_blank_frame_has_no_qr() -> None:
    blank = np.full((120, 160, 3), 200, dtype=np.uint8)
    assert detect_qr(blank) == []
