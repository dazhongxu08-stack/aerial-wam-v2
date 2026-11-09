"""Unit tests for TelloStreamCamera API (no hardware)."""
from __future__ import annotations

from experiments.aerial.deploy.tello_camera import TelloStreamCamera, TelloStreamCameraConfig


def test_config_defaults() -> None:
    cfg = TelloStreamCameraConfig()
    assert cfg.port == 11111
    assert cfg.wam_size == 224


def test_camera_construct_closed() -> None:
    cam = TelloStreamCamera(port=11111)
    assert cam.frame_seq == 0
    assert cam.frame_age_s == float("inf")
    assert cam.read_bgr_new() is None
    cam.close()
