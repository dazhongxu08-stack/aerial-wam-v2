"""Undistort the dog's SJCAM into the 90° pinhole the depth model was trained on."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .types import Intrinsics


class FisheyeRectifier:
    def __init__(self, path: str, out_size: int = 640, hfov_deg: float = 90.0) -> None:
        calib = json.loads(Path(path).read_text())
        self.src_size = (int(calib["width"]), int(calib["height"]))
        self.K = np.array(calib["K"], np.float64)
        self.D = np.array(calib["D"], np.float64).reshape(4, 1)
        self.out_size = int(out_size)
        self.hfov_deg = float(hfov_deg)
        f = 0.5 * self.out_size / np.tan(np.radians(self.hfov_deg) / 2.0)
        self.P = np.array([[f, 0.0, self.out_size / 2.0], [0.0, f, self.out_size / 2.0], [0.0, 0.0, 1.0]])
        self.intr = Intrinsics.from_hfov(self.out_size, self.out_size, self.hfov_deg)
        self._cached: tuple = ()

    def _maps(self, width: int, height: int):
        key = (width, height)
        if self._cached[:1] == (key,):
            return self._cached[1]
        K = self.K.copy()
        K[0, :] *= width / self.src_size[0]
        K[1, :] *= height / self.src_size[1]
        maps = cv2.fisheye.initUndistortRectifyMap(
            K, self.D, np.eye(3), self.P, (self.out_size, self.out_size), cv2.CV_16SC2)
        self._cached = (key, maps)
        return maps

    def apply(self, bgr: np.ndarray):
        """Return the 90° BGR image and a uint8 mask (255 where the sensor actually saw something)."""
        h, w = bgr.shape[:2]
        m1, m2 = self._maps(w, h)
        rect = cv2.remap(bgr, m1, m2, cv2.INTER_LINEAR)
        valid = cv2.remap(np.full((h, w), 255, np.uint8), m1, m2, cv2.INTER_NEAREST)
        return rect, valid
