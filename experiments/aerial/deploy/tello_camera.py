"""Tello H.264 UDP stream → BGR/RGB frames for vgoal deploy."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TelloStreamCameraConfig:
    port: int = 11111
    width: int = 960
    height: int = 720
    wam_size: int = 224
    open_timeout_s: float = 8.0


class TelloStreamCamera:
    """OpenCV capture of ``udp://0.0.0.0:<port>`` after Tello ``streamon``."""

    def __init__(
        self,
        port: int = 11111,
        width: int = 960,
        height: int = 720,
        wam_size: int = 224,
        open_timeout_s: float = 8.0,
    ) -> None:
        self.config = TelloStreamCameraConfig(
            port=port,
            width=width,
            height=height,
            wam_size=wam_size,
            open_timeout_s=open_timeout_s,
        )
        self._cap = None
        self._last_bgr: Optional[np.ndarray] = None

    def open(self) -> None:
        import cv2  # type: ignore

        url = f"udp://0.0.0.0:{int(self.config.port)}?overrun_nonfatal=1&fifo_size=50000000"
        # Fallback URL without ffmpeg options for builds that reject query args.
        urls = [
            url,
            f"udp://@0.0.0.0:{int(self.config.port)}",
            f"udp://0.0.0.0:{int(self.config.port)}",
        ]
        deadline = time.time() + float(self.config.open_timeout_s)
        last_err = ""
        while time.time() < deadline and self._cap is None:
            for u in urls:
                cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap.release()
                    last_err = f"open failed: {u}"
                    continue
                ok, frame = cap.read()
                if ok and frame is not None:
                    self._cap = cap
                    self._last_bgr = frame
                    logger.info("Tello stream open via %s shape=%s", u, frame.shape)
                    return
                cap.release()
                last_err = f"no frame yet: {u}"
            time.sleep(0.3)
        raise RuntimeError(f"Tello stream not ready ({last_err})")

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        import cv2  # type: ignore

        if self._cap is None:
            raise RuntimeError("camera not open")
        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._last_bgr = frame
        if self._last_bgr is None:
            raise RuntimeError("no Tello video frame")
        bgr = self._last_bgr
        h, w = int(self.config.wam_size), int(self.config.wam_size)
        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb_full, (w, h), interpolation=cv2.INTER_AREA)
        return bgr, np.ascontiguousarray(rgb, dtype=np.uint8)

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
        self._cap = None
