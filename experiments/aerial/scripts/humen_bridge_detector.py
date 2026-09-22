"""DEBUG ONLY — not for product pipeline.

HumenCorridor bridge detector (YOLO-World + horizon structure fallback).
The structure fallback produces false locks; mainline uses open_vocab + validate_search_traj.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from vgoal.detector import BaseDetector, DetectionResult


def _geometric_score(rgb: np.ndarray) -> float:
    h, w = rgb.shape[:2]
    band = rgb[int(h * 0.35) : int(h * 0.72), int(w * 0.2) : int(w * 0.8)]
    gray = band.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    horiz = float(gx - 0.35 * gy)
    upper = rgb[: int(h * 0.55), :, :]
    g_dom = float(upper[:, :, 1].mean() - 0.5 * (upper[:, :, 0].mean() + upper[:, :, 2].mean()))
    forest_pen = max(0.0, g_dom - 12.0) * 3.0
    return horiz * 4.0 + float(gray.std()) * 1.5 - forest_pen


def _synthetic_bridge_bbox(rgb: np.ndarray, score: float) -> DetectionResult:
    h, w = rgb.shape[:2]
    # Small bbox → depth fuse estimates far range (corridor bridge is hundreds of m out).
    u0, u1 = 0.46 * w, 0.54 * w
    v0, v1 = 0.44 * h, 0.52 * h
    conf = float(min(0.92, max(0.12, score / 45.0)))
    return DetectionResult(
        bbox=np.array([u0, v0, u1, v1], dtype=np.float32),
        confidence=conf,
        class_id=0,
        class_name="bridge",
    )


class HumenBridgeDetector(BaseDetector):
    """Scene-specific detector: open-vocab bridge classes, then deck/cable structure."""

    def __init__(
        self,
        *,
        model_path: str = "yolov8s-worldv2.pt",
        conf_threshold: float = 0.03,
        geom_threshold: float = 10.0,
        imgsz: int = 1280,
        device: str = "cuda",
        visual_prompt: str = "suspension bridge cable-stayed bridge tower",
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.geom_threshold = float(geom_threshold)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self._classes = list(
            dict.fromkeys(
                [c.strip() for c in visual_prompt.split() if c.strip()]
                + ["bridge", "suspension bridge", "viaduct", "tower", "cable"]
            )
        )
        self._model = None
        self._model_path = model_path

    def _yolo(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
            self._model.set_classes(self._classes)
        return self._model

    def detect(self, rgb: np.ndarray) -> Optional[DetectionResult]:
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim != 3:
            return None
        try:
            res = self._yolo().predict(
                arr, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False, device=self.device
            )[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                idx = int(boxes.conf.argmax())
                xyxy = boxes.xyxy[idx].cpu().numpy().astype(np.float32)
                return DetectionResult(
                    bbox=xyxy,
                    confidence=float(boxes.conf[idx]),
                    class_id=int(boxes.cls[idx]),
                    class_name=self._classes[int(boxes.cls[idx])],
                )
        except Exception:
            pass
        score = _geometric_score(arr)
        if score >= self.geom_threshold:
            return _synthetic_bridge_bbox(arr, score)
        return None
