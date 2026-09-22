"""HumenCorridor bridge detector — vision-only, no geometric fallback.

YOLO-World multi-class sweep tuned from open_vocab_probe on HumenCorridor:
  - bridge class prompts rarely fire (max ~0.01)
  - tower / compound prompts reach ~0.19–0.35 at corridor poses
  - ship is strong but not used as a lock target (optional ship-only reject)
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from vgoal.detector import BaseDetector, DetectionResult

# Probe-ranked bridge-related open-vocab classes (one predict, all classes set).
BRIDGE_CLASSES: List[str] = [
    "tower",
    "suspension bridge tower cable",
    "suspension bridge",
    "cable-stayed bridge",
    "bridge",
    "viaduct",
]

# Per-class minimum confidence (index-aligned with BRIDGE_CLASSES).
CLASS_MIN_CONF: List[float] = [0.08, 0.06, 0.05, 0.05, 0.04, 0.04]

SHIP_CLASSES: List[str] = ["ship", "cargo ship"]


def _bbox_passes_spatial(bbox: np.ndarray, h: int, w: int, cls_name: str) -> bool:
    """Tower locks must sit in upper/mid skyline — reject deck-level false positives."""
    cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
    if cls_name == "tower":
        return cy < 0.72 * h
    return True


class HumenCorridorDetector(BaseDetector):
    """Multi-class open-vocab detector for Humen corridor SEARCH (no structure fallback)."""

    def __init__(
        self,
        *,
        model_path: str = "yolov8s-worldv2.pt",
        conf_threshold: float = 0.04,
        imgsz: int = 1280,
        device: str = "cuda",
        visual_prompt: str = "",
        reject_ship_only: bool = True,
        ship_reject_conf: float = 0.45,
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.reject_ship_only = bool(reject_ship_only)
        self.ship_reject_conf = float(ship_reject_conf)
        extra = [c.strip() for c in str(visual_prompt or "").split() if c.strip()]
        self._classes = list(dict.fromkeys(extra + BRIDGE_CLASSES))
        self._class_min = {
            name: CLASS_MIN_CONF[i] if i < len(CLASS_MIN_CONF) else self.conf_threshold
            for i, name in enumerate(BRIDGE_CLASSES)
        }
        for name in extra:
            self._class_min.setdefault(name, self.conf_threshold)
        self._model = None
        self._model_path = model_path

    def _yolo(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
            self._model.set_classes(self._classes)
        return self._model

    def _ship_only(self, rgb: np.ndarray, bridge_best: Optional[DetectionResult]) -> bool:
        if not self.reject_ship_only or bridge_best is not None:
            return False
        try:
            from ultralytics import YOLO

            m = YOLO(self._model_path)
            m.set_classes(SHIP_CLASSES)
            res = m.predict(
                rgb, conf=self.ship_reject_conf, imgsz=self.imgsz, verbose=False, device=self.device
            )[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                return float(boxes.conf.max()) >= self.ship_reject_conf
        except Exception:
            pass
        return False

    def detect_all(self, rgb: np.ndarray) -> List[DetectionResult]:
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim != 3:
            return []
        h, w = arr.shape[:2]
        out: List[DetectionResult] = []
        try:
            res = self._yolo().predict(
                arr, conf=min(self.conf_threshold, 0.02), imgsz=self.imgsz, verbose=False, device=self.device
            )[0]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                return out
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                cls_name = self._classes[cls_id] if cls_id < len(self._classes) else str(cls_id)
                conf = float(boxes.conf[i])
                min_c = self._class_min.get(cls_name, self.conf_threshold)
                if conf < min_c:
                    continue
                xyxy = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                if not _bbox_passes_spatial(xyxy, h, w, cls_name):
                    continue
                out.append(
                    DetectionResult(
                        bbox=xyxy,
                        confidence=conf,
                        class_id=cls_id,
                        class_name="bridge",
                    )
                )
        except Exception:
            return out
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out

    def detect(self, rgb: np.ndarray) -> Optional[DetectionResult]:
        all_d = self.detect_all(rgb)
        best = all_d[0] if all_d else None
        if best is None and self._ship_only(rgb, None):
            return None
        return best
