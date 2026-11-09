"""Monocular depth + scene vector from the DA3 depth head (TorchScript export).

``tools/export_da3.py`` writes ``<name>.ts`` plus ``<name>.json`` (image size,
frame window). The scripted module maps a uint8 RGB window [1, L, S, S, 3] to
(depth [1, S, S] in meters with the forward calibration applied, env [1, D]).
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np


class Da3Depth:
    def __init__(self, path: str, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        meta = json.loads(Path(path).with_suffix(".json").read_text())
        self.size = int(meta["image_size"])
        self.n_frames = int(meta["n_frames"])
        self.device = device
        self.module = torch.jit.load(str(path), map_location=device).eval()
        self.fp16 = str(device).startswith("cuda") and os.environ.get("AERIAL_DA3_FP16") == "1"
        self._hist: Deque[np.ndarray] = deque(maxlen=self.n_frames)

    def reset(self) -> None:
        self._hist.clear()

    def __call__(self, rgb: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        torch = self.torch
        img = np.asarray(rgb, np.uint8)
        if img.shape[:2] != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        self._hist.append(img)
        frames = list(self._hist)
        while len(frames) < self.n_frames:
            frames.insert(0, frames[0])
        x = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            depth, env = self.module(x)
        return depth[0].float().cpu().numpy(), env[0].float().cpu().numpy()


class Da3Checkpoint:
    """DA3 depth from the training checkpoint via ``DepthMinPredictor`` (needs the training repo on PYTHONPATH).

    Same calibrated depth as the TorchScript export; no scene vector, so env-feature progress heads get none.
    """

    def __init__(self, path: str, device: str = "cuda", size: int = 224) -> None:
        from experiments.aerial.rl.depth_predictor import DepthMinPredictor
        from experiments.aerial.rl.env.obs import Observation

        if str(device).startswith("cuda"):
            os.environ.setdefault("AERIAL_DA3_FP16", "1")
        self._obs = Observation
        self.pred = DepthMinPredictor.from_checkpoint(path, device=device)
        self.size = int(size)

    def reset(self) -> None:
        self.pred.reset()

    def __call__(self, rgb: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img = cv2.resize(np.asarray(rgb, np.uint8), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        d = self.pred.predict_depth(self._obs(rgb=img, state=np.zeros(7, np.float32), collided=False))
        return np.asarray(d, np.float32), None


class MetricDa3:
    """Scale one camera's DA3 meters onto tape meters, and drop out-of-sensor pixels.

    DA3 paints the rectified black border as a near obstacle. Those pixels are outside
    this lens, so they are set to ``invalid_m`` (far, not an obstacle) after scaling.
    """

    uses_valid = True

    def __init__(self, inner, scale: float, invalid_m: float = 30.0) -> None:
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"metric scale must be positive, got {scale}")
        self.inner = inner
        self.scale = float(scale)
        self.invalid_m = float(invalid_m)
        self.size = getattr(inner, "size", 224)

    def reset(self) -> None:
        self.inner.reset()

    def __call__(self, rgb: np.ndarray, valid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        depth, env = self.inner(rgb)
        depth = np.asarray(depth, np.float32) * self.scale
        if valid is not None:
            mask = cv2.resize(np.asarray(valid), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            depth = np.where(mask, depth, self.invalid_m).astype(np.float32)
        return depth, env


class SensorDepth:
    """Use the backend's depth image (simulation ground truth); no scene vector."""

    size = 224
    needs_sensor = True

    def reset(self) -> None:
        pass

    def __call__(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if depth is None:
            raise RuntimeError("SensorDepth needs a backend depth image")
        return cv2.resize(np.asarray(depth, np.float32), (self.size, self.size), interpolation=cv2.INTER_NEAREST), None
