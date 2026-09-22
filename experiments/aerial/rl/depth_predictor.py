"""Runtime wrapper: DepthHead → ``obs.info['depth_min_pred']`` (frozen §4 ④ wiring).

Kept separate from ``dynamics_torch`` so the collector can depend on a tiny
protocol without importing torch at module import time on GPU-less hosts.
The real ``_DepthHead`` is loaded lazily inside ``from_checkpoint``.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np

from experiments.aerial.rl.depth_geometry import CONE_KEYS, azimuth_clearances, cone_clearances
from experiments.aerial.rl.env.obs import Observation


class DepthMinPredictor:
    """Maintain a short RGB history and emit scalar ``depth_min_pred``.

    Spec: collector must set ``obs.info['depth_min_pred']`` **before**
    ``safety.should_override``. When no checkpoint is loaded this is a no-op
    (returns None) so V0 default collection stays shield-inert.

    P0a adds :meth:`predict_cones` (five directional clearances on D̂). The
    collector/shield still consume only :meth:`predict_min` until P0b.
    """

    def __init__(self, *, n_frames: int = 4, device: str = "cpu") -> None:
        self.n_frames = int(n_frames)
        self.device = device
        self._hist: Deque[np.ndarray] = deque(maxlen=self.n_frames)
        self._model: Any = None

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        *,
        device: str = "cpu",
    ) -> "DepthMinPredictor":
        import torch
        from experiments.aerial.rl.dynamics_torch import build_depth_head

        payload = torch.load(str(path), map_location="cpu")
        n_frames = int(payload.get("n_frames", 4))
        # Factory dispatches on payload["backbone"] ("scratch" default → _DepthHead;
        # "da3" → DA3DepthHead). Canonical depth_step_5000.pt has no backbone key →
        # rebuilds as scratch, unchanged.
        model = build_depth_head(payload)
        model.load_state_dict(payload["model"], strict=True)
        model.to(device)
        model.eval()
        pred = cls(n_frames=n_frames, device=device)
        pred._model = model
        return pred

    def reset(self) -> None:
        self._hist.clear()

    def _run_depth_head(self, obs: Observation) -> Optional[np.ndarray]:
        """Push ``obs.rgb`` into history; return 2-D D̂ or None if unloaded/empty."""
        if self._model is None:
            return None
        import torch

        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        target = int(getattr(self._model, "image_size", 0) or 0)
        if target > 0 and rgb.shape[:2] != (target, target):
            import cv2  # lazy: only when resizing for DA3 / mismatched capture

            rgb = cv2.resize(rgb, (target, target), interpolation=cv2.INTER_LINEAR)
        self._hist.append(rgb)
        # Pad left with the oldest frame if history is still warming up.
        frames = list(self._hist)
        while len(frames) < self.n_frames:
            frames.insert(0, frames[0])
        stack = np.stack(frames[-self.n_frames :], axis=0)  # [L,H,W,3]
        tensor = torch.from_numpy(stack).unsqueeze(0)  # [1,L,H,W,3]
        with torch.no_grad():
            depth, _ = self._model.predict_from_window(tensor.to(self.device))
        return depth.squeeze(0).detach().float().cpu().numpy()

    def _min_from_depth(self, d: np.ndarray) -> Optional[float]:
        finite = d[np.isfinite(d) & (d > 0)]
        if finite.size == 0:
            return None
        return float(np.min(finite))

    def _cones_from_depth(
        self, d: np.ndarray, *, center_frac: float = 0.5
    ) -> Optional[Dict[str, float]]:
        cones = cone_clearances(d, center_frac=center_frac)
        if all(cones[k] == float("inf") for k in CONE_KEYS):
            return None
        return cones

    def predict_depth(self, obs: Observation) -> Optional[np.ndarray]:
        """Full D̂ map for open-vocab back-projection (eval / debug)."""
        return self._run_depth_head(obs)

    def predict_min(self, obs: Observation) -> Optional[float]:
        """Push ``obs.rgb`` into history; return min ``D̂`` or None if unloaded."""
        d = self._run_depth_head(obs)
        if d is None:
            return None
        return self._min_from_depth(d)

    def predict_cones(
        self,
        obs: Observation,
        *,
        center_frac: float = 0.5,
    ) -> Optional[Dict[str, float]]:
        """Push ``obs.rgb`` into history; return five-direction clearances on D̂.

        Keys (see :mod:`depth_geometry`): ``forward``, ``left``, ``right``,
        ``up``, ``down``. Each value is the min finite+positive depth in that
        region; ``inf`` means no obstacle seen there.

        Not wired into the collector/shield until P0b — :meth:`predict_min`
        remains available as the full-field min fallback. P0b collector fills
        ``obs.info['depth_cones_pred']`` from this method when present.
        """
        d = self._run_depth_head(obs)
        if d is None:
            return None
        return self._cones_from_depth(d, center_frac=center_frac)

    def predict_min_and_cones(
        self,
        obs: Observation,
        *,
        center_frac: float = 0.5,
    ) -> tuple[Optional[float], Optional[Dict[str, float]]]:
        """Single depth-head pass: full-field min + five cones on the same D̂."""
        d = self._run_depth_head(obs)
        if d is None:
            return None, None
        return self._min_from_depth(d), self._cones_from_depth(d, center_frac=center_frac)

    def predict_min_cones_and_azimuth(
        self,
        obs: Observation,
        *,
        center_frac: float = 0.5,
        azimuth_n_bins: int = 8,
        azimuth_hfov_deg: float = 90.0,
    ) -> tuple[Optional[float], Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
        """Single depth-head pass: full-field min + five cones + N-bin azimuth
        clearance on the SAME D̂.

        MUST call ``_run_depth_head`` at most once per step: it pushes
        ``obs.rgb`` into the sliding history window, so calling it twice in
        one step (e.g. once for cones, once for azimuth) would double-feed
        the same frame and desync the multi-frame history from real time.
        Use this instead of separately calling :meth:`predict_min_and_cones`
        and a standalone azimuth method.
        """
        d = self._run_depth_head(obs)
        if d is None:
            return None, None, None
        return (
            self._min_from_depth(d),
            self._cones_from_depth(d, center_frac=center_frac),
            azimuth_clearances(
                d, n_bins=azimuth_n_bins, center_frac=center_frac, hfov_deg=azimuth_hfov_deg
            ),
        )


class GTDepthAdapter:
    """Diagnostic-only: same ``predict_min_and_cones`` interface as
    :class:`DepthMinPredictor`, but reads AirSim's true DepthPlanar
    (``obs.depth``) instead of running the D̂ network.

    Privileged (ground-truth mesh depth) — never for deploy or for training
    the deployed π. Purpose: isolate whether a chronic-shield-intervention
    failure on a route is caused by D̂ bias/noise (this adapter flies clean)
    or by a genuine obstacle occupying the corridor (this adapter also gets
    intervened on heavily). Requires the env to grab per-step depth
    (``grab_depth=True``); otherwise ``obs.depth`` is None and this is a
    silent no-op (matches ``DepthMinPredictor`` with no checkpoint loaded).
    """

    def reset(self) -> None:
        return None

    def _min_from_depth(self, d: np.ndarray) -> Optional[float]:
        finite = d[np.isfinite(d) & (d > 0)]
        if finite.size == 0:
            return None
        return float(np.min(finite))

    def _cones_from_depth(
        self, d: np.ndarray, *, center_frac: float = 0.5
    ) -> Optional[Dict[str, float]]:
        cones = cone_clearances(d, center_frac=center_frac)
        if all(cones[k] == float("inf") for k in CONE_KEYS):
            return None
        return cones

    def predict_min(self, obs: Observation) -> Optional[float]:
        d = getattr(obs, "depth", None)
        if d is None:
            return None
        return self._min_from_depth(np.asarray(d, dtype=np.float32))

    def predict_cones(
        self, obs: Observation, *, center_frac: float = 0.5
    ) -> Optional[Dict[str, float]]:
        d = getattr(obs, "depth", None)
        if d is None:
            return None
        return self._cones_from_depth(np.asarray(d, dtype=np.float32), center_frac=center_frac)

    def predict_min_and_cones(
        self, obs: Observation, *, center_frac: float = 0.5
    ) -> tuple[Optional[float], Optional[Dict[str, float]]]:
        d = getattr(obs, "depth", None)
        if d is None:
            return None, None
        d = np.asarray(d, dtype=np.float32)
        return self._min_from_depth(d), self._cones_from_depth(d, center_frac=center_frac)

    def predict_min_cones_and_azimuth(
        self,
        obs: Observation,
        *,
        center_frac: float = 0.5,
        azimuth_n_bins: int = 8,
        azimuth_hfov_deg: float = 90.0,
    ) -> tuple[Optional[float], Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
        d = getattr(obs, "depth", None)
        if d is None:
            return None, None, None
        d = np.asarray(d, dtype=np.float32)
        return (
            self._min_from_depth(d),
            self._cones_from_depth(d, center_frac=center_frac),
            azimuth_clearances(
                d, n_bins=azimuth_n_bins, center_frac=center_frac, hfov_deg=azimuth_hfov_deg
            ),
        )
