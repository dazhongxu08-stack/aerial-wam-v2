"""Torch-free tests for P0a ``predict_cones`` and shared depth geometry.

``predict_min`` behaviour must stay identical (full-field min for shield wiring).
These tests stub the depth head so they run on Mac without GPU/torch.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pytest

from experiments.aerial.rl.depth_geometry import CONE_KEYS, cone_clearances
from experiments.aerial.rl.depth_predictor import DepthMinPredictor, read_forward_scale
from experiments.aerial.rl.env.obs import Observation


def _obs(rgb: Optional[np.ndarray] = None, *, size: int = 8) -> Observation:
    state = np.zeros(7, dtype=np.float32)
    if rgb is None:
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
    depth = np.full((size, size), 10.0, dtype=np.float32)
    return Observation(rgb=rgb, state=state, depth=depth)


def _depth_map(h: int = 8, w: int = 8) -> np.ndarray:
    """Synthetic map: one obstacle pixel per cone (non-overlapping at center_frac=0.5)."""
    d = np.full((h, w), 20.0, dtype=np.float64)
    d[0, w // 2] = 4.0  # up — top row
    d[h - 1, w // 2] = 2.0  # down — bottom row (global min)
    d[h // 2, 0] = 5.0  # left — left edge
    d[h // 2, w - 1] = 7.0  # right — right edge
    d[h // 2 - 1, w // 2] = 3.0  # forward — centre column inside crop
    return d


def _bind_stub_depth(pred: DepthMinPredictor, depth_map: np.ndarray) -> None:
    """Torch-free stub: push history like the real head, return fixed D̂."""
    pred._model = object()

    def _run(obs: Observation) -> np.ndarray:
        pred._hist.append(np.asarray(obs.rgb, dtype=np.uint8))
        return np.asarray(depth_map, dtype=np.float32)

    pred._run_depth_head = _run  # type: ignore[method-assign]


class _StubDepthModel:
    """Only used when torch is available (optional integration smoke)."""

    def __init__(self, depth_map: np.ndarray) -> None:
        self._depth = np.asarray(depth_map, dtype=np.float32)

    def predict_from_window(self, tensor: Any) -> Tuple[Any, Any]:
        import torch

        b, _l, h, w, _c = tensor.shape
        depth = torch.from_numpy(self._depth).unsqueeze(0).expand(b, h, w).clone()
        log_sigma = torch.zeros_like(depth)
        return depth, log_sigma


def test_cone_clearances_five_regions():
    d = _depth_map()
    cones = cone_clearances(d, center_frac=0.5)
    assert set(cones) == set(CONE_KEYS)
    h, w = d.shape
    mid_r, mid_c = h // 2, w // 2
    # left/right now use the same row band as forward (center rows only) so that
    # ground pixels in the lower part of the image don't dominate lateral clearance.
    cf = 0.5
    dh = max(1, int(h * cf))
    r0 = (h - dh) // 2
    from experiments.aerial.rl.depth_geometry import forward_min_depth

    finite = lambda reg: reg[np.isfinite(reg) & (reg > 0)]
    assert cones["forward"] == forward_min_depth(d, center_frac=0.5)
    assert cones["left"] == pytest.approx(float(np.min(finite(d[r0 : r0 + dh, :mid_c]))))
    assert cones["right"] == pytest.approx(float(np.min(finite(d[r0 : r0 + dh, mid_c:]))))
    assert cones["up"] == pytest.approx(float(np.min(finite(d[:mid_r, :]))))
    assert cones["down"] == pytest.approx(float(np.min(finite(d[mid_r:, :]))))
    assert cones["down"] == pytest.approx(2.0)  # global min lives in bottom half


def test_predict_cones_returns_directional_clearances():
    dmap = _depth_map()
    pred = DepthMinPredictor(n_frames=1)
    _bind_stub_depth(pred, dmap)
    obs = _obs(rgb=np.ones((8, 8, 3), dtype=np.uint8))
    cones = pred.predict_cones(obs, center_frac=0.5)
    assert cones is not None
    assert cones["down"] == pytest.approx(2.0)
    assert cones["forward"] == pytest.approx(3.0)


def test_predict_min_unchanged_full_field_min():
    dmap = _depth_map()
    pred = DepthMinPredictor(n_frames=1)
    _bind_stub_depth(pred, dmap)
    obs = _obs(rgb=np.ones((8, 8, 3), dtype=np.uint8))
    d_min = pred.predict_min(obs)
    assert d_min == pytest.approx(2.0)  # global min, not forward cone


def test_predict_min_same_on_identical_depth_map():
    """predict_min still returns full-field min independent of cone regions."""
    dmap = np.full((8, 8), 9.0, dtype=np.float32)
    dmap[0, 0] = 1.5  # corner only — outside forward centre at center_frac=0.5
    pred = DepthMinPredictor(n_frames=1)
    _bind_stub_depth(pred, dmap)
    obs = _obs(rgb=np.full((8, 8, 3), 42, dtype=np.uint8))
    assert pred.predict_min(obs) == pytest.approx(1.5)
    cones = pred.predict_cones(obs, center_frac=0.5)
    assert cones is not None
    assert cones["forward"] == pytest.approx(9.0)  # centre never sees the corner
    assert cones["up"] == pytest.approx(1.5)  # top-left corner in upper half


def test_predict_cones_none_when_unloaded():
    pred = DepthMinPredictor()
    assert pred.predict_cones(_obs()) is None


def test_predict_min_and_cones_single_depth_pass():
    dmap = _depth_map()
    pred = DepthMinPredictor(n_frames=1)
    calls = {"n": 0}

    def _run(obs: Observation) -> np.ndarray:
        calls["n"] += 1
        pred._hist.append(np.asarray(obs.rgb, dtype=np.uint8))
        return np.asarray(dmap, dtype=np.float32)

    pred._model = object()
    pred._run_depth_head = _run  # type: ignore[method-assign]
    obs = _obs(rgb=np.ones((8, 8, 3), dtype=np.uint8))
    d_min, cones = pred.predict_min_and_cones(obs)
    assert calls["n"] == 1
    assert d_min == pytest.approx(2.0)
    assert cones is not None
    assert cones["forward"] == pytest.approx(3.0)


def test_predict_cones_none_when_all_invalid():
    pred = DepthMinPredictor(n_frames=1)
    _bind_stub_depth(pred, np.full((4, 4), np.nan, dtype=np.float32))
    assert pred.predict_cones(_obs(size=4)) is None


def test_read_forward_scale_missing_is_one(tmp_path):
    path = tmp_path / "depth.pt"
    path.write_bytes(b"")
    assert read_forward_scale(path) == 1.0


def test_read_forward_scale_sidecar(tmp_path):
    path = tmp_path / "depth.pt"
    path.write_bytes(b"")
    (tmp_path / "depth.pt.forward_scale.json").write_text('{"scale": 1.6}')
    assert read_forward_scale(path) == 1.6


def test_forward_min_depth_min_crop_size():
    from experiments.aerial.rl.depth_geometry import forward_min_depth

    d_np = np.full((10, 10), 4.0, dtype=np.float64)
    d_np[4:6, 4:6] = 1.5
    assert forward_min_depth(d_np, center_frac=0.05) == 1.5

    torch = pytest.importorskip("torch")
    from experiments.aerial.rl.depth_geometry import forward_min_depth_torch

    d_t = torch.from_numpy(d_np).unsqueeze(0)
    hard = float(forward_min_depth_torch(d_t, center_frac=0.05).item())
    assert hard == 1.5
