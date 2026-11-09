"""Scene-object keep-out: detect the red stool, write it as a pillar, fly around it."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from experiments.aerial.rl.consequence_model import cones3_from_depth56, depth_input
from experiments.aerial.rl.semantic_keepout import (
    FWD_CONE_HALF_RAD,
    RED_STOOL,
    KeepoutTracker,
    KeepOut,
    SemanticKeepout,
    apply_keepouts,
    col_bearing,
    detect_color_objects,
)
from experiments.aerial.rl.tello_consequence_policy import TelloConsequencePolicy

REPO = Path(__file__).resolve().parents[4]
FRAMES = REPO / "artifacts/tello_gs"


def _hsv_rgb(h: int, s: int, v: int) -> np.ndarray:
    return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0, 0]


def _room(stool_u: float | None = 0.5, cloth: bool = False, h: int = 240, w: int = 320) -> np.ndarray:
    """Grey floor and wall; optional pink-seat / red-leg stool centred at column ``stool_u``."""
    rng = np.random.default_rng(3)
    im = np.full((h, w, 3), 150, np.float32) + rng.integers(-6, 7, (h, w, 3))
    im[: int(h * 0.55)] = 120 + rng.integers(0, 40, (int(h * 0.55), w, 3))
    if cloth:
        im[20:150, 10:70] = _hsv_rgb(176, 250, 100)
    if stool_u is not None:
        cx, sw = int(stool_u * w), 50
        im[150:165, cx - sw // 2: cx + sw // 2] = _hsv_rgb(172, 56, 228)
        for x in (cx - sw // 2, cx + sw // 2 - 6):
            im[165:200, x:x + 6] = _hsv_rgb(178, 200, 130)
    return np.clip(im, 0, 255).astype(np.uint8)


def test_detects_stool_and_ignores_red_cloth() -> None:
    dets = detect_color_objects(_room(0.5, cloth=True), RED_STOOL)
    assert len(dets) == 1
    u0, v0, u1, v1 = dets[0].bbox
    assert 0.38 < u0 < 0.5 < u1 < 0.62
    assert v1 > 0.8  # legs attached below the seat
    assert detect_color_objects(_room(None, cloth=True), RED_STOOL) == []


@pytest.mark.parametrize("rel, want", [
    ("run_20261015_124232/frames/000140.jpg", 1),
    ("run_20261015_124232/frames/000164.jpg", 1),
    ("run_20261015_121707/frames/000563.jpg", 0),  # red cloth on chair only
    ("run_20261014_144044/frames/038428.jpg", 0),  # red shelf boxes, pink coaster
])
def test_real_tello_frames(rel: str, want: int) -> None:
    p = FRAMES / rel
    if not p.is_file():
        pytest.skip("Tello frames not on this machine")
    rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    assert len(detect_color_objects(rgb, RED_STOOL)) == want


def test_pillar_is_full_height_and_blocks_forward_cone() -> None:
    far = np.full((56, 56), 6.0, np.float32)
    ko = KeepOut("red_stool", math.radians(-10), math.radians(10), 1.6, 1.1, "depth")
    d = apply_keepouts(far, [ko])
    centers = col_bearing((np.arange(56) + 0.5) / 56)
    inside = (centers >= ko.lo_rad) & (centers <= ko.hi_rad)
    assert np.all(d[:, inside] == pytest.approx(1.1)) and np.all(d[:, ~inside] == 6.0)
    fwd, left, right = cones3_from_depth56(d)
    assert fwd == pytest.approx(1.1) and left == right == 6.0


def test_low_stool_in_lower_fov_reads_clear_without_keepout() -> None:
    """Why the pillar exists: a stool under the cruise band does not touch the forward cone."""
    d = np.full((56, 56), 6.0, np.float32)
    d[44:, 24:32] = 1.5
    assert cones3_from_depth56(d)[0] == 6.0


def test_tracker_aims_at_one_tangent_edge_and_keeps_side() -> None:
    tr = KeepoutTracker(ttl_steps=3)
    tr.update(0.0, [KeepOut("s", math.radians(-15), math.radians(5), 1.5, 1.0, "")])
    b, detour = tr.steer(0.0, 0.0)
    assert detour and tr.side == 1  # goal at 0 is nearer the +5 deg edge
    assert b == pytest.approx(math.radians(5) + tr.margin_rad)
    # After yawing right the sector is out of view but remembered in world yaw; side stays.
    yaw = math.radians(30)
    tr.update(yaw, [])
    b2, detour2 = tr.steer(yaw, -yaw)
    assert detour2 and tr.side == 1
    assert b2 == pytest.approx(math.radians(5) + tr.margin_rad - yaw)
    for _ in range(4):
        tr.update(yaw, [])
    assert not tr.active and tr.side == 0
    assert tr.steer(yaw, -yaw) == (-yaw, False)


def test_margin_clears_forward_cone() -> None:
    assert KeepoutTracker().margin_rad > FWD_CONE_HALF_RAD


def _decide(pol: TelloConsequencePolicy, rgb: np.ndarray, depth56=None, yaw: float = 0.0):
    return pol.decide(rgb=rgb, height_m=1.1, yaw_rad=yaw, takeoff_yaw_rad=0.0, goal_bearing0_rad=0.0, depth56=depth56)


def _stool_depth(u: float = 0.5, r: float = 1.8) -> np.ndarray:
    """Depth head view: far room, stool only in the lower rows."""
    d = np.full((56, 56), 6.0, np.float32)
    c = int(u * 56)
    d[37:50, c - 6:c + 6] = r
    return d


def test_policy_without_avoid_creeps_into_low_stool() -> None:
    dec = _decide(TelloConsequencePolicy(), _room(0.5), _stool_depth())
    assert dec.family == "fwd" and dec.rc[1] > 0


def test_policy_with_avoid_turns_off_the_stool_line() -> None:
    pol = TelloConsequencePolicy(avoid=SemanticKeepout([RED_STOOL]))
    rgb, d = _room(0.5), _stool_depth()
    first = _decide(pol, rgb, d)
    assert first.keepout == ""  # one frame is not enough to confirm
    dec = _decide(pol, rgb, d)
    assert "red_stool" in dec.keepout and "(depth)" in dec.keepout
    assert dec.family.startswith("yaw") and dec.rc[1] == 0
    assert "detour" in dec.note


def test_stool_left_of_nose_passes_on_its_right() -> None:
    pol = TelloConsequencePolicy(avoid=SemanticKeepout([RED_STOOL]))
    rgb, d = _room(0.42), _stool_depth(0.42)
    _decide(pol, rgb, d)
    dec = _decide(pol, rgb, d)
    assert dec.family == "yaw_right" and pol.avoid.tracker.side == 1


def test_rgb_only_path_also_sees_pillar() -> None:
    pol = TelloConsequencePolicy(avoid=SemanticKeepout([RED_STOOL]))
    _decide(pol, _room(0.5))
    dec = _decide(pol, _room(0.5))
    assert dec.keepout and dec.rc[1] == 0


class _SpyModel(torch.nn.Module):
    horizon, frame_offsets, hist_actions, sem_channels = 4, (0,), 0, 0

    def forward(self, depth, goal, prev, actions, hist=None, sem=None, keep_radius=None):
        self.seen, self.sem, self.keep_radius, self.actions = depth, sem, keep_radius, actions
        out = {"hazard_logit": torch.zeros(actions.shape[:2]), "log_cones": torch.zeros(*actions.shape[:2], 3),
               "dgoal": torch.zeros(actions.shape[:2])}
        if self.sem_channels:
            out["keepout_logit"] = torch.zeros(actions.shape[:2])
        return out


def test_semantic_model_gets_mask_and_plain_depth() -> None:
    spy = _SpyModel()
    spy.sem_channels = 1
    pol = TelloConsequencePolicy(models=[spy], avoid=SemanticKeepout([RED_STOOL], radius_m=0.7))
    rgb, d = _room(0.5), _stool_depth()
    _decide(pol, rgb, d)
    _decide(pol, rgb, d)
    seen = spy.seen[0].numpy()
    assert seen[5, 28] == pytest.approx(float(depth_input(np.array([[6.0]]))[0, 0]), abs=1e-4)  # no pillar
    sem = spy.sem[0].numpy()
    assert sem.sum() > 0 and sem[:, 20:36].sum() == sem.sum()  # stool columns only
    assert sem[:28].sum() == 0  # stool sits in the lower half
    assert float(spy.keep_radius[0]) == pytest.approx(0.7)


def test_shield_candidates_use_model_yaw_sign() -> None:
    """The model is trained in AirSim NED: +dyaw turns right (clockwise)."""
    spy = _SpyModel()
    pol = TelloConsequencePolicy(models=[spy])
    _decide(pol, _room(None), np.full((56, 56), 6.0, np.float32))
    a = spy.actions.numpy()
    left, right = a[1, 0, 3], a[2, 0, 3]
    assert left < 0 < right


def test_consequence_shield_is_fed_the_pillar() -> None:
    spy = _SpyModel()
    pol = TelloConsequencePolicy(models=[spy], avoid=SemanticKeepout([RED_STOOL]))
    rgb, d = _room(0.5), _stool_depth()
    _decide(pol, rgb, d)
    _decide(pol, rgb, d)
    seen = spy.seen[0].numpy()
    pillar = float(depth_input(np.array([[pol._keepouts[0].pillar_m]]))[0, 0])
    assert seen[5, 28] == pytest.approx(pillar, abs=1e-4)  # top rows, centre column
    assert seen[5, 2] == pytest.approx(float(depth_input(np.array([[6.0]]))[0, 0]), abs=1e-4)
