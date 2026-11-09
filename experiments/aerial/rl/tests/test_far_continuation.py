"""Far continuation picks a subgoal. No weights and no flight."""
from __future__ import annotations

import math

import numpy as np

from experiments.aerial.rl.far_continuation import FlightContinuation
from experiments.aerial.rl.scene_memory import DepthView, SceneGuide, SceneMemory, open_guide


def _wall_on_the_chord(here, aim) -> bool:
    """Block the goal chord (east) a few meters out. Leave a side heading open."""
    return float(aim[0]) > 4.0 and abs(float(aim[1])) < 2.0


def test_blocked_chord_uses_the_far_open_heading():
    cont = FlightContinuation(horizon_m=8.0, far_m=40.0)
    here = np.zeros(3)
    goal = np.array([50.0, 0.0, 0.0])
    aim = cont.update(here, goal, _wall_on_the_chord)
    assert abs(float(np.linalg.norm(aim[:2])) - 8.0) < 0.2
    assert float(aim[1]) > 2.0
    assert cont.far_point is not None
    assert float(np.linalg.norm(cont.far_point[:2])) > 20.0


def test_remembered_continuation_is_kept_when_the_chord_later_looks_open():
    cont = FlightContinuation(horizon_m=8.0, far_m=40.0)
    goal = np.array([50.0, 0.0, 0.0])
    cont.update(np.zeros(3), goal, _wall_on_the_chord)
    remembered = cont.far_point.copy()

    def open_now(here, aim) -> bool:
        return False

    moved = np.array([2.0, 2.0, 0.0])
    aim = cont.update(moved, goal, open_now)
    assert np.allclose(cont.far_point, remembered)
    heading = math.atan2(aim[1] - moved[1], aim[0] - moved[0])
    remembered_heading = math.atan2(remembered[1] - moved[1], remembered[0] - moved[0])
    assert abs(heading - remembered_heading) < 0.15
    assert float(np.linalg.norm(aim[:2] - moved[:2])) <= 8.0 + 1e-6


def test_blocked_memory_picks_again():
    cont = FlightContinuation(horizon_m=8.0, far_m=40.0)
    goal = np.array([50.0, 0.0, 0.0])
    cont.update(np.zeros(3), goal, _wall_on_the_chord)

    def blocked_memory(here, aim) -> bool:
        return True

    aim = cont.update(np.array([1.0, 1.0, 0.0]), goal, blocked_memory)
    assert float(np.linalg.norm(aim[:2] - np.array([1.0, 1.0]))) <= 0.5 + 1e-6


def test_far_flag_is_off_unless_asked():
    mem = SceneMemory()
    mem.views.append(DepthView(pose=np.zeros(3), yaw=0.0, depth=np.full((8, 8), 30.0, np.float32)))
    state = np.zeros(7)
    goal = np.array([40.0, 0.0, 0.0])
    off = SceneGuide(mem, horizon_m=8.0, use_seen=True, use_far=False)
    on = SceneGuide(mem, horizon_m=8.0, use_seen=True, use_far=True)
    assert off.step_view(state, goal, None, None).source != "far"
    placed = on.step_view(state, goal, None, None)
    assert placed.source == "far"
    assert abs(float(np.linalg.norm(placed.aim[:2])) - 8.0) < 0.3


def test_unseen_space_does_not_count_as_a_far_opening():
    mem = SceneMemory()
    mem.views.append(DepthView(pose=np.zeros(3), yaw=0.0, depth=np.full((8, 8), 4.0, np.float32)))
    guide = SceneGuide(mem, horizon_m=8.0, use_seen=True, use_far=True)
    placed = guide.step_view(np.zeros(7), np.array([50.0, 0.0, 0.0]), None, None)
    assert placed.source == "far"
    assert guide.continuation.far_point is not None
    assert float(np.linalg.norm(guide.continuation.far_point[:2])) < 8.0


def test_open_guide_leaves_far_off(tmp_path):
    path = tmp_path / "scene.json"
    off = open_guide("", str(path), 8.0, "metric", "both")
    on = open_guide("", str(path), 8.0, "metric", "both", True)
    assert off is not None and on is not None
    assert off.use_far is False
    assert on.use_far is True
    assert open_guide("", "", 8.0, "metric", "both", True) is None
