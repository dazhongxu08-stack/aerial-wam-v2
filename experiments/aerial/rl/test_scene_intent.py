"""Unit tests for non-polyline scene intent (Phase-2 E0/E1)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.aerial.rl.scene_intent import (
    SceneIntentPlanner,
    TowardGoalIntent,
    clip_toward_goal,
)


def test_clip_toward_goal_far():
    p = np.array([0.0, 0.0, 10.0])
    g = np.array([100.0, 0.0, 10.0])
    c = clip_toward_goal(p, g, r_m=25.0)
    np.testing.assert_allclose(c, [25.0, 0.0, 10.0], atol=1e-5)


def test_clip_toward_goal_near_returns_g():
    p = np.array([0.0, 0.0, 10.0])
    g = np.array([10.0, 0.0, 10.0])
    c = clip_toward_goal(p, g, r_m=25.0)
    np.testing.assert_allclose(c, g, atol=1e-5)


def test_toward_g_never_uses_polyline_keys():
    intent = TowardGoalIntent(r_m=25.0, mode="toward_g")
    intent.reset()
    g_rel, info = intent.compute(
        curr_pos=np.zeros(3),
        curr_yaw=0.0,
        goal=np.array([80.0, 0.0, 0.0]),
    )
    assert "target_world" in info
    assert info.get("subgoal_source") == "toward_g"
    assert abs(float(g_rel[3]) - 25.0) < 1.0
    assert "cte_m" not in info or info["cte_m"] is None


def test_direct_g_keeps_full_distance():
    intent = TowardGoalIntent(r_m=25.0, mode="direct_g")
    intent.reset()
    g_rel, info = intent.compute(
        curr_pos=np.zeros(3),
        curr_yaw=0.0,
        goal=np.array([80.0, 0.0, 0.0]),
    )
    assert info["subgoal_source"] == "direct_g"
    assert float(g_rel[3]) == pytest.approx(80.0, abs=1e-3)


def test_scene_planner_picks_forward_when_clear():
    pl = SceneIntentPlanner(r_m=25.0)
    pl.reset()
    _g_rel, info = pl.compute(
        curr_pos=np.zeros(3),
        curr_yaw=0.0,
        goal=np.array([100.0, 0.0, 0.0]),
        d_fwd_hat=40.0,
    )
    assert info["subgoal_source"] == "scene"
    assert info["n_candidates"] >= 2
    tw = np.array(info["target_world"])
    assert tw[0] > 0.0


def test_scene_candidate0_no_hold():
    """Candidate 0 (toward_g) skips hold so the subgoal tracks position every
    step — identical to TowardGoalIntent.  Offaxis candidates do hold."""
    pl = SceneIntentPlanner(r_m=25.0, replan_period_s=2.0, step_hz=5.0)
    pl.reset()
    # Step 0: clear path → candidate 0 chosen
    _, info0 = pl.compute(np.zeros(3), 0.0, np.array([100.0, 0.0, 0.0]), 50.0)
    assert info0["chosen_idx"] == 0
    # Step 1: candidate 0 was chosen → must replan immediately (no hold)
    _, info1 = pl.compute(
        np.array([1.0, 0.0, 0.0]), 0.0, np.array([100.0, 0.0, 0.0]), 50.0
    )
    assert info1.get("replan") is True, "candidate 0 must not hold; must replan each step"

    # Offaxis case: when an offaxis candidate is chosen, the hold period applies.
    pl2 = SceneIntentPlanner(r_m=25.0, replan_period_s=2.0, step_hz=5.0, d_danger=3.0, d_clear=22.0, w_fwd=2.0)
    pl2.reset()
    _, i0 = pl2.compute(np.zeros(3), 0.0, np.array([100.0, 0.0, 0.0]), 4.0)
    assert i0["chosen_idx"] != 0, "offaxis candidate should win at d_fwd=4m"
    _, i1 = pl2.compute(np.array([0.3, 0.0, 0.0]), 0.0, np.array([100.0, 0.0, 0.0]), 4.0)
    assert i1.get("replan") is False, "offaxis candidate must hold between period steps"


def test_scene_no_replan_in_soft_zone():
    """d_fwd in soft zone [d_danger, d_clear) → offaxis candidate wins → hold
    applies → next step does NOT replan (d_fwd alone must not trigger replan)."""
    pl = SceneIntentPlanner(r_m=25.0, replan_period_s=2.0, step_hz=5.0)
    pl.reset()
    _, info0 = pl.compute(np.zeros(3), 0.0, np.array([100.0, 0.0, 0.0]), 15.0)
    assert info0.get("replan") is True  # first call always replans
    assert info0["chosen_idx"] != 0, "d_fwd=15 in soft zone should pick offaxis"
    _, info1 = pl.compute(
        np.array([0.3, 0.0, 0.0]), 0.0, np.array([100.0, 0.0, 0.0]), 15.0
    )
    assert info1.get("replan") is False, "offaxis hold: d_fwd alone must not trigger replan"


def test_scene_danger_triggers_emergency_replan():
    """d_fwd < d_danger overrides the hold period → immediate emergency replan."""
    pl = SceneIntentPlanner(r_m=25.0, replan_period_s=2.0, step_hz=5.0, d_danger=3.0)
    pl.reset()
    pl.compute(np.zeros(3), 0.0, np.array([100.0, 0.0, 0.0]), 40.0)
    _, info = pl.compute(
        np.array([0.3, 0.0, 0.0]), 0.0, np.array([100.0, 0.0, 0.0]), 2.0
    )
    assert info.get("replan") is True, "d_fwd < d_danger must trigger emergency replan"


def test_scene_clear_path_stays_on_goal_ray():
    """E1 observability: clear forward depth ⇒ candidate 0 ⇒ dev 0 ⇒ ≡ toward_g."""
    pl = SceneIntentPlanner(r_m=25.0)
    pl.reset()
    _, info = pl.compute(np.zeros(3), 0.0, np.array([100.0, 0.0, 0.0]), 40.0)
    assert info["chosen_idx"] == 0
    assert info["dev_deg"] == pytest.approx(0.0, abs=1e-6)
    assert info["replan_count"] == 1
    assert info["offaxis_count"] == 0


def test_scene_soft_zone_forces_offaxis():
    """With w_fwd=2.0 the penalty at d_fwd=4m is tight≈0.95 × r_m × 2 ≈ 47 m,
    which exceeds toward_g progress ≈ 25 m → offaxis candidate wins."""
    pl = SceneIntentPlanner(r_m=25.0, d_danger=3.0, d_clear=22.0, w_fwd=2.0)
    pl.reset()
    _, info = pl.compute(np.zeros(3), 0.0, np.array([100.0, 0.0, 0.0]), 4.0)
    assert info["chosen_idx"] != 0, "offaxis should beat toward_g when penalty > progress"
    assert info["offaxis_count"] == 1


def test_scene_near_clear_toward_g_still_wins():
    """Near d_clear tight→0 so penalty≈0 → toward_g wins on progress."""
    pl = SceneIntentPlanner(r_m=25.0, d_danger=3.0, d_clear=22.0, w_fwd=2.0)
    pl.reset()
    _, info = pl.compute(np.zeros(3), 0.0, np.array([100.0, 0.0, 0.0]), 21.0)
    assert info["chosen_idx"] == 0, "toward_g should win when d_fwd is near d_clear"
    assert info["offaxis_count"] == 0


def test_scene_imminent_danger_hard_blocks_forward():
    """When d_fwd < d_danger (< 3 m), forward candidates are hard-blocked
    → fan must pick a lateral candidate → offaxis counted."""
    pl = SceneIntentPlanner(r_m=25.0, d_danger=3.0, d_clear=22.0)
    pl.reset()
    # d_fwd=2.0 < d_danger=3.0 → hard block nose-aligned candidates
    _, info = pl.compute(np.zeros(3), 0.0, np.array([100.0, 0.0, 0.0]), 2.0)
    assert info["chosen_idx"] != 0, "forward candidate must be hard-blocked at d_fwd=2m"
    assert info["dev_deg"] > 1.0
    assert info["offaxis_count"] == 1


def test_scene_fan_reaches_past_danger_cone():
    """Regression: a fan narrower than the ±60° cone leaves nothing feasible,
    so every candidate is discarded and `scene` degenerates into `toward_g`."""
    pl = SceneIntentPlanner()
    assert max(abs(float(d)) for d in pl.yaw_offsets_deg) >= 60.0
    _, info = pl.compute(np.zeros(3), 0.0, np.array([100.0, 0.0, 0.0]), 3.0)
    assert info["n_feasible"] > 0
    assert info["n_fan_starved"] == 0


def test_scene_fan_starved_still_returns_target():
    pl = SceneIntentPlanner(r_m=25.0, yaw_offsets_deg=(0.0,))
    pl.reset()
    _, info = pl.compute(np.zeros(3), 0.0, np.array([100.0, 0.0, 0.0]), 1.0)
    assert info["n_feasible"] == 0
    assert info["n_fan_starved"] == 1
    assert np.all(np.isfinite(np.array(info["target_world"])))


def test_scene_counters_reset_per_route():
    pl = SceneIntentPlanner(r_m=25.0)
    pl.reset()
    pl.compute(np.zeros(3), 0.0, np.array([100.0, 0.0, 0.0]), 4.0)
    assert pl.replan_count == 1
    pl.reset()
    assert pl.replan_count == 0
    assert pl.offaxis_count == 0


# ---------------------------------------------------------------------------
# _cone_depth and _body_bearing_deg
# ---------------------------------------------------------------------------

def test_cone_depth_forward_sector():
    pl = SceneIntentPlanner()
    cones = {"forward": 10.0, "left": 50.0, "right": 50.0}
    # 0° bearing → forward cone
    assert pl._cone_depth(0.0, cones) == pytest.approx(10.0)
    # ±29° → still forward
    assert pl._cone_depth(29.0, cones) == pytest.approx(10.0)
    assert pl._cone_depth(-29.0, cones) == pytest.approx(10.0)


def test_cone_depth_side_sectors():
    pl = SceneIntentPlanner()
    cones = {"forward": 50.0, "left": 8.0, "right": 12.0}
    # +45° bearing (left) → left cone
    assert pl._cone_depth(45.0, cones) == pytest.approx(8.0)
    # -45° bearing (right) → right cone
    assert pl._cone_depth(-45.0, cones) == pytest.approx(12.0)
    # +75° bearing (far left) → left cone
    assert pl._cone_depth(75.0, cones) == pytest.approx(8.0)


def test_cone_depth_fallback_when_side_missing():
    pl = SceneIntentPlanner()
    cones = {"forward": 20.0}  # no left/right
    assert pl._cone_depth(60.0, cones) == pytest.approx(20.0)
    assert pl._cone_depth(-60.0, cones) == pytest.approx(20.0)


def test_cone_depth_none_when_cones_none():
    pl = SceneIntentPlanner()
    assert pl._cone_depth(0.0, None) is None


def test_body_bearing_forward():
    pl = SceneIntentPlanner()
    p = np.zeros(3)
    # Candidate directly in front (yaw=0 → x-axis)
    cand = np.array([25.0, 0.0, 0.0])
    bearing = pl._body_bearing_deg(p, cand, yaw=0.0)
    assert abs(bearing) < 1e-6


def test_body_bearing_left():
    pl = SceneIntentPlanner()
    p = np.zeros(3)
    # yaw=0, candidate at +y → 90° left
    cand = np.array([0.0, 25.0, 0.0])
    bearing = pl._body_bearing_deg(p, cand, yaw=0.0)
    assert bearing == pytest.approx(90.0, abs=1e-5)


def test_scene_planner_uses_left_cone_for_left_candidate():
    """When a left candidate faces a shallow left cone, it should be penalised
    even though d_fwd (forward cone) is clear.  Without depth_cones the test
    baseline shows the left candidate wins; with a blocked left cone the forward
    candidate should win instead."""
    # Setup: forward clear (50 m), left shallow (5 m → inside soft zone)
    # yaw = 0, goal far ahead → candidate 0 is toward_g (forward)
    # d_fwd=50 m (clear) so no forward penalty.
    # Left cone depth=5 → left candidates get penalised.
    pl = SceneIntentPlanner(r_m=25.0, d_danger=3.0, d_clear=40.0, w_fwd=2.0)
    pl.reset()
    cones_left_blocked = {"forward": 50.0, "left": 5.0, "right": 50.0}
    _, info = pl.compute(
        curr_pos=np.zeros(3),
        curr_yaw=0.0,
        goal=np.array([100.0, 0.0, 0.0]),
        d_fwd_hat=50.0,
        depth_cones=cones_left_blocked,
    )
    # Forward clear, left blocked → toward_g (candidate 0) should win
    assert info["chosen_idx"] == 0, (
        "left candidates should be penalised by left-cone depth; forward candidate wins"
    )


def test_stuck_escape_triggers_after_prolonged_no_progress():
    """Diagnosed hard134 route 0 (2026-09-18): a vehicle pinned in a corner
    stays within a few metres of the same distance-to-goal indefinitely — the
    ±75-105° fan + progress-weighted scoring keeps snapping back toward the
    blocked direct bearing. After ``stuck_escape_after_s`` seconds of no real
    progress, the planner must widen the fan (including near-reverse
    headings) and pick by clearance alone, ignoring goal progress."""
    pl = SceneIntentPlanner(
        r_m=25.0,
        d_danger=3.0,
        d_clear=40.0,
        step_hz=5.0,
        stuck_escape_after_s=2.0,  # short window for a fast test
        stuck_escape_hold_s=1.0,
    )
    pl.reset()
    p = np.zeros(3)
    goal = np.array([100.0, 0.0, 0.0])
    # Forward is blocked; the only real opening is far to the side (>90°,
    # beyond the base ±75° fan) — modelled as a much deeper "left" cone once
    # the bearing is >90°, which only the wide escape offsets can reach.
    cones = {"forward": 2.0, "left": 3.0, "right": 3.0}
    saw_escape = False
    for _ in range(40):  # 8s at 5 Hz — past the 2s no-progress threshold
        _, info = pl.compute(
            curr_pos=p,
            curr_yaw=0.0,
            goal=goal,
            d_fwd_hat=2.0,
            depth_cones=cones,
        )
        if info["in_escape"]:
            saw_escape = True
            break
    assert saw_escape, "expected stuck-escape to trigger after prolonged no-progress"
    assert pl.escape_count >= 1


def test_stuck_escape_disabled_by_default_zero():
    """stuck_escape_after_s=0 must fully preserve legacy behaviour (never
    widen the fan / override progress-based scoring)."""
    pl = SceneIntentPlanner(stuck_escape_after_s=0.0)
    pl.reset()
    p = np.zeros(3)
    goal = np.array([100.0, 0.0, 0.0])
    cones = {"forward": 2.0, "left": 3.0, "right": 3.0}
    for _ in range(200):
        _, info = pl.compute(
            curr_pos=p, curr_yaw=0.0, goal=goal, d_fwd_hat=2.0, depth_cones=cones
        )
        assert not info["in_escape"]
    assert pl.escape_count == 0
