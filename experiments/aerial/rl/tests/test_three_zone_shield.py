"""Tests for ThreeZoneSpeedShield deploy governor."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.aerial.rl.env.action import body_delta_limits
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.safety import ThreeZoneSpeedShield
from experiments.aerial.rl.three_zone import (
    ThreeZoneSpec,
    engage_outer_for_speed,
    planned_speed_m_s,
    resolve_v_ref_m_s,
)


def _obs(*, depth: float, v_fwd: float = 5.0, info=None):
    state = np.array([0.0, 0.0, 0.0, v_fwd, 0.0, 0.0, 0.0], dtype=np.float32)
    return Observation(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        state=state,
        depth=np.full((8, 8), depth, dtype=np.float32),
        info={"depth_min_pred": depth, **(info or {})},
    )


def test_planned_speed_at_boundaries():
    spec = ThreeZoneSpec()
    assert spec.v_cruise_m_s == pytest.approx(25.0)
    assert planned_speed_m_s(spec.engage_outer_m + 1.0, spec) == pytest.approx(25.0)
    assert planned_speed_m_s(spec.l1_m, spec) == pytest.approx(2.0, abs=0.05)
    assert planned_speed_m_s(spec.l2_m, spec) == pytest.approx(1.0, abs=0.05)
    assert planned_speed_m_s(spec.l3_m * 0.5, spec) == pytest.approx(0.2, abs=0.05)


def test_resolve_v_ref_and_dynamic_engage():
    spec = ThreeZoneSpec()
    assert resolve_v_ref_m_s(spec) == pytest.approx(25.0)
    assert resolve_v_ref_m_s(spec, v_now_m_s=5.0) == pytest.approx(5.0)
    assert resolve_v_ref_m_s(spec, v_now_m_s=5.0, v_cmd_m_s=12.0) == pytest.approx(12.0)
    assert resolve_v_ref_m_s(spec, v_now_m_s=30.0) == pytest.approx(25.0)
    eng5 = engage_outer_for_speed(spec, 5.0)
    eng25 = engage_outer_for_speed(spec, 25.0)
    assert eng5 == pytest.approx(12.65, abs=0.1)
    assert eng25 == pytest.approx(spec.engage_outer_m, abs=0.1)
    assert eng5 < eng25
    # Mid-range: slow open-air ceiling; fast already braking.
    assert planned_speed_m_s(50.0, spec, v_ref_m_s=5.0) == pytest.approx(25.0)
    assert planned_speed_m_s(50.0, spec, v_ref_m_s=25.0) < 25.0


def test_dynamic_shield_engage_scales_with_airspeed():
    """Option-1: same d̂, slow airspeed stays open-air; cruise airspeed caps."""
    shield = ThreeZoneSpeedShield(dynamic_v_ref=True)
    limits = body_delta_limits(0.2)
    action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    d = 50.0
    slow = _obs(depth=d, v_fwd=5.0)
    out_slow, ch_slow = shield.apply_action(action.copy(), slow, limits=limits)
    assert out_slow[0] == pytest.approx(1.0)
    assert not ch_slow
    assert slow.info["three_zone_v_ref_m_s"] == pytest.approx(5.0, abs=0.1)
    assert slow.info["three_zone_engage_outer_m"] < 20.0

    shield.reset()
    fast = _obs(depth=d, v_fwd=25.0)
    out_fast, ch_fast = shield.apply_action(action.copy(), fast, limits=limits)
    assert ch_fast
    assert out_fast[0] < 1.0
    assert fast.info["three_zone_v_ref_m_s"] == pytest.approx(25.0, abs=0.1)
    assert fast.info["three_zone_engage_outer_m"] == pytest.approx(
        shield.zone.engage_outer_m, abs=0.5
    )


def test_static_v_ref_flag_uses_full_cruise_engage():
    shield = ThreeZoneSpeedShield(dynamic_v_ref=False)
    limits = body_delta_limits(0.2)
    action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # Slow body but static engage ≈ 134 m → d=50 is inside band.
    obs = _obs(depth=50.0, v_fwd=5.0)
    out, changed = shield.apply_action(action, obs, limits=limits)
    assert changed
    assert out[0] < 1.0
    assert obs.info["three_zone_engage_outer_m"] == pytest.approx(
        shield.zone.engage_outer_m, abs=0.5
    )


def test_three_zone_caps_forward_at_5m():
    shield = ThreeZoneSpeedShield()
    limits = body_delta_limits(0.2)
    action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # max fwd @ 5 Hz
    obs = _obs(depth=5.0)
    capped, changed = shield.apply_action(action, obs, limits=limits)
    assert changed
    assert capped[0] < 1.0
    assert capped[0] == pytest.approx(1.0 * 0.2, abs=0.05)  # ~1 m/s * 0.2s


def test_three_zone_no_latch_on_depth():
    shield = ThreeZoneSpeedShield()
    limits = body_delta_limits(0.2)
    far_d = float(shield.zone.engage_outer_m + 10.0)
    obs_near = _obs(depth=1.0)
    obs_far = _obs(depth=far_d)
    out_near, changed = shield.apply_action(np.array([1.0, 0, 0, 0]), obs_near, limits=limits)
    assert changed
    assert out_near[0] < 0  # L3 active brake = retreat
    assert "three_zone_brake" in (obs_near.info.get("shield_channels") or [])
    assert not shield.should_override(obs_far)  # no episode latch
    capped, _ = shield.apply_action(np.array([1.0, 0, 0, 0]), obs_far, limits=limits)
    assert capped[0] == pytest.approx(1.0)


def test_l3_active_brake_no_latch_mid_zone_still_caps():
    """BA: d̂=3 → forward cap only; d̂=1 → −x; leaving L3 clears brake."""
    shield = ThreeZoneSpeedShield()
    limits = body_delta_limits(0.2)
    mid = _obs(depth=3.0)
    out_mid, ch_mid = shield.apply_action(np.array([1.0, 0, 0, 0]), mid, limits=limits)
    assert ch_mid
    assert out_mid[0] > 0
    assert "three_zone_brake" not in (mid.info.get("shield_channels") or [])
    near = _obs(depth=1.0)
    out_near, _ = shield.apply_action(np.array([1.0, 0, 0, 0]), near, limits=limits)
    assert out_near[0] < 0
    far = _obs(depth=float(shield.zone.engage_outer_m + 10.0))
    out_far, _ = shield.apply_action(np.array([1.0, 0, 0, 0]), far, limits=limits)
    assert out_far[0] == pytest.approx(1.0)
    assert not shield.should_override(far)


def test_three_zone_tau_emergency_latches():
    shield = ThreeZoneSpeedShield(min_tau_s=2.0)
    limits = body_delta_limits(0.2)
    far_d = float(shield.zone.engage_outer_m + 10.0)
    obs = _obs(depth=far_d, info={"tau_pred": 0.5})
    _, changed = shield.apply_action(np.zeros(4), obs, limits=limits)
    assert changed
    assert shield.last_channels == ("tau",)
    assert shield.should_override(_obs(depth=far_d))


def test_p_coll_emergency_vetoed_when_forward_clear():
    """False WM p_coll must not body-retreat when forward cone is clear (path-follow)."""
    from types import SimpleNamespace

    shield = ThreeZoneSpeedShield(max_p_coll=0.5)
    limits = body_delta_limits(0.2)
    # Side clutter can sink full-min while forward corridor is open (125 probe).
    obs = _obs(
        depth=5.5,
        info={
            "depth_cones_pred": {
                "forward": 22.0,
                "left": 5.0,
                "right": 5.0,
                "up": 20.0,
                "down": 20.0,
            }
        },
    )
    wm = SimpleNamespace(p_coll=0.55)
    out, changed = shield.apply_action(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), obs, wm_out=wm, limits=limits
    )
    assert not shield._emergency_engaged
    assert out[0] > 0.0
    # May still speed-cap, but must not flip to retreat.
    assert "p_coll" not in (obs.info.get("shield_channels") or [])


def test_p_coll_emergency_latches_when_forward_near():
    from types import SimpleNamespace

    shield = ThreeZoneSpeedShield(max_p_coll=0.5)
    limits = body_delta_limits(0.2)
    obs = _obs(
        depth=3.0,
        info={
            "depth_cones_pred": {
                "forward": 3.0,
                "left": 20.0,
                "right": 20.0,
                "up": 20.0,
                "down": 20.0,
            }
        },
    )
    wm = SimpleNamespace(p_coll=0.55)
    out, changed = shield.apply_action(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), obs, wm_out=wm, limits=limits
    )
    assert changed
    assert shield._emergency_engaged
    assert out[0] < 0.0
    assert "p_coll" in (obs.info.get("shield_channels") or [])


def test_collector_three_zone_intervention():
    from experiments.aerial.rl.buffer import ReplayBuffer
    from experiments.aerial.rl.collector import RolloutCollector

    class _Env:
        def __init__(self):
            self.config = type("C", (), {"step_hz": 5.0})()
            self.goal = np.array([10.0, 0.0, 0.0])

        def reset(self, episode=None):
            return _obs(depth=10.0)

        def step(self, action):
            return _obs(depth=10.0), {}

    class _Pred:
        def reset(self):
            pass

        def predict_min(self, obs):
            return 4.0  # inside L2 → cap < max fwd

        def predict_cones(self, obs):
            return {
                "forward": 4.0,
                "left": 20.0,
                "right": 20.0,
                "up": 20.0,
                "down": 20.0,
            }

    class _Policy:
        def act(self, view):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    col = RolloutCollector(
        _Env(),
        _Policy(),
        ReplayBuffer(capacity_episodes=1, seed=0),
        safety=ThreeZoneSpeedShield(),
        depth_predictor=_Pred(),
        max_steps=2,
        target_hz=0.0,
        skip_reset_collision=False,
    )
    ep, stats = col.collect_episode()
    assert stats.interventions >= 1
    assert ep[0].obs.info.get("depth_cones_pred") is not None
    assert ep[0].obs.info.get("three_zone_speed_cap_m_s") is not None


def test_p0b_forward_uses_min_of_cone_and_fullmin():
    """Side obstacle lowers full-min while forward crop stays far → still brake."""
    shield = ThreeZoneSpeedShield()
    limits = body_delta_limits(0.2)
    action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    fullmin_near = _obs(depth=1.0)
    out_full, _ = shield.apply_action(action.copy(), fullmin_near, limits=limits)
    shield.reset()
    mixed = _obs(
        depth=99.0,
        info={
            "depth_min_pred": 1.0,
            "depth_cones_pred": {
                "forward": 20.0,
                "left": 20.0,
                "right": 20.0,
                "up": 20.0,
                "down": 20.0,
            },
        },
    )
    out_mixed, changed = shield.apply_action(action.copy(), mixed, limits=limits)
    assert changed
    assert out_mixed[0] < 0
    assert out_mixed[0] == pytest.approx(out_full[0], abs=1e-6)
    assert "three_zone_brake" in (mixed.info.get("shield_channels") or [])


def test_p0b_forward_cone_matches_fullmin_when_equal():
    """When cones['forward'] == full-min, +x cap matches legacy full-min path."""
    shield = ThreeZoneSpeedShield()
    limits = body_delta_limits(0.2)
    action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    legacy = _obs(depth=5.0)
    out_legacy, ch_legacy = shield.apply_action(action, legacy, limits=limits)
    shield.reset()
    cones = _obs(
        depth=99.0,  # full-min decoy — shield must ignore when cones present
        info={
            "depth_min_pred": 99.0,
            "depth_cones_pred": {
                "forward": 5.0,
                "left": 20.0,
                "right": 20.0,
                "up": 20.0,
                "down": 20.0,
            },
        },
    )
    # _obs merges depth into depth_min_pred; override after construct
    cones.info["depth_min_pred"] = 99.0
    cones.info["depth_cones_pred"] = {
        "forward": 5.0,
        "left": 20.0,
        "right": 20.0,
        "up": 20.0,
        "down": 20.0,
    }
    out_cones, ch_cones = shield.apply_action(action.copy(), cones, limits=limits)
    assert ch_legacy and ch_cones
    assert out_cones[0] == pytest.approx(out_legacy[0], abs=1e-6)


def test_p0b_lateral_clamp_when_left_near():
    """Left cone ≤ L2 clamps +y (body left); forward may stay uncapped if far."""
    shield = ThreeZoneSpeedShield()
    limits = body_delta_limits(0.2)
    # v2=1 m/s → max |dy| = 0.2 m at dt=0.2
    action = np.array([0.05, 0.5, 0.0, 0.0], dtype=np.float64)
    obs = _obs(
        depth=20.0,
        info={
            "depth_cones_pred": {
                "forward": 20.0,
                "left": 3.0,  # ≤ L2=5
                "right": 20.0,
                "up": 20.0,
                "down": 20.0,
            },
        },
    )
    out, changed = shield.apply_action(action, obs, limits=limits)
    assert changed
    assert out[1] == pytest.approx(0.2, abs=1e-6)
    assert out[0] == pytest.approx(0.05, abs=1e-6)
    assert "three_zone_lat" in (obs.info.get("shield_channels") or [])
    assert "left" in (obs.info.get("three_zone_lat_axes") or [])


def test_l3_3d_norm_clamp_after_brake():
    """BB B1: L3 −x then ‖Δ‖₂ ≤ v_stop·dt (=0.04 m at 5 Hz)."""
    shield = ThreeZoneSpeedShield()
    limits = body_delta_limits(0.2)
    obs = _obs(depth=1.0, v_fwd=5.0)
    out, changed = shield.apply_action(np.array([1.0, 0, 0, 0]), obs, limits=limits)
    assert changed
    assert out[0] < 0
    max_norm = 0.2 * 0.2  # v_stop * dt
    assert float(np.linalg.norm(out)) <= max_norm + 1e-6
    assert "three_zone_3d" in (obs.info.get("shield_channels") or [])
    assert "three_zone_brake" in (obs.info.get("shield_channels") or [])


def test_l3_active_lateral_clamp_both_directions():
    """BB B2: left cone ≤ L2 clamps |dy| on L3 path (direct helper)."""
    shield = ThreeZoneSpeedShield()
    limits = body_delta_limits(0.2)
    max_axis = 0.2 * 0.2
    obs = _obs(
        depth=1.0,
        info={
            "depth_cones_pred": {
                "forward": 1.0,
                "left": 3.0,
                "right": 20.0,
                "up": 20.0,
                "down": 20.0,
            },
        },
    )
    action = np.array([-0.04, 0.3, 0.0, 0.0], dtype=np.float64)
    out, lat_ch = shield._cap_lateral_l3_active(action, obs, limits)
    assert lat_ch
    assert abs(out[1]) <= max_axis + 1e-6
    assert "three_zone_lat" in (obs.info.get("shield_channels") or [])


def test_l3_3d_skipped_outside_l3():
    """d̂=3 → forward cap only; no B1/B2 channels."""
    shield = ThreeZoneSpeedShield()
    limits = body_delta_limits(0.2)
    obs = _obs(depth=3.0)
    out, changed = shield.apply_action(np.array([1.0, 0.5, 0.0, 0.0]), obs, limits=limits)
    assert changed
    assert out[0] > 0
    ch = obs.info.get("shield_channels") or []
    assert "three_zone_brake" not in ch
    assert "three_zone_3d" not in ch


def test_tti_forward_hysteresis_holds_cap_near_boundary():
    shield = ThreeZoneSpeedShield(tti_coeff=4.0, tti_hysteresis_release_frac=0.15)
    limits = body_delta_limits(0.2)
    action = np.array([1.2, 0.0, 0.0, 0.0], dtype=np.float64)
    # v_ref≈6 → trigger=24m; release=27.6m
    obs_cap = _obs(depth=20.0, v_fwd=6.0, info={"depth_cones_pred": {"forward": 20.0}})
    out1, ch1 = shield.apply_action(action.copy(), obs_cap, limits=limits)
    assert ch1
    assert out1[0] < 1.2
    obs_mid = _obs(depth=23.0, v_fwd=6.0, info={"depth_cones_pred": {"forward": 23.0}})
    out2, ch2 = shield.apply_action(action.copy(), obs_mid, limits=limits)
    assert ch2
    assert out2[0] < 1.2
    obs_clear = _obs(depth=28.0, v_fwd=6.0, info={"depth_cones_pred": {"forward": 28.0}})
    out3, ch3 = shield.apply_action(action.copy(), obs_clear, limits=limits)
    assert not ch3
    assert out3[0] == pytest.approx(min(1.2, limits[0]))


def test_exclusion_forward_only_ignores_peripheral_full_fov_min():
    """Peripheral full-FOV min must not trigger exclusion when forward cone is open."""
    limits = body_delta_limits(0.2)
    obs = _obs(
        depth=2.0,
        v_fwd=5.0,
        info={
            "depth_min_pred": 2.0,
            "depth_cones_pred": {
                "forward": 8.0,
                "left": 2.0,
                "right": 20.0,
                "up": 20.0,
                "down": 20.0,
            },
        },
    )
    shield_full = ThreeZoneSpeedShield(exclusion_forward_only=False)
    out_full, ch_full = shield_full.apply_action(np.array([1.0, 0, 0, 0]), obs, limits=limits)
    assert ch_full
    assert out_full[0] < 0

    shield_fwd = ThreeZoneSpeedShield(exclusion_forward_only=True)
    out_fwd, ch_fwd = shield_fwd.apply_action(np.array([1.0, 0, 0, 0]), obs, limits=limits)
    assert not ch_fwd or out_fwd[0] >= 0


def test_emergency_retreat_is_speed_only_no_heading_decision():
    """Shield contract: clamp forward speed; preserve policy yaw/lateral.

    Clearer-cone retreat dyaw was a pilot decision (2026-09-18), then caused
    reverse-flight spins (2026-09-21). Default must not invent heading."""
    limits = body_delta_limits(0.2)

    obs_left_clear = _obs(
        depth=2.0,
        info={"depth_cones_pred": {"forward": 2.0, "left": 20.0, "right": 3.0, "up": 20.0, "down": 20.0}},
    )
    shield = ThreeZoneSpeedShield()
    assert shield.retreat_max_dyaw_rad == pytest.approx(0.0)
    out, changed = shield.apply_action(np.array([1.0, 0.2, -0.1, 0.25]), obs_left_clear, limits=limits)
    assert changed
    assert out[0] <= 0.0  # no forward into danger
    assert out[1] == pytest.approx(0.2, abs=1e-6)  # preserve lateral
    assert out[2] == pytest.approx(-0.1, abs=1e-6)
    assert out[3] == pytest.approx(0.25, abs=1e-6)  # preserve policy yaw
    assert obs_left_clear.info.get("shield_hard_brake") is True

    obs_right_clear = _obs(
        depth=2.0,
        info={"depth_cones_pred": {"forward": 2.0, "left": 3.0, "right": 20.0, "up": 20.0, "down": 20.0}},
    )
    out2, changed2 = ThreeZoneSpeedShield().apply_action(
        np.array([1.0, 0.0, 0.0, -0.25]), obs_right_clear, limits=limits
    )
    assert changed2 and out2[0] <= 0.0 and out2[3] == pytest.approx(-0.25, abs=1e-6)
    # Exclusion hard brake must not pretend to be the τ/p_coll emergency latch.
    assert obs_right_clear.info.get("shield_hard_brake") is True
    assert obs_right_clear.info.get("shield_emergency_override") is not True


def test_shield_terminal_soft_exclusion_creeps_when_aligned():
    """Near goal + aligned yaw: exclusion creeps instead of full hard-brake."""
    limits = body_delta_limits(0.2)
    shield = ThreeZoneSpeedShield()
    obs = _obs(
        depth=2.5,
        info={
            "depth_cones_pred": {
                "forward": 2.5,
                "left": 10.0,
                "right": 10.0,
                "up": 20.0,
                "down": 20.0,
            },
            "d_to_g": 12.0,
            "yaw_err_rad": 0.1,
        },
    )
    out, changed = shield.apply_action(np.array([1.0, 0.0, 0.0, 0.0]), obs, limits=limits)
    assert changed
    assert out[0] > 0.0
    assert out[0] <= shield.terminal_soft_exclusion_dx + 1e-6
    assert obs.info.get("shield_hard_brake") is False
    assert obs.info.get("shield_terminal_soft_exclusion") is True
    assert obs.info.get("shield_governor_cap") is True


def test_shield_terminal_still_hard_brakes_when_crash_imminent():
    limits = body_delta_limits(0.2)
    shield = ThreeZoneSpeedShield()
    obs = _obs(
        depth=1.0,
        info={
            "depth_cones_pred": {"forward": 1.0, "left": 10.0, "right": 10.0},
            "d_to_g": 12.0,
            "yaw_err_rad": 0.1,
        },
    )
    out, changed = shield.apply_action(np.array([1.0, 0.0, 0.0, 0.0]), obs, limits=limits)
    assert changed and out[0] <= 0.0
    assert obs.info.get("shield_hard_brake") is True


def test_shield_hard_brake_cleared_when_sky_opens():
    """Sticky hard_brake must not survive a clear frame (reward contamination).

    Collector carries last-step flags onto the next obs; apply_action must
    reset before recomputing so open-sky steps do not keep charging
    w_intervention.
    """
    limits = body_delta_limits(0.2)
    shield = ThreeZoneSpeedShield()
    obs = _obs(
        depth=2.0,
        info={
            "depth_cones_pred": {
                "forward": 2.0,
                "left": 20.0,
                "right": 3.0,
                "up": 20.0,
                "down": 20.0,
            },
            "shield_hard_brake": True,  # stale carry from prior step
            "shield_governor_cap": True,
            "shield_emergency_override": True,
        },
    )
    out, changed = shield.apply_action(np.array([1.0, 0.0, 0.0, 0.1]), obs, limits=limits)
    assert changed and out[0] <= 0.0
    assert obs.info.get("shield_hard_brake") is True  # still in exclusion

    obs_clear = _obs(
        depth=40.0,
        info={
            "depth_cones_pred": {
                "forward": 40.0,
                "left": 40.0,
                "right": 40.0,
                "up": 40.0,
                "down": 40.0,
            },
            "shield_hard_brake": True,
            "shield_governor_cap": True,
            "shield_emergency_override": True,
        },
    )
    out2, changed2 = shield.apply_action(
        np.array([1.0, 0.2, 0.0, 0.1]), obs_clear, limits=limits
    )
    assert not changed2
    assert out2[0] == pytest.approx(1.0, abs=1e-6)
    assert obs_clear.info.get("shield_hard_brake") is False
    assert obs_clear.info.get("shield_governor_cap") is False
    assert obs_clear.info.get("shield_emergency_override") is False


def test_retreat_dyaw_opt_in_ablation_still_gated():
    """Explicit retreat_max_dyaw_rad>0 keeps clearer-cone ablation + cum gate."""
    limits = body_delta_limits(0.2)
    cones = {"forward": 2.0, "left": 3.0, "right": 20.0, "up": 20.0, "down": 20.0}
    shield = ThreeZoneSpeedShield(retreat_max_dyaw_rad=0.35, retreat_max_cum_yaw_rad=0.70)
    dyaws = []
    for _ in range(8):
        obs = _obs(depth=2.0, info={"depth_cones_pred": cones, "goal": [50.0, 0.0, 0.0]})
        out, changed = shield.apply_action(np.array([1.0, 0.0, 0.0, 0.0]), obs, limits=limits)
        assert changed and out[0] <= 0.0  # forward cancelled / braked
        dyaws.append(float(out[3]))
    assert dyaws[0] < 0
    assert abs(sum(dyaws)) <= 0.70 + 1e-6
    assert any(abs(d) < 1e-9 for d in dyaws[2:])
