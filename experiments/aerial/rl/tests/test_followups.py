"""Tests for the review's follow-up (skeleton-honesty) items:

  A. RGB-only policy boundary is enforced by the type, not just documented.
  B. ``dynamics.kind`` is honored (``wan`` is rejected, not silently stubbed).
  C. Flipping the V1 gate ON with a stub dynamics reports ``noop``, not
     ``updated``.
  D. The online arrival radius comes from YAML and is tighter than the eval SR
     metric radius.
  E. Imagined reward mirrors the real reward (goal wired in, arrival bonus).
  + the safety shield's ``p_coll`` / depth trigger path (previously untested).
"""
import types

import numpy as np
import pytest

from experiments.aerial.rl.buffer import ReplayBuffer, Transition
from experiments.aerial.rl.collector import act_delta
from experiments.aerial.rl.corrector import CorrectorConfig, SerialCorrectorLoop
from experiments.aerial.rl.dynamics import DynamicsOutput, StubLatentDynamics
from experiments.aerial.rl.env.obs import Observation, PolicyObservation
from experiments.aerial.rl.imagination import imagine
from experiments.aerial.rl.reward import RewardConfig
from experiments.aerial.rl.safety import ThresholdSafetyShield
from experiments.aerial.rl.train_rl import _build_dynamics, build_from_config


def _obs(pos=(0.0, 0.0, 0.0), yaw=0.0, collided=False, info=None):
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, yaw], dtype=np.float32)
    return Observation(
        rgb=np.zeros((4, 4, 3), np.uint8),
        state=state,
        collided=collided,
        depth=np.ones((4, 4), np.float32),
        imu={"ang_vel": [0, 0, 0], "lin_acc": [0, 0, 9.807]},
        info=info or {},
    )


class _ForwardPolicy:
    def __init__(self, step=3.0):
        self.step = float(step)

    def act_latent(self, z):
        return np.array([self.step, 0.0, 0.0, 0.0])


# -- A: RGB-only boundary ------------------------------------------------
def test_policy_view_omits_privileged_fields():
    obs = _obs(pos=(1.0, 2.0, 3.0), yaw=0.5)
    view = obs.policy_view()
    assert isinstance(view, PolicyObservation)
    assert np.allclose(view.position, [1.0, 2.0, 3.0])
    assert view.yaw == pytest.approx(0.5)
    # The privileged / supervision fields must be structurally absent.
    for attr in ("depth", "imu", "collided", "velocity", "state"):
        assert not hasattr(view, attr), f"policy view leaked {attr}"


def test_act_delta_blocks_privileged_leak():
    class _LeakPolicy:
        def act(self, view):
            return view.depth  # not a policy-visible field -> AttributeError

    with pytest.raises(AttributeError):
        act_delta(_LeakPolicy(), _obs(), "instr")


def test_act_delta_passes_view_not_full_obs():
    seen = {}

    class _Spy:
        def act(self, view):
            seen["type"] = type(view)
            return np.zeros(4)

    act_delta(_Spy(), _obs(), "instr")
    assert seen["type"] is PolicyObservation


# -- B: dynamics.kind dispatch ------------------------------------------
def test_build_dynamics_stub_propagates_success_dist():
    dyn = _build_dynamics({"kind": "stub", "latent_dim": 8}, success_dist_m=2.5)
    assert isinstance(dyn, StubLatentDynamics)
    assert dyn._success_dist == pytest.approx(2.5)


def test_build_dynamics_wan_is_rejected_online():
    with pytest.raises(ValueError, match="(?i)offline"):
        _build_dynamics({"kind": "wan"}, success_dist_m=3.0)


def test_build_dynamics_unknown_rejected():
    with pytest.raises(ValueError, match="expected stub"):
        _build_dynamics({"kind": "bogus"}, success_dist_m=3.0)


# -- C: honest gate status ----------------------------------------------
def _prefill(buffer, n=4):
    obs = _obs()
    txns = [
        Transition(obs=obs, action=np.zeros(4), reward=0.0, done=False, next_obs=obs)
        for _ in range(n)
    ]
    buffer.add_episode(txns)


def test_wm_gate_on_with_stub_reports_noop_not_updated():
    buf = ReplayBuffer()
    _prefill(buf, n=4)
    loop = SerialCorrectorLoop(
        collector=types.SimpleNamespace(env=None, reward_cfg=RewardConfig()),
        buffer=buf,
        dynamics=StubLatentDynamics(latent_dim=8),
        config=CorrectorConfig(enable_wm_update=True, wm_batch=1, wm_window=2),
    )
    wm = loop._update_world_model()
    assert wm["status"] == "noop"          # NOT "updated"
    assert wm["skipped"] is True


# -- D: online success radius from YAML ---------------------------------
def _mock_cfg(**reward):
    return {
        "env": {"backend": "mock", "step_hz": 30.0},
        "reward": reward,
        "dynamics": {"kind": "stub", "latent_dim": 8},
    }


def test_success_dist_m_flows_from_yaml_to_reward_and_dynamics():
    loop = build_from_config(_mock_cfg(success_dist_m=2.5))
    assert loop.collector.reward_cfg.success_dist_m == pytest.approx(2.5)
    assert loop.dynamics._success_dist == pytest.approx(2.5)


def test_w_intervention_flows_from_yaml_to_reward_cfg():
    """CLI/yaml --w-intervention must not be dropped by build_from_config."""
    loop = build_from_config(_mock_cfg(w_intervention=0.1))
    assert loop.collector.reward_cfg.w_intervention == pytest.approx(0.1)
    # Default stays no-op when omitted.
    loop0 = build_from_config(_mock_cfg())
    assert loop0.collector.reward_cfg.w_intervention == pytest.approx(0.0)


def test_hard_brake_depth_m_defaults_to_safety_exclusion_m():
    cfg = {
        "env": {"backend": "mock", "step_hz": 30.0},
        "reward": {"w_intervention": 0.1},
        "safety": {"exclusion_m": 2.5},
        "dynamics": {"kind": "stub", "latent_dim": 8},
    }
    loop = build_from_config(cfg)
    assert loop.collector.reward_cfg.hard_brake_depth_m == pytest.approx(2.5)
    # Explicit reward override wins over safety.exclusion_m.
    cfg["reward"]["hard_brake_depth_m"] = 1.25
    loop2 = build_from_config(cfg)
    assert loop2.collector.reward_cfg.hard_brake_depth_m == pytest.approx(1.25)


def test_success_dist_m_default_is_tight_not_eval_radius():
    # Omitting it must fall back to the tight online default (3 m), never the
    # loose 20 m eval SR radius.
    loop = build_from_config(_mock_cfg())
    assert loop.collector.reward_cfg.success_dist_m == pytest.approx(3.0)


# -- E: imagined reward mirrors real ------------------------------------
def test_stub_arrival_sets_done_and_arrived():
    dyn = StubLatentDynamics(goal=np.array([0.3, 0.0, 0.0]), latent_dim=8, success_dist_m=1.0)
    z = dyn.encode(_obs())
    out = dyn.step(z, np.array([0.3, 0.0, 0.0, 0.0]))  # lands on goal
    assert out.arrived
    assert out.done


def test_imagine_adds_success_bonus_on_arrival():
    dyn = StubLatentDynamics(goal=np.array([0.3, 0.0, 0.0]), latent_dim=8, success_dist_m=1.0)
    z0 = dyn.encode(_obs())[None, :]
    cfg = RewardConfig(w_progress=1.0, w_maneuver=0.0, w_collision=0.0, success_bonus=10.0)
    roll = imagine(dyn, _ForwardPolicy(0.3), z0, horizon=3, reward_cfg=cfg)
    # step 0: progress 0.3 + success bonus 10 = 10.3; arrival terminates.
    assert roll.rewards[0, 0] == pytest.approx(0.3 + 10.0)
    assert roll.done[0, 0]


def test_corrector_sets_dynamics_goal_before_imagine():
    buf = ReplayBuffer()
    _prefill(buf, n=4)
    goal = np.array([5.0, 0.0, 0.0])
    loop = SerialCorrectorLoop(
        collector=types.SimpleNamespace(
            env=types.SimpleNamespace(goal=goal), reward_cfg=RewardConfig()
        ),
        buffer=buf,
        dynamics=StubLatentDynamics(latent_dim=8),  # goal=None initially
        imagination_policy=_ForwardPolicy(0.1),
        config=CorrectorConfig(enable_policy_update=True, imagine_batch=2, imagine_horizon=3),
    )
    rl = loop._update_policy()
    assert rl["status"] == "imagined"
    # The goal must have been pushed into the dynamics (else imagined progress≡0).
    assert loop.dynamics._goal is not None
    assert np.allclose(loop.dynamics._goal, goal)


# -- safety shield p_coll / depth trigger (test gap) --------------------
def test_threshold_shield_triggers_on_p_coll():
    # should_override LATCHES per instance, so use a fresh shield per case
    # (a triggered instance stays engaged — see the latch/hold test below).
    hi = DynamicsOutput(z_next=np.zeros(8), p_coll=0.9, progress=0.0, done=False)
    lo = DynamicsOutput(z_next=np.zeros(8), p_coll=0.1, progress=0.0, done=False)
    assert ThresholdSafetyShield(max_p_coll=0.5).should_override(_obs(), hi)
    assert not ThresholdSafetyShield(max_p_coll=0.5).should_override(_obs(), lo)


def test_threshold_shield_triggers_on_predicted_depth():
    # Fresh instance per case: the trigger latches (see latch/hold test below).
    near = _obs(info={"depth_min_pred": 1.0})   # < 1.5 -> brake
    far = _obs(info={"depth_min_pred": 5.0})
    assert ThresholdSafetyShield(min_depth_m=1.5).should_override(near)
    assert not ThresholdSafetyShield(min_depth_m=1.5).should_override(far)


def test_threshold_shield_safe_when_no_predictions():
    # No D̂ / τ / p_coll populated -> never override (safe to install early).
    assert not ThresholdSafetyShield().should_override(_obs())


def test_threshold_shield_bounded_state_feedback_retreat_after_latch():
    # v5: graduated retreat while speed must bleed; hold once clear + slow.
    shield = ThresholdSafetyShield(min_depth_m=3.0, retreat_step_m=3.0)
    assert shield.should_override(_obs(info={"depth_min_pred": 1.0}))  # trip + latch
    act = shield.override_action(_obs(info={"depth_min_pred": 1.0}))
    assert act[0] < 0.0 and np.allclose(act[1:], 0.0), f"expected −x retreat, got {act}"
    assert shield.should_override(_obs(info={"depth_min_pred": 99.0}))
    assert np.allclose(
        shield.override_action(_obs(info={"depth_min_pred": 99.0})), np.zeros(4)
    ), "clear of envelope and ~0 v_fwd → hold"
    shield.reset()
    assert not shield.should_override(_obs(info={"depth_min_pred": 99.0}))


def test_threshold_shield_kinematic_engage_before_standoff():
    """5 m/s @ 5 m: must engage before crossing 3 m standoff (5 < 3+5*1)."""
    state = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs = Observation(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        state=state,
        depth=np.full((8, 8), 5.0, dtype=np.float32),
        info={"depth_min_pred": 5.0},
    )
    shield = ThresholdSafetyShield(min_depth_m=3.0, min_tau_s=1.0)
    assert shield.should_override(obs)
    act = shield.override_action(obs)
    assert act[0] < 0.0, "must brake outside standoff when closing fast"


def test_threshold_shield_no_kinematic_engage_when_slow_and_far():
    state = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    obs = Observation(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        state=state,
        depth=np.full((8, 8), 5.0, dtype=np.float32),
        info={"depth_min_pred": 5.0},
    )
    shield = ThresholdSafetyShield(min_depth_m=3.0, min_tau_s=1.0)
    assert not shield.should_override(obs)
