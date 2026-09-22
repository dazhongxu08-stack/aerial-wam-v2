import numpy as np
import pytest

from experiments.aerial.rl.dynamics import StubLatentDynamics, DynamicsOutput
from experiments.aerial.rl.imagination import (
    MAX_IMAGINATION_HORIZON,
    imagine,
)
from experiments.aerial.rl.env.obs import Observation


class _ForwardPolicy:
    """Always drive +3 m forward in body frame."""

    def act_latent(self, z):
        return np.array([3.0, 0.0, 0.0, 0.0])


def _obs(pos, yaw=0.0):
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, yaw], dtype=np.float32)
    return Observation(rgb=np.zeros((4, 4, 3), np.uint8), state=state)


def test_encode_carries_proprio():
    dyn = StubLatentDynamics(latent_dim=8)
    z = dyn.encode(_obs([1.0, 2.0, 3.0], yaw=0.5))
    assert z.shape == (8,)
    assert np.allclose(z[:4], [1.0, 2.0, 3.0, 0.5])


def test_step_integrates_forward():
    dyn = StubLatentDynamics(goal=np.array([100.0, 0.0, 0.0]), latent_dim=8)
    z = dyn.encode(_obs([0.0, 0.0, 0.0]))
    out = dyn.step(z, np.array([3.0, 0.0, 0.0, 0.0]))
    assert isinstance(out, DynamicsOutput)
    assert np.allclose(out.z_next[:3], [3.0, 0.0, 0.0])
    assert out.progress == pytest.approx(3.0)  # moved 3m closer to goal


def test_p_coll_rises_near_obstacle():
    dyn = StubLatentDynamics(obstacle=np.array([3.0, 0.0, 0.0]), latent_dim=8, collide_radius_m=2.0)
    z = dyn.encode(_obs([0.0, 0.0, 0.0]))
    out = dyn.step(z, np.array([3.0, 0.0, 0.0, 0.0]))  # lands exactly on obstacle
    assert out.p_coll == pytest.approx(1.0)
    assert out.done


def test_imagine_shapes_and_returns():
    dyn = StubLatentDynamics(goal=np.array([100.0, 0.0, 0.0]), latent_dim=8)
    z0 = np.stack([dyn.encode(_obs([0.0, 0.0, 0.0])) for _ in range(5)])
    horizon = 6
    roll = imagine(dyn, _ForwardPolicy(), z0, horizon)
    assert roll.z.shape == (5, horizon + 1, 8)
    assert roll.actions.shape == (5, horizon, 4)
    assert roll.rewards.shape == (5, horizon)
    assert roll.p_coll.shape == (5, horizon)
    assert roll.returns.shape == (5,)
    # forward toward goal -> positive returns
    assert np.all(roll.returns > 0)


def test_imagine_done_masking_stops_accrual():
    dyn = StubLatentDynamics(obstacle=np.array([3.0, 0.0, 0.0]), latent_dim=8, collide_radius_m=2.0)
    z0 = dyn.encode(_obs([0.0, 0.0, 0.0]))[None, :]
    roll = imagine(dyn, _ForwardPolicy(), z0, horizon=5)
    # first step lands on obstacle -> done at t=0, all subsequent done
    assert roll.done[0, 0]
    assert np.all(roll.done[0])


def test_imagine_horizon_cap():
    dyn = StubLatentDynamics(latent_dim=8)
    z0 = dyn.encode(_obs([0.0, 0.0, 0.0]))[None, :]
    with pytest.raises(ValueError):
        imagine(dyn, _ForwardPolicy(), z0, horizon=MAX_IMAGINATION_HORIZON + 1)


def test_imagine_applies_f15_efficiency_when_weighted():
    """Imagination must subtract F15 efficiency (else H100 short-train is a no-op)."""
    from experiments.aerial.rl.reward import RewardConfig

    class _Fixed:
        def __init__(self, a):
            self._a = np.asarray(a, dtype=np.float64)

        def act_latent(self, z, goal_rel=None):
            return self._a.copy()

    dyn = StubLatentDynamics(latent_dim=8)
    z0 = dyn.encode(_obs([0.0, 0.0, 0.0]))[None, :]
    # Carrot 90° left → large body yaw error while thrusting forward.
    g_left = np.array([[0.0, 8.0, 0.0, 8.0]], dtype=np.float32)
    v = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    cfg0 = RewardConfig(w_progress=0.0, w_collision=0.0, w_maneuver=0.0)
    cfg1 = RewardConfig(
        w_progress=0.0, w_collision=0.0, w_maneuver=0.0, w_eff_heading=1.0,
    )
    r0 = imagine(
        dyn, _Fixed(a), z0, horizon=1,
        reward_cfg=cfg0, goal_rel0=g_left, body_vel0=v,
    ).rewards[0, 0]
    r1 = imagine(
        dyn, _Fixed(a), z0, horizon=1,
        reward_cfg=cfg1, goal_rel0=g_left, body_vel0=v,
    ).rewards[0, 0]
    assert r0 == pytest.approx(0.0, abs=1e-5)
    assert r1 < -0.5  # ≈ −π/2 heading cost


def test_dynamics_update_is_v1_gated():
    dyn = StubLatentDynamics(latent_dim=8)
    result = dyn.update(windows=[])
    assert result["skipped"] is True


def test_imagine_applies_w_intervention_via_hard_brake_band():
    """w_intervention imag proxy only when d_fwd_hat ≤ d_danger (hard brake)."""
    from experiments.aerial.rl.reward import RewardConfig

    class _NearDyn(StubLatentDynamics):
        def __init__(self, d_fwd, **kw):
            super().__init__(**kw)
            self._d = d_fwd

        def step(self, z, action, goal_rel=None, body_vel=None):
            out = super().step(z, action, goal_rel=goal_rel, body_vel=body_vel)
            return DynamicsOutput(
                z_next=out.z_next,
                p_coll=0.0,
                progress=out.progress,
                done=out.done,
                arrived=out.arrived,
                d_fwd_hat=self._d,
            )

    class _ZeroAct:
        def act_latent(self, z, goal_rel=None):
            return np.zeros(4, dtype=np.float64)

    z0 = StubLatentDynamics(latent_dim=8).encode(_obs([0.0, 0.0, 0.0]))[None, :]
    cfg0 = RewardConfig(w_progress=0.0, w_collision=0.0, w_maneuver=0.0, w_intervention=0.0)
    cfg1 = RewardConfig(w_progress=0.0, w_collision=0.0, w_maneuver=0.0, w_intervention=0.1)
    # Soft zone (5 m): no imag intervention charge
    r_soft = imagine(_NearDyn(5.0, latent_dim=8), _ZeroAct(), z0, horizon=1, reward_cfg=cfg1).rewards[0, 0]
    assert r_soft == pytest.approx(0.0, abs=1e-6)
    # Hard band (2 m): binary intervention charge
    r0 = imagine(_NearDyn(2.0, latent_dim=8), _ZeroAct(), z0, horizon=1, reward_cfg=cfg0).rewards[0, 0]
    r1 = imagine(_NearDyn(2.0, latent_dim=8), _ZeroAct(), z0, horizon=1, reward_cfg=cfg1).rewards[0, 0]
    assert r0 == pytest.approx(0.0, abs=1e-6)
    assert r1 == pytest.approx(-0.1, abs=1e-5)
    # Custom hard_brake_depth_m must move the imag charge band.
    cfg_tight = RewardConfig(
        w_progress=0.0, w_collision=0.0, w_maneuver=0.0,
        w_intervention=0.1, hard_brake_depth_m=1.5,
    )
    r_above = imagine(
        _NearDyn(2.0, latent_dim=8), _ZeroAct(), z0, horizon=1, reward_cfg=cfg_tight
    ).rewards[0, 0]
    r_below = imagine(
        _NearDyn(1.0, latent_dim=8), _ZeroAct(), z0, horizon=1, reward_cfg=cfg_tight
    ).rewards[0, 0]
    assert r_above == pytest.approx(0.0, abs=1e-6)
    assert r_below == pytest.approx(-0.1, abs=1e-5)
