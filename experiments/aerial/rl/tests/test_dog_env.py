from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.aerial.rl.collect_dog_consequence import (
    DOG_MACROS,
    route_point_ahead,
    sample_dog_macro,
)
from experiments.aerial.rl.env.airsim_dog_env import (
    AirSimDogEnv,
    AirSimDogEnvConfig,
    ground_drop_from_forward,
    ground_from_probe,
    mask_ground,
)
from experiments.aerial.rl.env.dog_dynamics import (
    MODEL_LIMITS,
    DogDynamics,
    DogDynamicsParams,
    DogDynamicsRanges,
    model_action_to_cmd,
)
from experiments.aerial.scripts.make_dog_airsim_settings import dog_settings


def test_yaw_has_delay_deadband_gain_and_lag():
    p = DogDynamicsParams(yaw_delay_s=0.2, yaw_deadband=0.2, yaw_gain=0.25, yaw_tau_s=0.3)
    d = DogDynamics(p)
    assert d.advance((0.0, 0.4), 0.2)[1] == pytest.approx(0.0, abs=1e-9)
    rates = [d.advance((0.0, 0.4), 0.2)[1] for _ in range(20)]
    assert 0.0 < rates[0] < rates[2] < 0.1
    assert rates[-1] == pytest.approx(0.1, rel=0.02)
    d.reset()
    assert max(abs(d.advance((0.0, 0.15), 0.2)[1]) for _ in range(20)) == 0.0
    d.reset()
    for _ in range(20):
        d.advance((0.0, 0.4), 0.2)
    d.advance((0.0, 0.0), 0.2)
    assert d.advance((0.0, 0.0), 0.2)[1] > 0.0


def test_forward_never_reverses_and_saturates():
    d = DogDynamics(DogDynamicsParams(vx_max=0.35, vx_gain=1.0, vx_deadband=0.0))
    vx = [d.advance((1.0, 0.0), 0.5)[0] for _ in range(20)]
    assert vx[-1] == pytest.approx(0.35, rel=0.02)
    d.reset()
    assert d.advance((-1.0, 0.0), 1.0)[0] == 0.0


def test_model_action_maps_to_dog_command_box():
    p = DogDynamicsParams(vx_max=0.35, vyaw_max=0.5)
    assert model_action_to_cmd(np.array([1.0, 0.4, 0.4, MODEL_LIMITS[3]]), p) == pytest.approx((0.35, 0.5))
    assert model_action_to_cmd(np.array([0.5, 0, 0, -MODEL_LIMITS[3] / 2]), p) == pytest.approx((0.175, -0.25))
    assert model_action_to_cmd(np.array([-1.0, 0, 0, 9.0]), p) == pytest.approx((0.0, 0.5))


def test_ranges_sample_inside_bounds():
    rng = np.random.default_rng(0)
    r = DogDynamicsRanges()
    for _ in range(50):
        p = r.sample(rng)
        assert r.yaw_gain[0] <= p.yaw_gain <= r.yaw_gain[1]
        assert r.vx_tau_s[0] <= p.vx_tau_s <= r.vx_tau_s[1]
        assert p.vx_max == DogDynamicsParams().vx_max


def test_dog_macros_are_planar_forward_only():
    rng = np.random.default_rng(3)
    seen = set()
    for _ in range(300):
        name, a = sample_dog_macro(rng, 8, float(rng.uniform(-math.pi, math.pi)), rad_per_step=0.06)
        seen.add(name)
        assert np.all(a[:, 0] >= 0.0) and np.all(a[:, 1:3] == 0.0)
        assert np.all(np.abs(a) <= MODEL_LIMITS + 1e-9)
    assert seen == set(DOG_MACROS)
    _, a = sample_dog_macro(rng, 10, -0.3, rad_per_step=0.06, name="goal_turn_fwd")
    assert np.sum(a[:, 3] < 0) == 5 and a[-1, 0] >= 0.4


def test_route_point_ahead_walks_polyline():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    p, h = route_point_ahead(xy, 0, 15.0)
    assert p == pytest.approx([10.0, 5.0]) and h == pytest.approx(math.pi / 2)
    p, _ = route_point_ahead(xy, 0, 50.0)
    assert p == pytest.approx([10.0, 10.0])


def _flat_forward_depth(h_cam: float, n: int = 224, hfov: float = 90.0) -> np.ndarray:
    f = 0.5 * n / math.tan(math.radians(hfov) / 2.0)
    cy = 0.5 * (n - 1)
    v = np.arange(n, dtype=np.float64)
    col = np.where(v > cy + 0.5, h_cam * f / np.maximum(v - cy, 1e-6), 100.0)
    return np.repeat(col[:, None], n, axis=1).astype(np.float32)


def test_ground_estimators():
    probe = np.full((224, 224), 40.0, np.float32)
    assert ground_from_probe(probe, 9) == pytest.approx((40.0, 0.0))
    assert ground_drop_from_forward(_flat_forward_depth(0.6), 90.0, 20) == pytest.approx(0.6, rel=0.02)
    wall = _flat_forward_depth(0.6)
    wall[150:, :] = 1.0
    assert math.isnan(ground_drop_from_forward(wall, 90.0, 20))


def test_mask_ground_keeps_obstacles_only():
    from experiments.aerial.rl.depth_geometry import cone_clearances

    flat = _flat_forward_depth(0.6)
    assert cone_clearances(flat)["forward"] < 1.3
    assert cone_clearances(mask_ground(flat, 90.0, 0.6))["forward"] > 50.0
    box = flat.copy()
    box[112:150, 100:124] = 2.0
    assert cone_clearances(mask_ground(box, 90.0, 0.6))["forward"] == pytest.approx(2.0)


def test_settings_move_camera_and_drop_lidar():
    base = {"ClockSpeed": 1.0, "Vehicles": {"drone_1": {
        "Sensors": {"Imu": {"SensorType": 2}, "L1": {"SensorType": 6}},
        "Cameras": {"front_custom": {"Z": 1.0, "CaptureSettings": [
            {"ImageType": 0, "Width": 1920, "Height": 1080}, {"ImageType": 1, "Width": 224, "Height": 224}]}}}}}
    s = dog_settings(base, port=41481, cam_z=-0.25, clock=2.0)
    v = s["Vehicles"]["drone_1"]
    assert s["ApiServerPort"] == 41481 and s["ClockSpeed"] == 2.0
    caps = v["Cameras"]["front_custom"]["CaptureSettings"]
    assert (caps[0]["Width"], caps[0]["Height"]) == (640, 360) and caps[1]["Width"] == 224
    assert v["Cameras"]["front_custom"]["Z"] == -0.25 and v["Cameras"]["front_custom"]["X"] == 0.35
    assert list(v["Sensors"]) == ["Imu"]
    assert base["Vehicles"]["drone_1"]["Cameras"]["front_custom"]["Z"] == 1.0


class _FakeWorld:
    """Flat street at ``ground`` (+up) with a wall at x = ``wall_x``; kinematic vehicle."""

    def __init__(self, ground: float = 2.0, wall_x: float = 1.0e9) -> None:
        self.ground, self.wall_x = ground, wall_x
        self.pos = np.zeros(3)  # +up
        self.yaw = 0.0
        self.pitch = 0.0
        self.cam_down = -0.15
        self.v = np.zeros(3)
        self.rate = 0.0
        self.t = time.perf_counter()
        self.stamp = 7
        self.hit = False
        self.moves = []
        self.poses = []

    def advance(self) -> None:
        now = time.perf_counter()
        dt, self.t = now - self.t, now
        self.pos = self.pos + self.v * dt
        self.yaw += self.rate * dt
        if self.pos[0] >= self.wall_x and not self.hit:
            self.hit, self.stamp = True, self.stamp + 1


class _FakeAirsim:
    @staticmethod
    def Vector3r(x, y, z):
        return SimpleNamespace(x_val=x, y_val=y, z_val=z)

    @staticmethod
    def to_quaternion(pitch, roll, yaw):
        return (pitch, roll, yaw)

    @staticmethod
    def Pose(p, q):
        return SimpleNamespace(position=p, orientation=q)

    @staticmethod
    def YawMode(is_rate, yaw_or_rate):
        return SimpleNamespace(is_rate=is_rate, yaw_or_rate=yaw_or_rate)


class _FakeClient:
    def __init__(self, w: _FakeWorld) -> None:
        self.w = w

    def enableApiControl(self, *a, **k): pass
    def armDisarm(self, *a, **k): pass
    def simPause(self, *a, **k): pass

    def simSetVehiclePose(self, pose, ignore, **k):
        w = self.w
        w.pos = np.array([pose.position.x_val, pose.position.y_val, -pose.position.z_val])
        w.pitch, _, w.yaw = pose.orientation
        w.v[:] = 0.0
        w.rate = 0.0
        w.t = time.perf_counter()
        w.poses.append((w.pos[0], w.pos[1], pose.position.z_val, w.pitch))
        if w.pos[0] >= w.wall_x and not w.hit:
            w.hit, w.stamp = True, w.stamp + 1

    def simGetVehiclePose(self, **k):
        self.w.advance()
        return SimpleNamespace(position=SimpleNamespace(z_val=-self.w.pos[2]))

    def simSetCameraPose(self, cam, pose, **k):
        self.w.pitch = pose.orientation[0]
        self.w.cam_down = pose.position.z_val

    def moveByVelocityZAsync(self, vx, vy, z_ned, dur, yaw_mode=None, **k):
        w = self.w
        w.advance()
        w.v = np.array([vx, vy, 0.0])
        w.pos[2] = -z_ned
        w.rate = math.radians(yaw_mode.yaw_or_rate) if yaw_mode.is_rate else 0.0
        w.moves.append((vx, vy, z_ned, yaw_mode.yaw_or_rate))

    def simGetCollisionInfo(self, **k):
        self.w.advance()
        return SimpleNamespace(has_collided=self.w.hit, time_stamp=self.w.stamp, object_name="wall")


class _FakeDogEnv(AirSimDogEnv):
    def __init__(self, world: _FakeWorld, **cfg) -> None:
        super().__init__(AirSimDogEnvConfig(**cfg))
        self.w = world
        self._client, self._airsim = _FakeClient(world), _FakeAirsim()

    def _connect(self):
        return self._client

    def _grab_scene_depth(self, client):
        w = self.w
        cam_up = w.pos[2] - w.cam_down
        if w.pitch:
            return np.zeros((224, 224, 3), np.uint8), np.full((224, 224), cam_up - w.ground, np.float32)
        d = _flat_forward_depth(cam_up - w.ground)
        return np.zeros((224, 224, 3), np.uint8), np.minimum(d, max(0.05, w.wall_x - w.pos[0])).astype(np.float32)

    def _grab_imu(self, client):
        return {}

    def observe_state(self):
        w = self.w
        w.advance()
        return np.array([*w.pos, *w.v, w.yaw], np.float32)


def test_env_spawns_at_dog_height_walks_and_turns():
    w = _FakeWorld(ground=2.0)
    p = DogDynamicsParams(vx_delay_s=0.0, vx_tau_s=0.05, yaw_delay_s=0.0, yaw_tau_s=0.05,
                          yaw_deadband=0.0, yaw_gain=1.0)
    env = _FakeDogEnv(w, step_hz=10.0, substeps=2, dynamics=p)
    obs = env.reset({"pos": [[0.0, 0.0, 30.0], [20.0, 0.0, 30.0]], "yaw": [0.0, 0.0]})
    assert obs.info["ground_z"] == pytest.approx(2.0, abs=1e-3)
    assert env.camera_z(obs.state[2]) == pytest.approx(2.6, abs=1e-3)
    assert not obs.collided
    w.poses.clear()
    for _ in range(10):
        obs, info = env.step(np.array([1.0, 0.4, 0.4, MODEL_LIMITS[3]]))
    assert info["cmd"] == pytest.approx([0.35, 0.5])
    assert 0.15 < obs.state[0] < 0.5 and obs.state[6] > 0.2
    assert env.ground_z == pytest.approx(2.0, abs=0.05)
    assert w.poses and all(p[2] == pytest.approx(-2.45) and p[3] == pytest.approx(0.0) for p in w.poses)
    stopped = env.step(np.array([0.0, 0.4, 0.4, 0.0]))[0]
    assert stopped.state[2] == pytest.approx(obs.state[2])


def test_env_reports_only_new_collisions():
    w = _FakeWorld(ground=0.0, wall_x=0.2)
    w.hit, w.stamp = True, 3
    p = DogDynamicsParams(vx_delay_s=0.0, vx_tau_s=0.05)
    env = _FakeDogEnv(w, step_hz=10.0, substeps=2, dynamics=p)
    obs = env.reset({"pos": [[0.0, 0.0, 30.0], [20.0, 0.0, 30.0]], "yaw": [0.0, 0.0]})
    assert not obs.collided
    w.hit = False
    hit = False
    for _ in range(20):
        obs, info = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
        if obs.collided:
            hit = True
            break
    assert hit and info["hit_object"] == "wall"


def test_stuck_needs_expected_motion_without_any():
    from experiments.aerial.rl.collect_dog_consequence import is_stuck

    buf = SimpleNamespace(dog_vel=[[0.3, 0.0]] * 10, state=[np.zeros(7, np.float32)] * 11)
    assert is_stuck(buf, 8, 0.5)
    buf.state = [np.array([0.1 * i, 0, 0, 0, 0, 0, 0], np.float32) for i in range(11)]
    assert not is_stuck(buf, 8, 0.5)
    buf = SimpleNamespace(dog_vel=[[0.0, 0.0]] * 10, state=[np.zeros(7, np.float32)] * 11)
    assert not is_stuck(buf, 8, 0.5)


def test_deploy_dog_sampler_matches_collector():
    from experiments.aerial.deploy.aerial_finder.nav.consequence_model import DOG_FAMILIES, sample_dog_macro as deploy

    for name in set(DOG_FAMILIES):
        for b in (-1.0, 0.2, 2.0):
            _, a = deploy(np.random.default_rng(5), 10, b, name, 0.06)
            _, c = sample_dog_macro(np.random.default_rng(5), 10, b, rad_per_step=0.06, name=name)
            assert np.allclose(a, c)
