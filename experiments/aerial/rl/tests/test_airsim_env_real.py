"""Offline tests for the *real* ``AirSimDroneEnv`` paths that the kinematic mock
cannot exercise: the NED<->+up world-frame conversion, per-reset health fail-fast,
and the lifecycle guarantee that a failed reset releases the single-consumer
renderer.

No real AirSim / cv2: a fake client + fake ``airsim`` module are injected, and the
cv2-dependent capture helpers are monkeypatched. This lets us assert the boundary
logic (z negation, health gating, close-on-exception) directly.
"""
import numpy as np
import pytest

from experiments.aerial.rl.env.airsim_env import AirSimDroneEnv, AirSimEnvConfig, decode_scene_bgr


def test_empty_scene_buffer_does_not_raise():
    assert decode_scene_bgr(np.array([], np.uint8)) is None


# -- fakes ---------------------------------------------------------------
class _Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x_val, self.y_val, self.z_val = float(x), float(y), float(z)


class _Kin:
    def __init__(self, pos, vel):
        self.position = pos
        self.linear_velocity = vel
        self.orientation = object()  # opaque; yaw comes from to_eularian_angles


class _State:
    def __init__(self, pos, vel):
        self.kinematics_estimated = _Kin(pos, vel)


class _Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


class _FakeAirsim:
    """Minimal stand-in for the ``airsim`` module surface reset/observe use."""

    Vector3r = _Vec
    Pose = _Pose

    def __init__(self, yaw=0.3):
        self._yaw = yaw

    def to_quaternion(self, roll, pitch, yaw):
        return ("quat", roll, pitch, yaw)

    def to_eularian_angles(self, _orientation):
        return (0.0, 0.0, self._yaw)

    class YawMode:
        def __init__(self, is_rate=True, yaw_or_rate=0.0):
            self.is_rate = is_rate
            self.yaw_or_rate = yaw_or_rate


class _FakeClient:
    """Records control toggles and the pose it was asked to set."""

    def __init__(self, ned_z=-5.0, ned_vz=-2.0):
        # NED state: z=-5 (5 m up), vz=-2 (climbing at 2 m/s).
        self._state = _State(_Vec(1.0, 2.0, ned_z), _Vec(0.5, 0.0, ned_vz))
        self.control_enabled = False
        self.armed = False
        self.set_poses = []

    def confirmConnection(self):
        return True

    def enableApiControl(self, enabled, vehicle_name=None):
        self.control_enabled = bool(enabled)

    def armDisarm(self, arm, vehicle_name=None):
        self.armed = bool(arm)

    def simSetVehiclePose(self, pose, ignore_collision, vehicle_name=None):
        self.set_poses.append(pose)

    def getMultirotorState(self, vehicle_name=None):
        return self._state


def _make_env(**cfg):
    env = AirSimEnvConfig(**cfg)
    e = AirSimDroneEnv(env)
    client = _FakeClient()
    e._client = client            # _connect() returns this, skips import airsim
    e._airsim = _FakeAirsim()      # _connect() early-returns, so set manually
    return e, client


_EPISODE = {"pos": [[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]], "yaw": [0.0, 0.0]}


# -- z-axis conversion (Fix #1) -----------------------------------------
def test_observe_state_negates_ned_z_to_up_world():
    e, _ = _make_env(health_check=False)
    st = e.observe_state()
    # NED z=-5 -> +up world z=+5; NED vz=-2 -> +up vz=+2.
    assert st[2] == pytest.approx(5.0)
    assert st[5] == pytest.approx(2.0)
    # Horizontal components pass through unchanged.
    assert st[0] == pytest.approx(1.0)
    assert st[1] == pytest.approx(2.0)
    assert st[3] == pytest.approx(0.5)
    assert st[6] == pytest.approx(0.3)  # yaw from to_eularian_angles


def test_reset_negates_start_z_into_ned_pose():
    e, client = _make_env(health_check=False, warmup_frames=0)
    e._grab_scene = e._grab_scene_native = lambda client: np.zeros((8, 8, 3), dtype=np.uint8)
    e._grab_depth = lambda client: None
    e._grab_imu = lambda client: {}
    e._grab_collision = lambda client: False

    e.reset(_EPISODE)
    assert len(client.set_poses) == 1
    # Episode start z = +5 (up world) must be sent as NED z = -5.
    assert client.set_poses[0].position.z_val == pytest.approx(-5.0)
    assert client.set_poses[0].position.x_val == pytest.approx(0.0)


# -- health fail-fast (Fix #3) ------------------------------------------
def test_health_check_raises_when_imu_missing():
    e, _ = _make_env(health_check=True, warmup_frames=0)
    e._grab_scene = e._grab_scene_native = lambda client: np.zeros((8, 8, 3), dtype=np.uint8)
    e._grab_depth = lambda client: np.zeros((8, 8), dtype=np.float32)
    e._grab_imu = lambda client: {}          # CV-only build -> no IMU
    e._grab_collision = lambda client: False
    with pytest.raises(RuntimeError):
        e.reset(_EPISODE)


def test_health_check_raises_when_depth_missing():
    e, _ = _make_env(health_check=True, warmup_frames=0)
    e._grab_scene = e._grab_scene_native = lambda client: np.zeros((8, 8, 3), dtype=np.uint8)
    e._grab_depth = lambda client: None      # no depth from renderer
    e._grab_imu = lambda client: {"ang_vel": [0, 0, 0], "lin_acc": [0, 0, 9.807]}
    e._grab_collision = lambda client: False
    with pytest.raises(RuntimeError):
        e.reset(_EPISODE)


# -- lifecycle: close on failed reset (Fix #4) --------------------------
def test_failed_reset_releases_control():
    e, client = _make_env(health_check=True, warmup_frames=0)
    e._grab_scene = e._grab_scene_native = lambda client: np.zeros((8, 8, 3), dtype=np.uint8)
    e._grab_depth = lambda client: None
    e._grab_imu = lambda client: {}          # fails health -> reset raises
    e._grab_collision = lambda client: False
    # Arm happens before the health check; the try/except must disarm on failure.
    with pytest.raises(RuntimeError):
        e.reset(_EPISODE)
    assert client.control_enabled is False
    assert client.armed is False


def test_context_manager_closes():
    e, client = _make_env(health_check=False)
    client.control_enabled = True
    client.armed = True
    with e:
        pass
    assert client.control_enabled is False
    assert client.armed is False
