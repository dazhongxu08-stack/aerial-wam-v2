"""``AirSimDogEnv`` — a street camera whose motion is the Lumos dog, not the quadrotor.

The packaged scene still has a multirotor mesh, but that vehicle is not flown.
Each step a model action becomes a dog command (``model_action_to_cmd``: ``dy``
and ``dz`` dropped), passes through the dog plant (delay, deadband, gain, lag),
and the body pose is integrated from that plant: forward speed along the current
heading, yaw rate about the vertical, height locked to the street. The pose is
written level, so the quadrotor controller never pitches, climbs, or strafes.
Ground height comes from a depth probe at reset (camera briefly posed nose-down
below the body, then put back) and is then tracked from the bottom rows of the
forward DepthPlanar frame.

That probe is the only camera-pose RPC, so run this env only against its own
renderer, started with ``make_dog_airsim_settings.py`` output, where the camera
sits ahead of and above a vehicle body that rides at dog-torso height; with the
drone settings (camera 1 m below the body) low obstacles would pass under the
body, and a camera over the body centre sees the rotors.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from experiments.aerial.rl.env.airsim_env import AirSimDroneEnv, AirSimEnvConfig
from experiments.aerial.rl.env.dog_dynamics import (
    DogDynamics,
    DogDynamicsParams,
    model_action_to_cmd,
)
from experiments.aerial.rl.env.obs import Observation


@dataclass
class AirSimDogEnvConfig(AirSimEnvConfig):
    step_hz: float = 2.0
    substeps: int = 5
    cam_height_m: float = 0.6
    # Camera offset from the vehicle centre as set in the renderer's settings (NED body frame: X ahead,
    # Z + = below). Collisions are against the vehicle body, so the dog settings put the camera 0.15 m
    # above a body riding at 0.45 m (clears the ~0.18 m kerbs a dog steps over), and 0.35 m ahead so
    # the rotors stay out of view.
    cam_forward_m: float = 0.35
    cam_offset_down_m: float = -0.15
    hfov_deg: float = 90.0
    probe_patch: int = 5
    # The probe turns the camera nose-down and moves it below the body so it does not see the vehicle.
    probe_cam_down_m: float = 0.4
    probe_max_m: float = 300.0
    ground_rows: int = 20
    ground_alpha: float = 0.3
    ground_max_jump_m: float = 0.3
    health_check: bool = False
    dynamics: DogDynamicsParams = field(default_factory=DogDynamicsParams)


def ground_from_probe(depth: np.ndarray, patch: int) -> Tuple[float, float]:
    """(median, spread) planar depth of the centre patch of a nose-down frame."""
    d = np.asarray(depth, dtype=np.float64)
    h, w = d.shape
    r = max(1, patch // 2)
    c = d[h // 2 - r: h // 2 + r + 1, w // 2 - r: w // 2 + r + 1]
    c = c[np.isfinite(c) & (c > 0.0)]
    if c.size == 0:
        return float("nan"), float("inf")
    return float(np.median(c)), float(np.percentile(c, 90) - np.percentile(c, 10))


def ground_drop_from_forward(depth: np.ndarray, hfov_deg: float, rows: int) -> float:
    """Camera height above the ground seen in the bottom ``rows`` of a level forward frame.

    A pixel row ``v`` below the principal point looks down by ``atan((v - cy) / f)``;
    planar depth ``d`` on flat ground then sits ``d * (v - cy) / f`` below the camera.
    Returns NaN when the bottom band is not ground-like (an obstacle fills it).
    """
    d = np.asarray(depth, dtype=np.float64)
    h, w = d.shape
    f = 0.5 * w / math.tan(math.radians(hfov_deg) / 2.0)
    cy = 0.5 * (h - 1)
    band = d[h - rows:, w // 3: 2 * w // 3]
    v = np.arange(h - rows, h, dtype=np.float64)[:, None]
    drop = band * (v - cy) / f
    ok = np.isfinite(drop) & (band > 0.05)
    if ok.sum() < band.size // 2:
        return float("nan")
    vals = drop[ok]
    med = float(np.median(vals))
    if med <= 0.0 or (np.percentile(vals, 90) - np.percentile(vals, 10)) / med > 0.08:
        return float("nan")
    return med


def mask_ground(depth: np.ndarray, hfov_deg: float, cam_above_ground_m: float, tol_m: float = 0.12) -> np.ndarray:
    """Copy of ``depth`` with ground pixels set to ``inf``, for clearance labels at dog height.

    The drone-era cone / column labels assume no ground in the central band; 0.6 m up it is
    always there (~1.2 m ahead at the band's lower edge) and would read as an obstacle.
    """
    d = np.asarray(depth, dtype=np.float64).copy()
    h, w = d.shape
    f = 0.5 * w / math.tan(math.radians(hfov_deg) / 2.0)
    below = d * ((np.arange(h, dtype=np.float64) - 0.5 * (h - 1)) / f)[:, None]
    d[below >= float(cam_above_ground_m) - float(tol_m)] = np.inf
    return d


def decode_raw_scene(buf: bytes, h: int, w: int) -> np.ndarray:
    """Uncompressed Scene bytes (BGR or BGRA, depending on the AirSim build) -> RGB uint8."""
    a = np.frombuffer(buf, dtype=np.uint8)
    ch = a.size // max(1, h * w)
    if ch not in (3, 4):
        raise RuntimeError(f"unexpected Scene buffer {a.size} bytes for {w}x{h}")
    return np.ascontiguousarray(a.reshape(h, w, ch)[:, :, 2::-1])


class AirSimDogEnv(AirSimDroneEnv):
    def __init__(self, config: Optional[AirSimDogEnvConfig] = None, **kwargs: Any) -> None:
        super().__init__(config or AirSimDogEnvConfig(**kwargs))
        self.config: AirSimDogEnvConfig
        self.dyn = DogDynamics(self.config.dynamics)
        self.ground_z: float = 0.0        # +up world
        self.ground_spread: float = 0.0
        self._coll_stamp: int = 0
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self.last_info: Dict[str, Any] = {}

    def set_dynamics(self, params: DogDynamicsParams) -> None:
        self.config.dynamics = params
        self.dyn = DogDynamics(params)

    def _target_z_ned(self) -> float:
        return -(self.ground_z + self.config.cam_height_m + self.config.cam_offset_down_m)

    def camera_z(self, body_z: float) -> float:
        return float(body_z) - self.config.cam_offset_down_m

    def _grab_scene_depth(self, client: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """One render request for an uncompressed Scene frame and a DepthPlanar frame."""
        import cv2  # type: ignore

        airsim = self._airsim
        cam = self.config.camera
        rs = client.simGetImages([airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False),
                                  airsim.ImageRequest(cam, airsim.ImageType.DepthPlanar, True, False)], **self._vk)
        rgb = decode_raw_scene(bytes(rs[0].image_data_uint8), int(rs[0].height), int(rs[0].width))
        if (rgb.shape[1], rgb.shape[0]) != (self.config.width, self.config.height):
            rgb = cv2.resize(rgb, (self.config.width, self.config.height), interpolation=cv2.INTER_LINEAR)
        depth = None
        if rs[1].width and rs[1].height:
            depth = np.asarray(rs[1].image_data_float, dtype=np.float32).reshape(rs[1].height, rs[1].width)
        return np.ascontiguousarray(rgb, dtype=np.uint8), depth

    def observe(self, *, force_depth: bool = False) -> Observation:
        client = self._connect()
        rgb, depth = self._grab_scene_depth(client)
        return Observation(rgb=rgb, state=self.observe_state(), collided=False, depth=depth,
                           imu=self._grab_imu(client), t=time.perf_counter() - self._t0,
                           info={"goal": None if self._goal is None else self._goal.tolist()})

    def _set_camera(self, client: Any, down: Optional[float] = None) -> None:
        airsim, cfg = self._airsim, self.config
        if down is None:
            pose = airsim.Pose(airsim.Vector3r(cfg.cam_forward_m, 0.0, cfg.cam_offset_down_m),
                               airsim.to_quaternion(0.0, 0.0, 0.0))
        else:
            pose = airsim.Pose(airsim.Vector3r(0.0, 0.0, down), airsim.to_quaternion(-math.pi / 2.0, 0.0, 0.0))
        client.simSetCameraPose(cfg.camera, pose, **self._vk)

    def _hold(self, client: Any, x: float, y: float, z_up: float, yaw: float, settle_s: float) -> None:
        """Teleport level to ``(x, y, z_up)`` and hover there. Physics stays running: on this AirSim
        build only the first teleport issued while paused takes effect."""
        airsim = self._airsim
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(float(x), float(y), -float(z_up)),
                                             airsim.to_quaternion(0.0, 0.0, float(yaw))), True, **self._vk)
        client.moveByVelocityZAsync(0.0, 0.0, -float(z_up), 1.0,
                                    yaw_mode=airsim.YawMode(False, math.degrees(yaw)), **self._vk)
        time.sleep(settle_s)

    def probe_ground(self, x: float, y: float, top_z: float) -> Tuple[float, float]:
        """(ground height +up, spread) under ``(x, y)`` seen from a level body near ``top_z`` (+up).

        NaN when nothing is seen below.
        """
        client = self._connect()
        cfg = self.config
        client.enableApiControl(True, **self._vk)
        client.armDisarm(True, **self._vk)
        self._hold(client, x, y, top_z, 0.0, 0.1)
        self._set_camera(client, cfg.probe_cam_down_m)
        try:
            self._grab_depth(client)
            z0 = -float(client.simGetVehiclePose(**self._vk).position.z_val)
            _, depth = self._grab_scene_depth(client)
            z1 = -float(client.simGetVehiclePose(**self._vk).position.z_val)
        finally:
            self._set_camera(client)
        if depth is None:
            return float("nan"), float("inf")
        dist, spread = ground_from_probe(depth, cfg.probe_patch)
        if not (0.0 < dist < cfg.probe_max_m) or abs(z1 - z0) > 0.3:
            return float("nan"), float("inf")
        return 0.5 * (z0 + z1) - cfg.probe_cam_down_m - dist, spread

    def reset(self, episode: Optional[Dict[str, Any]] = None, *,
              ground: Optional[Tuple[float, float]] = None) -> Observation:
        """Place the dog camera ``cam_height_m`` above the ground under ``episode['pos'][0]``.

        ``episode['pos'][0][2]`` is the probe height (+up) and ``episode['yaw'][0]`` the heading (rad,
        AirSim). ``ground`` = a ``probe_ground`` result for that point skips the probe. The returned
        obs carries ``info['ground_z']`` / ``info['ground_spread']``.
        """
        client = self._connect()
        if episode is None:
            raise ValueError("AirSimDogEnv.reset needs an episode with pos/yaw")
        start = np.asarray(episode["pos"], dtype=np.float64).reshape(-1, 3)[0]
        yaw0 = float(np.asarray(episode["yaw"], dtype=np.float64).reshape(-1)[0])
        self._goal = np.asarray(episode["pos"], dtype=np.float64).reshape(-1, 3)[-1].copy()
        client.enableApiControl(True, **self._vk)
        client.armDisarm(True, **self._vk)
        try:
            gz, spread = ground or self.probe_ground(float(start[0]), float(start[1]), float(start[2]))
            if not np.isfinite(gz):
                raise RuntimeError("ground probe returned no depth")
            self.ground_z, self.ground_spread = gz, spread
            self._x, self._y, self._yaw = float(start[0]), float(start[1]), yaw0
            self._hold(client, self._x, self._y, -self._target_z_ned(), self._yaw, 0.4)
            obs = self.observe(force_depth=True)
            self._coll_stamp = int(client.simGetCollisionInfo(**self._vk).time_stamp)
        except Exception:  # noqa: BLE001
            self.close()
            raise
        self.dyn.reset()
        obs.info.update({"ground_z": gz, "ground_spread": spread})
        return obs

    def _collided_since_reset(self, client: Any) -> Tuple[bool, str]:
        c = client.simGetCollisionInfo(**self._vk)
        hit = bool(c.has_collided) and int(c.time_stamp) != self._coll_stamp
        return hit, str(c.object_name) if hit else ""

    def _place_dog(self, client: Any) -> None:
        """Level pose from the dog plant. The flight controller is not asked to track it."""
        airsim = self._airsim
        pose = airsim.Pose(
            airsim.Vector3r(float(self._x), float(self._y), self._target_z_ned()),
            airsim.to_quaternion(0.0, 0.0, float(self._yaw)),
        )
        client.simSetVehiclePose(pose, False, **self._vk)

    def step(self, action: np.ndarray) -> Tuple[Observation, Dict[str, Any]]:
        """Collision info is latched with a timestamp, so one check after the substeps sees any contact."""
        client = self._connect()
        cfg = self.config
        dt = 1.0 / float(cfg.step_hz)
        sub = max(1, int(cfg.substeps))
        sdt = dt / sub
        cmd = model_action_to_cmd(action, cfg.dynamics)
        t0 = time.perf_counter()
        vx_m = wz_m = 0.0
        for i in range(sub):
            vx, wz = self.dyn.advance(cmd, sdt)
            vx_m += vx / sub
            wz_m += wz / sub
            # Translate along the heading, then yaw. No lateral or vertical channel.
            self._x += vx * math.cos(self._yaw) * sdt
            self._y += vx * math.sin(self._yaw) * sdt
            self._yaw = (self._yaw + wz * sdt + math.pi) % (2.0 * math.pi) - math.pi
            self._place_dog(client)
            budget = 0.2 if i == sub - 1 else 0.0
            rest = (i + 1) * sdt - budget - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)
        obs = self.observe(force_depth=True)
        hit, hit_obj = self._collided_since_reset(client)
        obs.collided = hit
        drop = float("nan")
        if obs.depth is not None:
            drop = ground_drop_from_forward(obs.depth, cfg.hfov_deg, cfg.ground_rows)
            seen = self.camera_z(float(obs.state[2])) - drop
            if np.isfinite(drop) and abs(seen - self.ground_z) <= cfg.ground_max_jump_m:
                self.ground_z += cfg.ground_alpha * (seen - self.ground_z)
        rest = dt - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)
        info = {"cmd": [cmd[0], cmd[1]], "dog_vel": [vx_m, wz_m], "ground_z": self.ground_z,
                "ground_drop": drop, "hit_object": hit_obj, "step_s": time.perf_counter() - t0}
        self.last_info = info
        return obs, info
