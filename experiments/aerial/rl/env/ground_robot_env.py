"""``CarlaGroundRobotEnv`` — G0 ground env bridge (CARLA via Avant-AirSim).

Planar 3-DOF: actions are 4-D ``(dx, dy, dz, dyaw)`` but ``dz`` is forced to 0.
Uses CARLA vehicle physics + RGB/depth sensors; mirrors ``AirSimDroneEnv`` surface.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.aerial.eval.run_closed_loop import apply_body_delta
from experiments.aerial.rl.env.action import GROUND_MAX_BODY_VELOCITY, body_delta_limits, clip_body_delta
from experiments.aerial.rl.env.obs import Observation


@dataclass
class GroundRobotEnvConfig:
    carla_host: str = "127.0.0.1"
    carla_port: int = 2200
    vehicle_bp: str = "vehicle.tesla.model3"
    width: int = 224
    height: int = 224
    step_hz: float = 10.0
    camera_height_m: float = 1.4
    camera_forward_m: float = 1.5


class CarlaGroundRobotEnv:
    """G0 CARLA ground robot env — ``reset / step / observe / close``."""

    def __init__(self, config: Optional[GroundRobotEnvConfig] = None, **kwargs: Any) -> None:
        self.config = config or GroundRobotEnvConfig(**kwargs)
        self._client = None
        self._world = None
        self._vehicle = None
        self._rgb_sensor = None
        self._dep_sensor = None
        self._col_sensor = None
        self._actors: List[Any] = []
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._collided = False
        self._goal: Optional[np.ndarray] = None
        self._t0 = 0.0

    def _connect(self) -> None:
        if self._client is not None:
            return
        import carla  # type: ignore

        self._carla = carla
        self._client = carla.Client(self.config.carla_host, self.config.carla_port)
        self._client.set_timeout(60.0)
        self._world = self._client.get_world()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(self.config.step_hz)
        self._world.apply_settings(settings)

    def reset(self, episode: Optional[Dict[str, Any]] = None) -> Observation:
        self._connect()
        self._destroy_actors()
        carla = self._carla
        bp_lib = self._world.get_blueprint_library()
        veh_bp = bp_lib.find(self.config.vehicle_bp)
        vehicle = None
        for sp in self._world.get_map().get_spawn_points():
            vehicle = self._world.try_spawn_actor(veh_bp, sp)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("CarlaGroundRobotEnv: spawn failed")
        self._vehicle = vehicle
        self._actors.append(vehicle)

        cam_tf = carla.Transform(
            carla.Location(x=self.config.camera_forward_m, z=self.config.camera_height_m)
        )
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(self.config.width))
        rgb_bp.set_attribute("image_size_y", str(self.config.height))
        self._rgb_sensor = self._world.spawn_actor(rgb_bp, cam_tf, attach_to=vehicle)
        dep_bp = bp_lib.find("sensor.camera.depth")
        dep_bp.set_attribute("image_size_x", str(self.config.width))
        dep_bp.set_attribute("image_size_y", str(self.config.height))
        self._dep_sensor = self._world.spawn_actor(dep_bp, cam_tf, attach_to=vehicle)
        col_bp = bp_lib.find("sensor.other.collision")
        self._col_sensor = self._world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        self._actors.extend([self._rgb_sensor, self._dep_sensor, self._col_sensor])

        self._latest_rgb = None
        self._latest_depth = None
        self._collided = False

        def on_rgb(img) -> None:
            h, w = self.config.height, self.config.width
            self._latest_rgb = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((h, w, 4))[:, :, :3].copy()

        def on_dep(img) -> None:
            h, w = self.config.height, self.config.width
            self._latest_depth = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((h, w, 4)).copy()

        def on_col(_evt) -> None:
            self._collided = True

        self._rgb_sensor.listen(on_rgb)
        self._dep_sensor.listen(on_dep)
        self._col_sensor.listen(on_col)

        if episode is not None:
            positions = np.asarray(episode["pos"], dtype=np.float64)
            self._goal = positions[-1].copy()
        else:
            self._goal = None

        self._world.tick()
        self._t0 = time.perf_counter()
        return self.observe()

    def step(self, action: np.ndarray) -> Tuple[Observation, Dict[str, Any]]:
        dt = 1.0 / float(self.config.step_hz)
        lim = body_delta_limits(dt, max_velocity=GROUND_MAX_BODY_VELOCITY)
        cmd = clip_body_delta(action, lim)
        cmd[2] = 0.0  # planar constraint

        pos, yaw = self._read_pose()
        new_pos, new_yaw = apply_body_delta(pos, yaw, cmd)
        dyaw = new_yaw - yaw
        speed = math.hypot(cmd[0], cmd[1]) / dt if dt > 0 else 0.0
        steer = float(np.clip(dyaw / max(dt, 1e-3), -1.0, 1.0))
        ctrl = self._carla.VehicleControl(
            throttle=float(np.clip(speed / 3.0, 0.0, 1.0)),
            steer=steer * 0.5,
            brake=0.0,
        )
        self._vehicle.apply_control(ctrl)
        self._world.tick()
        return self.observe(), {"cmd": cmd.tolist()}

    def _read_pose(self) -> Tuple[np.ndarray, float]:
        loc = self._vehicle.get_transform().location
        rot = self._vehicle.get_transform().rotation
        pos = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
        yaw = math.radians(rot.yaw)
        return pos, yaw

    def observe_state(self) -> np.ndarray:
        loc = self._vehicle.get_transform().location
        vel = self._vehicle.get_velocity()
        rot = self._vehicle.get_transform().rotation
        return np.array(
            [loc.x, loc.y, loc.z, vel.x, vel.y, vel.z, math.radians(rot.yaw)],
            dtype=np.float32,
        )

    def observe(self) -> Observation:
        if self._latest_rgb is None:
            rgb = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        else:
            rgb = self._latest_rgb
        depth = None
        if self._latest_depth is not None:
            depth = self._latest_depth[:, :, 2].astype(np.float32)  # placeholder channel
        return Observation(
            rgb=rgb,
            state=self.observe_state(),
            collided=self._collided,
            depth=depth,
            imu={},
            t=time.perf_counter() - self._t0,
            info={"goal": None if self._goal is None else self._goal.tolist()},
        )

    def _destroy_actors(self) -> None:
        for a in reversed(self._actors):
            try:
                a.stop()
                a.destroy()
            except Exception:  # noqa: BLE001
                pass
        self._actors.clear()
        self._vehicle = None

    def close(self) -> None:
        self._destroy_actors()
        if self._world is not None:
            try:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                self._world.apply_settings(settings)
            except Exception:  # noqa: BLE001
                pass
        self._client = None
        self._world = None

    def __enter__(self) -> "CarlaGroundRobotEnv":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


# Alias for chapter VII naming
GroundRobotBridge = CarlaGroundRobotEnv
