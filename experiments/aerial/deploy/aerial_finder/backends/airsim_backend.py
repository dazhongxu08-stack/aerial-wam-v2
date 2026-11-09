"""AirSim multirotor backend (simulation stage).

Same command path as the training env: a body delta becomes a world velocity +
yaw rate for ``moveByVelocityAsync`` over 1 / step_hz, rate-locked to wall time.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from ..geo import GeoRef
from ..types import Frame, Intrinsics, VehicleState
from .base import body_delta_to_world_velocity


class AirSimBackend:
    def __init__(self, host: str = "127.0.0.1", port: int = 41451, vehicle: str = "drone_1",
                 camera: str = "front_custom", hfov_deg: float = 90.0, step_hz: float = 5.0,
                 sensor_depth: bool = True, depth_hfov_deg: Optional[float] = None) -> None:
        import airsim

        self.airsim = airsim
        self.c = airsim.MultirotorClient(ip=host, port=port)
        self.c.confirmConnection()
        self.vk = {"vehicle_name": vehicle}
        self.camera = camera
        self.hfov_deg = float(hfov_deg)
        self.depth_hfov_deg = float(depth_hfov_deg or hfov_deg)
        self.step_hz = float(step_hz)
        self.sensor_depth = bool(sensor_depth)
        g = self.c.getHomeGeoPoint(**self.vk)
        self.geo_ref = GeoRef(float(g.latitude), float(g.longitude), float(g.altitude))
        self._t0 = time.perf_counter()
        self._col_ts = 0

    def arm(self) -> None:
        self.c.enableApiControl(True, **self.vk)
        self.c.armDisarm(True, **self.vk)
        self._mark_collisions()

    def _mark_collisions(self) -> None:
        """Collisions stamped at or before now (resting on the ground, setup teleports) are ignored."""
        self._col_ts = int(self.c.simGetCollisionInfo(**self.vk).time_stamp)

    def teleport(self, pos_world, yaw: float) -> None:
        """Place the vehicle (simulation scenario setup only)."""
        a = self.airsim
        pose = a.Pose(a.Vector3r(float(pos_world[0]), float(pos_world[1]), -float(pos_world[2])),
                      a.to_quaternion(0.0, 0.0, float(yaw)))
        self.c.simPause(True)
        try:
            self.c.simSetVehiclePose(pose, True, **self.vk)
        finally:
            self.c.simPause(False)
        self.c.simSetVehiclePose(pose, True, **self.vk)
        self.c.moveByVelocityAsync(0.0, 0.0, 0.0, 0.05, **self.vk)
        time.sleep(0.1)
        self.c.simSetVehiclePose(pose, True, **self.vk)
        self._mark_collisions()

    def state(self) -> VehicleState:
        k = self.c.getMultirotorState(**self.vk).kinematics_estimated
        p = k.position
        pitch, roll, yaw = self.airsim.to_eularian_angles(k.orientation)
        col = self.c.simGetCollisionInfo(**self.vk)
        return VehicleState(pos=np.array([p.x_val, p.y_val, -p.z_val]), yaw=float(yaw), roll=float(roll),
                            pitch=float(pitch), collided=bool(col.has_collided) and int(col.time_stamp) > self._col_ts,
                            t=time.perf_counter() - self._t0)

    def collision_object(self) -> str:
        return str(self.c.simGetCollisionInfo(**self.vk).object_name)

    def observe(self) -> Tuple[Frame, VehicleState]:
        a = self.airsim
        rq = [a.ImageRequest(self.camera, a.ImageType.Scene, False, True)]
        if self.sensor_depth:
            rq.append(a.ImageRequest(self.camera, a.ImageType.DepthPlanar, True, False))
        rs = self.c.simGetImages(rq, **self.vk)
        img = cv2.imdecode(np.frombuffer(bytes(rs[0].image_data_uint8), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("empty Scene buffer")
        rgb = np.ascontiguousarray(img[..., ::-1])
        depth, d_intr = None, None
        if self.sensor_depth and len(rs) > 1 and rs[1].width:
            depth = np.asarray(rs[1].image_data_float, np.float32).reshape(rs[1].height, rs[1].width)
            d_intr = Intrinsics.from_hfov(rs[1].width, rs[1].height, self.depth_hfov_deg)
        st = self.state()
        frame = Frame(rgb=rgb, rgb_intr=Intrinsics.from_hfov(rgb.shape[1], rgb.shape[0], self.hfov_deg),
                      depth=depth, depth_intr=d_intr, t=st.t)
        return frame, st

    def step(self, action: np.ndarray) -> None:
        dt = 1.0 / self.step_hz
        t0 = time.perf_counter()
        k = self.c.getMultirotorState(**self.vk).kinematics_estimated
        _, _, yaw = self.airsim.to_eularian_angles(k.orientation)
        vn, ve, vu, yr = body_delta_to_world_velocity(action, float(yaw), dt)
        self.c.moveByVelocityAsync(vn, ve, -vu, dt, yaw_mode=self.airsim.YawMode(True, math.degrees(yr)), **self.vk)
        rest = dt - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)

    def hover(self) -> None:
        self.c.moveByVelocityAsync(0.0, 0.0, 0.0, 1.0, **self.vk)

    def close(self) -> None:
        try:
            self.hover()
        except Exception:  # noqa: BLE001
            pass
