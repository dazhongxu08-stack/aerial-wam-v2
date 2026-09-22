"""``TelloDroneEnv`` — vgoal/Pixhawk-compatible env surface over Tello SDK.

Mirrors ``PixhawkDroneEnv``: ``reset / step / observe / close`` with body-frame
``(dx, dy, dz, dyaw)`` actions. Pose is dead-reckoned from commands + Tello
state (height, yaw); RGB from Tello H.264 stream or a local USB camera.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from experiments.aerial.rl.env.action import body_delta_limits, clip_body_delta
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.env.tello_client import TelloClient, guess_local_ip

logger = logging.getLogger(__name__)

# Indoor-safe default velocity scaling for EDU (m/s → full stick).
_DEFAULT_MAX_V_MPS = 0.6
_DEFAULT_MAX_YAW_RAD_S = math.pi / 4.0


def body_delta_to_rc(
    delta: np.ndarray,
    dt: float,
    *,
    max_v_mps: float = _DEFAULT_MAX_V_MPS,
    max_yaw_rad_s: float = _DEFAULT_MAX_YAW_RAD_S,
) -> Tuple[int, int, int, int]:
    """Body displacement → Tello ``rc a b c d`` stick units (−100…100).

    Body: x forward, y left, z up, yaw CCW+.
    Tello: a right+, b forward+, c up+, d yaw CW+ (negate our yaw).
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    d = np.asarray(delta, dtype=np.float64).reshape(4)
    vx, vy, vz, yaw_rate = float(d[0]) / dt, float(d[1]) / dt, float(d[2]) / dt, float(d[3]) / dt
    mv = max(1e-6, float(max_v_mps))
    my = max(1e-6, float(max_yaw_rad_s))
    a = int(round(100.0 * float(np.clip(-vy / mv, -1.0, 1.0))))  # left+ → stick left (−)
    b = int(round(100.0 * float(np.clip(vx / mv, -1.0, 1.0))))
    c = int(round(100.0 * float(np.clip(vz / mv, -1.0, 1.0))))
    d_yaw = int(round(100.0 * float(np.clip(-yaw_rate / my, -1.0, 1.0))))
    return (
        max(-100, min(100, a)),
        max(-100, min(100, b)),
        max(-100, min(100, c)),
        max(-100, min(100, d_yaw)),
    )


@dataclass
class TelloEnvConfig:
    tello_ip: str = "192.168.10.1"
    local_ip: str = ""
    local_cmd_port: int = 8889
    step_hz: float = 5.0
    camera_device: str = "tello"  # "tello" | V4L2 index/path | "mock"
    capture_w: int = 960
    capture_h: int = 720
    capture_fps: int = 30
    wam_encode_size: int = 224
    takeoff_on_reset: bool = False
    control_on_reset: bool = True
    mock_camera: bool = False
    max_v_mps: float = _DEFAULT_MAX_V_MPS
    max_yaw_rad_s: float = _DEFAULT_MAX_YAW_RAD_S
    stream_port: int = 11111


class TelloDroneEnv:
    """Tello velocity-step env for Orin + Tello EDU Wi-Fi."""

    def __init__(self, config: Optional[TelloEnvConfig] = None, **kwargs: Any) -> None:
        self.config = config or TelloEnvConfig(**kwargs)
        local = self.config.local_ip.strip() or guess_local_ip(self.config.tello_ip)
        self._client = TelloClient(
            local_ip=local,
            tello_ip=self.config.tello_ip,
            local_cmd_port=int(self.config.local_cmd_port),
        )
        self._camera: Any = None
        self._goal: Optional[np.ndarray] = None
        self._t0 = time.perf_counter()
        self._airborne = False
        self._control_active = False
        self._pos = np.zeros(3, dtype=np.float64)
        self._yaw = 0.0
        self._connected = False
        # Pixhawk-compat shim for deploy logging / record gate
        self._bridge = self
        self._offboard_active = False

    @property
    def goal(self) -> Optional[np.ndarray]:
        return self._goal

    def is_armed(self) -> bool:
        return self._airborne

    @property
    def connected(self) -> bool:
        return self._connected

    def rc_pwm_dict(self) -> Dict[str, int]:
        return {}

    def poll(self, timeout_s: float = 0.0) -> None:
        return None

    def status_dict(self, state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        bat = self._client.battery()
        return {
            "backend": "tello",
            "airborne": self._airborne,
            "battery": bat,
            "state_age_s": self._client.state_age_s,
            "pos": self._pos.tolist(),
            "yaw_deg": math.degrees(self._yaw),
        }

    def _open_camera(self) -> None:
        if self._camera is not None:
            return
        if self.config.mock_camera or str(self.config.camera_device).lower() == "mock":
            from experiments.aerial.deploy.real_camera import MockCamera

            self._camera = MockCamera(wam_size=self.config.wam_encode_size)
            self._camera.open()
            return

        if str(self.config.camera_device).lower() == "tello":
            self._client.stream_on()
            time.sleep(0.5)
            from experiments.aerial.deploy.tello_camera import TelloStreamCamera

            self._camera = TelloStreamCamera(
                port=int(self.config.stream_port),
                width=int(self.config.capture_w),
                height=int(self.config.capture_h),
                wam_size=int(self.config.wam_encode_size),
            )
            self._camera.open()
            return

        from experiments.aerial.deploy.real_camera import RealCamera, RealCameraConfig

        self._camera = RealCamera(
            RealCameraConfig(
                device=str(self.config.camera_device),
                width=int(self.config.capture_w),
                height=int(self.config.capture_h),
                wam_size=int(self.config.wam_encode_size),
                fps=int(self.config.capture_fps),
            )
        )
        self._camera.open()

    def _sync_pose_from_telemetry(self) -> None:
        st = self._client.state
        h = self._client.height_cm()
        if h is not None:
            self._pos[2] = float(h) / 100.0
        yaw_raw = st.get("yaw")
        if yaw_raw is not None:
            try:
                # Tello yaw is degrees; convert to rad. Sign: keep as reported.
                self._yaw = math.radians(float(yaw_raw))
            except ValueError:
                pass

    def _state_vec(self) -> np.ndarray:
        self._sync_pose_from_telemetry()
        return np.array(
            [
                self._pos[0],
                self._pos[1],
                self._pos[2],
                0.0,
                0.0,
                0.0,
                self._yaw,
            ],
            dtype=np.float32,
        )

    def reset(self, episode: Optional[Dict[str, Any]] = None) -> Observation:
        if not self._client.connect(retries=5):
            raise RuntimeError(
                f"Tello SDK handshake failed at {self.config.tello_ip} "
                f"(local {self._client.local_ip})"
            )
        self._connected = True
        self._client.start_state_listener()
        time.sleep(0.3)

        self._goal = None
        if episode is not None:
            positions = np.asarray(episode["pos"], dtype=np.float64)
            self._goal = positions[-1].copy()

        self._pos[:] = 0.0
        self._yaw = 0.0
        self._sync_pose_from_telemetry()

        self._control_active = bool(self.config.control_on_reset)
        self._offboard_active = self._control_active

        if self.config.takeoff_on_reset:
            logger.info("Tello takeoff…")
            resp = self._client.takeoff()
            if resp != "ok":
                raise RuntimeError(f"Tello takeoff failed: {resp}")
            self._airborne = True
            time.sleep(1.0)
            self._sync_pose_from_telemetry()

        self._open_camera()
        bat = self._client.battery()
        logger.info(
            "Tello reset airborne=%s control=%s battery=%s local=%s",
            self._airborne,
            self._control_active,
            bat,
            self._client.local_ip,
        )
        self._t0 = time.perf_counter()
        return self.observe()

    def observe(self) -> Observation:
        state = self._state_vec()
        bgr, rgb = self._camera.read()
        import cv2  # type: ignore

        rgb_yolo = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        info: Dict[str, Any] = {
            "capture_shape": list(bgr.shape),
            "bgr_native": bgr,
            "mavlink": self.status_dict(state),
            "tello": self.status_dict(state),
        }
        if self._goal is not None:
            info["goal"] = self._goal.copy()
        return Observation(
            rgb=rgb,
            state=state,
            collided=False,
            depth=None,
            imu={},
            t=time.perf_counter() - self._t0,
            info=info,
            rgb_yolo=rgb_yolo,
            baro_alt=float(self._pos[2]),
            agl_m=float(self._pos[2]),
        )

    def step(self, action: np.ndarray) -> tuple[Observation, Dict[str, Any]]:
        dt = 1.0 / float(self.config.step_hz)
        t0 = time.perf_counter()
        cmd = clip_body_delta(action, body_delta_limits(dt))
        a = b = c = d = 0
        if self._control_active and self._airborne:
            a, b, c, d = body_delta_to_rc(
                cmd,
                dt,
                max_v_mps=float(self.config.max_v_mps),
                max_yaw_rad_s=float(self.config.max_yaw_rad_s),
            )
            self._client.rc(a, b, c, d)
            # Dead-reckon horizontal in body → world
            c_y, s_y = math.cos(self._yaw), math.sin(self._yaw)
            self._pos[0] += c_y * float(cmd[0]) - s_y * float(cmd[1])
            self._pos[1] += s_y * float(cmd[0]) + c_y * float(cmd[1])
            self._yaw += float(cmd[3])
        elif self._control_active:
            self._client.rc(0, 0, 0, 0)

        remaining = dt - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
        obs = self.observe()
        info = {
            "cmd": cmd.tolist(),
            "rc": [a, b, c, d],
            "offboard": self._control_active,
            "armed": self._airborne,
        }
        return obs, info

    def close(self) -> None:
        try:
            self._client.rc(0, 0, 0, 0)
            if self._airborne:
                logger.info("Tello land…")
                self._client.land()
                self._airborne = False
            if str(self.config.camera_device).lower() == "tello" and not self.config.mock_camera:
                self._client.stream_off()
        except Exception:  # noqa: BLE001
            pass
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:  # noqa: BLE001
                pass
        self._camera = None
        self._client.close()
        self._connected = False
        self._control_active = False
        self._offboard_active = False

    def __enter__(self) -> "TelloDroneEnv":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False
