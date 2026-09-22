"""``AirSimDroneEnv`` — V0 runnable AirSim env (Plan-A continuous-velocity stepping).

This unifies the *tested* probe call sequences from
``experiments/aerial/sim_verify/probes/{t2_capability.py, l2f_clean_retest.py}``
into a gym-like ``reset / step / observe / close`` interface. It replaces the
teleport+discrete ``OpenFlyBridge`` (``eval/run_closed_loop.py``) with a physics
step: ``moveByVelocityAsync(vx, vy, vz, dt).join()`` at ``step_hz``.

The renderer is SINGLE-CONSUMER (one client on :41451) — this env holds exactly
one connection; parallelism lives on the imagination side, not here.

Real-AirSim deps (``airsim``, ``cv2``) are imported lazily inside methods so this
module (and the whole ``rl`` package) imports cleanly on hosts without them, and
the mock env / unit tests never touch AirSim. For offline use see
``mock_env.MockAirSimDroneEnv`` (same surface).
"""
from __future__ import annotations

import math
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from experiments.aerial.rl.env.action import (
    body_delta_limits,
    body_delta_to_velocity_ned,
    clip_body_delta,
)
from experiments.aerial.rl.env.obs import Observation, depth_sanity_detail

# Reuse the already-validated numerical gates rather than re-deriving them.
_SIM_VERIFY = pathlib.Path(__file__).resolve().parents[2] / "sim_verify"
if str(_SIM_VERIFY) not in sys.path:
    sys.path.insert(0, str(_SIM_VERIFY))
try:  # pragma: no cover - import shim; exercised on the 4090 host
    from lib import sanity  # type: ignore
except Exception:  # noqa: BLE001
    sanity = None  # type: ignore


@dataclass
class AirSimEnvConfig:
    """Mirrors the ``sim_verify/config.env`` schema (+ RL step rate)."""

    host: str = "127.0.0.1"
    port: int = 41451
    camera: str = "front_custom"
    vehicle: str = "drone_1"
    width: int = 224
    height: int = 224
    step_hz: float = 30.0
    takeoff_z: float = -3.0            # NED start altitude (climb 3 m)
    health_check: bool = True          # sanity-gate IMU + depth on reset
    warmup_frames: int = 1
    # Per-step DepthPlanar grab. Cross-net DepthPlanar@224 is ~250 ms (vs Scene
    # JPEG ~30 ms), so enabling it drops the closed loop from ~14 Hz to ~3 Hz on
    # the H100→4090 path. Health-check on reset still grabs depth once when
    # health_check=True. Set False for Plan-A rate smokes / RGB-first collection.
    grab_depth: bool = True
    fanout_rgb: bool = False
    wam_encode_size: int = 224

    @classmethod
    def from_env(cls, overrides: Optional[Dict[str, Any]] = None) -> "AirSimEnvConfig":
        """Read from ``AIRSIM_*`` / ``L2F_*`` env vars (as ``config.env`` sets)."""
        cfg = cls(
            host=os.environ.get("AIRSIM_HOST", cls.host),
            port=int(os.environ.get("AIRSIM_PORT", cls.port)),
            camera=os.environ.get("AIRSIM_CAMERA", cls.camera),
            vehicle=os.environ.get("AIRSIM_VEHICLE", cls.vehicle),
            width=int(os.environ.get("L2F_W", cls.width)),
            height=int(os.environ.get("L2F_H", cls.height)),
            step_hz=float(os.environ.get("RL_STEP_HZ", cls.step_hz)),
        )
        if overrides:
            for k, v in overrides.items():
                setattr(cfg, k, v)
        return cfg


class AirSimDroneEnv:
    """Continuous-velocity AirSim drone env producing rich observations."""

    def __init__(self, config: Optional[AirSimEnvConfig] = None, **kwargs: Any) -> None:
        self.config = config or AirSimEnvConfig(**kwargs)
        self._client: Any = None
        self._airsim: Any = None
        self._goal: Optional[np.ndarray] = None
        self._t0 = time.perf_counter()

    # -- connection -------------------------------------------------------
    @property
    def _vk(self) -> Dict[str, str]:
        """vehicle-name kwarg splat (l2f_clean_retest.py pattern)."""
        return {"vehicle_name": self.config.vehicle}

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client
        import airsim  # type: ignore  # lazy: absent off the 4090

        self._airsim = airsim
        client = airsim.MultirotorClient(ip=self.config.host, port=self.config.port)
        client.confirmConnection()
        self._client = client
        return client

    # -- lifecycle --------------------------------------------------------
    def reset(self, episode: Optional[Dict[str, Any]] = None) -> Observation:
        """Arm + place the drone at the episode start; return the first obs.

        ``episode`` uses the OpenFly annotation shape (``pos`` [N,3], ``yaw``
        [N]); ``pos[0]`` is the start and ``pos[-1]`` the goal (for the reward).
        """
        client = self._connect()
        airsim = self._airsim

        start = np.zeros(3, dtype=np.float64)
        yaw0 = 0.0
        if episode is not None:
            positions = np.asarray(episode["pos"], dtype=np.float64)
            yaws = np.asarray(episode["yaw"], dtype=np.float64).reshape(-1)
            start = positions[0].copy()
            yaw0 = float(yaws[0])
            self._goal = positions[-1].copy()
        else:
            self._goal = None

        client.enableApiControl(True, **self._vk)
        client.armDisarm(True, **self._vk)
        # Past this point control is armed; guarantee release on any failure so a
        # crashed reset never leaves the single-consumer renderer occupied.
        paused = False
        try:
            # Episode positions are +up world; AirSim pose is NED, so negate z.
            pose = airsim.Pose(
                airsim.Vector3r(float(start[0]), float(start[1]), -float(start[2])),
                airsim.to_quaternion(0.0, 0.0, float(yaw0)),
            )
            # Pause physics while teleporting + first observe. On some Outdoor
            # hosts (.110) an unpaused reset drops through collision (~25 m Z)
            # during the depth grab RPC and fails spawn gates.
            if hasattr(client, "simPause"):
                client.simPause(True)
                paused = True
            client.simSetVehiclePose(pose, True, **self._vk)

            for _ in range(max(0, self.config.warmup_frames)):
                self._grab_scene(client)

            # Only force a depth frame on reset when grab_depth is enabled.
            # On Outdoor hosts (.110) DepthPlanar can hang indefinitely, which
            # stalls every episode reset and blocks the entire eval.
            obs = self.observe(force_depth=self.config.grab_depth)
            if episode is not None:
                p0 = np.asarray(obs.position, dtype=np.float64).reshape(3)
                if float(np.linalg.norm(p0 - start)) > 5.0:
                    client.simSetVehiclePose(pose, True, **self._vk)
                    obs = self.observe(force_depth=self.config.grab_depth)
            if paused:
                client.simPause(False)
                paused = False
                # Paused IMU is often near-zero (fails lin_acc gate). Unpause,
                # re-seat, and refresh IMU + kinematics only — do NOT re-grab
                # depth (that RPC is long enough for another free-fall).
                client.simSetVehiclePose(pose, True, **self._vk)
                try:
                    client.moveByVelocityAsync(0.0, 0.0, 0.0, 0.05, **self._vk)
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.05)
                client.simSetVehiclePose(pose, True, **self._vk)
                imu = self._grab_imu(client)
                if imu:
                    obs.imu = imu
                obs.state = np.asarray(self.observe_state(), dtype=np.float32)
            if self.config.health_check:
                self._assert_healthy(obs)
            return obs
        except Exception:  # noqa: BLE001 - release control, then re-raise
            self.close()
            raise
        finally:
            if paused:
                try:
                    client.simPause(False)
                except Exception:  # noqa: BLE001
                    pass

    def step(self, action: np.ndarray) -> tuple[Observation, Dict[str, Any]]:
        """Execute a 4-D body delta as a velocity command over ``1/step_hz``.

        Wall-clock rate lock: ``moveByVelocityAsync(..., duration=dt).join()``
        makes wall time ≥ dt + RPC, so ``achieved_hz`` can never match
        ``step_hz`` (V0 commanded 12 → got 7.9; command-8 probes got ~6).
        Fire the velocity command without joining, sleep to the labeled
        deadline, then observe — commanded dt == wall dt. The next step's
        velocity command replaces this one on the renderer.
        """
        client = self._connect()
        airsim = self._airsim
        dt = 1.0 / float(self.config.step_hz)
        cmd = clip_body_delta(action, body_delta_limits(dt))

        # Rate-lock the whole step (yaw read + async move + observe) to `dt`.
        # Putting t0 after observe_state left a naked RPC outside the pad and
        # capped achieved Hz at ~7 when commanding 8.
        t0 = time.perf_counter()
        yaw = self.observe_state()[6]
        vx, vy, vz_ned, yaw_rate_deg = body_delta_to_velocity_ned(cmd, yaw, dt)
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_deg)
        client.moveByVelocityAsync(
            vx, vy, vz_ned, dt, yaw_mode=yaw_mode, **self._vk
        )
        # Leave headroom for observe() so the labeled deadline is hit after the
        # frame returns. RGB Scene JPEG is ~20 ms local / ~30 ms cross-net;
        # DepthPlanar@224 is ~100 ms local (4090 loopback 2026-08-04 bench) so
        # the old fixed 40 ms budget overshot by ~80 ms/step whenever
        # grab_depth=True and falsely capped achieved Hz below commanded.
        observe_budget = 0.15 if self.config.grab_depth else 0.04
        remaining = dt - (time.perf_counter() - t0)
        if remaining > observe_budget:
            time.sleep(remaining - observe_budget)
        obs = self.observe()
        remaining = dt - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)

        info = {"cmd": cmd.tolist(), "vx": vx, "vy": vy, "vz_ned": vz_ned}
        return obs, info

    def observe(self, *, force_depth: bool = False) -> Observation:
        import cv2  # type: ignore

        client = self._connect()
        native_rgb = self._grab_scene_native(client)
        rgb_yolo: Optional[np.ndarray] = None
        rgb_vio: Optional[np.ndarray] = None
        if self.config.fanout_rgb:
            wam_sz = int(self.config.wam_encode_size)
            rgb = cv2.resize(
                native_rgb,
                (wam_sz, wam_sz),
                interpolation=cv2.INTER_LINEAR,
            )
            det = native_rgb
            if (det.shape[1], det.shape[0]) != (self.config.width, self.config.height):
                det = cv2.resize(
                    det,
                    (self.config.width, self.config.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            rgb_yolo = np.ascontiguousarray(det, dtype=np.uint8)
            rgb_vio = rgb_yolo
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        else:
            rgb = native_rgb
            if (rgb.shape[1], rgb.shape[0]) != (self.config.width, self.config.height):
                rgb = cv2.resize(
                    rgb,
                    (self.config.width, self.config.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        # Depth is optional per-step (see ``grab_depth``); force it for the
        # one-shot health check on reset even when per-step grabs are off.
        want_depth = force_depth or self.config.grab_depth
        depth = self._grab_depth(client) if want_depth else None
        imu = self._grab_imu(client)
        collided = self._grab_collision(client)
        state = self.observe_state()
        return Observation(
            rgb=rgb,
            state=state,
            collided=collided,
            depth=depth,
            imu=imu,
            t=time.perf_counter() - self._t0,
            info={"goal": None if self._goal is None else self._goal.tolist()},
            rgb_yolo=rgb_yolo,
            rgb_vio=rgb_vio,
        )

    def observe_state(self) -> np.ndarray:
        client = self._connect()
        k = client.getMultirotorState(**self._vk).kinematics_estimated
        p, v, o = k.position, k.linear_velocity, k.orientation
        _, _, yaw = self._airsim.to_eularian_angles(o)
        # AirSim is NED (z down); expose the +up world frame the rest of the
        # stack (action dz>0=up, mock, reward) uses. Negate z and vz only.
        return np.array(
            [p.x_val, p.y_val, -p.z_val, v.x_val, v.y_val, -v.z_val, yaw],
            dtype=np.float32,
        )

    def close(self) -> None:
        if self._client is None:
            return
        try:  # best-effort disarm (t2_capability.py:312-313)
            self._client.armDisarm(False, **self._vk)
            self._client.enableApiControl(False, **self._vk)
        except Exception:  # noqa: BLE001
            pass
        self._client = None

    def __enter__(self) -> "AirSimDroneEnv":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    @property
    def goal(self) -> Optional[np.ndarray]:
        return self._goal

    # -- capture helpers (lifted from the probes) -------------------------
    def _grab_scene_native(self, client: Any) -> np.ndarray:
        import cv2  # type: ignore

        airsim = self._airsim
        rq = [airsim.ImageRequest(self.config.camera, airsim.ImageType.Scene, False, True)]
        r = client.simGetImages(rq, **self._vk)[0]
        raw = np.frombuffer(bytes(r.image_data_uint8), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise RuntimeError("empty/undecodable Scene buffer")
        return np.ascontiguousarray(img[..., ::-1], dtype=np.uint8)  # RGB

    def _grab_scene(self, client: Any) -> np.ndarray:
        import cv2  # type: ignore

        rgb = self._grab_scene_native(client)
        if (rgb.shape[1], rgb.shape[0]) != (self.config.width, self.config.height):
            rgb = cv2.resize(
                rgb,
                (self.config.width, self.config.height),
                interpolation=cv2.INTER_LINEAR,
            )
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def _grab_depth(self, client: Any) -> Optional[np.ndarray]:
        airsim = self._airsim
        try:
            rq = [airsim.ImageRequest(self.config.camera, airsim.ImageType.DepthPlanar, True, False)]
            r = client.simGetImages(rq, **self._vk)[0]
            if not (r.width and r.height):
                return None
            depth = np.asarray(r.image_data_float, dtype=np.float32).reshape(r.height, r.width)
            return depth
        except Exception:  # noqa: BLE001
            return None

    def _grab_imu(self, client: Any) -> Dict[str, Any]:
        try:
            d = client.getImuData(**self._vk)
            return {
                "ang_vel": [d.angular_velocity.x_val, d.angular_velocity.y_val, d.angular_velocity.z_val],
                "lin_acc": [d.linear_acceleration.x_val, d.linear_acceleration.y_val, d.linear_acceleration.z_val],
            }
        except Exception:  # noqa: BLE001
            return {}

    def _grab_collision(self, client: Any) -> bool:
        try:
            return bool(client.simGetCollisionInfo(**self._vk).has_collided)
        except Exception:  # noqa: BLE001
            return False

    def _assert_healthy(self, obs: Observation) -> None:
        """Fail fast on a dead/CV-only renderer.

        Missing sensors are a *failure*, not a pass: a CV-only build returns no
        IMU/depth, and silently skipping the gate there would green-light a
        renderer that can't produce the reward/supervision signals V0 needs.
        """
        if sanity is None:
            raise RuntimeError(
                "health_check requested but sim_verify sanity gates are "
                "unavailable — cannot verify the renderer; run on the 4090 host "
                "or set health_check=False deliberately."
            )
        if not obs.imu:
            raise RuntimeError(
                "no IMU data on reset — renderer/sensors unavailable (CV-only "
                "mode or dead bridge)."
            )
        ok, note = sanity.imu_ok(obs.imu)
        if not ok:
            raise RuntimeError(f"IMU sanity failed on reset: {note}")
        if self.config.grab_depth:
            if obs.depth is None:
                raise RuntimeError(
                    "no depth data on reset — renderer/sensors unavailable (CV-only "
                    "mode or dead bridge)."
                )
            ok, note = sanity.depth_ok(depth_sanity_detail(obs.depth))
            if not ok:
                raise RuntimeError(f"depth sanity failed on reset: {note}")
