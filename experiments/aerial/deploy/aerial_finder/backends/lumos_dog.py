"""Lumos robot dog backend: SJCAM frames, IMU yaw, walk commands through h12_lud_bridge.

``step`` turns a body delta into a walk command: dx -> forward speed, dyaw -> a heading target that the
10 Hz send loop tracks on the IMU yaw (the dog turns at ~1/4 of the commanded rate and ignores rates
below ~0.2 rad/s, so open-loop turn rates under- and then over-shoot). Forward motion is held while the
heading error is above ``turn_gate_deg``. dy / dz are dropped. ``cmd_mode="direct"`` sends the action as a command: dx / 1.0 -> vx / vx_max and dyaw / 0.314 ->
clockwise vyaw / vyaw_max. The navigator scores the measured displacement of that command, not the raw box value.
Commands go to the bridge as ``{"src": "script"}`` with the gait and the speed in one packet.
The bridge holds that gait: the handle's repeated stand packets do not flip the mode. E-stop, a
moved stick, or the gait switch actually changing still takes over. Bridge sign convention
(confirmed on the dog): vx > 0 forward, vyaw > 0 turns left.

Pose is dead reckoning: position integrates ``vx_gain`` times the forward speed the bridge accepted, along the IMU yaw.
Height is fixed at ``cam_height_m`` (the camera; set the mission's home_agl_m to the same value).
"""
from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ..geo import GeoRef, wrap
from ..types import Frame, Intrinsics, VehicleState

log = logging.getLogger("aerial_finder.lumos_dog")
IMU_CHANNEL = "lcm_imu_data"
MODEL_LIMITS = (1.0, 0.4, 0.4, 0.3141592653589793)


class _Camera:
    def __init__(self, dev: str, width: int, height: int) -> None:
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.at = 0.0
        self.n = 0
        self.run = True
        self.cap = None
        self.remote = dev.startswith("http://") or dev.startswith("https://")
        if self.remote:
            self.url = dev
            threading.Thread(target=self._http_loop, daemon=True).start()
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with self.lock:
                    ready = self.frame is not None
                if ready:
                    break
                time.sleep(0.05)
            else:
                self.run = False
                raise RuntimeError(f"cannot open camera {dev}")
            return
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {dev}")
        threading.Thread(target=self._loop, daemon=True).start()

    def _store(self, bgr: np.ndarray) -> None:
        with self.lock:
            self.frame, self.at, self.n = bgr, time.time(), self.n + 1

    def _loop(self) -> None:
        while self.run:
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                time.sleep(0.02)
                continue
            self._store(bgr)

    def _http_loop(self) -> None:
        """Pull the newest JPEG. A frame the Mac already considers old is not stored."""
        while self.run:
            try:
                req = urllib.request.Request(self.url, headers={"Cache-Control": "no-cache"})
                with urllib.request.urlopen(req, timeout=0.8) as resp:
                    age = float(resp.headers.get("X-Frame-Age", "0") or 0)
                    buf = resp.read()
                if age > 0.8:
                    time.sleep(0.05)
                    continue
                bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    time.sleep(0.02)
                    continue
                self._store(bgr)
            except Exception:
                time.sleep(0.05)

    def latest(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self.lock:
            return self.frame, self.at, self.n

    def close(self) -> None:
        self.run = False
        time.sleep(0.1)
        if self.cap is not None:
            self.cap.release()


class _Imu:
    def __init__(self, url: str, sdk_roots: Sequence[str]) -> None:
        for root in sdk_roots:
            p = Path(root).expanduser() / "lcm_typedef" / "python"
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        import lcm
        from imu_data_lcmt import imu_data_lcmt

        self.type = imu_data_lcmt
        self.lc = lcm.LCM(url)
        self.lc.subscribe(IMU_CHANNEL, self._on)
        self.lock = threading.Lock()
        self.rpy: Optional[Tuple[float, float, float]] = None
        self.at = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _on(self, _ch: str, data: bytes) -> None:
        m = self.type.decode(data)
        with self.lock:
            self.rpy, self.at = tuple(float(v) for v in m.navRPY), time.time()

    def _loop(self) -> None:
        while True:
            self.lc.handle_timeout(200)

    def get(self) -> Optional[Tuple[float, float, float]]:
        with self.lock:
            return None if self.rpy is None or time.time() - self.at > 0.5 else self.rpy


class LumosDogBackend:
    def __init__(self, camera: str = "/dev/video0", width: int = 1280, height: int = 720, hfov_deg: float = 110.0,
                 crop_bottom_frac: float = 0.1, step_hz: float = 5.0, bridge: str = "http://127.0.0.1:8092",
                 lcm_url: str = "udpm://239.255.76.67:7667?ttl=255",
                 sdk_roots: Sequence[str] = ("/home/ubuntu/lumos_sdk", "~/lumos_sdk"),
                 vx_max: float = 0.3, vyaw_max: float = 0.4, cam_height_m: float = 0.6,
                 imu_yaw_sign: Optional[float] = None, calib_vyaw: float = 0.25, calib_s: float = 2.0,
                 ready_wait_s: float = 30.0, dry: bool = False, geo: Sequence[float] = (0.0, 0.0, 0.0),
                 save_dir: str = "", save_every_s: float = 1.0, yaw_k: float = 1.5, vyaw_min: float = 0.25,
                 yaw_tol_deg: float = 3.0, turn_gate_deg: float = 15.0, max_lead_deg: float = 45.0,
                 cmd_mode: str = "heading", cmd_ttl_s: float = 0.6, fisheye_calib: str = "",
                 vx_gain: float = 0.57) -> None:
        if cmd_mode not in ("heading", "direct"):
            raise ValueError(f"cmd_mode must be heading|direct, got {cmd_mode!r}")
        self.cmd_mode = cmd_mode
        self.cmd_ttl_s = float(cmd_ttl_s)
        self.save_dir = Path(save_dir).expanduser() / time.strftime("%Y%m%d_%H%M%S") if save_dir else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_s = float(save_every_s)
        self._t_save = 0.0
        self.step_hz = float(step_hz)
        self.bridge = bridge.rstrip("/")
        self.vx_max, self.vyaw_max = float(vx_max), float(vyaw_max)
        self.vx_gain = float(vx_gain)
        self.cam_height_m = float(cam_height_m)
        self.crop_bottom_frac = float(crop_bottom_frac)
        self.hfov_deg = float(hfov_deg)
        self.yaw_sign = imu_yaw_sign
        self.calib_vyaw, self.calib_s, self.ready_wait_s = float(calib_vyaw), float(calib_s), float(ready_wait_s)
        self.dry = bool(dry)
        self.yaw_k, self.vyaw_min = float(yaw_k), float(vyaw_min)
        self.yaw_tol, self.turn_gate = math.radians(yaw_tol_deg), math.radians(turn_gate_deg)
        self.max_lead = math.radians(max_lead_deg)
        self.track = False
        self.vx_des = 0.0
        self.yaw_tgt = 0.0
        self._exec_from: Optional[Tuple[float, float, float]] = None
        self.geo_ref = GeoRef(*(float(v) for v in geo))
        self.rectifier = None
        if fisheye_calib:
            from ..fisheye import FisheyeRectifier
            self.rectifier = FisheyeRectifier(fisheye_calib)
        self.cam = _Camera(camera, width, height)
        self.imu = _Imu(lcm_url, sdk_roots)
        self.lock = threading.Lock()
        self.cmd = (0.0, 0.0)
        self.cmd_at = 0.0
        # Walk with a zero speed backs the dog up, so idle is STAND and WALK is only used while a command is live.
        self.mode = "stand"
        self._STAND = 2
        self._WALK = 12
        self.accepted = False
        self.last_reply: dict = {}
        self.pos = np.zeros(2)
        self.yaw0: Optional[float] = None
        self._t_odom: Optional[float] = None
        self._t0 = time.perf_counter()
        self._run = True
        threading.Thread(target=self._send_loop, daemon=True).start()
        self._wait_sensors()

    # -- plumbing -----------------------------------------------------------
    def _wait_sensors(self) -> None:
        t0 = time.time()
        while time.time() - t0 < 15.0:
            if self.cam.latest()[2] >= 30 and self.imu.get() is not None:
                return
            time.sleep(0.1)
        raise RuntimeError("camera or lcm_imu_data not available")

    def _post_json(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.bridge + "/api/cmd", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=0.3).read())

    def _post_gait(self, state: int, vx: float, vyaw: float) -> dict:
        return self._post_json({
            "src": "script", "state": int(state), "vx": vx, "vy": 0.0, "vyaw": vyaw, "stop": False,
        })

    def health(self) -> dict:
        return json.loads(urllib.request.urlopen(self.bridge + "/api/health", timeout=0.5).read())

    def _send_loop(self) -> None:
        """Re-sends the newest command at 10 Hz; goes quiet when it is older than ``cmd_ttl_s`` (bridge then zeros)."""
        while self._run:
            t = time.time()
            with self.lock:
                if self.track and self.mode == "walk":
                    self.cmd = self._track()
                mode = self.mode
                (vx, vyaw), at = self.cmd, self.cmd_at
            self._integrate(t)
            moving = abs(vx) > 1e-4 or abs(vyaw) > 1e-4
            fresh = t - at <= self.cmd_ttl_s
            try:
                if not self.dry and mode == "walk" and moving and fresh:
                    rep = self._post_gait(self._WALK, vx, vyaw)
                    self.last_reply, self.accepted = rep, bool(rep.get("script_accepted"))
                elif not self.dry:
                    self.last_reply = self._post_gait(self._STAND, 0.0, 0.0)
                    self.accepted = False
                else:
                    self.accepted = False
            except Exception as exc:  # noqa: BLE001
                self.last_reply, self.accepted = {"error": str(exc)}, False
            time.sleep(max(0.0, 0.1 - (time.time() - t)))

    def _set(self, vx: float, vyaw: float) -> None:
        with self.lock:
            self.track = False
            self.cmd, self.cmd_at = (float(vx), float(vyaw)), time.time()
            self.mode = "walk" if abs(vx) > 1e-4 or abs(vyaw) > 1e-4 else "stand"

    def _track(self) -> Tuple[float, float]:
        """(vx, bridge vyaw) toward yaw_tgt; called with the lock held."""
        yaw = self._yaw()
        err = wrap(self.yaw_tgt - yaw)
        if abs(err) > self.max_lead:
            self.yaw_tgt = wrap(yaw + math.copysign(self.max_lead, err))
            err = math.copysign(self.max_lead, err)
        rate = 0.0
        if abs(err) > self.yaw_tol:
            rate = float(np.clip(self.yaw_k * abs(err), self.vyaw_min, self.vyaw_max))
        vx = self.vx_des if abs(err) <= self.turn_gate else 0.0
        # Heading is clockwise positive; the bridge's vyaw > 0 turns left.
        return vx, -math.copysign(rate, err)

    def _raw_yaw(self) -> Optional[float]:
        r = self.imu.get()
        return None if r is None else r[2]

    def _yaw(self) -> float:
        """World yaw, north -> east (clockwise from above), zero at the start heading."""
        y = self._raw_yaw()
        if y is None or self.yaw0 is None or self.yaw_sign is None:
            return 0.0
        return wrap(-self.yaw_sign * (y - self.yaw0))

    def _integrate(self, t: float) -> None:
        if self._t_odom is not None and self.accepted:
            dt = min(0.5, t - self._t_odom)
            yaw = self._yaw()
            # Bridge vx is the speed it actually published. The first 0.3 s of a
            # gait change is zero, so integrating the requested command would invent motion.
            vx = float(self.last_reply.get("vx") or 0.0)
            self.pos += vx * self.vx_gain * dt * np.array([math.cos(yaw), math.sin(yaw)])
        self._t_odom = t

    # -- Backend API ----------------------------------------------------------
    def prepare(self) -> None:
        """Mission start: wait for H12 hand-off, find the IMU yaw sign with a short left turn, zero the pose."""
        if self.dry:
            self.yaw_sign = 1.0 if self.yaw_sign is None else self.yaw_sign
        else:
            # Stay in STAND until a real command. A zero walk speed makes this dog step backward.
            if self.yaw_sign is None:
                y0 = self._raw_yaw()
                t1 = time.time()
                while time.time() - t1 < self.calib_s:
                    self._set(0.0, self.calib_vyaw)
                    time.sleep(0.1)
                self._set(0.0, 0.0)
                time.sleep(0.8)
                y1 = self._raw_yaw()
                dyaw = math.degrees(wrap(y1 - y0)) if y0 is not None and y1 is not None else 0.0
                if abs(dyaw) < 4.0:
                    raise RuntimeError(f"left turn moved IMU yaw only {dyaw:.1f} deg; is the dog walking?")
                self.yaw_sign = 1.0 if dyaw > 0 else -1.0
                log.info("imu yaw sign %+.0f (left turn %.1f deg)", self.yaw_sign, dyaw)
        self.yaw0 = self._raw_yaw()
        self.pos[:] = 0.0

    def observe(self) -> Tuple[Frame, VehicleState]:
        bgr, at, _n = self.cam.latest()
        if bgr is None or time.time() - at > 1.0:
            raise RuntimeError("stale camera frame")
        h, w = bgr.shape[:2]
        valid = None
        if self.rectifier is not None:
            bgr, valid = self.rectifier.apply(bgr)
            intr = self.rectifier.intr
        else:
            keep = int(round(h * (1.0 - self.crop_bottom_frac)))
            bgr = bgr[:keep]
            f = 0.5 * w / math.tan(math.radians(self.hfov_deg) / 2.0)
            intr = Intrinsics(f, f, 0.5 * w, 0.5 * bgr.shape[0], int(w), int(bgr.shape[0]))
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        st = VehicleState(pos=np.array([self.pos[0], self.pos[1], self.cam_height_m]), yaw=self._yaw(),
                          t=time.perf_counter() - self._t0)
        if self.save_dir is not None and time.time() - self._t_save >= self.save_every_s:
            self._t_save = time.time()
            cv2.imwrite(str(self.save_dir / f"{st.t:07.1f}_x{st.pos[0]:+.1f}_y{st.pos[1]:+.1f}"
                                            f"_yaw{math.degrees(st.yaw):+.0f}.jpg"), bgr)
        return Frame(rgb=rgb, rgb_intr=intr, valid=valid, t=st.t), st

    def step(self, action: np.ndarray) -> None:
        t0 = time.perf_counter()
        dx, _dy, _dz, dyaw = (float(v) for v in np.asarray(action, np.float64))
        if self.cmd_mode == "direct":
            vx = float(np.clip(dx / MODEL_LIMITS[0], 0.0, 1.0)) * self.vx_max
            vyaw_cw = float(np.clip(dyaw / MODEL_LIMITS[3], -1.0, 1.0)) * self.vyaw_max
            self._set(vx, -vyaw_cw)
            rest = 1.0 / self.step_hz - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)
            # Leave the command running through the next depth pass. A zero command
            # is STAND, because WALK at zero speed steps backward.
            return
        with self.lock:
            if not self.track:
                self.yaw_tgt = self._yaw()
            self.track = True
            self.vx_des = float(np.clip(dx * self.step_hz, 0.0, self.vx_max))
            self.yaw_tgt = wrap(self.yaw_tgt + dyaw)
            self.cmd, self.cmd_at = self._track(), time.time()
        rest = 1.0 / self.step_hz - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)

    def executed(self, action: np.ndarray) -> np.ndarray:
        """What the dog did since the previous call, as a model action (dx forward m, dyaw clockwise rad)."""
        now = (self.pos[0], self.pos[1], self._yaw())
        prev, self._exec_from = self._exec_from, now
        a = np.zeros(4)
        if prev is not None:
            a[0] = math.hypot(now[0] - prev[0], now[1] - prev[1])
            a[3] = wrap(now[2] - prev[2])
        return a

    def hover(self) -> None:
        self._set(0.0, 0.0)

    def status(self) -> dict:
        return {"cmd": [round(v, 3) for v in self.cmd], "accepted": self.accepted, "pos": self.pos.round(2).tolist(),
                "yaw_deg": round(math.degrees(self._yaw()), 1), "bridge": self.last_reply}

    def close(self) -> None:
        self.hover()
        time.sleep(0.8)
        self._run = False
        self.cam.close()
