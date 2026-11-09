#!/usr/bin/env python3
"""Tello ground-console API backed by the Lumos dog, so Tello mission scripts run unchanged.

Serves the endpoints ``tello_consequence_collect`` / ``tello_find_object`` use:
  GET  /api/status      height fixed at --height-cm while "airborne", yaw from lcm_imu_data,
                        vgx/vgy from the accepted walk command (dead reckoning, Tello sign convention)
  GET  /snapshot.jpg    newest camera frame
  POST /api/rc          b (forward) and d (yaw, clockwise+) -> {"src": "auto"} walk commands to h12_lud_bridge
  POST /api/cmd         takeoff = calibrate the IMU yaw sign with a short left turn, then accept rc;
                        land / emergency = zero speed and stop accepting rc
  POST /api/record, /api/control_mode   bookkeeping only

The bridge only lets these commands through while H12 is in WALK with centred sticks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lud_policy_walk import LCM_URL, Camera, Imu, Sender  # noqa: E402

RC_HOLD_S = 1.0


class Dog:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cam = Camera(args.camera, args.width, args.height)
        self.imu = Imu(LCM_URL)
        self.sender = Sender(args.bridge, 10.0, enabled=True)
        self.lock = threading.Lock()
        self.airborne = False
        self.yaw_sign: Optional[float] = None
        self.rc = (0, 0, 0, 0)
        self.rc_at = 0.0
        self.cmd = (0.0, 0.0)
        self.health: Dict[str, Any] = {}
        self.run_dir = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(self.run_dir / "shim.jsonl", "a", buffering=1)
        threading.Thread(target=self._drive_loop, daemon=True).start()
        threading.Thread(target=self._health_loop, daemon=True).start()

    def note(self, kind: str, **kw: Any) -> None:
        row = {"t": round(time.time(), 3), "kind": kind, **kw}
        self.log.write(json.dumps(row) + "\n")
        print(kind, json.dumps(kw), flush=True)

    def _health_loop(self) -> None:
        url = self.args.bridge.rstrip("/") + "/api/health"
        while True:
            try:
                self.health = json.loads(urllib.request.urlopen(url, timeout=0.5).read())
            except Exception as exc:  # noqa: BLE001
                self.health = {"error": str(exc)}
            time.sleep(0.5)

    def _drive_loop(self) -> None:
        while True:
            with self.lock:
                a, b, c, d = self.rc
                fresh = self.airborne and time.time() - self.rc_at <= RC_HOLD_S
            if fresh:
                vx = min(self.args.vx_max, max(0.0, b / 100.0 * self.args.vx_full))
                vyaw = max(-self.args.vyaw_max, min(self.args.vyaw_max, -d / 100.0 * self.args.vyaw_full))
            else:
                vx = vyaw = 0.0
            self.cmd = (vx, vyaw)
            self.sender.set(vx, vyaw)
            time.sleep(0.1)

    def raw_yaw(self) -> Optional[float]:
        y, age = self.imu.yaw()
        return None if y is None or age > 0.5 else y

    def tello_yaw_deg(self) -> float:
        y = self.raw_yaw()
        if y is None or self.yaw_sign is None:
            return 0.0
        # Tello yaw is clockwise positive.
        return math.degrees(-self.yaw_sign * y)

    def takeoff(self) -> Dict[str, Any]:
        t0 = time.time()
        while not self.health.get("auto_ready"):
            if time.time() - t0 > self.args.ready_wait_s:
                return {"ok": False, "error": "H12 not in WALK with centred sticks"}
            time.sleep(0.2)
        y0 = self.raw_yaw()
        if y0 is None:
            return {"ok": False, "error": "no imu"}
        t1 = time.time()
        while time.time() - t1 < 2.0:
            self.sender.set(0.0, 0.25)
            time.sleep(0.1)
        self.sender.set(0.0, 0.0)
        time.sleep(0.8)
        y1 = self.raw_yaw()
        dyaw = math.degrees(math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))) if y1 is not None else 0.0
        if abs(dyaw) < 4.0:
            self.note("takeoff_fail", dyaw_deg=dyaw, bridge=self.sender.last_reply)
            return {"ok": False, "error": f"left turn moved imu yaw only {dyaw:.1f} deg"}
        with self.lock:
            self.yaw_sign = 1.0 if dyaw > 0 else -1.0
            self.airborne = True
            self.rc, self.rc_at = (0, 0, 0, 0), time.time()
        self.note("takeoff", dyaw_left_deg=round(dyaw, 1), imu_yaw_sign=self.yaw_sign)
        return {"ok": True, "resp": "ok", "imu_yaw_sign": self.yaw_sign}

    def land(self, why: str) -> Dict[str, Any]:
        with self.lock:
            self.airborne = False
            self.rc = (0, 0, 0, 0)
        self.sender.set(0.0, 0.0)
        self.note("land", why=why)
        return {"ok": True, "resp": "ok"}

    def status(self) -> Dict[str, Any]:
        _bgr, f_at, _n = self.cam.latest()
        h = self.health
        pct = h.get("battery_pct")
        if pct is None and h.get("battery_v"):
            pct = int(max(0.0, min(100.0, (float(h["battery_v"]) - 48.0) / (54.6 - 48.0) * 100.0)))
        yaw_deg = self.tello_yaw_deg()
        vx, _ = self.cmd
        moving = vx if (self.airborne and self.sender.accepted) else 0.0
        psi = math.radians(yaw_deg)
        imu_ok = self.raw_yaw() is not None
        ok = imu_ok and "error" not in h and (not self.airborne or h.get("auto_ready", False))
        return {
            "ok": bool(ok),
            "battery": pct,
            "wifi_rssi_dbm": None,
            "allow_fly": True,
            "airborne": self.airborne,
            "temp": 40,
            "frame_age_s": round(time.time() - f_at, 2) if f_at else None,
            "height_cm": self.args.height_cm if self.airborne else 0,
            "tof": self.args.height_cm if self.airborne else 10,
            "yaw": round(yaw_deg, 1),
            "pitch": 0.0,
            "tello": {"vgx": -moving * 10.0 * math.cos(psi), "vgy": -moving * 10.0 * math.sin(psi), "vgz": 0.0},
            "dog": {"auto_ready": h.get("auto_ready"), "auto_active": h.get("auto_active"),
                    "accepted": self.sender.accepted, "cmd": [round(v, 3) for v in self.cmd]},
        }


def make_handler(dog: Dog):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return

        def _json(self, obj: Dict[str, Any], code: int = 200) -> None:
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path.startswith("/api/status"):
                self._json(dog.status())
            elif self.path.startswith("/snapshot.jpg"):
                bgr, _at, _n = dog.cam.latest()
                if bgr is None:
                    self._json({"ok": False, "error": "no frame"}, 503)
                    return
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                raw = buf.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            else:
                self._json({"ok": False, "error": "not found"}, 404)

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path.startswith("/api/rc"):
                with dog.lock:
                    dog.rc = tuple(int(body.get(k, 0)) for k in "abcd")
                    dog.rc_at = time.time()
                self._json({"ok": True})
            elif self.path.startswith("/api/cmd"):
                c = str(body.get("cmd", ""))
                if c == "takeoff":
                    self._json(dog.takeoff())
                elif c in ("land", "emergency") or c.startswith("forward"):
                    self._json(dog.land(c))
                else:
                    self._json({"ok": True, "resp": "ignored"})
            elif self.path.startswith("/api/record"):
                self._json({"ok": True, "dir": str(dog.run_dir)})
            elif self.path.startswith("/api/control_mode"):
                self._json({"ok": True, "mode": body.get("mode")})
            else:
                self._json({"ok": False, "error": "not found"}, 404)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--bridge", default="http://127.0.0.1:8092")
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--height-cm", type=int, default=120, help="reported while airborne; keeps altitude logic idle")
    ap.add_argument("--vx-full", type=float, default=0.7, help="dog m/s at Tello forward stick 100")
    ap.add_argument("--vx-max", type=float, default=0.35)
    ap.add_argument("--vyaw-full", type=float, default=1.0, help="dog rad/s at Tello yaw stick 100")
    ap.add_argument("--vyaw-max", type=float, default=0.4)
    ap.add_argument("--ready-wait-s", type=float, default=20.0)
    ap.add_argument("--out", default=str(Path.home() / "lud_find_runs"))
    args = ap.parse_args()
    dog = Dog(args)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(dog))
    print(f"lud_tello_shim :{args.port} -> bridge {args.bridge}, run_dir {dog.run_dir}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dog.land("shutdown")
        dog.sender.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
