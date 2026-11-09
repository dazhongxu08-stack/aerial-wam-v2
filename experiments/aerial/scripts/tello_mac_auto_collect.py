#!/usr/bin/env python3
"""Indoor-safe Tello auto flight through Mac ground console HTTP.

Collects the real-world fields the WAM stack actually trains on:
  policy/WM: RGB JPEG + proprio4 (x,y,z,yaw) + executed body-delta (from RC)
  supervision: IMU (agx/agy/agz), ToF/h, battery, wifi RSSI
Tello has no GT depth — do not invent it.

Safety: battery/RSSI/link gates, max height, no backward stick, auto land.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

BASE = "http://127.0.0.1:8090"
MIN_BAT = 40
MAX_H_CM = 180
MIN_RSSI = -72
RC_HZ = 10.0


def _req(method: str, path: str, body: Optional[dict] = None, timeout: float = 8.0) -> dict:
    data = None
    headers = {"Connection": "close"}
    if body is not None:
        raw = json.dumps(body).encode()
        data = raw
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def status() -> dict:
    return _req("GET", "/api/status")


def cmd(name: str) -> dict:
    return _req("POST", "/api/cmd", {"cmd": name}, timeout=25.0)


def rc(a: int, b: int, c: int, d: int) -> dict:
    return _req("POST", "/api/rc", {"a": a, "b": b, "c": c, "d": d, "source": "auto"})


def hold(seconds: float, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> None:
    t0 = time.time()
    dt = 1.0 / RC_HZ
    while time.time() - t0 < seconds:
        st = status()
        bat = st.get("battery") or 0
        h = st.get("height_cm") or 0
        rssi = st.get("wifi_rssi_dbm")
        if not st.get("ok"):
            raise RuntimeError("link lost")
        if bat and int(bat) < 25:
            raise RuntimeError(f"battery {bat}")
        if h and int(h) > MAX_H_CM:
            raise RuntimeError(f"too high {h}cm")
        if rssi is not None and int(rssi) < MIN_RSSI:
            raise RuntimeError(f"rssi {rssi}")
        # altitude trim toward ~1.1 m
        cc = c
        if h is not None and c == 0:
            if int(h) < 90:
                cc = 18
            elif int(h) > 140:
                cc = -12
        rc(a, max(0, b), cc, d)  # never backward
        time.sleep(dt)


PROFILES = {
    "yaw_fwd": lambda: (
        hold(3.0),
        hold(4.0, d=-28),
        hold(4.0, d=28),
        hold(1.5),
        hold(3.5, b=22),
        hold(1.5),
        hold(3.5, b=22),
        hold(2.0),
    ),
    "fwd_yaw": lambda: (
        hold(3.0),
        hold(4.0, b=20),
        hold(2.0),
        hold(5.0, d=30),
        hold(5.0, d=-30),
        hold(2.0),
        hold(3.0, b=22),
        hold(2.0),
    ),
    "creep_yaw": lambda: (
        hold(3.5),
        hold(5.0, b=16, d=-20),
        hold(1.5),
        hold(5.0, b=16, d=20),
        hold(2.0),
        hold(3.0, b=18),
        hold(2.0),
    ),
}


def _preflight() -> dict:
    st = status()
    print("preflight", {k: st.get(k) for k in ("ok", "battery", "wifi_rssi_dbm", "allow_fly", "airborne", "temp")})
    if not st.get("ok"):
        raise RuntimeError("Tello link down")
    bat = int(st.get("battery") or 0)
    if bat < MIN_BAT:
        raise RuntimeError(f"battery {bat}% < {MIN_BAT}%")
    rssi = st.get("wifi_rssi_dbm")
    if rssi is not None and int(rssi) < MIN_RSSI:
        raise RuntimeError(f"rssi {rssi}")
    return st


def fly_one(name: str) -> int:
    _preflight()
    rec = _req("POST", "/api/record", {"on": True})
    print(f"[{name}] record", rec.get("msg"))
    try:
        print(f"[{name}] takeoff", cmd("takeoff"))
        PROFILES[name]()
        print(f"[{name}] land", cmd("land"))
        time.sleep(5)
        return 0
    except Exception as exc:
        print(f"[{name}] ABORT fly:", exc, file=sys.stderr)
        try:
            cmd("land")
        except Exception:
            try:
                cmd("emergency")
            except Exception:
                pass
        return 1
    finally:
        try:
            rc(0, 0, 0, 0)
        except Exception:
            pass
        stop = _req("POST", "/api/record", {"on": False})
        print(f"[{name}] saved", stop)


def main() -> int:
    names = sys.argv[1:] or ["yaw_fwd", "fwd_yaw", "creep_yaw"]
    for n in names:
        if n not in PROFILES:
            print("unknown profile", n, file=sys.stderr)
            return 2
    _req("POST", "/api/control_mode", {"mode": "auto"})
    rc_code = 0
    try:
        for i, name in enumerate(names):
            print(f"=== flight {i+1}/{len(names)} {name} ===")
            try:
                _preflight()
            except Exception as exc:
                print("ABORT remaining:", exc, file=sys.stderr)
                rc_code = 2
                break
            if fly_one(name) != 0:
                rc_code = 1
                break
            if i + 1 < len(names):
                print("rest 8s")
                time.sleep(8)
    finally:
        try:
            _req("POST", "/api/control_mode", {"mode": "manual"})
        except Exception:
            pass
    return rc_code


if __name__ == "__main__":
    raise SystemExit(main())
