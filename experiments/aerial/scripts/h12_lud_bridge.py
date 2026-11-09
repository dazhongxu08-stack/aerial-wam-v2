#!/usr/bin/env python3
"""H12 HTTP stick commands → Lumos lcm_robot_cmd.

Run on Orin (eth0 on the robot LAN, Wi-Fi to H12):

  cd /home/ubuntu/lumos_sdk
  python3 /path/to/h12_lud_bridge.py --bind 0.0.0.0 --port 8092
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

THIS = Path(__file__).resolve()
roots = [
    Path("/home/ubuntu/lumos_sdk"),
    Path.home() / "lumos_sdk",
    Path("/Users/xudazhong/Projects/lumos_sdk"),
]
if len(THIS.parents) > 3:
    roots.append(THIS.parents[3] / "lumos_sdk")
for root in roots:
    for p in (root / "python", root / "lcm_typedef" / "python"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

from lud_send_robot_cmd import (  # noqa: E402
    DEFAULT_LCM_URL,
    ROBOT_CMD_CHANNEL,
    STATES,
    LudCommandPublisher,
    clamp as sdk_clamp,
)
from robot_battery_lcmt import robot_battery_lcmt  # noqa: E402

BATTERY_CHANNEL = "lcm_robot_battery"

LOCK = threading.Lock()
PUBLISHER: LudCommandPublisher | None = None
LAST_STATE = STATES["RESET"]
LAST_VX = 0.0
LAST_VY = 0.0
LAST_VYAW = 0.0
LAST_PUB_AT = 0.0
LAST_SWITCH_AT = 0.0
# A state change is ignored by the controller if the same message carries speed.
# Hold zero speed briefly so the transition is actually accepted.
SWITCH_HOLD_S = 0.30
LAST: Dict[str, Any] = {
    "ok": False,
    "state": STATES["RESET"],
    "vx": 0.0,
    "vy": 0.0,
    "vyaw": 0.0,
    "ts": 0.0,
    "battery_pct": None,
    "battery_v": None,
    "battery_a": None,
    "battery_ts": 0.0,
}

# Auto commands ({"src": "auto"}) drive only while H12 is fresh, in walk, not
# stopped, sticks centred. Any stick, CH6 change or CH5 stop hands control back.
# Script commands ({"src": "script"}) hold stand/walk themselves. The handle's
# repeated stand packets are the switch sitting still, not a new mode command,
# so they must not flip the dog back to stand on every tick. E-stop, a moved
# stick, or the gait switch actually changing still takes over. A quiet script
# that left the dog walking is stood, because walk-at-zero steps backward.
H12_TTL_S = 0.6
AUTO_TTL_S = 0.5
SCRIPT_TTL_S = 0.5
STICK_CENTRED = 0.05
AUTO_VX_MAX = 0.35
AUTO_VYAW_MAX = 0.5
H12: Dict[str, Any] = {"at": 0.0, "state": STATES["RESET"], "stop": True, "vx": 0.0, "vy": 0.0, "vyaw": 0.0}
AUTO: Dict[str, Any] = {"at": 0.0, "vx": 0.0, "vyaw": 0.0}
SCRIPT: Dict[str, Any] = {"at": 0.0, "seen": 0.0}
AUTO_ACTIVE = False
SCRIPT_HELD = False


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class QuietPublisher(LudCommandPublisher):
    """Same LCM payload as the SDK helper, without per-tick stdout."""

    def publish(
        self,
        state: int,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        policy_type: int = 0,
    ) -> None:
        vx = sdk_clamp(vx, -0.5, 0.5)
        vy = sdk_clamp(vy, -0.3, 0.3)
        vyaw = sdk_clamp(vyaw, -0.5, 0.5)
        if self.dry_run:
            return
        message = self._message_type()
        message.state = state
        message.x = vx
        message.y = vy
        message.yaw = vyaw
        message.policy_type = policy_type
        self._lcm.publish(ROBOT_CMD_CHANNEL, message.encode())


def _h12_allows_auto(now: float) -> bool:
    return (
        now - H12["at"] <= H12_TTL_S
        and H12["state"] == STATES["RL_WALK_AMP"]
        and not H12["stop"]
        and max(abs(H12["vx"]), abs(H12["vy"]), abs(H12["vyaw"])) < STICK_CENTRED
    )


def _auto_live(now: float) -> bool:
    return now - AUTO["at"] <= AUTO_TTL_S and _h12_allows_auto(now)


def _script_live(now: float) -> bool:
    return now - float(SCRIPT["at"]) <= SCRIPT_TTL_S


def _parse_motion(body: Dict[str, Any]) -> tuple:
    state = int(body.get("state", STATES["RESET"]))
    if state not in STATES.values():
        raise ValueError(f"unsupported state {state}")
    stop = bool(body.get("stop", False))
    vx = 0.0 if stop else clamp(float(body.get("vx", 0.0)), -0.5, 0.5)
    vy = 0.0 if stop else clamp(float(body.get("vy", 0.0)), -0.3, 0.3)
    vyaw = 0.0 if stop else clamp(float(body.get("vyaw", 0.0)), -0.5, 0.5)
    if state != STATES["RL_WALK_AMP"]:
        vx = vy = vyaw = 0.0
    return state, vx, vy, vyaw, stop


def apply_cmd(body: Dict[str, Any]) -> Dict[str, Any]:
    if body.get("src") == "auto":
        return _apply_auto(body)
    if body.get("src") == "script":
        return _apply_script(body)
    state, vx, vy, vyaw, stop = _parse_motion(body)
    now = time.time()
    with LOCK:
        prev_state = int(H12["state"])
        prev_at = float(H12["at"])
        H12.update(at=now, state=state, stop=stop, vx=vx, vy=vy, vyaw=vyaw)
        script_live = _script_live(now)
    # While a script is live, the handle's unchanged stand/walk stream is not a
    # new command. Estop, a moved stick, or the gait switch actually moving
    # cancels the script and drives immediately. With no script, every handle
    # packet drives.
    if script_live:
        sticks = max(abs(vx), abs(vy), abs(vyaw)) >= STICK_CENTRED
        switched = prev_at > 0.0 and state != prev_state
        if stop or sticks or switched:
            return _takeover(state, vx, vy, vyaw, stop, now)
        with LOCK:
            rec = dict(LAST)
        rec["h12_ignored"] = True
        return rec
    if _auto_live(now):
        with LOCK:
            return dict(LAST)
    return _drive(state, vx, vy, vyaw, stop, "h12")


def _takeover(state: int, vx: float, vy: float, vyaw: float, stop: bool, now: float) -> Dict[str, Any]:
    """Operator estop, stick, or gait-switch change. The script stays out until it goes quiet."""
    global SCRIPT_HELD
    with LOCK:
        SCRIPT["at"] = 0.0
        SCRIPT["seen"] = now
        SCRIPT_HELD = True
    return _drive(state, vx, vy, vyaw, stop, "h12")


def _apply_script(body: Dict[str, Any]) -> Dict[str, Any]:
    global SCRIPT_HELD
    state, vx, vy, vyaw, stop = _parse_motion(body)
    now = time.time()
    with LOCK:
        if SCRIPT_HELD:
            if now - float(SCRIPT["seen"]) <= SCRIPT_TTL_S:
                SCRIPT["seen"] = now
                rec = dict(LAST)
                rec["script_accepted"] = False
                return rec
            SCRIPT_HELD = False
        SCRIPT["at"] = now
        SCRIPT["seen"] = now
    rec = _drive(state, vx, vy, vyaw, stop, "script")
    rec["script_accepted"] = True
    return rec


def _apply_auto(body: Dict[str, Any]) -> Dict[str, Any]:
    global AUTO_ACTIVE
    vx = clamp(float(body.get("vx", 0.0)), -AUTO_VX_MAX, AUTO_VX_MAX)
    vyaw = clamp(float(body.get("vyaw", 0.0)), -AUTO_VYAW_MAX, AUTO_VYAW_MAX)
    now = time.time()
    with LOCK:
        AUTO.update(at=now, vx=vx, vyaw=vyaw)
        walking = LAST_STATE == STATES["RL_WALK_AMP"]
    if not (walking and _h12_allows_auto(now)):
        with LOCK:
            AUTO_ACTIVE = False
            rec = dict(LAST)
        rec["auto_accepted"] = False
        return rec
    with LOCK:
        AUTO_ACTIVE = True
    rec = _drive(STATES["RL_WALK_AMP"], vx, 0.0, vyaw, False, "auto")
    rec["auto_accepted"] = True
    return rec


def _drive(state: int, vx: float, vy: float, vyaw: float, stop: bool, src: str) -> Dict[str, Any]:
    global LAST_STATE, LAST_VX, LAST_VY, LAST_VYAW, LAST_PUB_AT, LAST_SWITCH_AT
    assert PUBLISHER is not None
    now = time.time()
    with LOCK:
        prev = LAST_STATE
        last_vx, last_vy, last_vyaw, last_at = LAST_VX, LAST_VY, LAST_VYAW, LAST_PUB_AT
        changed = state != prev
        if changed or now - LAST_SWITCH_AT < SWITCH_HOLD_S:
            vx = vy = vyaw = 0.0
        same = (
            not changed
            and abs(vx - last_vx) < 1e-4
            and abs(vy - last_vy) < 1e-4
            and abs(vyaw - last_vyaw) < 1e-4
        )
        min_dt = 0.04 if state == STATES["RL_WALK_AMP"] else 0.12
        do_switch = changed
        do_hold = (not changed) and (not same or (now - last_at) >= min_dt)
        if do_switch:
            LAST_STATE = state
            LAST_SWITCH_AT = now
            LAST_PUB_AT = now
        elif do_hold:
            LAST_PUB_AT = now
        LAST_VX, LAST_VY, LAST_VYAW = vx, vy, vyaw
    if do_switch:
        if last_at > 0:
            PUBLISHER.publish(prev, 0.0, 0.0, 0.0)
        PUBLISHER.publish(state, 0.0, 0.0, 0.0)
        print(f"state {prev} -> {state}")
    elif do_hold:
        PUBLISHER.publish(state, vx, vy, vyaw)
    rec = {
        "ok": True,
        "state": state,
        "vx": vx,
        "vy": vy,
        "vyaw": vyaw,
        "stop": stop,
        "src": src,
        "ts": time.time(),
    }
    with LOCK:
        LAST.update(rec)
    return rec


def watchdog_loop() -> None:
    """Zero the walk speed when the driving source goes quiet.

    A script that stops while the dog is still walking is stood instead. Leaving
    WALK at zero speed makes this dog step backward.
    """
    global AUTO_ACTIVE
    script_was_live = False
    while True:
        time.sleep(0.1)
        now = time.time()
        with LOCK:
            moving = max(abs(LAST_VX), abs(LAST_VY), abs(LAST_VYAW)) > 1e-4
            walking = LAST_STATE == STATES["RL_WALK_AMP"]
            was_auto = AUTO_ACTIVE
            h12_fresh = now - H12["at"] <= H12_TTL_S
            script_live = _script_live(now)
            src = LAST.get("src")
            held_out = SCRIPT_HELD
        if script_was_live and not script_live and walking and src == "script" and not held_out:
            print("script released -> stand", flush=True)
            _drive(STATES["STAND"], 0.0, 0.0, 0.0, False, "watchdog")
            script_was_live = False
            continue
        script_was_live = script_live
        if was_auto and not _auto_live(now):
            with LOCK:
                AUTO_ACTIVE = False
            print("auto released", flush=True)
            if walking:
                _drive(STATES["RL_WALK_AMP"], 0.0, 0.0, 0.0, True, "watchdog")
            continue
        if moving and walking and not was_auto and not script_live and not h12_fresh:
            print("h12 silent -> stand", flush=True)
            _drive(STATES["STAND"], 0.0, 0.0, 0.0, False, "watchdog")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, code: int, obj: Dict[str, Any]) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/api/health", "/", "/api/status"):
            now = time.time()
            with LOCK:
                rec = dict(LAST)
                rec["auto_active"] = AUTO_ACTIVE
                rec["h12_age_s"] = round(now - H12["at"], 2)
                rec["auto_age_s"] = round(now - AUTO["at"], 2)
                rec["script_age_s"] = round(now - float(SCRIPT["at"]), 2)
                rec["script_held"] = SCRIPT_HELD
            rec["auto_ready"] = _h12_allows_auto(now)
            rec["ok"] = True
            rec["service"] = "h12_lud_bridge"
            self._send(200, rec)
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/cmd":
            self._send(404, {"ok": False, "error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(body, dict):
                raise ValueError("json object required")
            rec = apply_cmd(body)
            self._send(200, rec)
        except Exception as exc:
            self._send(400, {"ok": False, "error": str(exc)})


def on_battery(_channel: str, data: bytes) -> None:
    try:
        msg = robot_battery_lcmt.decode(data)
    except Exception:
        return
    pct = int(msg.percentage)
    with LOCK:
        LAST["battery_pct"] = pct if 0 <= pct <= 100 else None
        LAST["battery_v"] = round(float(msg.voltage), 2)
        LAST["battery_a"] = round(float(msg.current), 2)
        LAST["battery_ts"] = time.time()


def lcm_loop() -> None:
    assert PUBLISHER is not None and PUBLISHER._lcm is not None
    PUBLISHER._lcm.subscribe(BATTERY_CHANNEL, on_battery)
    print(f"lcm sub {BATTERY_CHANNEL}")
    while True:
        PUBLISHER._lcm.handle_timeout(200)


def udp_loop(bind: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind, port))
    print(f"udp cmd {bind}:{port}")
    while True:
        raw, _addr = sock.recvfrom(2048)
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
            if isinstance(body, dict):
                apply_cmd(body)
        except Exception as exc:
            sys.stderr.write(f"udp cmd: {exc}\n")


def main() -> int:
    global PUBLISHER
    p = argparse.ArgumentParser(description="H12 → Lumos LCM bridge")
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--lcm-url", default=DEFAULT_LCM_URL)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    PUBLISHER = QuietPublisher(args.lcm_url, dry_run=args.dry_run)
    threading.Thread(target=udp_loop, args=(args.bind, args.port), daemon=True).start()
    threading.Thread(target=watchdog_loop, daemon=True).start()
    if not args.dry_run:
        threading.Thread(target=lcm_loop, daemon=True).start()
    httpd = ThreadingHTTPServer((args.bind, args.port), Handler)
    httpd.request_queue_size = 64
    print(f"h12_lud_bridge http://{args.bind}:{args.port} udp:{args.port} lcm={args.lcm_url} dry_run={args.dry_run}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
