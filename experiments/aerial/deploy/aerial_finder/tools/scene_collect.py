#!/usr/bin/env python3
"""Record a few clips in the current scene.

The dog steps backward if it is left in WALK with zero speed. This script holds
STAND while the camera is still, switches to WALK once for the moving clips, and
returns to STAND when finished. Commands use src=script so the handle's stand
stream does not flip the gait on every tick.
"""
from __future__ import annotations

import csv
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2

sys.path.insert(0, "/home/ubuntu/lumos_sdk/lcm_typedef/python")
import lcm  # noqa: E402
from imu_data_lcmt import imu_data_lcmt  # noqa: E402

STAND = 2
WALK = 12
OUT = Path("/home/ubuntu/aerial_finder_runs/scene_%s" % time.strftime("%Y%m%d_%H%M%S"))
FRAMES = OUT / "frames"


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        "http://127.0.0.1:8092/api/cmd",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1.0) as resp:
        return json.loads(resp.read().decode())


def hold_state(state: int, seconds: float, vx: float = 0.0, vyaw: float = 0.0) -> dict:
    last = {}
    t0 = time.time()
    while time.time() - t0 < seconds:
        last = post({
            "src": "script", "state": state, "vx": vx, "vy": 0.0, "vyaw": vyaw, "stop": False,
        })
        time.sleep(0.05)
    return last


def main() -> int:
    FRAMES.mkdir(parents=True, exist_ok=True)
    rows: list[tuple] = []
    lock = threading.Lock()
    stop = threading.Event()
    t0 = time.time()

    def imu_loop() -> None:
        node = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

        def on_imu(_channel: str, data: bytes) -> None:
            msg = imu_data_lcmt.decode(data)
            with lock:
                rows.append((
                    time.time() - t0, time.time(),
                    *[float(v) for v in msg.navRPY],
                    *[float(v) for v in msg.acc],
                    *[float(v) for v in msg.omega],
                ))

        node.subscribe("lcm_imu_data", on_imu)
        while not stop.is_set():
            try:
                node.handle_timeout(40)
            except Exception as exc:  # noqa: BLE001
                print("imu", exc, flush=True)
                time.sleep(0.05)

    thread = threading.Thread(target=imu_loop, daemon=True)
    thread.start()
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(3):
        cap.read()
    cmd_log: list[tuple] = []
    nframes = 0
    rejected = 0
    next_frame = 0.0

    def grab(t: float) -> None:
        nonlocal nframes, next_frame
        if t < next_frame:
            return
        ok, frame = cap.read()
        if ok and frame is not None:
            cv2.imwrite(str(FRAMES / f"{t:07.3f}.jpg"), frame)
            nframes += 1
        next_frame = t + 0.2

    clips = (
        ("g1_stand", 1.0, STAND, 0.0, 0.0),
        ("g1_fwd", 3.0, WALK, 0.25, 0.0),
        ("g1_yaw", 2.0, WALK, 0.0, 0.40),
        ("g1_fwd2", 2.0, WALK, 0.25, 0.0),
        ("g2_stand", 0.6, STAND, 0.0, 0.0),
        ("g2_fwd", 2.0, WALK, 0.25, 0.0),
        ("g2_yaw", 1.5, WALK, 0.0, 0.40),
    )

    def run_clip(phase: str, seconds: float, state: int, vx: float, vyaw: float) -> int:
        nonlocal rejected
        t_end = time.time() + seconds
        sent = held = 0
        while time.time() < t_end:
            t = time.time() - t0
            rec = post({
                "src": "script", "state": state, "vx": vx, "vy": 0.0, "vyaw": vyaw, "stop": False,
            })
            sent += 1
            got_vx = float(rec.get("vx") or 0.0)
            got_yaw = float(rec.get("vyaw") or 0.0)
            same = int(rec.get("state") or 0) == state and abs(got_vx - vx) < 0.05 and abs(got_yaw - vyaw) < 0.05
            held += int(same)
            if state == WALK and not same:
                rejected += 1
            cmd_log.append((round(t, 3), vx, vyaw, phase, int(same), rec.get("state"), round(got_vx, 3), round(got_yaw, 3)))
            grab(t)
            time.sleep(0.04)
        print(f"{phase} sent {sent} held {held}", flush=True)
        return held

    try:
        held_motion = 0
        for phase, seconds, state, vx, vyaw in clips:
            held = run_clip(phase, seconds, state, vx, vyaw)
            if state == WALK:
                held_motion += held
        return 0 if held_motion > 10 else 3
    finally:
        hold_state(STAND, 0.5)
        print("stand_done", flush=True)
        stop.set()
        cap.release()
        thread.join(timeout=1.0)
        with lock:
            imu_copy = list(rows)
        with open(OUT / "imu.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "wall", "roll", "pitch", "yaw", "ax", "ay", "az", "wx", "wy", "wz"])
            writer.writerows(imu_copy)
        with open(OUT / "cmd.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "vx", "vyaw", "phase", "held", "state", "out_vx", "out_vyaw"])
            writer.writerows(cmd_log)
        print("frames", nframes, "imu", len(imu_copy), "rejected", rejected, flush=True)
        print("OUT", OUT, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
