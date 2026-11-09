#!/usr/bin/env python3
"""Standalone Tello PyAV video diagnostic (no takeoff)."""
from __future__ import annotations

import time

from experiments.aerial.deploy.tello_camera import TelloStreamCamera
from experiments.aerial.rl.env.tello_client import TelloClient


def main() -> int:
    c = TelloClient(local_ip="192.168.10.2", tello_ip="192.168.10.1", local_cmd_port=9000)
    print("connect", c.connect(retries=5))
    c.start_state_listener()
    print("bat", c.battery())
    try:
        c.stream_off()
    except Exception as exc:
        print("streamoff", type(exc).__name__, exc)
    time.sleep(0.4)

    cam = TelloStreamCamera(port=11111, local_ip="0.0.0.0", open_timeout_s=15.0)
    got = 0
    try:
        cam.start_worker()
        time.sleep(0.15)
        print("streamon", c.stream_on())
        cam.wait_ready()
        print("ready", cam.frame_seq, cam.frame_age_s)
        last = -1
        t0 = time.time()
        while time.time() - t0 < 8:
            fr = cam.read_bgr_new()
            if fr is not None and cam.frame_seq != last:
                last = cam.frame_seq
                got += 1
                if got == 1:
                    print("shape", fr.shape)
            time.sleep(0.05)
        print("frames", got, "seq", cam.frame_seq, "fps", round(got / 8.0, 2))
    finally:
        cam.close()
    try:
        c.stream_off()
    except Exception:
        pass
    c.close()
    return 0 if got >= 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
