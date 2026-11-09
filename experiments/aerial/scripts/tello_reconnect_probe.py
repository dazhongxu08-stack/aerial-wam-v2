#!/usr/bin/env python3
"""Quick reconnect probe: SDK + PyAV."""
from __future__ import annotations

import time

from experiments.aerial.deploy.tello_camera import TelloStreamCamera
from experiments.aerial.rl.env.tello_client import TelloClient


def main() -> int:
    c = TelloClient(local_ip="192.168.10.2", tello_ip="192.168.10.1", local_cmd_port=9000)
    print("connect", c.connect(retries=10))
    c.start_state_listener()
    time.sleep(0.4)
    print("bat", c.battery(), "age", round(c.state_age_s, 2))
    cam = TelloStreamCamera(port=11111, open_timeout_s=12.0)
    try:
        cam.start_worker()
        time.sleep(0.2)
        print("streamon", c.stream_on())
        t0 = time.time()
        got = 0
        while time.time() - t0 < 10:
            fr = cam.read_bgr_new()
            if fr is not None:
                got += 1
                if got == 1:
                    print("shape", fr.shape)
                if got >= 20:
                    break
            time.sleep(0.05)
        print("frames", got, "seq", cam.frame_seq)
        return 0 if got >= 10 else 1
    finally:
        cam.close()
        c.close()


if __name__ == "__main__":
    raise SystemExit(main())
