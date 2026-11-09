#!/usr/bin/env python3
"""Bind video first, streamon, count UDP + decode with PyAV (no restart)."""
from __future__ import annotations

import socket
import time

import av

from experiments.aerial.rl.env.tello_client import TelloClient


def main() -> int:
    c = TelloClient(local_ip="192.168.10.2", tello_ip="192.168.10.1", local_cmd_port=9000)
    print("connect", c.connect(retries=6))
    c.start_state_listener()
    time.sleep(0.3)
    print("bat", c.battery())

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    sock.bind(("0.0.0.0", 11111))
    sock.settimeout(1.0)
    print("bound 11111")
    print("streamon", c.stream_on())
    time.sleep(0.2)
    print("streamon2", c.stream_on())

    codec = av.CodecContext.create("h264", "r")
    pkts = 0
    nbytes = 0
    frames = 0
    t0 = time.time()
    while time.time() - t0 < 20:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        pkts += 1
        nbytes += len(data)
        try:
            for packet in codec.parse(data):
                for frame in codec.decode(packet):
                    frames += 1
                    if frames == 1:
                        img = frame.to_ndarray(format="bgr24")
                        print("first_frame", img.shape, "after", round(time.time() - t0, 2), "s")
        except Exception as exc:
            print("decode_err", type(exc).__name__, exc)
            continue
        if frames >= 30:
            break
    print("pkts", pkts, "bytes", nbytes, "frames", frames)
    sock.close()
    c.close()
    return 0 if frames >= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
