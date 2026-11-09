#!/usr/bin/env python3
"""Decode Tello H.264 UDP with PyAV only (no OpenCV) → JPEG file for the console."""
from __future__ import annotations

import logging
import os
import socket
import time

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("tello_video_sidecar")

PORT = int(os.environ.get("TELLO_STREAM_PORT", "11111"))
OUT = os.environ.get("TELLO_FRAME_JPG", "/tmp/tello_latest.jpg")
TMP = OUT + ".part"


def main() -> int:
    import av  # type: ignore
    from PIL import Image  # type: ignore

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(1.0)
    codec = av.CodecContext.create("h264", "r")
    log.info("listen 0.0.0.0:%s → %s", PORT, OUT)
    pkts = 0
    frames = 0
    last = time.time()
    while True:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            now = time.time()
            if now - last >= 3.0:
                log.info("waiting pkts=%s frames=%s", pkts, frames)
                last = now
            continue
        pkts += 1
        try:
            packets = codec.parse(data)
        except Exception:  # noqa: BLE001
            continue
        for packet in packets:
            try:
                decoded = codec.decode(packet)
            except Exception:  # noqa: BLE001
                continue
            for frame in decoded:
                rgb = frame.to_ndarray(format="rgb24")
                img = Image.fromarray(rgb)
                if img.width > 640:
                    img = img.resize((640, 480))
                img.save(TMP, "JPEG", quality=70)
                os.replace(TMP, OUT)
                frames += 1
                if frames == 1 or frames % 60 == 0:
                    log.info("frame %s %sx%s pkts=%s", frames, img.width, img.height, pkts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
