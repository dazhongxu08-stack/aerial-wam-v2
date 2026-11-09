#!/usr/bin/env python3
"""Serve the SJCAM 4000 plugged into this Mac as the dog's latest JPEG.

The Orin pulls ``/frame.jpg``. The capture thread keeps only the newest frame,
so a slow reader cannot build up a queue.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

PORT = 8093
WIDTH = 1280
HEIGHT = 720
GRAB = Path(__file__).with_name("usb_cam_grab.swift")
FRAME_PATH = Path("/tmp/sjcam_usb.jpg")


class _Frames:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpg = b""
        self.at = 0.0
        self.n = 0
        self.shape = (0, 0)
        self.run = True

    def publish(self, bgr) -> None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        with self.lock:
            self.jpg = buf.tobytes()
            self.at = time.time()
            self.n += 1
            self.shape = (int(bgr.shape[1]), int(bgr.shape[0]))

    def snapshot(self):
        with self.lock:
            age = 0.0 if self.at == 0.0 else time.time() - self.at
            return self.jpg, age, self.n, self.shape


FRAMES = _Frames()


def capture() -> None:
    """Read the USB camera named 4000. OpenCV's index list mixes in the phone and the Mac."""
    proc = subprocess.Popen(["swift", str(GRAB)], stderr=subprocess.PIPE, text=True)
    seen = 0.0
    try:
        while FRAMES.run:
            if proc.poll() is not None:
                err = proc.stderr.read() if proc.stderr is not None else ""
                raise SystemExit(err.strip() or f"usb camera grab exited {proc.returncode}")
            if FRAME_PATH.exists():
                st = FRAME_PATH.stat()
                if st.st_mtime != seen and st.st_size > 1000:
                    bgr = cv2.imdecode(
                        np_from_file(FRAME_PATH), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        FRAMES.publish(bgr)
                    seen = st.st_mtime
            time.sleep(0.03)
    finally:
        proc.terminate()


def np_from_file(path: Path):
    import numpy as np
    return np.frombuffer(path.read_bytes(), np.uint8)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            jpg, age, n, shape = FRAMES.snapshot()
            body = json.dumps({
                "ok": n > 0 and age < 1.0,
                "n": n,
                "age_s": round(age, 3),
                "width": shape[0],
                "height": shape[1],
                "bytes": len(jpg),
            }).encode()
            self._send(200, "application/json", body, age)
            return
        if path == "/frame.jpg":
            jpg, age, n, _shape = FRAMES.snapshot()
            if n == 0 or not jpg:
                self.send_error(503, "no frame")
                return
            self._send(200, "image/jpeg", jpg, age)
            return
        self.send_error(404)

    def _send(self, code: int, ctype: str, body: bytes, age: float) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Age", f"{age:.3f}")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    try:
        FRAME_PATH.unlink()
    except OSError:
        pass
    threading.Thread(target=capture, daemon=True).start()
    deadline = time.time() + 20.0
    while time.time() < deadline:
        jpg, _age, n, shape = FRAMES.snapshot()
        if n > 0 and jpg:
            break
        time.sleep(0.05)
    else:
        raise SystemExit("USB camera 4000 produced no frame")
    bgr = cv2.imdecode(np_from_file(FRAME_PATH), cv2.IMREAD_COLOR)
    mean = float(bgr.mean()) if bgr is not None else 0.0
    if mean < 60.0:
        raise SystemExit(f"USB camera frame is still blank, mean {mean:.1f}")
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"listening 0.0.0.0:{PORT} usb 4000 frame {shape[0]}x{shape[1]} mean {mean:.0f}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        FRAMES.run = False
        httpd.server_close()


if __name__ == "__main__":
    main()
