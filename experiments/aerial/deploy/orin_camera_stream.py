"""MJPEG HTTP stream for Orin USB camera — view from Mac via SSH tunnel."""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from http import server
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_JPEG_QUALITY = 80
_FrameFn = Callable[[], Tuple[np.ndarray, np.ndarray]]


def _encode_jpeg(bgr: np.ndarray, quality: int) -> bytes:
    import cv2  # type: ignore

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _index_html(port: int) -> bytes:
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Orin camera</title>
<style>
  body {{ margin: 0; background: #111; color: #ddd; font-family: sans-serif; }}
  main {{ padding: 12px; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #333; }}
  .meta {{ margin-top: 8px; font-size: 14px; color: #aaa; }}
</style></head>
<body><main>
  <h1>Orin live camera</h1>
  <img src="/stream.mjpg" alt="live stream">
  <p class="meta">MJPEG on port {port} · <a href="/snapshot.jpg">snapshot</a></p>
</main></body></html>"""
    return body.encode("utf-8")


class _OrinCameraStreamHandler(BaseHTTPRequestHandler):
    server_version = "OrinCameraStream/1.0"
    frame_fn: Optional[_FrameFn] = None
    jpeg_quality: int = _JPEG_QUALITY
    min_frame_interval_s: float = 1.0 / 15.0
    _latest_jpeg: Optional[bytes] = None
    _latest_lock = threading.Lock()

    def log_message(self, fmt: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_bytes(200, "text/html; charset=utf-8", _index_html(self.server.server_port))
            return
        if self.path == "/snapshot.jpg":
            self._send_snapshot()
            return
        if self.path == "/stream.mjpg":
            self._send_mjpeg()
            return
        self.send_error(404, "not found")

    def _send_bytes(self, code: int, content_type: str, payload: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _read_frame_jpeg(self) -> bytes:
        if self.frame_fn is None:
            raise RuntimeError("frame_fn not configured")
        bgr, _ = self.frame_fn()
        jpeg = _encode_jpeg(bgr, self.jpeg_quality)
        with self._latest_lock:
            self._latest_jpeg = jpeg
        return jpeg

    def _send_snapshot(self) -> None:
        try:
            jpeg = self._read_frame_jpeg()
        except Exception as exc:
            logger.error("snapshot failed: %s", exc)
            self.send_error(500, str(exc))
            return
        self._send_bytes(200, "image/jpeg", jpeg)

    def _send_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                t0 = time.perf_counter()
                jpeg = self._read_frame_jpeg()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                elapsed = time.perf_counter() - t0
                sleep_s = self.min_frame_interval_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except (BrokenPipeError, ConnectionResetError):
            logger.info("client disconnected: %s", self.address_string())
        except Exception as exc:
            logger.error("stream ended: %s", exc)


def _open_camera(args: argparse.Namespace):
    if args.mock:
        from experiments.aerial.deploy.real_camera import MockCamera

        cam = MockCamera(wam_size=int(args.wam_size))
        cam.open()
        return cam, cam.read

    from experiments.aerial.deploy.real_camera import RealCamera, RealCameraConfig

    preset = None if str(args.v4l2_preset).lower() in ("", "none", "off") else str(args.v4l2_preset)
    cam = RealCamera(
        RealCameraConfig(
            device=str(args.camera),
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            wam_size=int(args.wam_size),
            fourcc=str(args.fourcc),
            v4l2_preset=preset,
            v4l2_profile=str(args.v4l2_profile),
        )
    )
    cam.open()
    return cam, cam.read


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Orin USB camera MJPEG stream")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (use 127.0.0.1 + SSH tunnel)")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--camera", default="0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stream-fps", type=float, default=15.0, help="max MJPEG publish rate")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--wam-size", type=int, default=224)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--v4l2-preset", default="none")
    parser.add_argument("--v4l2-profile", default="sjcam", choices=["sjcam", "hj", "c922"])
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    cam, frame_fn = _open_camera(args)
    handler_cls = _OrinCameraStreamHandler
    handler_cls.frame_fn = frame_fn
    handler_cls.jpeg_quality = int(args.jpeg_quality)
    handler_cls.min_frame_interval_s = 1.0 / max(0.1, float(args.stream_fps))

    httpd = ThreadingHTTPServer((str(args.host), int(args.port)), handler_cls)
    httpd.daemon_threads = True
    logger.info(
        "Orin camera stream http://%s:%d/ (camera=%s mock=%s)",
        args.host,
        args.port,
        args.camera,
        args.mock,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        httpd.server_close()
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
