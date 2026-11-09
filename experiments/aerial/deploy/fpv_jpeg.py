"""Front-camera JPEG for the office page, with the search target boxed.

The driving process writes /tmp/yhs_mark.json. A held box stays after the
find. This process only serves the picture; it does not command the chassis.
"""
from __future__ import annotations

import json
import threading
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

latest = b""
lock = threading.Lock()
_MARK = "/tmp/yhs_mark.json"


def _overlay(img: np.ndarray) -> np.ndarray:
    try:
        body = json.loads(open(_MARK, encoding="utf-8").read())
        box = body.get("box")
        stamp = float(body.get("t") or 0.0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return img
    if not box or len(box) != 4:
        return img
    if not body.get("hold") and time.time() - stamp > 1.0:
        return img
    height, width = img.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box)
    p0 = (max(0, min(width - 1, int(x0 * width))), max(0, min(height - 1, int(y0 * height))))
    p1 = (max(0, min(width - 1, int(x1 * width))), max(0, min(height - 1, int(y1 * height))))
    if p1[0] <= p0[0] or p1[1] <= p0[1]:
        return img
    cv2.rectangle(img, p0, p1, (0, 220, 0), 3)
    label = str(body.get("label") or "")
    if label:
        _draw_label(img, label, p0[0], max(0, p0[1] - 46))
    return img


_FONT = None


def _font():
    global _FONT
    if _FONT is not None:
        return _FONT
    try:
        from PIL import ImageFont
        _FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 28)
    except (ImportError, OSError):
        _FONT = False
    return _FONT


def _draw_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    font = _font()
    if not font:
        return
    from PIL import Image, ImageDraw
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plate = Image.fromarray(rgb)
    draw = ImageDraw.Draw(plate)
    edge = draw.textbbox((x, y), text, font=font)
    draw.rectangle((edge[0] - 4, edge[1] - 2, edge[2] + 4, edge[3] + 2), fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(80, 255, 80))
    img[:] = cv2.cvtColor(np.asarray(plate), cv2.COLOR_RGB2BGR)


class Cam(Node):
    def __init__(self) -> None:
        super().__init__("fpv_jpeg")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, "/camera1/camera1/color/image_raw", self.on_image, qos)

    def on_image(self, msg: Image) -> None:
        global latest
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding.lower() == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = np.ascontiguousarray(img)
        img = _overlay(img)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with lock:
            latest = buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if not self.path.startswith("/jpg"):
            self.send_error(404)
            return
        with lock:
            frame = latest
        if not frame:
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        self.wfile.write(frame)


def main() -> None:
    rclpy.init()
    node = Cam()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 9100), Handler).serve_forever()


if __name__ == "__main__":
    main()
