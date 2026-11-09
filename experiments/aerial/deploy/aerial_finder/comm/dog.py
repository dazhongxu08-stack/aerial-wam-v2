"""Generic robot-dog target report.

Message (``schema: aerial_finder.target/v1``), sent as JSON by HTTP POST to
``<base_url>/api/v1/targets`` or published on an MQTT topic::

    {
      "schema": "aerial_finder.target/v1",
      "mission_id": "...", "drone_id": "...", "timestamp": "2026-10-15T12:00:00Z",
      "query": {"text": "...", "label": "stool", "color": "red"},
      "position": {"lat": 47.64, "lon": -122.14, "alt_m": 122.3, "uncertainty_m": 0.8},
      "local": {"frame": "home_NEU", "north_m": 1.0, "east_m": 2.0, "up_m": 0.3,
                "home": {"lat": ..., "lon": ..., "alt_m": ...}},
      "confidence": 0.71, "observations": 7,
      "image_jpeg_b64": "..."   # optional best crop
    }
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..geo import GeoRef

log = logging.getLogger("aerial_finder.dog")
SCHEMA = "aerial_finder.target/v1"


def target_message(*, mission_id: str, drone_id: str, query: Dict[str, Any], pos_local: np.ndarray, geo: GeoRef,
                   uncertainty_m: float, confidence: float, observations: int,
                   crop_rgb: Optional[np.ndarray] = None) -> Dict[str, Any]:
    n, e, u = (float(v) for v in pos_local[:3])
    lat, lon, alt = geo.to_gps(n, e, u)
    msg: Dict[str, Any] = {
        "schema": SCHEMA, "mission_id": mission_id, "drone_id": drone_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "query": query,
        "position": {"lat": round(lat, 8), "lon": round(lon, 8), "alt_m": round(alt, 2),
                     "uncertainty_m": round(float(uncertainty_m), 2)},
        "local": {"frame": "home_NEU", "north_m": round(n, 2), "east_m": round(e, 2), "up_m": round(u, 2),
                  "home": {"lat": geo.lat, "lon": geo.lon, "alt_m": geo.alt}},
        "confidence": round(float(confidence), 3), "observations": int(observations),
    }
    if crop_rgb is not None and crop_rgb.size:
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(crop_rgb[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            msg["image_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
    return msg


class HttpReporter:
    def __init__(self, base_url: str, timeout_s: float = 3.0, retries: int = 5, token: str = "") -> None:
        self.url = base_url.rstrip("/") + "/api/v1/targets"
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.token = token

    def send(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(msg).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        err = ""
        for i in range(self.retries):
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    body = r.read().decode("utf-8", "replace")
                    return {"ok": 200 <= r.status < 300, "status": r.status, "body": body[:500], "attempts": i + 1}
            except (urllib.error.URLError, OSError) as exc:
                err = str(exc)
                log.warning("dog report attempt %d failed: %s", i + 1, exc)
                time.sleep(min(2.0, 0.3 * 2 ** i))
        return {"ok": False, "error": err, "attempts": self.retries}


class MqttReporter:
    def __init__(self, host: str, port: int = 1883, topic: str = "aerial_finder/targets", qos: int = 1) -> None:
        import paho.mqtt.client as mqtt

        self.topic, self.qos = topic, int(qos)
        self.client = mqtt.Client()
        self.client.connect(host, int(port))
        self.client.loop_start()

    def send(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        info = self.client.publish(self.topic, json.dumps(msg), qos=self.qos)
        info.wait_for_publish(timeout=5.0)
        return {"ok": info.is_published(), "topic": self.topic}


class MockDogServer:
    """Accepts target reports and lists them (``GET /api/v1/targets``) — integration tests and demos."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        received: List[Dict[str, Any]] = []
        self.received = received

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D401
                pass

            def _reply(self, code: int, obj: Any) -> None:
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                if self.path != "/api/v1/targets":
                    return self._reply(404, {"error": "not found"})
                n = int(self.headers.get("Content-Length", 0))
                try:
                    msg = json.loads(self.rfile.read(n))
                except json.JSONDecodeError:
                    return self._reply(400, {"error": "bad json"})
                received.append(msg)
                p = msg.get("position", {})
                log.info("dog got target %s at %.7f, %.7f", msg.get("query", {}).get("label"), p.get("lat", 0),
                         p.get("lon", 0))
                self._reply(200, {"ok": True, "n": len(received)})

            def do_GET(self):
                if self.path != "/api/v1/targets":
                    return self._reply(404, {"error": "not found"})
                self._reply(200, [{k: v for k, v in m.items() if k != "image_jpeg_b64"} for m in received])

        self.httpd = ThreadingHTTPServer((host, int(port)), H)
        self.port = self.httpd.server_address[1]
        self.url = f"http://{host}:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> "MockDogServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
