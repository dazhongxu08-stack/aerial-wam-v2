"""Build the stack from a JSON config and serve the task HTTP API.

API (JSON):
  GET  /api/v1/health                     -> {"ok", "busy", "state"}
  POST /api/v1/parse    {"instruction"}   -> parsed MissionSpec (dry run)
  POST /api/v1/missions {"instruction"}   -> 202 {"mission_id", "spec"}; 409 when busy
  GET  /api/v1/missions/current           -> live status (state, pose, gps, task, tracks, result)
  POST /api/v1/missions/current/abort     -> hover and stop
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .mission.parser import MissionSpec, ParseError, parse_instruction
from .mission.runner import MissionConfig, MissionRunner

log = logging.getLogger("aerial_finder.service")


def _path(base: Path, p: str) -> str:
    q = Path(p).expanduser()
    return str(q if q.is_absolute() else base / q)


def build(cfg: Dict[str, Any], base: Path) -> MissionRunner:
    from .perception.detector import ColorHints, TargetDetector

    bc = dict(cfg.get("backend") or {"type": "fake"})
    kind = bc.pop("type")
    fake_world = None
    if kind == "airsim":
        from .backends.airsim_backend import AirSimBackend
        backend = AirSimBackend(**bc)
        backend.arm()
    elif kind == "lumos_dog":
        from .backends.lumos_dog import LumosDogBackend
        for key in ("fisheye_calib", "save_dir"):
            if bc.get(key):
                bc[key] = _path(base, str(bc[key]))
        backend = LumosDogBackend(**bc)
    elif kind == "fake":
        import numpy as np

        from .backends.fake import FakeBackend, FakeTarget, FakeWorld
        fake_world = FakeWorld([FakeTarget(np.asarray(t["pos"], float), t.get("label", "stool"), t.get("color", "red"))
                                for t in bc.pop("targets", [])])
        backend = FakeBackend(fake_world, **bc)
    else:
        raise ValueError(f"unknown backend {kind}")

    dc = dict(cfg.get("depth") or {"type": "sensor"})
    if dc["type"] == "da3":
        from .nav.depth import Da3Depth
        depth = Da3Depth(_path(base, dc["path"]), device=dc.get("device", "cuda"))
    elif dc["type"] in ("da3_ckpt", "vis_dpth"):
        from .nav.depth import Da3Checkpoint, MetricDa3
        depth = Da3Checkpoint(_path(base, dc["path"]), device=dc.get("device", "cuda"))
        if dc.get("metric_scale") is not None:
            depth = MetricDa3(depth, float(dc["metric_scale"]), float(dc.get("invalid_m", 30.0)))
    else:
        from .nav.depth import SensorDepth
        depth = SensorDepth()

    nc = dict(cfg.get("nav") or {"type": "goal_seek"})
    if nc.pop("type") == "consequence":
        from .nav.navigator import ConsequenceNavigator, NavConfig
        surv = [_path(base, p) for p in nc.pop("surv_ckpts")]
        prog = [_path(base, p) for p in nc.pop("prog_ckpts", [])]
        dev = nc.pop("device", "cuda")
        nav = ConsequenceNavigator(surv, prog, device=dev, cfg=NavConfig(**nc))
    else:
        from .backends.fake import GoalSeekNavigator
        nav = GoalSeekNavigator()

    tc = dict(cfg.get("detector") or {"type": "fake"})
    if tc["type"] in ("yolo_world", "vis_open_v"):
        from .perception.detector import YoloWorld
        yolo = YoloWorld(_path(base, tc.get("model", "yolov8s-worldv2.pt")), device=tc.get("device", "cuda"),
                         imgsz=int(tc.get("imgsz", 1280)), conf=float(tc.get("conf", 0.05)),
                         tiles=tc.get("tiles"), tile_imgsz=int(tc.get("tile_imgsz", 960)))
    elif tc["type"] == "yolo_world_ts":
        from .perception.detector import YoloWorldTS
        yolo = YoloWorldTS(_path(base, tc["model"]), tc["classes"], device=tc.get("device", "cuda"),
                           imgsz=int(tc.get("imgsz", 640)), conf=float(tc.get("conf", 0.05)))
    else:
        from .backends.fake import FakeYolo
        yolo = FakeYolo(backend, max_range_m=float(tc.get("max_range_m", 35.0)))
    detector = TargetDetector(yolo, min_color_frac=float(tc.get("min_color_frac", 0.2)),
                              hints=ColorHints() if tc.get("color_hints", True) else None,
                              min_conf=float(tc.get("conf", 0.05)))

    reporters = []
    dog = cfg.get("dog") or {}
    if dog.get("http"):
        from .comm.dog import HttpReporter
        reporters.append(HttpReporter(dog["http"], token=dog.get("token", "")))
    if dog.get("mqtt"):
        from .comm.dog import MqttReporter
        reporters.append(MqttReporter(**dog["mqtt"]))

    mcfg = MissionConfig(**(cfg.get("mission") or {}))
    return MissionRunner(backend, depth, nav, detector, reporters, mcfg, Path(_path(base, cfg.get("out_dir", "runs"))),
                         drone_id=cfg.get("drone_id", "drone_1"))


def serve(runner: MissionRunner, host: str = "0.0.0.0", port: int = 8088,
          parse_fn: Optional[Callable[[str], MissionSpec]] = None) -> ThreadingHTTPServer:
    parse = parse_fn or parse_instruction
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            log.debug(fmt, *a)

        def _reply(self, code: int, obj: Any) -> None:
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self) -> Tuple[Dict[str, Any], str]:
            try:
                n = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(n) or b"{}"), ""
            except (ValueError, json.JSONDecodeError) as exc:
                return {}, str(exc)

        def do_GET(self):
            if self.path == "/api/v1/health":
                st = runner.status()
                return self._reply(200, {"ok": True, "busy": runner.busy(), "state": st.get("state")})
            if self.path == "/api/v1/missions/current":
                return self._reply(200, runner.status())
            self._reply(404, {"error": "not found"})

        def do_POST(self):
            body, err = self._body()
            if err:
                return self._reply(400, {"error": f"bad json: {err}"})
            if self.path in ("/api/v1/parse", "/api/v1/missions"):
                try:
                    spec = parse(str(body.get("instruction", "")))
                except ParseError as exc:
                    return self._reply(422, {"error": str(exc)})
                if self.path == "/api/v1/parse":
                    return self._reply(200, spec.to_dict())
                if runner.busy():
                    return self._reply(409, {"error": "mission running", "status": runner.status()})
                mid = runner.start(spec)
                return self._reply(202, {"mission_id": mid, "spec": spec.to_dict()})
            if self.path == "/api/v1/missions/current/abort":
                runner.abort()
                return self._reply(200, {"ok": True})
            self._reply(404, {"error": "not found"})

    httpd = ThreadingHTTPServer((host, int(port)), H)
    log.info("serving on %s:%d", host, httpd.server_address[1])
    return httpd
