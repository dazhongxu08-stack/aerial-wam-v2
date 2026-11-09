"""Single HTTP/CLI surface for ground, air, and link."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from .adapters import AirAdapter, GroundAdapter, LinkAdapter, make_parser
from .ports import Facade
from .schema import AirTaskIn, GroundTaskIn, OPENAPI, ParseIn


def _json(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def build_facade(config: Optional[str] = None, *, load_parser: bool = True) -> Facade:
    parser = make_parser(config) if load_parser else make_parser(None)
    return Facade(parser, GroundAdapter(), AirAdapter(), LinkAdapter())


def serve(facade: Facade, host: str = "0.0.0.0", port: int = 8100) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:
            return

        def _reply(self, code: int, obj: Any) -> None:
            b = _json(obj)
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

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/v1/health", "/health"):
                return self._reply(200, facade.health().to_dict())
            if path in ("/v1/openapi.json", "/openapi.json"):
                return self._reply(200, OPENAPI)
            if path == "/v1/ground/task":
                if not facade.ground:
                    return self._reply(503, {"error": "ground offline"})
                return self._reply(200, facade.ground.status())
            if path == "/v1/air/status":
                if not facade.air:
                    return self._reply(503, {"error": "air offline"})
                return self._reply(200, facade.air.status())
            if path == "/v1/link/status":
                if not facade.link:
                    return self._reply(503, {"error": "link offline"})
                return self._reply(200, facade.link.status())
            self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            body, err = self._body()
            if err:
                return self._reply(400, {"error": err})
            try:
                if path == "/v1/parse":
                    return self._reply(200, facade.parser.parse(ParseIn.from_body(body)).to_dict())
                if path == "/v1/ground/tasks":
                    if not facade.ground:
                        return self._reply(503, {"error": "ground offline"})
                    rec = facade.ground.start(GroundTaskIn.from_body(body))
                    return self._reply(202 if rec.get("mission_id") else 400, rec)
                if path == "/v1/ground/abort":
                    if not facade.ground:
                        return self._reply(503, {"error": "ground offline"})
                    return self._reply(200, facade.ground.abort())
                if path == "/v1/air/tasks":
                    if not facade.air:
                        return self._reply(503, {"error": "air offline"})
                    rec = facade.air.start(AirTaskIn.from_body(body))
                    code = 202 if rec.get("ok") else 400
                    return self._reply(code, rec)
                if path == "/v1/link/cmd":
                    if not facade.link:
                        return self._reply(503, {"error": "link offline"})
                    return self._reply(200, facade.link.cmd(body))
            except ValueError as exc:
                return self._reply(422, {"error": str(exc)})
            self._reply(404, {"error": "not found"})

    return ThreadingHTTPServer((host, int(port)), H)
