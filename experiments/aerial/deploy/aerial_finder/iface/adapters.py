"""Adapters from public ports onto the installed product."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .schema import AirTaskIn, GroundTaskIn, ParseIn, ParseOut


def _http(method: str, url: str, body: Optional[dict] = None, timeout: float = 4.0) -> Dict[str, Any]:
    data = None
    headers = {"Connection": "close"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
    return json.loads(raw) if raw else {}


class ParserAdapter:
    def __init__(self, parse_fn: Callable[[str], Any]) -> None:
        self._fn = parse_fn

    def parse(self, req: ParseIn) -> ParseOut:
        spec = self._fn(req.text)
        d = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
        return ParseOut.from_spec_dict(d)


class GroundAdapter:
    """Talks to the existing find HTTP on 8088, or holds a local runner."""

    def __init__(self, base: str = "http://127.0.0.1:8088") -> None:
        self.base = base.rstrip("/")

    def ready(self) -> str:
        try:
            st = _http("GET", self.base + "/api/v1/health")
            if st.get("busy"):
                return "busy"
            return "idle" if st.get("ok") else "down"
        except (urllib.error.URLError, TimeoutError, OSError):
            return "offline"

    def start(self, req: GroundTaskIn) -> Dict[str, Any]:
        if not req.arm:
            return {"ok": False, "error": "arm required", "armed": False}
        try:
            return _http("POST", self.base + "/api/v1/missions", {"instruction": req.text}, timeout=30.0)
        except urllib.error.HTTPError as exc:
            return {"ok": False, "error": exc.read().decode()[:400]}

    def status(self) -> Dict[str, Any]:
        try:
            return _http("GET", self.base + "/api/v1/missions/current")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"ok": False, "error": str(exc)}

    def abort(self) -> Dict[str, Any]:
        try:
            return _http("POST", self.base + "/api/v1/missions/current/abort", {})
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"ok": False, "error": str(exc)}


class AirAdapter:
    def __init__(self, hub: str = "") -> None:
        self.hub = (hub or os.environ.get("AIR_HUB") or "http://192.168.54.2:8090").rstrip("/")

    def ready(self) -> str:
        try:
            st = _http("GET", self.hub + "/api/status")
            if not st.get("ok"):
                return "down"
            if st.get("airborne"):
                return "busy"
            return "idle"
        except (urllib.error.URLError, TimeoutError, OSError):
            return "offline"

    def status(self) -> Dict[str, Any]:
        try:
            st = _http("GET", self.hub + "/api/status")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": bool(st.get("ok")),
            "airborne": bool(st.get("airborne")),
            "battery": st.get("battery"),
            "height_cm": st.get("height_cm"),
            "frame_age_s": st.get("frame_age_s"),
            "allow": bool(st.get("allow_fly")),
        }

    def start(self, req: AirTaskIn) -> Dict[str, Any]:
        if not req.arm:
            return {"ok": False, "error": "arm required", "armed": False}
        return {"ok": False, "error": "air start is CLI-only; use rt-air with arm after operator 飞/采"}


class LinkAdapter:
    def __init__(self, base: str = "http://127.0.0.1:8092") -> None:
        self.base = base.rstrip("/")

    def ready(self) -> str:
        try:
            st = _http("GET", self.base + "/api/status")
            return "up" if st.get("ok") else "down"
        except (urllib.error.URLError, TimeoutError, OSError):
            return "offline"

    def status(self) -> Dict[str, Any]:
        try:
            return _http("GET", self.base + "/api/status")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"ok": False, "error": str(exc)}

    def cmd(self, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _http("POST", self.base + "/api/cmd", body, timeout=8.0)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"ok": False, "error": str(exc)}


def make_parser(config_path: Optional[str]) -> ParserAdapter:
    from .. import __main__ as m
    from ..mission.llm_parser import make_parse_fn
    from ..mission.parser import parse_instruction

    if not config_path:
        return ParserAdapter(parse_instruction)
    cfg_path = Path(config_path).expanduser().resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    fn = make_parse_fn(cfg.get("parser"), m._install_root(cfg_path))
    return ParserAdapter(fn)
