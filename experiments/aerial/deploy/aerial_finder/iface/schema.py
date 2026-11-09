"""Public I/O records. No stack-specific type names."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParseIn:
    text: str

    @classmethod
    def from_body(cls, body: Dict[str, Any]) -> "ParseIn":
        t = str(body.get("text") or body.get("instruction") or "").strip()
        if not t:
            raise ValueError("text required")
        return cls(text=t)


@dataclass
class ParseOut:
    text: str
    object: str
    prompts: List[str]
    color: Optional[str] = None
    range_m: Optional[float] = None
    range_tol_m: Optional[float] = None
    bearing_rad: Optional[float] = None
    heading_rad: Optional[float] = None
    notify: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_spec_dict(cls, d: Dict[str, Any]) -> "ParseOut":
        notify = list(d.get("report_to") or [])
        notify = ["ground" if x in ("dog", "ugv") else "air" if x in ("drone", "uav") else x for x in notify]
        return cls(
            text=str(d.get("text") or ""),
            object=str(d.get("label") or ""),
            prompts=list(d.get("prompts") or []),
            color=d.get("color"),
            range_m=d.get("distance_m"),
            range_tol_m=d.get("tolerance_m"),
            bearing_rad=d.get("body_bearing"),
            heading_rad=d.get("world_heading"),
            notify=notify,
        )


@dataclass
class HealthOut:
    ok: bool
    ground: str = "unknown"
    air: str = "unknown"
    link: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroundTaskIn:
    text: str
    arm: bool = False

    @classmethod
    def from_body(cls, body: Dict[str, Any]) -> "GroundTaskIn":
        t = str(body.get("text") or body.get("instruction") or "").strip()
        if not t:
            raise ValueError("text required")
        return cls(text=t, arm=bool(body.get("arm", False)))


@dataclass
class AirTaskIn:
    text: str
    range_m: float
    bearing_deg: float
    arm: bool = False
    air_s: float = 75.0

    @classmethod
    def from_body(cls, body: Dict[str, Any]) -> "AirTaskIn":
        t = str(body.get("text") or body.get("goal") or "").strip()
        if not t:
            raise ValueError("text required")
        if body.get("range_m") is None or body.get("bearing_deg") is None:
            raise ValueError("range_m and bearing_deg required")
        return cls(
            text=t,
            range_m=float(body["range_m"]),
            bearing_deg=float(body["bearing_deg"]),
            arm=bool(body.get("arm", False)),
            air_s=float(body.get("air_s", 75.0)),
        )


OPENAPI: Dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "rt API", "version": "1"},
    "paths": {
        "/v1/health": {"get": {"summary": "Platform health", "responses": {"200": {"description": "ok"}}}},
        "/v1/parse": {"post": {"summary": "Parse instruction to a task spec",
                               "requestBody": {"content": {"application/json": {"schema": {
                                   "type": "object", "required": ["text"],
                                   "properties": {"text": {"type": "string"}}}}}}}},
        "/v1/ground/tasks": {"post": {"summary": "Start a ground find task",
                                      "description": "Requires arm=true. Does not move unless armed."}},
        "/v1/ground/task": {"get": {"summary": "Current ground task"}},
        "/v1/ground/abort": {"post": {"summary": "Abort ground task"}},
        "/v1/air/status": {"get": {"summary": "Air platform / hub status"}},
        "/v1/air/tasks": {"post": {"summary": "Start an air task",
                                   "description": "Requires arm=true. Does not take off unless armed."}},
        "/v1/link/status": {"get": {"summary": "Command-link status"}},
        "/v1/link/cmd": {"post": {"summary": "Forward a stick/command object to the link"}},
    },
}
