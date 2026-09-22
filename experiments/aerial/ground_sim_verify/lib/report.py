"""JSON report merge helper (mirrors aerial sim_verify)."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

_ARTIFACT = pathlib.Path(__file__).resolve().parents[1] / "artifacts"
_ARTIFACT.mkdir(parents=True, exist_ok=True)
_DEFAULT_PATH = _ARTIFACT / "ground_sim_capability_report.json"


def merge(section: str, data: Dict[str, Any], path: pathlib.Path | None = None) -> pathlib.Path:
    out = path or _DEFAULT_PATH
    existing: Dict[str, Any] = {}
    if out.exists():
        existing = json.loads(out.read_text())
    existing[section] = data
    out.write_text(json.dumps(existing, indent=2))
    return out
