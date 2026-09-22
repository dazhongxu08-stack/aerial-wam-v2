#!/usr/bin/env python3
"""Ground sim Fork decision: G-A / G-A⁻ / G-B.

Exit: 0=G-A, 2=G-A⁻, 3=G-B
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Tuple

_ROOT = pathlib.Path(__file__).resolve().parent
_REPORT = _ROOT / "artifacts" / "ground_sim_capability_report.json"


def _ok(node: dict) -> bool:
    return isinstance(node, dict) and bool(node.get("pass"))


def decide(data: dict) -> Tuple[str, int, str, dict]:
    t0 = data.get("t0_connectivity", {})
    t1 = data.get("t1_carla_ground", {})

    connected = _ok(t0)
    rgb = _ok(t1.get("rgb"))
    depth = _ok(t1.get("depth"))
    odom = _ok(t1.get("odom"))
    coll = _ok(t1.get("collision"))
    phys = _ok(t1.get("physics"))
    continuous = _ok(t1.get("continuous_rgb"))

    caps = {
        "connected": connected,
        "rgb": rgb,
        "depth": depth,
        "odom": odom,
        "collision": coll,
        "physics": phys,
        "continuous_rgb": continuous,
    }

    if not connected:
        fork, code, why = "G-B", 3, "CARLA connect/spawn failed"
    elif rgb and depth and odom and coll and phys and continuous:
        fork, code, why = "G-A", 0, "full ground RGB+depth+odom+collision+physics"
    elif rgb and odom:
        missing = [k for k, v in caps.items() if not v and k not in ("connected",)]
        fork, code, why = "G-A⁻", 2, f"partial: missing {missing}"
    else:
        fork, code, why = "G-B", 3, "insufficient ground sensors"

    return fork, code, why, caps


def main() -> int:
    if not _REPORT.exists():
        print(f"Missing {_REPORT}", file=sys.stderr)
        return 3
    data = json.loads(_REPORT.read_text())
    fork, code, why, caps = decide(data)
    print(f"FORK={fork} exit={code}")
    print(f"why: {why}")
    for k, v in caps.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
