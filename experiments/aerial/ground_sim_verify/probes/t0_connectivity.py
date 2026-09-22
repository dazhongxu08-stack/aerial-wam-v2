#!/usr/bin/env python3
"""T0 — CARLA TCP + world + vehicle blueprint availability."""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import report  # noqa: E402

HOST = os.environ.get("CARLA_HOST", "127.0.0.1")
PORT = int(os.environ.get("CARLA_PORT", "2200"))
VEHICLE_BP = os.environ.get("GROUND_VEHICLE_BP", "vehicle.tesla.model3")

res: dict = {"host": HOST, "port": PORT, "vehicle_bp": VEHICLE_BP}

try:
    import carla  # type: ignore

    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)
    world = client.get_world()
    map_name = world.get_map().name
    bp = world.get_blueprint_library().find(VEHICLE_BP)
    res["connected"] = True
    res["map"] = map_name
    res["vehicle_bp_found"] = bp is not None
    res["pass"] = bool(bp is not None)
except Exception as e:  # noqa: BLE001
    res["connected"] = False
    res["pass"] = False
    res["error"] = repr(e)

report.merge("t0_connectivity", res)
print("[T0]", res)
raise SystemExit(0 if res.get("pass") else 1)
