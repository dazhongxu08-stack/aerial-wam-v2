#!/usr/bin/env python3
"""Write an AirSim settings.json for the dog stand-in from an existing drone one.

Changes only: the API port, clock speed, Scene capture size, every camera's X / Z
offset (ahead of the rotors and above the body, so the body rides at dog-torso
height), and drops lidar sensors. Pass the
output to the renderer with ``-settings=<path>``; the base file is not touched.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def _scene_size(captures: Any, wh: Tuple[int, int]) -> None:
    for cs in captures or []:
        if int(cs.get("ImageType", -1)) == 0:
            cs["Width"], cs["Height"] = int(wh[0]), int(wh[1])


def dog_settings(base: Dict[str, Any], *, port: int, cam_z: float, clock: float, cam_x: float = 0.35,
                 scene_wh: Tuple[int, int] = (640, 360)) -> Dict[str, Any]:
    s = copy.deepcopy(base)
    s["ApiServerPort"] = int(port)
    s["ClockSpeed"] = float(clock)
    _scene_size(s.get("CameraDefaults", {}).get("CaptureSettings"), scene_wh)
    for veh in s.get("Vehicles", {}).values():
        for cam in veh.get("Cameras", {}).values():
            cam["X"], cam["Z"] = float(cam_x), float(cam_z)
            _scene_size(cam.get("CaptureSettings"), scene_wh)
        # Body-relative chase view: behind and above, looking forward and slightly down.
        # Requested only when recording, so the policy camera stays front_custom.
        scene = next(
            (c for c in (veh.get("Cameras", {}).get("front_custom", {}).get("CaptureSettings") or [])
             if int(c.get("ImageType", -1)) == 0),
            {"ImageType": 0, "Width": int(scene_wh[0]), "Height": int(scene_wh[1]), "FOV_Degrees": 90},
        )
        veh.setdefault("Cameras", {})["chase"] = {
            "CaptureSettings": [dict(scene)],
            "X": -4.0, "Y": 0.0, "Z": -2.0,
            "Pitch": -18.0, "Roll": 0.0, "Yaw": 0.0,
        }
        sensors = veh.get("Sensors", {})
        for name in [k for k, v in sensors.items() if int(v.get("SensorType", -1)) == 6]:
            del sensors[name]
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=41481)
    ap.add_argument("--cam-z", type=float, default=-0.15)
    ap.add_argument("--cam-x", type=float, default=0.35)
    ap.add_argument("--clock", type=float, default=1.0)
    ap.add_argument("--scene-wh", type=int, nargs=2, default=(640, 360))
    a = ap.parse_args()
    base = json.loads(Path(a.base).read_text(encoding="utf-8"))
    out = dog_settings(base, port=a.port, cam_z=a.cam_z, clock=a.clock, cam_x=a.cam_x, scene_wh=tuple(a.scene_wh))
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {a.out} port={a.port} cam_x={a.cam_x} cam_z={a.cam_z} clock={a.clock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
