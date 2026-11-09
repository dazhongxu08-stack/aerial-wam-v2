"""AirSim end-to-end demo: place a target (+ distractors), run one mission, score the report.

    python -m aerial_finder.tools.sim_demo --config sim_airsim.json --scene scene.json \
        --instruction "去前方200m左右找一个红色的凳子，然后把位置发给机器狗"

scene.json::

    {"start": [x, y, z_up], "yaw_deg": 30,
     "objects": [{"name": "finder_target", "asset": "SM_Stool_1", "pos": [x, y, z_up], "scale": 1.0,
                  "color": "red", "target": true},
                 {"name": "finder_distr_1", "asset": "SM_Wood_Barrel_01", "pos": [...], "color": "red"}]}

Colours are applied as flat textures written next to the scene file (the UE
host must see the same path). Spawned objects are removed at the end unless
``--keep-objects``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

COLOR_RGB = {"red": (200, 20, 20), "blue": (20, 40, 200), "green": (30, 160, 40), "yellow": (230, 200, 20),
             "white": (235, 235, 235), "black": (15, 15, 15), "orange": (240, 120, 10)}


def set_texture(client, name: str, path: str) -> None:
    """Older AirSim servers take (object, texture) without the component id the newer client sends."""
    try:
        if client.client.call("simSetObjectMaterialFromTexture", name, path) is False:
            raise RuntimeError(f"texture not applied to {name}")
    except Exception:  # noqa: BLE001
        if client.simSetObjectMaterialFromTexture(name, path) is False:
            raise RuntimeError(f"texture not applied to {name}")


def place_objects(client, objects, tex_dir: Path):
    import airsim

    names = []
    tag = time.strftime("%H%M%S")
    for o in objects:
        x, y, z = (float(v) for v in o["pos"])
        pose = airsim.Pose(airsim.Vector3r(x, y, -z), airsim.to_quaternion(0.0, 0.0, math.radians(o.get("yaw_deg", 0))))
        s = float(o.get("scale", 1.0))
        # A duplicate actor name is a fatal UE error on this server, so names are made unique per run.
        name = client.simSpawnObject(f"{o['name']}_{tag}", o["asset"], pose, airsim.Vector3r(s, s, s), False)
        names.append(name)
        if o.get("color"):
            p = tex_dir / f"tex_{o['color']}.png"
            if not p.is_file():
                img = np.zeros((64, 64, 3), np.uint8)
                img[:] = COLOR_RGB[o["color"]][::-1]
                cv2.imwrite(str(p), img)
            set_texture(client, name, str(p.resolve()))
        got = client.simGetObjectPose(name).position
        logging.info("spawned %s (%s) at %.1f %.1f %.1f", name, o["asset"], got.x_val, got.y_val, -got.z_val)
    return names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--keep-objects", action="store_true")
    ap.add_argument("--dog", default="", help="dog URL; default starts a local mock dog")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(message)s")
    from ..comm.dog import HttpReporter, MockDogServer
    from ..mission.parser import parse_instruction
    from ..service import build

    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    scene_path = Path(args.scene).resolve()
    scene = json.loads(scene_path.read_text())
    dog = None
    if not args.dog:
        dog = MockDogServer().start()
        cfg["dog"] = {"http": dog.url}
    else:
        cfg["dog"] = {"http": args.dog}
    runner = build(cfg, cfg_path.parent)
    be = runner.backend
    names = place_objects(be.c, scene.get("objects", []), scene_path.parent)
    try:
        be.teleport(scene["start"], math.radians(scene.get("yaw_deg", 0.0)))
        time.sleep(1.0)
        if "ground_z" in scene:
            runner.cfg.ground_z = float(scene["ground_z"])
        res = runner.run(parse_instruction(args.instruction))
        truth = [o for o in scene["objects"] if o.get("target")]
        out = {"result": res, "truth": None, "error_m": None}
        if truth:
            t = np.asarray(truth[0]["pos"], float)
            lat, lon, alt = be.geo_ref.to_gps(*t)
            out["truth"] = {"local": t.tolist(), "lat": lat, "lon": lon}
            msg = (res.get("result") or {}).get("message")
            if msg:
                n, e, _ = be.geo_ref.to_local(msg["position"]["lat"], msg["position"]["lon"], msg["position"]["alt_m"])
                out["error_m"] = round(math.hypot(n - t[0], e - t[1]), 2)
        if dog is not None:
            out["dog_received"] = len(dog.received)
        mid = res.get("mission_id")
        if mid:
            (runner.out_root / mid / "demo_score.json").write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps({k: v for k, v in out.items() if k != "result"} | {"state": res.get("state"),
                                                                          "reason": res.get("reason")}, indent=1))
        return 0 if res.get("state") == "DONE" and (out["error_m"] is None or out["error_m"] < 3.0) else 2
    finally:
        be.hover()
        if not args.keep_objects:
            for n in names:
                be.c.simDestroyObject(n)
        if dog is not None:
            dog.stop()


if __name__ == "__main__":
    sys.exit(main())
