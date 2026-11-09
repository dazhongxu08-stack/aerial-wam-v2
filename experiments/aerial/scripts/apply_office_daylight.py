#!/usr/bin/env python3
"""Re-apply the office daylight scene. Idempotent: existing DayFill lights are kept.

Spawned lights cannot be moved. Setting object_mobility on a light crashes this editor.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/simworld/run")
import collect_office_train as col

CFG_PATH = Path("/data/simworld/run/office_daylight.json")
LOCAL_CFG = Path(__file__).with_name("office_daylight.json")


def load_cfg() -> dict:
    path = CFG_PATH if CFG_PATH.is_file() else LOCAL_CFG
    return json.loads(path.read_text())


def cmd(ucv, text: str) -> str:
    ans = str(col.req(ucv, text))
    print(f"{text} -> {ans}", flush=True)
    return ans


def apply(ucv, cfg: dict) -> None:
    fog = cfg["fog"]
    cmd(ucv, f"vrun r.Fog {fog['r.Fog']}")
    cmd(ucv, f"vrun r.VolumetricFog {fog['r.VolumetricFog']}")
    for name in fog["hide"]:
        cmd(ucv, f"vset /object/{name}/hide")
    exp = cfg["exposure"]
    cmd(ucv, f"vrun r.EyeAdaptationQuality {exp['r.EyeAdaptationQuality']}")
    cmd(ucv, f"vrun r.DefaultFeature.AutoExposure {exp['r.DefaultFeature.AutoExposure']}")
    cmd(ucv, f"vrun r.ExposureOffset {exp['r.ExposureOffset']}")

    present = set(str(col.req(ucv, "vget /objects")).split())
    lights = cfg["lights"]
    color = " ".join(str(int(v)) for v in lights["color"])
    for name in lights["names"]:
        if name not in present:
            spawned = cmd(ucv, f"vset /objects/spawn {lights['class']} {name}")
            if "error" in spawned.lower():
                raise RuntimeError(spawned)
        cmd(ucv, f"vset /object/{name}/color {color}")

    barrels = cfg["barrels"]
    props = json.loads(Path(barrels["manifest"]).read_text())
    z = float(barrels["z"])
    for item in props:
        name, x, y = item["name"], float(item["x"]), float(item["y"])
        if barrels.get("mobility"):
            cmd(ucv, f"vset /object/{name}/object_mobility true")
        cmd(ucv, f"vset /object/{name}/physics false")
        cmd(ucv, f"vset /object/{name}/location {x} {y} {z}")
        got = cmd(ucv, f"vget /object/{name}/location")
        gx, gy, gz = [float(p) for p in str(got).split()[:3]]
        if abs(gx - x) > 1 or abs(gy - y) > 1 or abs(gz - z) > 1:
            raise RuntimeError(f"{name} stayed at {got}")


def check_gap(ucv) -> None:
    cams = str(col.req(ucv, "vget /cameras")).split()
    if "FusionCameraSensor" not in cams:
        cmd(ucv, "vset /cameras/spawn")
        time.sleep(0.3)
        cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.ensure_dog(ucv)
    col.place(ucv, -160, -200, 0)
    col.look(ucv, cam, -160, -200, 0)
    time.sleep(0.12)
    rgb, depth = col.grab(ucv, cam)
    image, _seen, _warped = col.warp_pair(rgb, depth)
    center = image[72:152, 72:152]
    print(
        f"GAP center {float(center.mean()):.1f} rgb {np.round(center.mean(axis=(0, 1)), 1).tolist()} "
        f"warp {float(image.mean()):.1f}",
        flush=True,
    )


def main() -> int:
    cfg = load_cfg()
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    apply(ucv, cfg)
    check_gap(ucv)
    print("SAVED_SETUP_APPLIED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
