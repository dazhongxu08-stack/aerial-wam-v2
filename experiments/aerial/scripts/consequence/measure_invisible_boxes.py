"""Measure the collision extent of known invisible volumes by hovering the drone.

A point counts as inside only when simGetCollisionInfo reports that object with a
new timestamp. Extent is the last inside point along each axis, so the box is the
region where the drone body overlaps the volume.
"""
from __future__ import annotations

import json
import time

import airsim

OBJECTS = {
    "SM_NYCB_578": [(140.0, 88.0)],
    "SM_NYCB_587": [(64.6, 80.8), (70.0, 80.0), (60.0, 82.0), (65.0, 75.0)],
    "SM_NYCB_869": [(-861.0, 39.0), (-870.0, 40.0), (-850.0, 40.0)],
    "SM_NYCB_867": [(-890.0, 39.0), (-885.0, 30.0), (-895.0, 20.0)],
    "SM_NYCB_22": [(-986.0, 85.0), (-976.0, 80.0), (-990.0, 90.0), (-1000.0, 75.0), (-970.0, 70.0)],
    "SM_NYCB_1-5_2406": [(-1085.0, 98.0), (-1080.0, 90.0), (-1075.0, 85.0), (-1088.0, 80.0)],
}
Z_LEVELS = [-16.0, -20.0, -24.0, -28.0, -32.0]
LIMIT_M = 80.0
TOL_M = 0.5
SETTLE_S = 0.25


def main() -> int:
    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.enableApiControl(True)

    def name_at(x: float, y: float, z: float) -> str:
        prev = int(c.simGetCollisionInfo().time_stamp)
        pose = airsim.Pose(airsim.Vector3r(float(x), float(y), float(z)), airsim.to_quaternion(0, 0, 0))
        c.simSetVehiclePose(pose, True)
        c.moveByVelocityAsync(0, 0, 0, SETTLE_S).join()
        ci = c.simGetCollisionInfo()
        if ci.has_collided and int(ci.time_stamp) != prev:
            return str(ci.object_name)
        return ""

    def inside(name: str, x: float, y: float, z: float) -> bool:
        return name_at(x, y, z) == name

    def edge(name: str, origin, axis: int, sign: int):
        lo = 0.0
        step = 1.0
        hi = None
        while step <= LIMIT_M:
            p = list(origin)
            p[axis] += sign * step
            if inside(name, *p):
                lo = step
                step *= 2
            else:
                hi = step
                break
        if hi is None:
            return None
        while hi - lo > TOL_M:
            mid = (lo + hi) / 2
            p = list(origin)
            p[axis] += sign * mid
            if inside(name, *p):
                lo = mid
            else:
                hi = mid
        p = list(origin)
        p[axis] += sign * lo
        return [round(v, 2) for v in p]

    out = {}
    for name, xys in OBJECTS.items():
        levels = []
        for z in Z_LEVELS:
            origins = [[x, y, z] for x, y in xys if inside(name, x, y, z)]
            if not origins:
                levels.append({"z": z, "inside": False})
                print(name, "z", z, "no inside seed", flush=True)
                continue
            boxes = []
            for origin in origins:
                faces = {}
                for axis, sign, key in ((0, -1, "x_lo"), (0, 1, "x_hi"), (1, -1, "y_lo"), (1, 1, "y_hi")):
                    faces[key] = edge(name, origin, axis, sign)
                boxes.append(faces)
                print(name, "z", z, "from", [round(v, 1) for v in origin], faces, flush=True)
            levels.append({"z": z, "origins": [[round(v, 2) for v in o] for o in origins], "boxes": boxes})
        z_faces = {}
        origin = None
        for lv in levels:
            if lv.get("origins") and abs(lv["z"] + 24.0) < 1:
                origin = lv["origins"][0]
        if origin is None:
            origin = next((lv["origins"][0] for lv in levels if lv.get("origins")), None)
        if origin is not None:
            for sign, key in ((-1, "z_lo"), (1, "z_hi")):
                z_faces[key] = edge(name, origin, 2, sign)
            print(name, "z faces", z_faces, flush=True)
        out[name] = {"levels": levels, "z_faces": z_faces}
    path = "experiments/aerial/rl/artifacts/invisible_boxes_flight_20261012.json"
    open(path, "w").write(json.dumps(out, indent=1))
    print("WROTE", path, flush=True)
    return 0


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    finally:
        print("elapsed_s", round(time.time() - t0, 1), flush=True)
