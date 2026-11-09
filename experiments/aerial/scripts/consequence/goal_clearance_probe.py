"""How close is each route goal to a wall? Hover at the goal and read DepthPlanar at 8 yaws (no flight).

Goal = last pos of each episode; AirSim NED z = -pos z. Contacts use a fresh simGetCollisionInfo timestamp
(has_collided stays latched after teleports). clearance_m = min centre-window depth over the 8 yaws.
usage: goal_clearance_probe.py OUT_JSON ANNOTATION [ANNOTATION ...] [--routes id,id]
"""
from __future__ import annotations

import argparse
import json
import math

import airsim
import numpy as np

CAM = "front_custom"
NEAR_WALL_M = 4.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_json")
    ap.add_argument("annotations", nargs="+")
    ap.add_argument("--routes", default="", help="comma list of route_id to keep (default: all)")
    args = ap.parse_args()
    keep = set(filter(None, args.routes.split(",")))
    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.enableApiControl(True)

    def place(p, yaw: float) -> str:
        prev = int(c.simGetCollisionInfo().time_stamp)
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*map(float, p)), airsim.to_quaternion(0, 0, yaw)), True)
        c.moveByVelocityAsync(0, 0, 0, 0.6).join()
        ci = c.simGetCollisionInfo()
        return str(ci.object_name) if ci.has_collided and int(ci.time_stamp) != prev else ""

    def depth():
        r = c.simGetImages([airsim.ImageRequest(CAM, airsim.ImageType.DepthPlanar, True, False)])[0]
        d = np.array(r.image_data_float, np.float32).reshape(r.height, r.width)
        h, w = d.shape
        win = d[h // 2 - h // 8:h // 2 + h // 8, w // 2 - w // 8:w // 2 + w // 8]
        return round(float(win.min()), 2), round(float(d.min()), 2)

    rows = []
    for ann in args.annotations:
        for ep in json.load(open(ann))["episodes"]:
            rid = ep["route_id"]
            if keep and rid not in keep:
                continue
            gx, gy, gz = ep["pos"][-1]
            goal = np.array([gx, gy, -gz], float)
            yaws, contact = [], ""
            for k in range(8):
                yaw = k * math.pi / 4
                place(goal + [0, 0, -8], yaw)
                contact = contact or place(goal, yaw)
                cmin, fmin = depth()
                yaws.append({"yaw_deg": k * 45, "center_min": cmin, "full_min": fmin})
            best = min(yaws, key=lambda y: y["center_min"])
            row = {"annotation": ann, "route_id": rid, "goal_ned": goal.tolist(), "contact": contact,
                   "clearance_m": best["center_min"], "clearance_yaw_deg": best["yaw_deg"],
                   "near_wall": best["center_min"] < NEAR_WALL_M, "yaws": yaws}
            rows.append(row)
            print(rid, {k: row[k] for k in ("contact", "clearance_m", "clearance_yaw_deg", "near_wall")}, flush=True)
    json.dump({"near_wall_m": NEAR_WALL_M, "goals": rows}, open(args.out_json, "w"), indent=1)
    print("WROTE", args.out_json, len(rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
