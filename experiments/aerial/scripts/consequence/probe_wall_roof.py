"""Where a wall renders vs where it collides, by height: walk the drone toward it along one axis.

For each (lateral position, height): render = start + center-pixel DepthPlanar distance looking along the scan
direction; hit = first placed position (0.1 m steps) that gives a new collision. A hit with no rendered surface
within ~1 m of it means collision geometry the camera cannot see (e.g. SM_NYCB_579 above its roof).
usage: probe_wall_roof.py --axis +y --start -860,14 --lateral=-860,-852,-845 --zs 27,28,29,30,31,33,36 --out o.json
"""
import argparse
import json
import math
import time

import airsim
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--axis", choices=["+x", "-x", "+y", "-y"], required=True)
ap.add_argument("--start", required=True, help="x,y of the first scan; the scan coordinate starts here")
ap.add_argument("--lateral", required=True, help="positions on the other axis")
ap.add_argument("--zs", required=True)
ap.add_argument("--max-m", type=float, default=6.0)
ap.add_argument("--out", required=True)
ap.add_argument("--vehicle", default="drone_1")
ap.add_argument("--camera", default="front_custom")
args = ap.parse_args()

VK = {"vehicle_name": args.vehicle}
c = airsim.MultirotorClient()
c.confirmConnection()
c.enableApiControl(True, **VK)
c.armDisarm(True, **VK)
along_x = args.axis[1] == "x"
sgn = 1.0 if args.axis[0] == "+" else -1.0
yaw = (0.0 if sgn > 0 else math.pi) if along_x else (math.pi / 2 if sgn > 0 else -math.pi / 2)
sx, sy = (float(v) for v in args.start.split(","))
s0 = sx if along_x else sy


def place(s, lat, z):
    x, y = (s, lat) if along_x else (lat, s)
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, -z), airsim.to_quaternion(0, 0, yaw)), True, **VK)


def ahead():
    r = c.simGetImages([airsim.ImageRequest(args.camera, airsim.ImageType.DepthPlanar, True, False)], **VK)[0]
    d = np.asarray(r.image_data_float, np.float32).reshape(r.height, r.width)
    h, w = d.shape
    return float(d[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3].min())


rows = []
for lat in (float(v) for v in args.lateral.split(",")):
    for z in (float(v) for v in args.zs.split(",")):
        place(s0, lat, z)
        c.hoverAsync(**VK)
        time.sleep(0.4)
        place(s0, lat, z)
        time.sleep(0.2)
        render = s0 + sgn * ahead()
        ts0 = c.simGetCollisionInfo(**VK).time_stamp
        hit, obj = None, None
        for i in range(1, int(args.max_m / 0.1) + 1):
            s = s0 + sgn * 0.1 * i
            place(s, lat, z)
            time.sleep(0.12)
            ci = c.simGetCollisionInfo(**VK)
            if ci.has_collided and ci.time_stamp != ts0:
                hit, obj = round(s, 2), ci.object_name
                break
        row = {"lateral": lat, "z": z, "render": round(render, 2), "hit": hit, "obj": obj,
               "render_minus_hit_m": None if hit is None else round(sgn * (render - hit), 2)}
        rows.append(row)
        print(row, flush=True)
place(s0, float(args.lateral.split(",")[0]), 40.0)
json.dump(rows, open(args.out, "w"), indent=1)
