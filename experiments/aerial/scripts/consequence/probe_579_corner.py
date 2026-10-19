"""Map where SM_NYCB_579 collides vs where it renders near the od_hard_02 crash corner (x~123, y~88.4).

For each (y, z_up): rendered = center-row DepthPlanar distance looking west from x=127;
collides = largest x (scanning west from 127 in 0.1 m) where placing the drone gives a new 579 collision.
"""
import json
import math
import time

import airsim
import numpy as np

VK = {"vehicle_name": "drone_1"}
c = airsim.MultirotorClient()
c.confirmConnection()
c.enableApiControl(True, **VK)
c.armDisarm(True, **VK)


def place(x, y, z_up, yaw):
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, -z_up), airsim.to_quaternion(0, 0, yaw)), True, **VK)


def col():
    ci = c.simGetCollisionInfo(**VK)
    return ci.has_collided, ci.object_name, ci.time_stamp


def depth_ahead():
    r = c.simGetImages([airsim.ImageRequest("front_custom", airsim.ImageType.DepthPlanar, True, False)], **VK)[0]
    d = np.asarray(r.image_data_float, np.float32).reshape(r.height, r.width)
    h, w = d.shape
    return float(d[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3].min()), float(d[h // 2:, :].min())


rows = []
X0 = 127.0
for z in (23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0):
    for y in (87.6, 88.0, 88.4, 88.8, 89.2):
        place(X0, y, z, math.pi)
        c.hoverAsync(**VK)
        time.sleep(0.4)
        place(X0, y, z, math.pi)
        time.sleep(0.2)
        ahead, low = depth_ahead()
        hit_x, obj = None, None
        _, _, ts0 = col()
        for i in range(1, 61):
            x = X0 - 0.1 * i
            place(x, y, z, math.pi)
            time.sleep(0.12)
            h, o, ts = col()
            if h and ts != ts0:
                hit_x, obj = round(x, 2), o
                break
        row = {"z": z, "y": y, "render_x": round(X0 - ahead, 2), "hit_x": hit_x, "obj": obj,
               "gap_m": None if hit_x is None else round(hit_x - (X0 - ahead), 2)}
        rows.append(row)
        print(row, flush=True)
place(X0, 92.0, 30.0, math.pi)
json.dump(rows, open("/tmp/corner579_probe.json", "w"), indent=1)
