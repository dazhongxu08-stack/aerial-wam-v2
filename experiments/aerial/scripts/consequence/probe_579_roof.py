"""579 south face near the od_hard_01 spawn (x 104-106, wall y~88): rendered vs collision, scanning north from y=85."""
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


def ahead():
    r = c.simGetImages([airsim.ImageRequest("front_custom", airsim.ImageType.DepthPlanar, True, False)], **VK)[0]
    d = np.asarray(r.image_data_float, np.float32).reshape(r.height, r.width)
    h, w = d.shape
    return float(d[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3].min())


rows = []
Y0 = 85.0
for x in (104.2, 106.0, 112.0, 120.0):
    for z in (24.5, 25.0, 25.2, 25.5, 26.0, 27.0, 30.0):
        place(x, Y0, z, math.pi / 2)
        c.hoverAsync(**VK)
        time.sleep(0.4)
        place(x, Y0, z, math.pi / 2)
        time.sleep(0.2)
        a = ahead()
        ts0 = c.simGetCollisionInfo(**VK).time_stamp
        hit_y, obj = None, None
        for i in range(1, 41):
            y = Y0 + 0.1 * i
            place(x, y, z, math.pi / 2)
            time.sleep(0.12)
            ci = c.simGetCollisionInfo(**VK)
            if ci.has_collided and ci.time_stamp != ts0:
                hit_y, obj = round(y, 2), ci.object_name
                break
        row = {"x": x, "z": z, "render_y": round(Y0 + a, 2), "hit_y": hit_y, "obj": obj}
        rows.append(row)
        print(row, flush=True)
place(105.0, 85.0, 30.0, 0.0)
json.dump(rows, open("/tmp/roof579_probe.json", "w"), indent=1)
