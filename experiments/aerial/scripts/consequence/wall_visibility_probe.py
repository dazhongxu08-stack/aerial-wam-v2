"""Check whether a collision face is visible: hover facing it and compare camera depth to the contact distance.

Contact uses a fresh simGetCollisionInfo timestamp. Depth is DepthPlanar from front_custom; the centre
window should read about (face_x - x) if the face is rendered. Scene frames are saved for inspection.
usage: wall_visibility_probe.py OUT_DIR
"""
from __future__ import annotations

import json
import os
import sys

import airsim
import numpy as np

FACE_X = 123.3
POINTS = [(118.3, 95.0, -24.0), (120.3, 97.0, -24.0), (121.8, 97.5, -23.5), (122.8, 98.0, -23.5),
          (118.3, 99.0, -29.0), (120.3, 92.0, -24.0)]
CONTACT_XS = [122.8, 123.3, 123.8, 124.3]
CAM = "front_custom"


def main() -> int:
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.enableApiControl(True)

    def pose(x, y, z, yaw=0.0):
        prev = int(c.simGetCollisionInfo().time_stamp)
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, yaw)), True)
        c.moveByVelocityAsync(0, 0, 0, 0.3).join()
        ci = c.simGetCollisionInfo()
        return str(ci.object_name) if ci.has_collided and int(ci.time_stamp) != prev else ""

    rows = []
    for i, (x, y, z) in enumerate(POINTS):
        contact = pose(x, y, z)
        r_depth, r_rgb = c.simGetImages([
            airsim.ImageRequest(CAM, airsim.ImageType.DepthPlanar, True, False),
            airsim.ImageRequest(CAM, airsim.ImageType.Scene, False, True)])
        d = np.array(r_depth.image_data_float, np.float32).reshape(r_depth.height, r_depth.width)
        h, w = d.shape
        win = d[h // 2 - h // 10:h // 2 + h // 10, w // 2 - w // 10:w // 2 + w // 10]
        open(os.path.join(out_dir, f"scene_{i}.png"), "wb").write(r_rgb.image_data_uint8)
        row = {"pos": [x, y, z], "expected_face_m": round(FACE_X - x, 2), "contact": contact,
               "depth_center_min": round(float(win.min()), 2), "depth_center_median": round(float(np.median(win)), 2),
               "depth_full_min": round(float(d.min()), 2)}
        rows.append(row)
        print(row, flush=True)
    contacts = []
    for x in CONTACT_XS:
        pose(118.0, 97.0, -24.0)
        name = pose(x, 97.0, -24.0)
        contacts.append({"x": x, "contact": name})
        print("contact", x, name or "-", flush=True)
    json.dump({"face_x": FACE_X, "rows": rows, "contacts": contacts},
              open(os.path.join(out_dir, "WALL_VIS.json"), "w"), indent=1)
    print("WROTE", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
