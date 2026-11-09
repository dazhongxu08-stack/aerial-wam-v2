"""Is a route goal reachable without the model? Hover at the goal, step toward a suspected box face, and fly
straight at the goal from a ring of starts.

Contacts use a fresh simGetCollisionInfo timestamp (has_collided stays latched after teleports). Depth is
DepthPlanar from front_custom. Positions are AirSim NED (z down).
usage: goal_ring_probe.py OUT_DIR GOAL_X GOAL_Y GOAL_Z_NED [--face-dir south|north|east|west] [--extra-start x,y,z ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import airsim
import numpy as np

CAM = "front_custom"
DT = 0.2
SPEED = 1.0
SUCCESS_M = 3.0
MAX_STEPS = 120
FACE_VEC = {"south": (-1.0, 0.0), "north": (1.0, 0.0), "east": (0.0, 1.0), "west": (0.0, -1.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("gx", type=float)
    ap.add_argument("gy", type=float)
    ap.add_argument("gz", type=float, help="NED z (negative = up)")
    ap.add_argument("--face-dir", default="south", choices=sorted(FACE_VEC))
    ap.add_argument("--ring-m", type=float, default=10.0)
    ap.add_argument("--extra-start", action="append", default=[], help="x,y,z NED")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    goal = np.array([args.gx, args.gy, args.gz])
    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.enableApiControl(True)

    def ts() -> int:
        return int(c.simGetCollisionInfo().time_stamp)

    def fresh(prev: int) -> str:
        ci = c.simGetCollisionInfo()
        return str(ci.object_name) if ci.has_collided and int(ci.time_stamp) != prev else ""

    def place(p, yaw: float) -> str:
        prev = ts()
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*map(float, p)), airsim.to_quaternion(0, 0, yaw)), True)
        c.moveByVelocityAsync(0, 0, 0, 0.6).join()
        return fresh(prev)

    def depth(save: str = ""):
        reqs = [airsim.ImageRequest(CAM, airsim.ImageType.DepthPlanar, True, False)]
        if save:
            reqs.append(airsim.ImageRequest(CAM, airsim.ImageType.Scene, False, True))
        rs = c.simGetImages(reqs)
        d = np.array(rs[0].image_data_float, np.float32).reshape(rs[0].height, rs[0].width)
        if save:
            open(os.path.join(args.out_dir, save), "wb").write(rs[1].image_data_uint8)
        h, w = d.shape
        win = d[h // 2 - h // 8:h // 2 + h // 8, w // 2 - w // 8:w // 2 + w // 8]
        return round(float(win.min()), 2), round(float(np.median(win)), 2), round(float(d.min()), 2)

    out = {"goal_ned": goal.tolist(), "at_goal": [], "face_steps": [], "approaches": []}

    place(goal + [0, 0, -8], 0.0)
    for k in range(8):
        yaw = k * math.pi / 4
        place(goal + [0, 0, -8], yaw)
        contact = place(goal, yaw)
        cmin, cmed, fmin = depth(f"goal_yaw{k * 45:03d}.png" if k % 2 == 0 else "")
        row = {"yaw_deg": k * 45, "contact": contact, "center_min": cmin, "center_med": cmed, "full_min": fmin}
        out["at_goal"].append(row)
        print("at_goal", row, flush=True)

    fx, fy = FACE_VEC[args.face_dir]
    face_yaw = math.atan2(fy, fx)
    for s in np.arange(0.0, 4.01, 0.5):
        place(goal + [0, 0, -8], face_yaw)
        p = goal + [fx * s, fy * s, 0.0]
        contact = place(p, face_yaw)
        cmin, cmed, _ = depth()
        row = {"offset_m": float(s), "pos": [round(float(v), 2) for v in p], "contact": contact,
               "center_min": cmin, "center_med": cmed}
        out["face_steps"].append(row)
        print("face", row, flush=True)

    starts = [goal + [args.ring_m * math.cos(b), args.ring_m * math.sin(b), 0.0]
              for b in np.arange(8) * math.pi / 4]
    starts += [np.array([float(v) for v in s.split(",")]) for s in args.extra_start]
    for i, s in enumerate(starts):
        yaw0 = math.atan2(goal[1] - s[1], goal[0] - s[0])
        place(s + [0, 0, -8], yaw0)
        start_contact = place(s, yaw0)
        rec = {"start": [round(float(v), 2) for v in s], "start_d": round(float(np.linalg.norm(s - goal)), 2),
               "start_contact": start_contact, "end": "max_steps", "object": "", "steps": 0}
        if start_contact:
            rec["end"] = "start_embedded"
        else:
            prev = ts()
            dmin = depth()[0]
            for step in range(MAX_STEPS):
                p = c.getMultirotorState().kinematics_estimated.position
                cur = np.array([p.x_val, p.y_val, p.z_val])
                v = goal - cur
                dist = float(np.linalg.norm(v))
                if dist <= SUCCESS_M:
                    rec["end"] = "arrived"
                    break
                v = v / max(dist, 1e-6) * SPEED
                yaw_deg = math.degrees(math.atan2(v[1], v[0]))
                c.moveByVelocityAsync(float(v[0]), float(v[1]), float(v[2]), DT,
                                      yaw_mode=airsim.YawMode(False, yaw_deg)).join()
                dmin = min(dmin, depth()[0])
                obj = fresh(prev)
                if obj:
                    rec["end"], rec["object"] = "collision", obj
                    break
                rec["steps"] = step + 1
            p = c.getMultirotorState().kinematics_estimated.position
            rec["d_final"] = round(float(np.linalg.norm(np.array([p.x_val, p.y_val, p.z_val]) - goal)), 2)
            rec["center_depth_min"] = dmin
        out["approaches"].append(rec)
        print("approach", rec, flush=True)

    json.dump(out, open(os.path.join(args.out_dir, "GOAL_RING.json"), "w"), indent=1)
    print("WROTE", args.out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
