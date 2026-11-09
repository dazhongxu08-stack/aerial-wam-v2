"""Top-down surface height map around each route (for shortest-path references).

Tiles a grid over the route box, puts the vehicle ALT_M above flight level with front_custom pitched straight down,
and back-projects DepthPlanar to surface NED z. Only pixels within NADIR_DEG of the optical axis are used (less
roof-edge occlusion); per 0.5 m cell the highest surface wins. The camera pose is restored at the end.
Invisible collision meshes do not render, so they are absent from these maps.
usage: topdown_height.py OUT_DIR ROUTES_JSON route_id[,route_id...]
       topdown_height.py OUT_DIR --box NAME:X0:X1:Y0:Y1 [--box ...] [--flight-z-up 24]   (writes top_box_NAME.npz)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import airsim
import numpy as np

CAM = "front_custom"
ALT_M, TILE_M, RES_M, PAD_M, NADIR_DEG = 45.0, 25.0, 0.5, 30.0, 32.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("routes_json", nargs="?")
    ap.add_argument("route_ids", nargs="?")
    ap.add_argument("--box", action="append", default=[])
    ap.add_argument("--flight-z-up", type=float, default=24.0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.routes_json:
        eps = {e["route_id"]: e for e in json.loads(Path(args.routes_json).read_text())["episodes"]}
        for rid in args.route_ids.split(","):
            e = eps[rid]
            s = np.array([e["pos"][0][0], e["pos"][0][1], -e["pos"][0][2]])
            g = np.array([e["pos"][-1][0], e["pos"][-1][1], -e["pos"][-1][2]])
            jobs.append((f"top_{rid}", min(s[0], g[0]) - PAD_M, max(s[0], g[0]) + PAD_M,
                         min(s[1], g[1]) - PAD_M, max(s[1], g[1]) + PAD_M, float(g[2]), s, g))
    for b in args.box:
        name, *lim = b.split(":")
        nan3 = np.full(3, np.nan)
        jobs.append((f"top_box_{name}", *map(float, lim), -args.flight_z_up, nan3, nan3))
    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.enableApiControl(True)
    def q(a):
        return np.array([a.w_val, a.x_val, a.y_val, a.z_val], float)

    def qmul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    info = c.simGetCameraInfo(CAM)
    veh = c.simGetVehiclePose()
    qv_inv = q(veh.orientation) * np.array([1, -1, -1, -1])
    off = info.pose.position - veh.position
    rp = qmul(qmul(qv_inv, [0.0, off.x_val, off.y_val, off.z_val]), q(veh.orientation))
    qr = qmul(qv_inv, q(info.pose.orientation))
    pose0 = airsim.Pose(airsim.Vector3r(*map(float, rp[1:])), airsim.Quaternionr(*map(float, (qr[1], qr[2], qr[3], qr[0]))))
    print("camera relative pose", pose0, flush=True)
    fov = math.radians(float(info.fov))
    try:
        c.simSetCameraPose(CAM, airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-math.pi / 2, 0, 0)))
        for name, x0, x1, y0, y1, zf, s, g in jobs:
            nx, ny = int(math.ceil((x1 - x0) / RES_M)), int(math.ceil((y1 - y0) / RES_M))
            top = np.full((nx, ny), np.inf, np.float32)
            for tx in np.arange(x0, x1 + 1e-6, TILE_M):
                for ty in np.arange(y0, y1 + 1e-6, TILE_M):
                    cz = zf - ALT_M
                    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(float(tx), float(ty), float(cz)),
                                                    airsim.to_quaternion(0, 0, 0)), True)
                    c.moveByVelocityAsync(0, 0, 0, 0.3).join()
                    p = c.simGetVehiclePose().position
                    r = c.simGetImages([airsim.ImageRequest(CAM, airsim.ImageType.DepthPlanar, True, False)])[0]
                    d = np.array(r.image_data_float, np.float32).reshape(r.height, r.width)
                    h, w = d.shape
                    f = (w / 2.0) / math.tan(fov / 2.0)
                    v, u = np.mgrid[0:h, 0:w]
                    du, dv = (u - (w - 1) / 2.0) / f, ((h - 1) / 2.0 - v) / f
                    ok = (d > 1.0) & (d < 500.0) & (np.degrees(np.arctan(np.hypot(du, dv))) <= NADIR_DEG)
                    wx, wy, wz = p.x_val + dv * d, p.y_val + du * d, p.z_val + d
                    i = ((wx - x0) / RES_M).astype(int)
                    j = ((wy - y0) / RES_M).astype(int)
                    ok &= (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
                    np.minimum.at(top, (i[ok], j[ok]), wz[ok].astype(np.float32))
            top[np.isinf(top)] = np.nan
            np.savez_compressed(out / f"{name}.npz", top_ned_z=top, x0=x0, y0=y0, res=RES_M,
                                spawn=s, goal=g, flight_z=zf)
            print(name, "cells", nx * ny, "filled", float(np.isfinite(top).mean()), flush=True)
    finally:
        c.simSetCameraPose(CAM, pose0)
        c.simSetVehiclePose(veh, True)
        c.moveByVelocityAsync(0, 0, 0, 0.3).join()
        now = c.simGetCameraInfo(CAM).pose
        print("camera restored: world orientation now", now.orientation, "before", info.pose.orientation, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
