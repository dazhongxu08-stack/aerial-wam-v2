"""Map collision volumes at flight altitude by placing the vehicle on a grid (finds meshes that do not render).

At every grid point the vehicle is put 8 m above, then at the point; a fresh contact marks the cell and records
the object name. Output npz: hit (nx, ny) bool, names (nx, ny) str, x0, y0, res, alt (z up).
usage: invisible_probe.py OUT_NPZ --box X0:X1:Y0:Y1 [--alt 24] [--res 2]
"""
from __future__ import annotations

import argparse
import collections

import airsim
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_npz")
    ap.add_argument("--box", required=True)
    ap.add_argument("--alt", type=float, default=24.0)
    ap.add_argument("--res", type=float, default=2.0)
    args = ap.parse_args()
    x0, x1, y0, y1 = map(float, args.box.split(":"))
    xs, ys = np.arange(x0, x1 + 1e-6, args.res), np.arange(y0, y1 + 1e-6, args.res)
    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.enableApiControl(True)
    veh = c.simGetVehiclePose()

    def place(x: float, y: float, z: float) -> str:
        prev = int(c.simGetCollisionInfo().time_stamp)
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0)), True)
        c.moveByVelocityAsync(0, 0, 0, 0.3).join()
        ci = c.simGetCollisionInfo()
        return str(ci.object_name) if ci.has_collided and int(ci.time_stamp) != prev else ""

    hit = np.zeros((len(xs), len(ys)), bool)
    names = np.full((len(xs), len(ys)), "", dtype=object)
    try:
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                place(float(x), float(y), -args.alt - 8.0)
                n = place(float(x), float(y), -args.alt)
                hit[i, j], names[i, j] = bool(n), n
            print(f"row {i + 1}/{len(xs)} x={x:.0f} hits {int(hit[i].sum())}", flush=True)
            np.savez_compressed(args.out_npz, hit=hit, names=names.astype(str), x0=x0, y0=y0, res=args.res, alt=args.alt)
    finally:
        c.simSetVehiclePose(veh, True)
    print("objects", dict(collections.Counter(n for n in names.ravel() if n)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
