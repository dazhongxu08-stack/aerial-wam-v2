#!/usr/bin/env python3
"""Map where the rendered scene is actually city blocks, on a coarse grid.

The ``urban_interior`` polygons in ``env_airsim16_regions.json`` include open
water in the rendered map (west_central, west_southwest, part of east), so a
polygon test alone lets waterfront starts through. At each grid cell this grabs
GT DepthPlanar at cruise height in four headings and measures the fraction of
above-horizon pixels nearer than ``near_m``. Over water that band holds sky and
a distant skyline; among buildings it is mostly facade.

Writes ``urban_mask.json`` (per-cell fractions and the urban flag) and a PNG.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def above_horizon_near_frac(depth: np.ndarray, near_m: float) -> float:
    d = np.asarray(depth, dtype=np.float64)
    top = d[: d.shape[0] // 2]
    v = np.isfinite(top) & (top > 0.05)
    if not np.any(v):
        return 0.0
    return float(np.mean(top[v] < float(near_m)))


def scan_boxes(regions: Dict[str, Any], margin_m: float) -> List[Tuple[str, float, float, float, float]]:
    out = []
    for r in regions.get("urban_regions", []):
        p = np.asarray(r["polygon_xy"], dtype=np.float64)
        out.append((
            r["id"],
            float(p[:, 0].min() - margin_m), float(p[:, 0].max() + margin_m),
            float(p[:, 1].min() - margin_m), float(p[:, 1].max() + margin_m),
        ))
    return out


def grab_depth(client: Any, airsim: Any, x: float, y: float, z_up: float, yaw_deg: float, settle_s: float) -> np.ndarray:
    quat = airsim.to_quaternion(0.0, 0.0, math.radians(yaw_deg))
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, -z_up), quat), True)
    time.sleep(settle_s)
    (resp,) = client.simGetImages(
        [airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)]
    )
    return np.asarray(
        airsim.list_to_2d_float_array(resp.image_data_float, resp.width, resp.height),
        dtype=np.float32,
    )


def draw(cells: List[Dict[str, Any]], out: Path, grid_m: float) -> None:
    import cv2

    xs = np.array([c["x"] for c in cells])
    ys = np.array([c["y"] for c in cells])
    x0, y0 = xs.min(), ys.min()
    px = 6
    w = int(round((xs.max() - x0) / grid_m)) + 1
    h = int(round((ys.max() - y0) / grid_m)) + 1
    img = np.full((h * px, w * px, 3), 40, dtype=np.uint8)
    for c in cells:
        i = int(round((c["x"] - x0) / grid_m))
        j = int(round((c["y"] - y0) / grid_m))
        f = float(c["near_frac_mean"])
        col = (60, int(80 + 175 * min(1.0, f / 0.5)), 60) if c["urban"] else (160, 90, 40)
        img[(h - 1 - j) * px : (h - j) * px, i * px : (i + 1) * px] = col
    cv2.imwrite(str(out), img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", default="experiments/aerial/phase3_unified/env_airsim16_regions.json")
    ap.add_argument("--out-dir", default="experiments/aerial/rl/artifacts/interior_od_routes_20260930")
    ap.add_argument("--grid-m", type=float, default=30.0)
    ap.add_argument("--margin-m", type=float, default=60.0)
    ap.add_argument("--z-up", type=float, default=24.0)
    ap.add_argument("--near-m", type=float, default=150.0)
    ap.add_argument("--urban-frac", type=float, default=0.15)
    ap.add_argument("--settle-s", type=float, default=0.12)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import airsim

    regions = json.loads((root / args.regions).read_text(encoding="utf-8"))
    client = airsim.MultirotorClient(ip=args.host)
    client.confirmConnection()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cells: List[Dict[str, Any]] = []
    seen = set()
    t0 = time.time()
    for rid, xmin, xmax, ymin, ymax in scan_boxes(regions, float(args.margin_m)):
        for x in np.arange(xmin, xmax + 1e-6, args.grid_m):
            for y in np.arange(ymin, ymax + 1e-6, args.grid_m):
                key = (round(float(x) / args.grid_m), round(float(y) / args.grid_m))
                if key in seen:
                    continue
                seen.add(key)
                fr = [
                    above_horizon_near_frac(
                        grab_depth(client, airsim, float(x), float(y), args.z_up, yaw, args.settle_s),
                        args.near_m,
                    )
                    for yaw in (0.0, 90.0, 180.0, 270.0)
                ]
                cells.append({
                    "x": float(x),
                    "y": float(y),
                    "box": rid,
                    "near_frac": [round(f, 3) for f in fr],
                    "near_frac_mean": round(float(np.mean(fr)), 3),
                    "near_frac_min": round(float(np.min(fr)), 3),
                    "urban": bool(np.mean(fr) >= float(args.urban_frac)),
                })
                if len(cells) % 50 == 0:
                    n_u = sum(c["urban"] for c in cells)
                    print(f"{len(cells)} cells, urban {n_u}, {time.time() - t0:.0f}s", flush=True)
    payload = {
        "grid_m": float(args.grid_m),
        "z_up": float(args.z_up),
        "near_m": float(args.near_m),
        "urban_frac": float(args.urban_frac),
        "cells": cells,
    }
    (out_dir / "urban_mask.json").write_text(json.dumps(payload), encoding="utf-8")
    draw(cells, out_dir / "urban_mask.png", float(args.grid_m))
    by_box: Dict[str, List[int]] = {}
    for c in cells:
        by_box.setdefault(c["box"], [0, 0])
        by_box[c["box"]][0] += 1
        by_box[c["box"]][1] += int(c["urban"])
    print("DONE", len(cells), "cells", {k: f"{u}/{n} urban" for k, (n, u) in by_box.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
