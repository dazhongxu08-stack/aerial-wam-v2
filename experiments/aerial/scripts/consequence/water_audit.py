"""Audit routes and consequence data against the measured open-water raster (water_map_measured_*.npz).

The raster comes from looking straight down from 150 m in AirSim (sea segmentation colour, blobs >= 1% of the frame)
and replaces the hand-drawn water_exclusion rectangles, which miss the real shoreline by up to ~80 m.
Rule (urban interior): spawn >= 100 m from open water, every path point >= 80 m.

usage: water_audit.py --raster R.npz --routes a.json,b.json --data-dirs d1,d2 --out AUDIT.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

SPAWN_MIN_M, PATH_MIN_M = 100.0, 80.0


class Shore:
    def __init__(self, path: str):
        z = np.load(path)
        self.x0, self.x1, self.y0, self.y1, self.res = [float(v) for v in z["extent"]]
        land = (z["water"] == 0).astype(np.uint8)
        self.dist = cv2.distanceTransform(land, cv2.DIST_L2, 5) * self.res

    def __call__(self, x: float, y: float) -> float:
        i = int((x - self.x0) / self.res)
        j = int((y - self.y0) / self.res)
        if not (0 <= i < self.dist.shape[0] and 0 <= j < self.dist.shape[1]):
            return 0.0
        return float(self.dist[i, j])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raster", required=True)
    ap.add_argument("--routes", default="")
    ap.add_argument("--data-dirs", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shore = Shore(args.raster)
    out = {"rule": {"spawn_min_m": SPAWN_MIN_M, "path_min_m": PATH_MIN_M}, "raster": args.raster,
           "routes": {}, "data": {}}

    for ann in [p for p in args.routes.split(",") if p]:
        rows = {}
        for e in json.loads(Path(ann).read_text())["episodes"]:
            pos = e["pos"]
            d = [shore(p[0], p[1]) for p in pos]
            ok = d[0] >= SPAWN_MIN_M and min(d) >= PATH_MIN_M
            rows[e["route_id"]] = {"spawn_m": round(d[0], 1), "path_min_m": round(min(d), 1), "keep": ok}
        bad = sorted(r for r, v in rows.items() if not v["keep"])
        out["routes"][ann] = {"n": len(rows), "n_fail": len(bad), "fail": bad, "per_route": rows}
        print(f"{Path(ann).name}: {len(bad)}/{len(rows)} fail -> {bad}")

    for dd in [p for p in args.data_dirs.split(",") if p]:
        review = json.loads((Path(dd) / "AUTO_REVIEW.json").read_text())
        dists, per_route = [], {}
        if "probes" in review:
            for rec in review["probes"]:
                z = np.load(Path(dd) / "probes" / rec["file"])
                s = np.asarray(z["state"]).reshape(-1)
                dv = shore(float(s[0]), float(s[1]))
                dists.append(dv)
                per_route.setdefault(rec["route_id"], []).append(dv)
        else:
            for f in sorted(glob.glob(f"{dd}/episodes/episode_*.npz")):
                if f.endswith(".da3.npz"):
                    continue
                z = np.load(f)
                dv = [shore(float(a[0]), float(a[1])) for a in np.asarray(z["state"])]
                dists.extend(dv)
                rid = json.loads(bytes(z["meta"]).decode()).get("route_id", Path(f).stem)
                per_route.setdefault(rid, []).extend(dv)
        d = np.asarray(dists)
        out["data"][dd] = {"samples": int(d.size), "frac_lt_80m": float((d < PATH_MIN_M).mean()) if d.size else None,
                           "frac_lt_100m": float((d < SPAWN_MIN_M).mean()) if d.size else None,
                           "min_m": float(d.min()) if d.size else None,
                           "routes_with_samples_lt_80m": sorted(r for r, v in per_route.items() if min(v) < PATH_MIN_M)}
        print(f"{Path(dd).name}: samples={d.size} <80m={out['data'][dd]['frac_lt_80m']:.3f} "
              f"<100m={out['data'][dd]['frac_lt_100m']:.3f} min={d.min():.0f} "
              f"routes<80m={out['data'][dd]['routes_with_samples_lt_80m']}")
    Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
