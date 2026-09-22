#!/usr/bin/env python3
"""Nudge inland y=11 spawns (+30m Y) for collision-prone routes."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.aerial.phase3_unified.region_geometry import DEFAULT_REGIONS_PATH, load_regions
from experiments.aerial.scripts.select_outdoor_complex_routes import score_spawn

PATCH_IDX = {6, 7, 8, 9, 19}
DY_M = 30.0


def patch_annotation(src: Path, dst: Path, *, dy_m: float = DY_M) -> dict:
    data = json.loads(src.read_text(encoding="utf-8"))
    regions = load_regions(DEFAULT_REGIONS_PATH)
    out_eps = []
    patched = []
    for ep in data["episodes"]:
        ep2 = copy.deepcopy(ep)
        ri = int(ep2["route_idx"])
        if ri in PATCH_IDX:
            pos = np.asarray(ep2["pos"], dtype=np.float64).copy()
            pos[:, 1] += float(dy_m)
            ep2["pos"] = [[float(x), float(y), float(z)] for x, y, z in pos]
            row = score_spawn(
                ri, ep2, regions=regions, min_spawn_z=12.0,
                min_spawn_inland_m=100.0, min_path_inland_m=80.0,
            )
            if row is None:
                raise SystemExit(f"route {ri}: patch dy={dy_m} failed geo gate")
            ep2["spawn_patch_dy_m"] = float(dy_m)
            patched.append(ri)
        out_eps.append(ep2)
    out = {**data, "episodes": out_eps, "spawn_patch": {"dy_m": dy_m, "route_indices": sorted(patched)}}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return {"patched": sorted(patched), "dst": str(dst)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json",
    )
    p.add_argument(
        "--dst",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json",
    )
    p.add_argument("--dy-m", type=float, default=DY_M)
    args = p.parse_args()
    rep = patch_annotation(_REPO / args.src, _REPO / args.dst, dy_m=float(args.dy_m))
    print(f"[patch-spawn] {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
