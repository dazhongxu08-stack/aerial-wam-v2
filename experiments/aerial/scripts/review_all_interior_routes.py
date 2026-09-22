#!/usr/bin/env python3
"""Review all outdoor_complex routes: geo, raw NPZ, curated, eval."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.aerial.phase3_unified.region_geometry import (
    DEFAULT_REGIONS_PATH,
    classify_spawn_xy,
    load_regions,
    path_inland_metrics,
)

INLAND_IDX = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19}


def _path_m(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _tier(inland: str, cur_n: int, cur_a: int, prog: float, raw_arr: int) -> str:
    if inland == "water":
        return "EXCL"
    if cur_a > 0 and prog >= 60:
        return "STRONG"
    if cur_a > 0 or prog >= 50:
        return "MED"
    if cur_n > 0 or raw_arr > 0:
        return "WEAK"
    if inland == "inland":
        return "NO_DATA"
    return "EDGE"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--annotation",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json",
    )
    p.add_argument(
        "--raw",
        default="experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_20260916_night",
    )
    p.add_argument(
        "--curated",
        default="experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_20260916_night_curated",
    )
    p.add_argument(
        "--eval-json",
        default="artifacts/wam_phase2_v11_interior_ft_sr_eval_20260916_night/eval_all.json",
    )
    p.add_argument("--out", default="artifacts/review_all_interior_routes_20260917.json")
    args = p.parse_args()

    ann_path = _REPO / args.annotation
    raw_dir = _REPO / args.raw
    cur_path = _REPO / args.curated
    eval_path = _REPO / args.eval_json
    out_path = _REPO / args.out

    regions = load_regions(DEFAULT_REGIONS_PATH)
    routes = {int(e["route_idx"]): e for e in json.loads(ann_path.read_text())["episodes"]}

    manifest = json.loads((raw_dir / "manifest.json").read_text()) if (raw_dir / "manifest.json").is_file() else {"episodes": []}
    raw_by: dict[int, dict] = defaultdict(lambda: {"n": 0, "arr": 0, "best_path": 0.0})
    for ent in manifest.get("episodes", []):
        ri = int(ent.get("route_idx", -1))
        if ri < 0:
            continue
        raw_by[ri]["n"] += 1
        if ent.get("arrived"):
            raw_by[ri]["arr"] += 1
        raw_by[ri]["best_path"] = max(raw_by[ri]["best_path"], float(ent.get("path_length_m", 0)))

    cur_by: dict[int, list] = defaultdict(list)
    if cur_path.joinpath("manifest.json").is_file():
        for e in json.loads(cur_path.joinpath("manifest.json").read_text()).get("episodes", []):
            cur_by[int(e["route_idx"])].append(e)

    eval_by: dict[int, dict] = {}
    if eval_path.is_file():
        for e in json.loads(eval_path.read_text()).get("episodes", []):
            eval_by[int(e["route_idx"])] = e

    rows = []
    print(
        f"{'idx':>3} {'y':>4} {'zone':6} {'tier':7} "
        f"{'cur':>3} {'cA':>2} {'raw':>3} {'rA':>2} "
        f"{'eval%':>6} {'arr':>5} spawn_xy"
    )
    for ri in sorted(routes):
        ep = routes[ri]
        pos = np.asarray(ep["pos"], dtype=np.float64)
        spawn = pos[0]
        y = float(spawn[1])
        if y < 0 or ri in range(12, 19):
            zone = "water"
        elif ri in INLAND_IDX:
            zone = "inland"
        else:
            zone = "edge"
        spawn_meta = classify_spawn_xy(float(spawn[0]), float(spawn[1]), regions)
        inland = path_inland_metrics(pos, regions)
        cn = len(cur_by.get(ri, []))
        ca = sum(1 for x in cur_by.get(ri, []) if x.get("arrived"))
        rb = raw_by[ri]
        ev = eval_by.get(ri)
        prog = float(ev.get("progress_ratio", 0)) * 100 if ev else -1.0
        ea = bool(ev.get("arrived")) if ev else False
        tier = _tier(zone, cn, ca, prog, rb["arr"])
        row = {
            "route_idx": ri,
            "route_id": ep.get("route_id"),
            "spawn_y": y,
            "zone": zone,
            "tier": tier,
            "path_m": _path_m(pos),
            "spawn_inland_m": float(spawn_meta["spawn_inland_m"]),
            "path_min_inland_m": float(inland["path_min_inland_m"]),
            "curated_n": cn,
            "curated_arrived": ca,
            "raw_n": rb["n"],
            "raw_arrived": rb["arr"],
            "raw_best_path_m": rb["best_path"],
            "eval_progress_pct": prog,
            "eval_arrived": ea,
            "spawn_xy": [float(spawn[0]), float(spawn[1]), float(spawn[2])],
        }
        rows.append(row)
        print(
            f"{ri:3d} {y:4.0f} {zone:6} {tier:7} "
            f"{cn:3d} {ca:2d} {rb['n']:3d} {rb['arr']:2d} "
            f"{prog:6.1f} {str(ea):>5} ({spawn[0]:.0f},{spawn[1]:.0f})"
        )

    summary = {
        "inland_routes": sum(1 for r in rows if r["zone"] == "inland"),
        "curated_routes": sum(1 for r in rows if r["curated_n"] > 0),
        "curated_arrived_routes": sum(1 for r in rows if r["curated_arrived"] > 0),
        "eval_routes": sum(1 for r in rows if r["eval_progress_pct"] >= 0),
        "strong": [r["route_idx"] for r in rows if r["tier"] == "STRONG"],
        "med": [r["route_idx"] for r in rows if r["tier"] == "MED"],
        "weak": [r["route_idx"] for r in rows if r["tier"] == "WEAK"],
        "no_data": [r["route_idx"] for r in rows if r["tier"] == "NO_DATA"],
        "excl": [r["route_idx"] for r in rows if r["tier"] == "EXCL"],
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[review] wrote {out_path}")
    print(
        f"[review] inland={summary['inland_routes']} curated_routes={summary['curated_routes']} "
        f"strong={summary['strong']} med={summary['med']} weak={summary['weak']} "
        f"no_data={summary['no_data']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
