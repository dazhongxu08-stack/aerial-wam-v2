#!/usr/bin/env python3
"""Verify baseline20 arrived routes: inland urban airspace + traj sanity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
from experiments.aerial.phase3_unified.region_geometry import (
    classify_spawn_xy,
    load_regions,
    path_inland_metrics,
)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else _REPO / "artifacts/baseline20_arrived_verify"
    eval_path = root / "eval_all.json"
    if not eval_path.is_file():
        eval_path = _REPO / "artifacts/urban_complex_baseline20_20260917_z38base/eval_all.json"
    ann_path = _REPO / "experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json"
    traj_dir = root / "traj"
    if not traj_dir.is_dir():
        traj_dir = _REPO / "artifacts/urban_complex_baseline20_20260917_z38base/traj"

    regions = load_regions()
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    arrived = [e for e in eval_data.get("episodes", []) if e.get("arrived")]

    print(f"ARRIVED {len(arrived)} routes\n")
    print(f"{'ri':>3s} {'inland':>7s} {'in_urb':>6s} {'in_int':>6s} {'z_rng':>12s} {'xy_span':>10s} {'steps':>5s} {'min_d':>6s} VERDICT")
    rows = []
    for e in sorted(arrived, key=lambda x: int(x["route_idx"])):
        ri = int(e["route_idx"])
        ep = ann["episodes"][ri]
        start = np.asarray(ep["pos"][0], dtype=float)
        goal = np.asarray(ep["pos"][-1], dtype=float)
        cls = classify_spawn_xy(float(start[0]), float(start[1]), regions)
        inland_spawn = float(cls["spawn_inland_m"])

        tpath = traj_dir / f"route{ri:02d}.jsonl"
        pts = []
        if tpath.is_file():
            for line in tpath.read_text(encoding="utf-8").strip().splitlines():
                o = json.loads(line)
                p = o.get("position") or o.get("pos")
                if p:
                    pts.append(p)
        pts_arr = np.asarray(pts, dtype=float) if pts else np.zeros((0, 3))
        path_m = path_inland_metrics(pts_arr, regions) if len(pts_arr) else {}
        path_min_inland = float(path_m.get("path_min_inland_m", 0))

        z_rng = f"{pts_arr[:,2].min():.0f}-{pts_arr[:,2].max():.0f}" if len(pts_arr) else "?"
        xy_span = (
            f"{np.ptp(pts_arr[:,0]):.0f}x{np.ptp(pts_arr[:,1]):.0f}" if len(pts_arr) else "?"
        )

        ok = (
            inland_spawn >= 100.0
            and path_min_inland >= 80.0
            and not cls["in_water"]
            and float(e.get("d_min_m", 999)) <= 5.0
            and len(pts_arr) >= 50
        )
        verdict = "URBAN_INLAND_OK" if ok else "CHECK"
        rows.append({
            "route_idx": ri,
            "verdict": verdict,
            "spawn_inland_m": inland_spawn,
            "path_min_inland_m": path_min_inland,
            "in_urban": cls["in_urban"],
            "in_interior": cls["in_interior"],
            "z_range": z_rng,
            "xy_span": xy_span,
            "steps": len(pts_arr),
            "min_d_m": e.get("d_min_m"),
            "start": start.tolist(),
            "goal": goal.tolist(),
        })
        print(
            f"{ri:3d} {inland_spawn:7.1f} {str(cls['in_urban']):>6s} {str(cls['in_interior']):>6s} "
            f"{z_rng:>12s} {xy_span:>10s} {len(pts_arr):5d} {str(e.get('d_min_m')):>6s} {verdict}"
        )

    out = root / "geography_verify.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
