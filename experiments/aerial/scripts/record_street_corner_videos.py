#!/usr/bin/env python3
"""Replay three frozen-corner tracks into dual-view videos."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path("/data/aerial-wam-v2")
PY = "/data/venvs/sim_verify/bin/python"
ANNO = "experiments/aerial/phase3_unified/annotations/street_corners_frozen.json"
TRAJ = (
    "experiments/aerial/rl/artifacts/"
    "eval_street_corners_streetdepth_20261005/road_center/traj"
)
OUT = (
    "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/"
    "eval_street_corners_streetdepth_20261005/videos"
)
JOBS = (
    (2, "CORNER 02  [-784, 48]  turned and arrived", "corner02"),
    (4, "CORNER 04  [-936, 180]  turned and arrived", "corner04"),
    (10, "CORNER 10  [-780, 228]  turned, stopped 22 m", "corner10"),
)


def main() -> None:
    Path(OUT).mkdir(parents=True, exist_ok=True)
    for idx, label, prefix in JOBS:
        cmd = [
            PY,
            "-u",
            "experiments/aerial/scripts/wam_phase2_dual_view_from_traj.py",
            "--traj-jsonl",
            f"{TRAJ}/route{idx:02d}.jsonl",
            "--ref-polyline-json",
            ANNO,
            "--route-idx",
            str(idx),
            "--route-label",
            label,
            "--map-title",
            "road center, no polyline",
            "--plan-mode",
            "toward_g",
            "--out-dir",
            OUT,
            "--out-prefix",
            prefix,
            "--fps",
            "5",
            "--source-hz",
            "5",
            "--ego-width",
            "960",
            "--ego-height",
            "540",
        ]
        print("RUN", prefix, flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
