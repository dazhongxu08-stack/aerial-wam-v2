#!/usr/bin/env python3
"""Fly one two-turn detour. Polyline first, then road_center with no polyline."""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/data/aerial-wam-v2")
PY = "/data/venvs/sim_verify/bin/python"
ANNO = ROOT / "experiments/aerial/phase3_unified/annotations/two_turn_detour.json"
OUT = ROOT / "experiments/aerial/rl/artifacts/eval_two_turn_forced_20261005"
DEPTH = "experiments/aerial/rl/artifacts/da3_yaw_street_20261005/depth_yaw_street.pt"

EVAL_BASE = [
    PY,
    "-m",
    "experiments.aerial.scripts.wam_phase2_long_eval",
    "--wm-ckpt",
    "experiments/aerial/rl/artifacts/wm_ckpt_coll_pair_retrain_20260930/wm_step_24600.pt",
    "--actor-ckpt",
    "experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_soft_20260926_soft_noshield_h84/v4_ac_best.pt",
    "--planner",
    "--planner-horizon",
    "5",
    "--planner-rollout",
    "open_loop",
    "--tti-coeff",
    "2.5",
    "--heading-reentry-cos",
    "0.5",
    "--cte-reentry-m",
    "1.5",
    "--cruise-speed",
    "8.0",
    "--success-dist",
    "3.0",
    "--terminal-pin-rem-m",
    "20",
    "--min-spawn-z",
    "14",
    "--spawn-z-retry-m",
    "2",
    "--spawn-z-max-retries",
    "1",
    "--min-spawn-clear-m",
    "2",
    "--shield-exclusion-forward-only",
    "--r-m-intent",
    "100",
    "--device",
    "cuda",
    "--depth-ckpt",
    DEPTH,
    "--annotation",
    str(ANNO.relative_to(ROOT)),
    "--episodes",
    "1",
    "--max-steps",
    "400",
]


def _heading(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)], dtype=np.float64)


def score_corner(rows: list, corner_xy, yaw_out: float, leg_out_m: float) -> dict:
    pos = np.asarray([r["pos"][:2] for r in rows], dtype=np.float64)
    yaw = np.asarray([float(r.get("yaw_deg") or 0.0) for r in rows], dtype=np.float64)
    corner = np.asarray(corner_xy, dtype=np.float64)
    dist = np.linalg.norm(pos - corner, axis=1)
    visit = int(np.argmin(dist))
    along = float(np.dot(pos[-1] - corner, _heading(yaw_out)))
    yaw_err = abs((yaw[-1] - yaw_out + 180.0) % 360.0 - 180.0)
    turned = bool(dist[visit] < 18.0 and along > 0.6 * float(leg_out_m))
    return {
        "turned": turned,
        "min_corner_m": round(float(dist[visit]), 1),
        "along_out_m": round(along, 1),
        "end_yaw_err_deg": round(float(yaw_err), 1),
    }


def fly(protocol: str) -> None:
    dest = OUT / protocol
    dest.mkdir(parents=True, exist_ok=True)
    cmd = EVAL_BASE + [
        "--subgoal-source",
        protocol,
        "--traj-out",
        str((dest / "traj").relative_to(ROOT)),
        "--out",
        str((dest / "eval.json").relative_to(ROOT)),
    ]
    print("FLY", protocol, flush=True)
    log_path = dest / "run.log"
    with log_path.open("w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    print("exit", protocol, proc.returncode, flush=True)


def report() -> None:
    doc = json.loads(ANNO.read_text())
    route = doc["routes"][0]
    goal = np.asarray(route["pos"][-1][:2], dtype=np.float64)
    summary = []
    for protocol in ("polyline", "road_center"):
        traj = OUT / protocol / "traj" / "route00.jsonl"
        rows = [json.loads(line) for line in traj.read_text().splitlines() if line.strip()]
        end = np.asarray(rows[-1]["pos"][:2], dtype=np.float64)
        d_final = float(np.linalg.norm(end - goal))
        arrived = d_final <= 3.0
        corners = []
        for spec in route["corners"]:
            item = score_corner(rows, spec["xy"], spec["yaw_out"], spec["leg_out_m"])
            item["xy"] = spec["xy"]
            corners.append(item)
        both = arrived and all(c["turned"] for c in corners)
        rec = {
            "protocol": protocol,
            "arrived": arrived,
            "d_final_m": round(d_final, 2),
            "both_turns_and_arrived": both,
            "end_xy": [round(float(end[0]), 1), round(float(end[1]), 1)],
            "corners": corners,
            "steps": len(rows),
        }
        summary.append(rec)
        print(
            "%s arrived=%s d=%.2f both=%s corners=%s"
            % (
                protocol,
                arrived,
                d_final,
                both,
                [(c["xy"], c["turned"], c["min_corner_m"], c["along_out_m"]) for c in corners],
            ),
            flush=True,
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fly("polyline")
    fly("road_center")
    report()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
