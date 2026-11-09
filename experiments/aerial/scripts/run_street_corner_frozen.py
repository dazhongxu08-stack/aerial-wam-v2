#!/usr/bin/env python3
"""Re-fly the frozen 13 street corners. Polyline first, then road_center.

The exam is street_corners_frozen.json. A route counts only when the vehicle
arrives (eval d_final <= 3 m) and its track passes the corner onto the exit
leg. Road-center starts only if polyline still clears the previous 8/13.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/aerial-wam-v2")
PY = "/data/venvs/sim_verify/bin/python"
ANNO = ROOT / "experiments/aerial/phase3_unified/annotations/street_corners_frozen.json"
OUT = ROOT / "experiments/aerial/rl/artifacts/eval_street_corners_frozen_20261005"
POLYLINE_FLOOR = 8

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
]


def log(msg: str) -> None:
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "suite.log").open("a") as f:
        f.write(line + "\n")


def write_status(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.json").write_text(json.dumps(payload, indent=2))


def _heading_vec(deg: float) -> tuple[float, float]:
    r = np.deg2rad(deg)
    return float(np.cos(r)), float(np.sin(r))


def score_turn(route: dict, traj_path: Path, arrived: bool) -> dict:
    rows = []
    if traj_path.is_file():
        for line in traj_path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    corner = np.asarray(route["corner"], dtype=np.float64)
    if not rows:
        return {"arrived": bool(arrived), "turned": False, "reason": "no_traj"}
    pos = np.asarray([r["pos"][:2] for r in rows], dtype=np.float64)
    z = np.asarray([r["pos"][2] for r in rows], dtype=np.float64)
    yaw = np.asarray([float(r.get("yaw_deg") or 0.0) for r in rows], dtype=np.float64)
    dist_corner = np.linalg.norm(pos - corner, axis=1)
    visit = int(np.argmin(dist_corner))
    end = pos[-1]
    out_dx, out_dy = _heading_vec(float(route["yaw_out"]))
    along_out = float(np.dot(end - corner, np.array([out_dx, out_dy])))
    yaw_err = abs((yaw[-1] - float(route["yaw_out"]) + 180.0) % 360.0 - 180.0)
    turned = bool(dist_corner[visit] < 18.0 and along_out > 0.6 * float(route["leg_out_m"]))
    return {
        "arrived": bool(arrived),
        "turned": turned,
        "passed_and_arrived": bool(arrived and turned),
        "min_corner_m": round(float(dist_corner[visit]), 1),
        "along_out_m": round(along_out, 1),
        "end_xy": [round(float(end[0]), 1), round(float(end[1]), 1)],
        "end_yaw_deg": round(float(yaw[-1]), 1),
        "yaw_err_deg": round(float(yaw_err), 1),
        "z_min": round(float(z.min()), 1),
        "z_max": round(float(z.max()), 1),
        "steps": len(rows),
        "d_to_g": rows[-1].get("d_to_g"),
    }


def fly(routes: list[dict], protocol: str) -> list[dict]:
    proto_dir = OUT / protocol
    proto_dir.mkdir(parents=True, exist_ok=True)
    n = len(routes)
    longest = max(len(r["pos"]) for r in routes)
    max_steps = str(max(450, int(longest * 3)))
    cmd = EVAL_BASE + [
        "--annotation",
        str(ANNO.relative_to(ROOT)),
        "--episodes",
        str(n),
        "--subgoal-source",
        protocol,
        "--max-steps",
        max_steps,
        "--traj-out",
        str((proto_dir / "traj").relative_to(ROOT)),
        "--out",
        str((proto_dir / "eval_all.json").relative_to(ROOT)),
    ]
    log("fly %s n=%d max_steps=%s" % (protocol, n, max_steps))
    write_status({"phase": protocol, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    log_path = proto_dir / "run.log"
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=max(7200, 600 * n))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
            raise
    log("fly %s exit %s" % (protocol, proc.returncode))
    eval_doc = {}
    eval_path = proto_dir / "eval_all.json"
    if eval_path.is_file():
        eval_doc = json.loads(eval_path.read_text())
    results = eval_doc.get("results") or eval_doc.get("episodes") or []
    if not results:
        log("eval produced no episodes")
        write_status({"phase": "error", "protocol": protocol, "reason": "eval_no_episodes"})
        raise SystemExit(1)
    scored = []
    for i, route in enumerate(routes):
        rec = next((r for r in results if int(r.get("route_idx", -1)) == i), {})
        arrived = bool(rec.get("arrived"))
        turn = score_turn(route, proto_dir / "traj" / ("route%02d.jsonl" % i), arrived)
        turn.update(
            {
                "protocol": protocol,
                "route_idx": i,
                "corner": route["corner"],
                "yaw_in": route["yaw_in"],
                "yaw_out": route["yaw_out"],
                "d_final_m": rec.get("d_final_m"),
                "severe_collision": rec.get("severe_collision"),
                "progress_ratio": rec.get("progress_ratio"),
                "steps_eval": rec.get("steps"),
                "fail_tag": rec.get("fail_tag"),
            }
        )
        scored.append(turn)
        log(
            "  %s r%d corner %s arrived=%s turned=%s d=%s yaw_err=%s"
            % (
                protocol,
                i,
                route["corner"],
                turn["arrived"],
                turn["turned"],
                turn.get("d_final_m"),
                turn.get("yaw_err_deg"),
            )
        )
    (proto_dir / "turn_scores.json").write_text(json.dumps(scored, indent=2))
    return scored


def main() -> None:
    global OUT
    if os.environ.get("STREET_CORNER_OUT"):
        OUT = Path(os.environ["STREET_CORNER_OUT"])
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    doc = json.loads(ANNO.read_text())
    routes = doc["routes"]
    only = os.environ.get("STREET_CORNER_ONLY", "")
    depth = os.environ.get("STREET_CORNER_DEPTH", "").strip()
    if depth:
        EVAL_BASE.extend(["--depth-ckpt", depth])
    log("frozen routes %d" % len(routes))
    if depth:
        log("depth ckpt %s" % depth)
    if only == "road_center":
        road = fly(routes, "road_center")
        road_both = sum(1 for r in road if r.get("passed_and_arrived"))
        log("road_center passed_and_arrived %d/%d" % (road_both, len(road)))
        write_status(
            {
                "phase": "done",
                "road_center_passed_and_arrived": road_both,
                "n": len(routes),
            }
        )
        return
    write_status({"phase": "polyline", "n": len(routes)})
    poly = fly(routes, "polyline")
    both = sum(1 for r in poly if r.get("passed_and_arrived"))
    log("polyline passed_and_arrived %d/%d floor %d" % (both, len(poly), POLYLINE_FLOOR))
    write_status(
        {
            "phase": "polyline_done",
            "passed_and_arrived": both,
            "n": len(poly),
        }
    )
    if both < POLYLINE_FLOOR:
        log("polyline below the previous 8/13, not starting road_center")
        write_status({"phase": "stopped", "reason": "polyline_regressed", "passed_and_arrived": both})
        return
    road = fly(routes, "road_center")
    road_both = sum(1 for r in road if r.get("passed_and_arrived"))
    log("road_center passed_and_arrived %d/%d (polyline %d)" % (road_both, len(road), both))
    write_status(
        {
            "phase": "done",
            "polyline_passed_and_arrived": both,
            "road_center_passed_and_arrived": road_both,
            "n": len(routes),
        }
    )


if __name__ == "__main__":
    main()
