#!/usr/bin/env python3
"""Find street corners that require a turn, then fly each one two ways.

polyline: the carrot follows the L.
toward_g: the carrot points at the goal, so the turn is not pre-drawn.

Writes a rolling STATUS.json after every route. Meant to run on the AirSim host.
"""

from __future__ import annotations

import gc
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/aerial-wam-v2")
PY = "/data/venvs/sim_verify/bin/python"
OUT = ROOT / "experiments/aerial/rl/artifacts/eval_street_corners_overnight_20261005"
Z = 14.0
STREET_FWD = 16.0
LEG_IN_MIN = 48.0
LEG_OUT_MIN = 40.0
MAX_ROUTES_PER_TILE = 6

TILES = [
    # Known west canyon, then the blocks around it.
    (-1280.0, -900.0, 0.0, 180.0, 20.0),
    (-1600.0, -1280.0, 0.0, 180.0, 24.0),
    (-900.0, -560.0, 0.0, 180.0, 24.0),
    (-1280.0, -900.0, 180.0, 360.0, 24.0),
    (-1280.0, -900.0, -180.0, 0.0, 24.0),
]
if os.environ.get("STREET_CORNER_OUT"):
    OUT = Path(os.environ["STREET_CORNER_OUT"])
if os.environ.get("STREET_CORNER_TILES"):
    TILES = [tuple(float(v) for v in row) for row in json.loads(os.environ["STREET_CORNER_TILES"])]

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
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "suite.log").open("a") as f:
        f.write(line + "\n")


def write_status(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.json").write_text(json.dumps(payload, indent=2))


def _heading_vec(deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    return math.cos(r), math.sin(r)


class StreetProbe:
    def __init__(self) -> None:
        import airsim
        import cv2
        from experiments.aerial.rl.depth_geometry import approach_clearances

        self.airsim = airsim
        self.cv2 = cv2
        self.approach_clearances = approach_clearances
        self.client = airsim.MultirotorClient(ip="127.0.0.1")
        self.client.confirmConnection()

    def meas(self, x: float, y: float, yaw_deg: float) -> tuple[float, float, float]:
        q = self.airsim.to_quaternion(0.0, 0.0, math.radians(yaw_deg))
        self.client.simSetVehiclePose(
            self.airsim.Pose(self.airsim.Vector3r(float(x), float(y), -Z), q), True
        )
        time.sleep(0.02)
        (resp,) = self.client.simGetImages(
            [self.airsim.ImageRequest("front_custom", self.airsim.ImageType.DepthPlanar, True, False)]
        )
        gt = np.asarray(
            self.airsim.list_to_2d_float_array(resp.image_data_float, resp.width, resp.height),
            np.float32,
        )
        gt = self.cv2.resize(gt, (224, 224), interpolation=self.cv2.INTER_NEAREST)
        f, l, r = self.approach_clearances(gt, center_frac=0.12)
        return float(f), float(l), float(r)


def is_street(sample: tuple[float, float, float]) -> bool:
    f, l, r = sample
    return f >= STREET_FWD and 4.0 <= l <= 32.0 and 4.0 <= r <= 32.0


def walk_leg(probe: StreetProbe, x: float, y: float, heading: float, limit: float = 120.0) -> float:
    dx, dy = _heading_vec(heading)
    dist = 0.0
    ok = 0.0
    step = 8.0
    while dist + step <= limit + 1e-6:
        dist += step
        sample = probe.meas(x + dx * dist, y + dy * dist, heading)
        if not is_street(sample):
            break
        ok = dist
    return ok


def refine_corner(probe: StreetProbe, x: float, y: float) -> tuple[float, float, float] | None:
    best = None
    for dy in np.arange(-12, 13, 4):
        for dx in np.arange(-12, 13, 4):
            px, py = x + float(dx), y + float(dy)
            east = probe.meas(px, py, 0)
            north = probe.meas(px, py, 90)
            if not (is_street(east) and is_street(north)):
                continue
            score = east[0] + north[0] - abs(east[1] - east[2]) - abs(north[1] - north[2])
            if best is None or score > best[0]:
                best = (score, px, py)
    if best is None:
        return None
    return best[1], best[2], best[0]


def polyline(start, corner, goal, yaw_in: float, yaw_out: float, step: float = 4.0):
    pts = []
    yaws = []

    def append_leg(a, b, yaw):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        dist = float(np.linalg.norm(b[:2] - a[:2]))
        n = max(1, int(math.floor(dist / step)))
        for i in range(n):
            p = a + (b - a) * (i / n)
            pts.append([round(float(p[0]), 2), round(float(p[1]), 2), Z])
            yaws.append(float(yaw))

    append_leg(start, corner, yaw_in)
    append_leg(corner, goal, yaw_out)
    pts.append([round(float(goal[0]), 2), round(float(goal[1]), 2), Z])
    yaws.append(float(yaw_out))
    return pts, yaws


def diagonal_is_open(probe: StreetProbe, start, goal) -> bool:
    mx = 0.5 * (start[0] + goal[0])
    my = 0.5 * (start[1] + goal[1])
    bearing = math.degrees(math.atan2(goal[1] - start[1], goal[0] - start[0]))
    return is_street(probe.meas(mx, my, bearing))


def discover_tile(probe: StreetProbe, tile, seen_corners: list[tuple[float, float]]) -> list[dict]:
    x0, x1, y0, y1, step = tile
    log(f"scan tile x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}] step {step:.0f}")
    hits = []
    xs = np.arange(x0, x1 + 1e-6, step)
    ys = np.arange(y0, y1 + 1e-6, step)
    n = 0
    for y in ys:
        for x in xs:
            n += 1
            east = probe.meas(float(x), float(y), 0)
            north = probe.meas(float(x), float(y), 90)
            if is_street(east) and is_street(north):
                hits.append((float(x), float(y)))
        log(f"  row y={y:.0f} hits {len(hits)} scanned {n}")
    routes = []
    for hx, hy in hits:
        if any(math.hypot(hx - sx, hy - sy) < 30.0 for sx, sy in seen_corners):
            continue
        refined = refine_corner(probe, hx, hy)
        if refined is None:
            continue
        cx, cy, _ = refined
        if any(math.hypot(cx - sx, cy - sy) < 30.0 for sx, sy in seen_corners):
            continue
        candidates = []
        for yaw_in, yaw_out in ((0.0, 90.0), (0.0, -90.0), (90.0, 0.0), (90.0, 180.0)):
            in_len = walk_leg(probe, cx, cy, yaw_in + 180.0)
            out_len = walk_leg(probe, cx, cy, yaw_out)
            if in_len < LEG_IN_MIN or out_len < LEG_OUT_MIN:
                continue
            in_dx, in_dy = _heading_vec(yaw_in)
            out_dx, out_dy = _heading_vec(yaw_out)
            # Use a fixed leg so routes stay comparable, not the whole canyon.
            use_in = min(in_len, 80.0)
            use_out = min(out_len, 56.0)
            start = (cx - in_dx * use_in, cy - in_dy * use_in, Z)
            goal = (cx + out_dx * use_out, cy + out_dy * use_out, Z)
            if math.hypot(goal[0] - start[0], goal[1] - start[1]) < 40.0:
                continue
            if diagonal_is_open(probe, start, goal):
                log(f"  drop open diagonal at ({cx:.0f},{cy:.0f}) {yaw_in:.0f}->{yaw_out:.0f}")
                continue
            pts, yaws = polyline(start, (cx, cy, Z), goal, yaw_in, yaw_out)
            candidates.append(
                {
                    "corner": [round(cx, 1), round(cy, 1)],
                    "yaw_in": yaw_in,
                    "yaw_out": yaw_out,
                    "leg_in_m": round(use_in, 1),
                    "leg_out_m": round(use_out, 1),
                    "pos": pts,
                    "yaw": yaws,
                }
            )
        if not candidates:
            continue
        candidates.sort(key=lambda r: r["leg_in_m"] + r["leg_out_m"], reverse=True)
        chosen = candidates[0]
        seen_corners.append((cx, cy))
        routes.append(chosen)
        log(
            f"  route corner ({cx:.0f},{cy:.0f}) "
            f"yaw {chosen['yaw_in']:.0f}->{chosen['yaw_out']:.0f} "
            f"legs {chosen['leg_in_m']}+{chosen['leg_out_m']}"
        )
        if len(routes) >= MAX_ROUTES_PER_TILE:
            break
    log(f"tile routes {len(routes)}")
    return routes


def score_turn(route: dict, traj_path: Path, arrived: bool) -> dict:
    rows = []
    if traj_path.is_file():
        for line in traj_path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    corner = np.asarray(route["corner"], dtype=np.float64)
    goal = np.asarray(route["pos"][-1][:2], dtype=np.float64)
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


def fly_protocol(routes: list[dict], protocol: str, stamp: str) -> list[dict]:
    anno = {"routes": []}
    for i, route in enumerate(routes):
        anno["routes"].append(
            {
                "route_idx": i,
                "route_id": f"corner_{stamp}_{i:02d}",
                "source_kind": "depth_traced_corner",
                "pos": route["pos"],
                "yaw": route["yaw"],
            }
        )
    proto_dir = OUT / stamp / protocol
    proto_dir.mkdir(parents=True, exist_ok=True)
    anno_path = proto_dir / "routes.json"
    anno_path.write_text(json.dumps(anno))
    meta_path = proto_dir / "route_meta.json"
    meta_path.write_text(json.dumps(routes))
    n = len(routes)
    longest = max(len(r["pos"]) for r in routes)
    max_steps = str(max(450, int(longest * 3)))
    cmd = EVAL_BASE + [
        "--annotation",
        str(anno_path.relative_to(ROOT)),
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
    log(f"fly {protocol} n={n} max_steps={max_steps}")
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
            proc.wait(timeout=max(1800, 600 * n))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
            raise
    log(f"fly {protocol} exit {proc.returncode}")
    eval_doc = {}
    eval_path = proto_dir / "eval_all.json"
    if eval_path.is_file():
        eval_doc = json.loads(eval_path.read_text())
    results = eval_doc.get("results") or eval_doc.get("episodes") or []
    scored = []
    for i, route in enumerate(routes):
        rec = next((r for r in results if int(r.get("route_idx", -1)) == i), {})
        arrived = bool(rec.get("arrived"))
        turn = score_turn(route, proto_dir / "traj" / f"route{i:02d}.jsonl", arrived)
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
            f"  {protocol} r{i} corner {route['corner']} arrived={turn['arrived']} "
            f"turned={turn['turned']} d={turn.get('d_final_m')} yaw_err={turn.get('yaw_err_deg')}"
        )
    (proto_dir / "turn_scores.json").write_text(json.dumps(scored, indent=2))
    return scored


def main() -> None:
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    write_status({"phase": "scan", "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    probe = StreetProbe()
    log("airsim connected")
    seen: list[tuple[float, float]] = []
    seen_path = os.environ.get("STREET_CORNER_SEEN")
    if seen_path and Path(seen_path).is_file():
        for row in json.loads(Path(seen_path).read_text()):
            corner = row.get("corner")
            if corner and len(corner) >= 2:
                seen.append((float(corner[0]), float(corner[1])))
        log(f"skip {len(seen)} corners already flown")
    all_scores: list[dict] = []
    for ti, tile in enumerate(TILES):
        routes = discover_tile(probe, tile, seen)
        if not routes:
            write_status({"phase": "scan", "tile": ti, "routes": 0, "scores": all_scores})
            continue
        # Drop the AirSim pose client before the eval process connects.
        del probe
        gc.collect()
        stamp = f"tile{ti:02d}"
        (OUT / stamp).mkdir(parents=True, exist_ok=True)
        (OUT / stamp / "routes_full.json").write_text(json.dumps(routes))
        write_status({"phase": "fly", "tile": ti, "n_routes": len(routes), "scores": all_scores})
        for protocol in ("polyline", "toward_g"):
            try:
                scored = fly_protocol(routes, protocol, stamp)
            except subprocess.TimeoutExpired:
                log(f"{protocol} tile {ti} timed out")
                scored = [{"protocol": protocol, "tile": ti, "error": "timeout"}]
            except Exception as exc:  # keep the night running
                log(f"{protocol} tile {ti} failed: {exc}")
                scored = [{"protocol": protocol, "tile": ti, "error": str(exc)}]
            for row in scored:
                row["tile"] = ti
            all_scores.extend(scored)
            (OUT / "turn_scores_all.json").write_text(json.dumps(all_scores, indent=2))
            write_status(
                {
                    "phase": "fly",
                    "tile": ti,
                    "protocol_done": protocol,
                    "n_scored": len(all_scores),
                    "scores": all_scores,
                }
            )
        probe = StreetProbe()
        log("airsim reconnected for next tile")
    write_status({"phase": "done", "n_scored": len(all_scores), "scores": all_scores})
    log(f"SUITE_DONE n={len(all_scores)}")


if __name__ == "__main__":
    main()
