#!/usr/bin/env python3
"""Tello runs -> consequence-model training windows (hazard labels only).

Resamples each qualifying run to the sim model step (5 Hz), runs DA3 on the
frame at each step, and converts the logged sticks into body deltas in the
sim action convention (AirSim NED: +dy right, +dyaw clockwise, bearing
clockwise+). Goal features match the runtime policy: yaw-only bearing and a
fixed 4 m distance. Cones and dgoal have no metric ground truth on the real
drone, so ``valid`` is 0 and only the hazard head is supervised.

Event ``t_s`` is relative to the post-takeoff air clock; telemetry ``t_s`` is
relative to record start, so events are shifted by the takeoff event time.

Run on the Mac (frames live under artifacts/tello_gs):

    PYTHONPATH=$PWD python3 experiments/aerial/scripts/tello_posttrain_windows.py \\
        --out experiments/aerial/rl/artifacts/tello_posttrain_20261014/windows.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEP_HZ = 5.0
# rc_calib_report.json (2026-10-13): stick 30 -> 0.405 m/s on tape, linear fit.
V_PER_STICK = 0.0135
GOAL_DIST_M = 4.0


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def qualifying_runs(gs_dir: Path, day: str) -> List[Path]:
    out = []
    for r in sorted(gs_dir.glob(f"run_{day}_*")):
        m = r / "meta.json"
        if not m.is_file():
            continue
        tier = str(json.loads(m.read_text(encoding="utf-8")).get("qc_tier") or "")
        if tier.startswith("可用"):
            out.append(r)
    return out


def load_run(run: Path) -> Dict[str, Any]:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (run / "telemetry.jsonl").read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines() if l.strip()]
    take = [e for e in events if e.get("event") == "takeoff"]
    land = [e for e in events if e.get("event") == "land"]
    if not take:
        raise ValueError(f"{run.name}: no takeoff event")
    t_air0 = float(take[0]["t_s"])
    coll = sorted(t_air0 + float(e["t_s"]) for e in events if e.get("event") == "collision")
    t_end = float(land[0]["t_s"]) if land else float(rows[-1]["t_s"])
    return {"meta": meta, "rows": rows, "t_air0": t_air0, "t_end": t_end, "coll": coll}


def resample(run: Dict[str, Any], settle_s: float) -> Dict[str, np.ndarray]:
    """Step times on a 5 Hz grid from takeoff+settle to land or first contact."""
    rows = [r for r in run["rows"] if r.get("airborne")]
    t = np.array([r["t_s"] for r in rows])
    t0 = run["t_air0"] + settle_s
    t1 = min(run["t_end"], run["coll"][0]) if run["coll"] else run["t_end"]
    grid = np.arange(t0, t1 + 1e-6, 1.0 / STEP_HZ)
    idx = np.clip(np.searchsorted(t, grid), 0, len(rows) - 1)
    yaw = np.unwrap(np.radians([float(r["pose"]["yaw_deg"]) for r in rows]))
    h = np.array([float(r["tello"].get("h", 0)) / 100.0 for r in rows])
    rc = np.array([r["rc"][:4] for r in rows], np.float64)
    seq = np.array([int(r["frame_seq"]) for r in rows])
    n = len(grid)
    act = np.zeros((max(0, n - 1), 4), np.float32)
    dt = 1.0 / STEP_HZ
    for i in range(n - 1):
        m = (t >= grid[i]) & (t < grid[i + 1])
        a_st, b_st = (rc[m, 0].mean(), rc[m, 1].mean()) if m.any() else (rc[idx[i], 0], rc[idx[i], 1])
        act[i] = (b_st * V_PER_STICK * dt,
                  a_st * V_PER_STICK * dt,
                  h[idx[i + 1]] - h[idx[i]],
                  yaw[idx[i + 1]] - yaw[idx[i]])
    collided = np.zeros(n, bool)
    if run["coll"]:
        collided[-1] = True
    return {"t": grid, "idx": idx, "yaw": yaw[idx], "seq": seq[idx], "action": act, "collided": collided}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gs-dir", default="artifacts/tello_gs")
    ap.add_argument("--day", default="20261014")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-ckpt",
                    default="experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/da3_ft_head_fp16.pt")
    ap.add_argument("--device", default="")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--settle-s", type=float, default=1.0)
    args = ap.parse_args()

    import torch

    from experiments.aerial.rl.collect_consequence_dataset import pool_min
    from experiments.aerial.rl.consequence_model import LIMITS, depth_input, goal_features
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation

    dev = args.device or ("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    pred = DepthMinPredictor.from_checkpoint(ROOT / args.depth_ckpt, device=dev)
    H = int(args.horizon)
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal")
    parts: Dict[str, List[np.ndarray]] = {k: [] for k in keys + ("run", "step", "depth56")}
    report = []
    runs = qualifying_runs(ROOT / args.gs_dir, args.day)
    for ri, rd in enumerate(runs):
        run = load_run(rd)
        s = resample(run, args.settle_s)
        n = len(s["t"])
        if n < 2:
            report.append({"run": rd.name, "steps": n, "skip": "too short"})
            continue
        pred.reset()
        d56 = []
        for q in s["seq"]:
            f = rd / "frames" / f"{q:06d}.jpg"
            rgb = np.asarray(Image.open(f).convert("RGB").resize((224, 224), Image.BILINEAR), np.uint8)
            d = pred.predict_depth(Observation(rgb=rgb, state=np.zeros(7, np.float32), collided=False))
            d56.append(pool_min(d))
        meta = run["meta"]
        yaw0 = float(s["yaw"][0])
        rows = run["rows"]
        air = [r for r in rows if r.get("airborne")]
        tk = next(i for i, r in enumerate(air) if r["t_s"] >= run["t_air0"]) if air else 0
        takeoff_yaw = math.radians(float(air[tk]["pose"]["yaw_deg"])) if air else yaw0
        b0 = math.radians(float(meta.get("goal_bearing_deg", 0.0)))
        act = np.clip(s["action"], -LIMITS, LIMITS)
        T = n - 1
        for t in range(T):
            acts = np.zeros((H, 4), np.float32)
            hit = np.zeros(H, np.float32)
            alive = np.zeros(H, np.float32)
            for k in range(H):
                i = t + k
                if i >= T:
                    break
                acts[k] = act[i]
                alive[k] = 1.0
                if s["collided"][i + 1]:
                    hit[k] = 1.0
                    break
            bearing = wrap(takeoff_yaw + b0 - math.atan2(math.sin(s["yaw"][t]), math.cos(s["yaw"][t])))
            goal = np.array([GOAL_DIST_M * math.cos(bearing), GOAL_DIST_M * math.sin(bearing), 0.0])
            parts["depth"].append(depth_input(d56[t]))
            parts["goal"].append(goal_features(np.zeros(7), goal))
            parts["prev_action"].append(act[t - 1] if t > 0 else np.zeros(4, np.float32))
            parts["actions"].append(acts)
            parts["hit"].append(hit)
            parts["alive"].append(alive)
            parts["valid"].append(np.zeros(H, np.float32))
            parts["log_cones"].append(np.zeros((H, 3), np.float32))
            parts["dgoal"].append(np.zeros(H, np.float32))
            parts["run"].append(np.int16(ri))
            parts["step"].append(np.int32(t))
            parts["depth56"].append(np.asarray(d56[t], np.float16))
        pos = int(sum(1 for h in parts["hit"][-T:] if h.sum() > 0))
        rec = {"run": rd.name, "steps": n, "windows": T, "hit_windows": pos, "contact": bool(run["coll"]),
               "dx_mean": float(act[:, 0].mean()), "dyaw_abs_mean": float(np.abs(act[:, 3]).mean()),
               "min_fwd_depth_last": float(np.min(np.asarray(d56[-1], np.float32)[18:38, 18:38]))}
        report.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, runs=np.array([r.name for r in runs]),
                        **{k: np.stack(v) for k, v in parts.items()})
    (out.parent / "windows_report.json").write_text(
        json.dumps({"step_hz": STEP_HZ, "v_per_stick": V_PER_STICK, "goal_dist_m": GOAL_DIST_M,
                    "convention": "AirSim NED: +dy right, +dyaw CW, bearing CW+", "runs": report},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out, sum(len(v) for v in [parts["hit"]]), "windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
