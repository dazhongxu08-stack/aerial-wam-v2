#!/usr/bin/env python3
"""Survival-only walk out of the west dead end. Does not change the wall judge."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/aerial-wam-v2/experiments/aerial/scripts")

import run_office_cons_loop as loop
import sim_verify_gap as sim
import train_office_cam60_around as base
from experiments.aerial.rl.consequence_model import load_checkpoint
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

OUT = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts/indoor_office_cam60_20261105/c_s218/consequence_cam60_exit.pt")
WALL_GOAL = (80.0, -250.0)


def score_next(models, depth56, state, goal, cands, prev, da_f=0.0):
    curve, prog = sim._curves(models, depth56, state, goal, cands, prev)
    return curve[:, 0], prog


def pick_learned(da, low, names, cands, surv, allowed, bearing):
    surv = np.asarray(surv, np.float64)
    names = list(names)
    cands = np.asarray(cands, np.float64)
    ok = np.flatnonzero(surv >= loop.SURV_MIN)
    if ok.size:
        moving = [int(i) for i in ok if "spin" not in names[int(i)]]
        if moving:
            dx = cands[moving, 0, 0]
            chosen = int(moving[int(np.argmax(dx))])
        else:
            chosen = int(ok[int(np.argmax(surv[ok]))])
        return loop.first_cmd(cands[chosen]), str(names[chosen]), float(surv[chosen])
    return np.zeros(4, np.float64), "stop", float(np.max(surv))


def walk_out(ucv, cam, depth_model, model, name, x, y, yaw, goal_xy, steps):
    print(f"WALK {name} start {x:.0f},{y:.0f} yaw {yaw:.0f}", flush=True)
    rng = np.random.default_rng(0)
    prev = np.zeros(4, np.float32)
    rows = []
    gx, gy = goal_xy
    goal = np.array([gx / 100.0, gy / 100.0, 0.52], np.float64)
    step = 0
    yaw_start = float(yaw)
    while step < steps:
        seen = loop.sense(ucv, cam, depth_model, x, y, yaw)
        if seen is None:
            print(f"  s={step} grab_fail", flush=True)
            break
        state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, math.radians(yaw)], np.float64)
        bearing = sim.wrap(math.atan2(goal[1] - state[1], goal[0] - state[0]) - state[6])
        names, cands, allowed = sim.candidates(rng, bearing)
        surv, _prog = sim.score([model], seen["depth"], state, goal, cands, prev, seen["da"][0])
        action, family, s_pick = pick_learned(seen["da"], seen["low"], names, cands, surv, allowed, bearing)
        # One scored spin is 18°. Ten of the same yaw step is a half turn in place.
        burst = [np.asarray(action, np.float64)]
        facing_end = abs((yaw - yaw_start + 180.0) % 360.0 - 180.0) < 30.0
        if facing_end and "spin" in family and abs(float(action[0])) < 0.05 and abs(float(action[3])) > 0.1:
            burst = [np.asarray(action, np.float64)] * 10
            print(f"  spin_run {len(burst)}", flush=True)
        for act in burst:
            if step >= steps:
                break
            gt, da = seen["gt"], seen["da"]
            dist = math.hypot(gx - x, gy - y)
            print(
                f"  s={step} x={x:.0f} y={y:.0f} yaw={yaw:.1f} dist={dist:.0f} "
                f"gt={gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f} da={da[0]:.2f},{da[1]:.2f},{da[2]:.2f} "
                f"surv_max={float(np.max(surv)):.3f} pick={s_pick:.3f} "
                f"{family} dx={float(act[0]):.2f} dyaw={float(act[3]):.3f}",
                flush=True,
            )
            rows.append({
                "x": x, "y": y, "yaw": yaw, "family": family,
                "dx": float(act[0]), "dyaw": float(act[3]),
                "dist": dist, "surv_pick": s_pick,
            })
            step += 1
            if family == "stop" or min(gt) < loop.HIT_M or dist < 40.0:
                return rows
            x, y, yaw = loop.integrate(x, y, yaw, float(act[0]), float(act[3]))
            prev = np.asarray(act, np.float32)
    return rows


def came_back(rows) -> bool:
    turned_at = None
    for row in rows:
        yaw = float(row["yaw"])
        if abs(abs(yaw) - 180.0) <= 30.0 or abs(yaw) >= 150.0:
            turned_at = float(row["x"])
            break
    if turned_at is None:
        return False
    return any(float(row["x"]) < turned_at - 25.0 for row in rows)


def main() -> int:
    if not OUT.is_file():
        print("EXIT_NO_WEIGHTS", flush=True)
        return 1
    loop.sim.score = score_next
    loop.pick = pick_learned
    depth_model = DepthMinPredictor.from_checkpoint(str(base.DEPTH), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    model = load_checkpoint(str(OUT), "cuda")
    model.eval()
    print(f"DEPTH {base.DEPTH}\nCONS {OUT}", flush=True)
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(loop.col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    try:
        loop.restore_boxes(ucv)
        for name, x, steps in (("face", -260.0, 28), ("entrance", -520.0, 16)):
            rows = walk_out(ucv, cam, depth_model, model, name, x, -250.0, 0.0, WALL_GOAL, steps)
            yaws = [float(r["yaw"]) for r in rows]
            xs = [float(r["x"]) for r in rows]
            back = came_back(rows)
            print(
                f"EXIT_WALK {name} steps={len(rows)} yaw0={yaws[0]:.0f} yawN={yaws[-1]:.0f} "
                f"x0={xs[0]:.0f} xN={xs[-1]:.0f} xmin={min(xs):.0f} back={back} "
                f"end={rows[-1]['family']}",
                flush=True,
            )
    finally:
        loop.restore_boxes(ucv)
        print("RESTORED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
