#!/usr/bin/env python3
"""Three survival-only rounds of the office gates on consequence_cam60_drive.pt.

Uses the existing judges. Also records whether the west corridor entrance
is driven through. Does not change those judges.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/aerial-wam-v2/experiments/aerial/scripts")

import run_office_cons_loop as loop
import run_office_front_scene as front
import sim_verify_gap as sim
from experiments.aerial.rl.consequence_model import load_checkpoint
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

ART = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts"
DEPTH = f"{ART}/da3_office_cam60_20261105/depth_office_cone_cam60.pt"
CONS = f"{ART}/indoor_office_cam60_20261105/c_s218/consequence_cam60_drive.pt"
ROUNDS = int(os.environ.get("OFFICE_ROUNDS", "3"))


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


def entrance_ok(rows) -> bool:
    if not rows:
        return False
    early = any(r["family"] == "stop" and float(r["x"]) < -400.0 for r in rows)
    return (not early) and max(float(r["x"]) for r in rows) > -450.0


def main() -> int:
    loop.sim.score = score_next
    sim.score = score_next
    loop.pick = pick_learned
    depth_model = DepthMinPredictor.from_checkpoint(DEPTH, device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    model = load_checkpoint(CONS, "cuda")
    model.eval()
    print(f"DEPTH {DEPTH}\nCONS {CONS}\nROUNDS {ROUNDS}", flush=True)
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(loop.col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    tally = []
    try:
        for rnd in range(1, ROUNDS + 1):
            print(f"ROUND {rnd}", flush=True)
            loop.restore_boxes(ucv)
            jobs = [
                ("center_lane", "aisle", -480.0, -200.0, 0.0, (200.0, -200.0), 36),
                ("block_west", "wall", -520.0, -250.0, 0.0, (80.0, -250.0), 40),
                ("gap", "through", -160.0, -200.0, 0.0, (200.0, -200.0), 36),
            ]
            for name, kind, x, y, yaw, goal, steps in jobs:
                rows = loop.walk(ucv, cam, depth_model, model, name, kind, x, y, yaw, goal, steps)
                ok = loop.judge(name, kind, rows)
                tally.append((rnd, name, ok))
                if name == "block_west":
                    near = min(rows, key=lambda r: abs(float(r["x"]) + 462.0))
                    ent = entrance_ok(rows)
                    print(
                        f"ENTRANCE {'PASS' if ent else 'FAIL'} round={rnd} "
                        f"near_x={float(near['x']):.0f} surv={float(near['surv_pick']):.3f} "
                        f"{near['family']} end_x={float(rows[-1]['x']):.0f} end={rows[-1]['family']}",
                        flush=True,
                    )
                    tally.append((rnd, "entrance", ent))
            loop.boxes.place_boxes(ucv)
            rows = loop.walk(
                ucv, cam, depth_model, model, "right_box", "side",
                -480.0, -170.0, 0.0, (220.0, -170.0), 32,
            )
            tally.append((rnd, "right_box", loop.judge("right_box", "side", rows)))
            loop.restore_boxes(ucv)
            front.place_front_barrel(ucv)
            rows = loop.walk(
                ucv, cam, depth_model, model, "front_barrel", "front",
                front.START[0], front.START[1], front.START[2], front.GOAL, front.STEPS,
            )
            tally.append((rnd, "front_barrel", front.judge_front(rows)))
            loop.restore_boxes(ucv)
            print(f"ROUND {rnd} done", flush=True)
    finally:
        loop.restore_boxes(ucv)
        print("RESTORED", flush=True)
    names = ["center_lane", "block_west", "entrance", "gap", "right_box", "front_barrel"]
    for name in names:
        marks = ["PASS" if ok else "FAIL" for rnd, n, ok in tally if n == name]
        print(f"SUM {name} {' '.join(marks)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
