#!/usr/bin/env python3
"""Closed loop with the short-box depth head.

Uses the same office choice as the car, including the moving turn around a
low side object. Does not call sim_verify main and does not change exposure.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")

import collect_office_train as col
import sim_verify_gap as sim
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation

DEPTH = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_lowbox_20261103/depth_office_cone_lowbox.pt"
CONS = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/indoor_office_mix_angle_f_20261102/c_s218/consequence_best.pt"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


office = _load(Path("/data/simworld/run/office_choice_side.py"), "office_choice_side")
wire = _load(Path("/data/simworld/run/low_band.py"), "low_band")


def sense(ucv, cam, depth_model, x, y, yaw):
    col.place(ucv, x, y, yaw)
    col.look(ucv, cam, x, y, yaw)
    packed = col.grab(ucv, cam)
    if packed is None:
        return None
    rgb, depth = packed
    image, seen, warped = col.warp_pair(rgb, depth)
    gt = col.cones3(col.pool_min(col.hide(warped, seen)))
    depth_model.reset()
    pred = depth_model.predict_depth(Observation(
        rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False))
    pred = np.asarray(pred, np.float32)
    if pred.shape[:2] != warped.shape[:2]:
        import cv2
        pred = cv2.resize(pred, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
    low = wire.low_clearance(pred, col.FY, col.CY, col.SRC_H)
    hid = col.pool_min(col.hide(pred, seen))
    da = col.cones3(hid)
    return {
        "gt": [float(v) for v in gt],
        "da": [float(v) for v in da],
        "depth": hid,
        "low": low,
        "mean": float(rgb.mean()),
    }


def walk(ucv, cam, depth_model, models, name, x, y, yaw, goal_xy, steps):
    print(f"WALK {name} start {x:.0f},{y:.0f} yaw {yaw:.0f} goal {goal_xy}", flush=True)
    rng = np.random.default_rng(0)
    prev = np.zeros(4, np.float32)
    passage = None
    passage_yaw = None
    near_low = 0
    rows = []
    goal = np.array([goal_xy[0] / 100.0, goal_xy[1] / 100.0, 0.52], np.float64)
    for step in range(steps):
        seen = sense(ucv, cam, depth_model, x, y, yaw)
        if seen is None:
            print(f"  s={step} grab_fail", flush=True)
            break
        state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, math.radians(yaw)], np.float64)
        bearing = sim.wrap(math.atan2(goal[1] - state[1], goal[0] - state[0]) - state[6])
        names, cands, allowed = sim.candidates(rng, bearing)
        surv, prog = sim.score(models, seen["depth"], state, goal, cands, prev, seen["da"][0])
        _pick, action, family, _n = office.resolve(
            names, cands, surv, allowed, bearing, seen["da"], prog, None, None, sim.SURV_MIN)
        side = office.side_passage(seen["da"], seen["low"])
        if side is None and passage in ("arc_left", "arc_right"):
            turned = abs(sim.wrap(math.radians(yaw) - passage_yaw))
            side = office.hold_arc(seen["da"], seen["low"], passage, turned)
        if side is not None:
            action, family = side
        if family in ("arc_left", "arc_right"):
            if passage != family:
                passage_yaw = math.radians(yaw)
            passage = family
        elif family == "stop" or (
            passage is not None
            and abs(sim.wrap(math.radians(yaw) - passage_yaw)) >= office.ARC_HOLD_RAD
            and float(seen["da"][0]) >= office.NEAR_M
        ):
            passage = None
            passage_yaw = None
        nose_close = family == "stop" and float(seen["da"][0]) < office.NEAR_M
        if side is not None or nose_close:
            near_low += 1
        else:
            near_low = 0
        if near_low == 1:
            action = np.zeros(4, np.float64)
            family = "confirm"
        gt = seen["gt"]
        da = seen["da"]
        print(
            f"  s={step} x={x:.0f} y={y:.0f} yaw={yaw:.1f} "
            f"gt={gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f} da={da[0]:.2f},{da[1]:.2f},{da[2]:.2f} "
            f"low={seen['low']} {family} dx={float(action[0]):.2f} dyaw={float(action[3]):.3f}",
            flush=True,
        )
        rows.append({
            "gt": gt, "da": da, "family": family,
            "dx": float(action[0]), "dyaw": float(action[3]), "yaw": yaw,
        })
        if family == "stop" or gt[0] < 0.55:
            break
        prev = np.asarray(action, np.float32)
        x, y, yaw = sim.integrate(x, y, yaw, action)
    return rows


def judge(name, rows, kind):
    if not rows:
        print(f"RESULT {name} no_frames", flush=True)
        return
    last = rows[-1]
    driven = [r for r in rows if r["dx"] > 0.2]
    min_fwd = min(r["gt"][0] for r in rows)
    min_side = min(min(r["gt"][1], r["gt"][2]) for r in rows)
    yaw0, yaw1 = rows[0]["yaw"], rows[-1]["yaw"]
    drove_into = any(r["dx"] > 0.2 and min(r["gt"][0], r["gt"][1], r["gt"][2]) < 0.55 for r in rows)
    print(
        f"RESULT {name} kind={kind} steps={len(rows)} end={last['family']} "
        f"min_fwd={min_fwd:.2f} min_side={min_side:.2f} yaw={yaw0:.1f}->{yaw1:.1f} "
        f"drove_into={drove_into} moved={bool(driven)}",
        flush=True,
    )


def main() -> None:
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.ensure_dog(ucv)
    depth_model = DepthMinPredictor.from_checkpoint(DEPTH, device="cuda")
    models = [sim.load_checkpoint(CONS, "cuda")]
    print(f"DEPTH {DEPTH}", flush=True)
    jobs = [
        ("right_box", "side", -480, -170, 0, (220, -170), 32),
        ("center_lane", "side", -480, -200, 0, (220, -200), 32),
        ("gap", "through", -160, -200, 0, (200, -200), 36),
        ("block_west", "wall", -520, -250, 0, (80, -250), 24),
    ]
    for name, kind, x, y, yaw, goal, steps in jobs:
        rows = walk(ucv, cam, depth_model, models, name, x, y, yaw, goal, steps)
        judge(name, rows, kind)


if __name__ == "__main__":
    main()
