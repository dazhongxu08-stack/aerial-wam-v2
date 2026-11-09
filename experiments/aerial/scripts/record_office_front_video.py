#!/usr/bin/env python3
"""Save the frontal-barrel test as camera frames. Restores the barrel after."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/aerial-wam-v2/experiments/aerial/scripts")

import run_office_cons_loop as loop
import run_office_front_scene as front
import sim_verify_gap as sim
from experiments.aerial.rl.consequence_model import load_checkpoint
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

OUT = Path("/data/simworld/run/office_open_front_20261105/video")


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


def paint(image: np.ndarray, line1: str, line2: str, line3: str) -> np.ndarray:
    frame = np.ascontiguousarray(image)
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, h - 78), (w, h), (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(bar, 0.55, frame, 0.45, 0)
    cv2.putText(frame, line1, (12, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (12, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, line3, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def main() -> int:
    art = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts"
    loop.DEPTH = os.environ.get(
        "OFFICE_DEPTH",
        f"{art}/da3_office_align_20261104/depth_office_cone_align.pt",
    )
    loop.CONS = os.environ.get(
        "OFFICE_CONS",
        f"{art}/indoor_office_align_20261104/c_s218/consequence_open.pt",
    )
    loop.sim.score = score_next
    loop.pick = pick_learned
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("frame_*.jpg"):
        old.unlink()
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(loop.col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    loop.boxes.save_restore(ucv)
    depth_model = DepthMinPredictor.from_checkpoint(loop.DEPTH, device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    model = load_checkpoint(loop.CONS, "cuda")
    model.eval()
    loop.restore_boxes(ucv)
    front.place_front_barrel(ucv)
    x, y, yaw = front.START
    gx, gy = front.GOAL
    goal = np.array([gx / 100.0, gy / 100.0, 0.52], np.float64)
    rng = np.random.default_rng(0)
    prev = np.zeros(4, np.float32)
    saved = 0
    try:
        for step in range(front.STEPS):
            seen = loop.sense(ucv, cam, depth_model, x, y, yaw)
            if seen is None:
                print(f"s={step} grab_fail", flush=True)
                break
            packed = loop.col.grab(ucv, cam)
            if packed is None:
                print(f"s={step} image_fail", flush=True)
                break
            state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, np.deg2rad(yaw)], np.float64)
            bearing = sim.wrap(np.arctan2(goal[1] - state[1], goal[0] - state[0]) - state[6])
            names, cands, allowed = sim.candidates(rng, bearing)
            surv, _prog = sim.score([model], seen["depth"], state, goal, cands, prev, seen["da"][0])
            action, family, s_pick = pick_learned(seen["da"], seen["low"], names, cands, surv, allowed, bearing)
            gt, da = seen["gt"], seen["da"]
            frame = paint(
                packed[0],
                f"front barrel  s={step}  {family}  surv={s_pick:.2f}",
                f"GT fwd {gt[0]:.2f}  left {gt[1]:.2f}  right {gt[2]:.2f}",
                f"DA fwd {da[0]:.2f}  left {da[1]:.2f}  right {da[2]:.2f}  yaw {yaw:.0f}",
            )
            path = OUT / f"frame_{saved:03d}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            print(
                f"s={step} x={x:.0f} y={y:.0f} yaw={yaw:.0f} {family} "
                f"gt={gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f}",
                flush=True,
            )
            saved += 1
            if family == "stop" or min(gt) < loop.HIT_M:
                break
            x, y, yaw = loop.integrate(x, y, yaw, float(action[0]), float(action[3]))
            prev = np.asarray(action, np.float32)
    finally:
        loop.restore_boxes(ucv)
        print("front_restored", saved, flush=True)
    return 0 if saved else 2


if __name__ == "__main__":
    raise SystemExit(main())
