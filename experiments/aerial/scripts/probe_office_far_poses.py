#!/usr/bin/env python3
"""Live p20 GT/DA along the west approach; find a true near-wall pose."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_office_lowbox as boxes
import collect_office_train as col
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation
from office_bands import cones3_forward

CKPT = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_far_20261103/depth_office_cone_far.pt"


def sense(ucv, cam, depth_model, x, y, yaw):
    col.place(ucv, x, y, yaw)
    col.look(ucv, cam, x, y, yaw)
    packed = col.grab(ucv, cam)
    rgb, depth = packed
    image, seen, warped = col.warp_pair(rgb, depth)
    gt = cones3_forward(col.pool_min(col.hide(warped, seen)))
    depth_model.reset()
    pred = np.asarray(
        depth_model.predict_depth(Observation(rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False)),
        np.float32,
    )
    if pred.shape[:2] != warped.shape[:2]:
        import cv2

        pred = cv2.resize(pred, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
    da = cones3_forward(col.pool_min(col.hide(pred, seen)))
    return [float(v) for v in gt], [float(v) for v in da]


def restore(ucv):
    rows = json.loads((boxes.OUT / "lowbox_restore.json").read_text())
    for row in rows:
        n = row["name"]
        col.req(ucv, f"vset /object/{n}/object_mobility true")
        col.req(ucv, f"vset /object/{n}/physics false")
        col.req(ucv, f"vset /object/{n}/scale {row['scale']}")
        col.req(ucv, f"vset /object/{n}/rotation {row['rotation']}")
        col.req(ucv, f"vset /object/{n}/location {row['location']}")


def main() -> int:
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    col.ensure_dog(ucv)
    boxes.save_restore(ucv)
    restore(ucv)
    pred = DepthMinPredictor.from_checkpoint(CKPT, device="cuda")
    print("CKPT", CKPT, flush=True)
    near = None
    for x in range(-520, 80, 40):
        gt, da = sense(ucv, cam, pred, float(x), -250.0, 0.0)
        print(f"west x={x} gt_f={gt[0]:.2f} da_f={da[0]:.2f} gt={gt} da={da}", flush=True)
        if near is None and gt[0] <= 1.5:
            near = (x, gt, da)
    print("NEAR", near, flush=True)
    for name, pose in (("aisle", (-480.0, -200.0, 0.0)), ("gap", (-160.0, -200.0, 0.0))):
        gt, da = sense(ucv, cam, pred, *pose)
        print(f"{name} gt_f={gt[0]:.2f} da_f={da[0]:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
