#!/usr/bin/env python3
"""Dump GT/DA row profiles at the two failed poses. No motion."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_office_train as col
import collect_office_lowbox as boxes
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation
from office_bands import cones3_forward, horizon_row_224, horizon_row_56

DEPTH = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_lowbox_20261103/depth_office_cone_lowbox.pt"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


wire = _load(Path("/data/simworld/run/low_band.py"), "low_band")


def row_mins(d, r0, r1):
    h, w = d.shape
    r0, r1 = max(0, r0), min(h, r1)
    band = d[r0:r1]
    if band.size == 0:
        return (None, None, None)

    def p(x):
        v = x[np.isfinite(x) & (x > 0.15) & (x < 25)]
        if v.size < 8:
            return None
        return (float(v.min()), float(np.percentile(v, 15)), float(np.median(v)))

    return p(band[:, : w // 3]), p(band[:, w // 4 : 3 * w // 4]), p(band[:, 2 * w // 3 :])


def dump(tag, gt, da, src_h=480):
    h = gt.shape[0]
    hz = horizon_row_224(h) if h == 224 else horizon_row_56(out=h)
    hide = col.HIDE_ROW if h == 224 else col.HIDE_ROW * h // 224
    print(f"== {tag} shape={gt.shape} horizon={hz} hide={hide}", flush=True)
    for r0, r1, name in (
        (0, hz, "above_hz"),
        (max(0, hz - 8), hz, "hz_minus"),
        (hz, hide, "hz_to_hide"),
        (hide, min(h, hide + 16), "below_hide"),
    ):
        print(f"  GT {name} [{r0}:{r1}] L/C/R={row_mins(gt, r0, r1)}", flush=True)
        print(f"  DA {name} [{r0}:{r1}] L/C/R={row_mins(da, r0, r1)}", flush=True)
    print("  cones_fwd GT", [float(v) for v in cones3_forward(col.pool_min(col.hide(gt)))], flush=True)
    print("  cones_fwd DA", [float(v) for v in cones3_forward(col.pool_min(col.hide(da)))], flush=True)
    print("  cones_old GT", [float(v) for v in col.cones3(col.pool_min(col.hide(gt)))], flush=True)
    print("  cones_old DA", [float(v) for v in col.cones3(col.pool_min(col.hide(da)))], flush=True)
    print("  low DA", wire.low_clearance(da, col.FY, col.CY, src_h), flush=True)
    print("  low GT", wire.low_clearance(gt, col.FY, col.CY, src_h), flush=True)


def grab_pair(ucv, cam, depth_model, x, y, yaw):
    col.place(ucv, x, y, yaw)
    col.look(ucv, cam, x, y, yaw)
    packed = col.grab(ucv, cam)
    rgb, depth = packed
    image, seen, warped = col.warp_pair(rgb, depth)
    gt = np.where(np.isfinite(warped) & (warped > 0.05), warped, np.nan).astype(np.float32)
    depth_model.reset()
    pred = np.asarray(
        depth_model.predict_depth(
            Observation(rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False)
        ),
        np.float32,
    )
    if pred.shape[:2] != gt.shape[:2]:
        import cv2

        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = pred.copy()
    pred[~seen] = np.nan
    return gt, pred, float(rgb.mean())


def restore(ucv):
    rows = json.loads((boxes.OUT / "lowbox_restore.json").read_text())
    for row in rows:
        n = row["name"]
        col.req(ucv, f"vset /object/{n}/object_mobility true")
        col.req(ucv, f"vset /object/{n}/physics false")
        col.req(ucv, f"vset /object/{n}/scale {row['scale']}")
        col.req(ucv, f"vset /object/{n}/rotation {row['rotation']}")
        col.req(ucv, f"vset /object/{n}/location {row['location']}")


def main():
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    col.ensure_dog(ucv)
    depth_model = DepthMinPredictor.from_checkpoint(DEPTH, device="cuda")
    restore(ucv)
    for tag, pose in (
        ("gap_s3", (-78, -217, 0)),
        ("gap_s0", (-160, -200, 0)),
        ("wall_s0", (-520, -250, 0)),
        ("aisle_s0", (-480, -200, 0)),
    ):
        gt, da, mean = grab_pair(ucv, cam, depth_model, *pose)
        print(f"POSE {tag} {pose} mean={mean:.1f}", flush=True)
        dump(tag, gt, da)
    print("PLACE_BOXES", flush=True)
    boxes.place_boxes(ucv)
    for tag, pose in (
        ("box_s0", (-480, -170, 0)),
        ("box_s6", (-332, -228, -36)),
        ("box_s7", (-312, -243, -36)),
    ):
        gt, da, mean = grab_pair(ucv, cam, depth_model, *pose)
        print(f"POSE {tag} {pose} mean={mean:.1f}", flush=True)
        dump(tag, gt, da)


if __name__ == "__main__":
    main()
