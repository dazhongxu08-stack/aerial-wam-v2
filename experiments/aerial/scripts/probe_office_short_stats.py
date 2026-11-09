#!/usr/bin/env python3
"""Print short-strip percentiles at the DA-only failure poses. No training."""
from __future__ import annotations

import json
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
from office_bands import FAR, GROUND_FRAC, cones3_forward, floor_range_map, horizon_row_224

DEPTH = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_far_20261103/depth_office_cone_far.pt"

POSES = [
    ("wall400", -400, -250, 0, False),
    ("wall390", -390, -250, 0, False),
    ("wall380", -380, -250, 0, False),
    ("wall370", -370, -250, 0, False),
    ("wall360", -360, -250, 0, False),
    ("wall350", -350, -250, 0, False),
    ("wall340", -340, -250, 0, False),
    ("wall330", -330, -250, 0, False),
    ("gap146", -146, -200, 0, False),
    ("gap100", -100, -200, 0, False),
    ("gap70", -70, -200, 0, False),
    ("gap40", -40, -200, 0, False),
    ("gap0", 0, -200, 0, False),
    ("aisle426", -426, -200, 0, False),
    ("side400", -400, -170, 0, True),
    ("side382", -382, -170, 0, True),
    ("side360", -360, -170, 0, True),
]


def bands(depth, gt=False):
    d = np.asarray(depth, np.float32)
    h, w = d.shape
    hz = horizon_row_224(h)
    hide = col.HIDE_ROW
    r0, r1 = int(hz), int(min(h, hide))
    band = d[r0:r1]
    fl = floor_range_map(h, w)[r0:r1]
    slices = {
        "L": (slice(None), slice(0, w // 3)),
        "C": (slice(None), slice(w // 4, 3 * w // 4)),
        "R": (slice(None), slice(2 * w // 3, None)),
    }
    out = {}
    for name, sl in slices.items():
        x = band[sl]
        floor = fl[sl]
        v = x[np.isfinite(x) & (x > 0.05) & (x < FAR - 0.5) & (x < GROUND_FRAC * floor)]
        if v.size < 8:
            out[name] = {"n": int(v.size)}
            continue
        out[name] = {
            "n": int(v.size),
            "p15": round(float(np.percentile(v, 15)), 2),
            "p50": round(float(np.percentile(v, 50)), 2),
            "p80": round(float(np.percentile(v, 80)), 2),
            "span": round(float(np.percentile(v, 80) - np.percentile(v, 50)), 2),
        }
    return out


def main() -> int:
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.ensure_dog(ucv)
    rows = json.loads((boxes.OUT / "lowbox_restore.json").read_text())
    for row in rows:
        name = row["name"]
        col.req(ucv, f"vset /object/{name}/location {row['location']}")
    depth_model = DepthMinPredictor.from_checkpoint(DEPTH, device="cuda")
    placed = False
    for name, x, y, yaw, side in POSES:
        if side and not placed:
            boxes.place_boxes(ucv)
            placed = True
        if not side and placed:
            for row in rows:
                col.req(ucv, f"vset /object/{row['name']}/location {row['location']}")
            placed = False
        col.place(ucv, x, y, yaw)
        col.look(ucv, cam, x, y, yaw)
        packed = col.grab(ucv, cam)
        rgb, depth = packed
        image, seen, warped = col.warp_pair(rgb, depth)
        gt56 = col.pool_min(col.hide(warped, seen))
        depth_model.reset()
        pred = depth_model.predict_depth(
            Observation(rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False)
        )
        pred = np.asarray(pred, np.float32)
        if pred.shape[:2] != warped.shape[:2]:
            import cv2

            pred = cv2.resize(pred, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
        hid = col.pool_min(col.hide(pred, seen))
        da = cones3_forward(hid)
        gt = cones3_forward(gt56)
        print(
            name,
            f"xy={x},{y} yaw={yaw}",
            "da_cone",
            [round(float(v), 2) for v in da],
            "gt_cone",
            [round(float(v), 2) for v in gt],
            flush=True,
        )
        print("  DA", bands(pred), flush=True)
        print("  GT", bands(warped), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
