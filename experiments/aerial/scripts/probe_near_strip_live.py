#!/usr/bin/env python3
"""Place the six barrels, then compare raw and thin-strip depth at the gap start."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")

import collect_office_train as col
import fit_near_strip as fit
import sim_verify_gap as gap

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
CFG = json.loads((ART / "near_strip_20261102/near_strip.json").read_text())["best"]
DEPTH = ART / "da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt"
CONS = ART / "indoor_office_mix_angle_f_20261102/c_s218/consequence_best.pt"


def place(ucv) -> None:
    props = json.loads(Path("/data/simworld/run/aisle_props.json").read_text())
    for item in props:
        name, x, y = item["name"], float(item["x"]), float(item["y"])
        mob = col.req(ucv, f"vset /object/{name}/object_mobility true")
        phys = col.req(ucv, f"vset /object/{name}/physics false")
        loc = col.req(ucv, f"vset /object/{name}/location {x} {y} -48")
        got = col.req(ucv, f"vget /object/{name}/location")
        print("place", name, "mob", mob, "phys", phys, "set", loc, "got", got, flush=True)


def column_mins(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, np.float32)
    h, w = d.shape
    return d[h // 4 : h - h // 4].min(axis=0)


def main() -> None:
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    col.req(ucv, "vrun r.Fog 0")
    col.req(ucv, "vrun r.VolumetricFog 0")
    col.req(ucv, "vset /object/ExponentialHeightFog_1/hide")
    col.req(ucv, "vrun r.ExposureOffset 8")
    cams = str(col.req(ucv, "vget /cameras")).split()
    if "FusionCameraSensor" not in cams:
        col.req(ucv, "vset /cameras/spawn")
        cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.ensure_dog(ucv)
    place(ucv)
    print("loading", flush=True)
    depth_model = gap.DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    model = gap.load_checkpoint(str(CONS), "cuda")
    poses = (
        ("gap", -160.0, -200.0, 0.0),
        ("yaw_left", -170.0, -200.0, 18.0),
        ("west", -520.0, -250.0, 0.0),
    )
    for name, x, y, yaw in poses:
        seen = gap.sense(ucv, cam, depth_model, x, y, yaw, False)
        raw = np.asarray(seen["depth"], np.float32)
        # sense already applied the patched metric_depth if a previous process patched
        # this process. Recompute from the model directly so raw is uncorrected.
        packed = None
        col.place(ucv, x, y, yaw)
        col.look(ucv, cam, x, y, yaw)
        packed = col.grab(ucv, cam)
        rgb, depth = packed
        image, seen_m, warped = col.warp_pair(rgb, depth)
        gt = col.cones3(col.pool_min(col.hide(warped, seen_m)))
        hid, _scale, _raw_c, _da = gap.metric_depth.__wrapped__(depth_model, image, seen_m, warped) if hasattr(gap.metric_depth, "__wrapped__") else (None, None, None, None)
        if hid is None:
            depth_model.reset()
            from experiments.aerial.rl.env.obs import Observation
            pred = depth_model.predict_depth(Observation(
                rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False))
            pred = np.asarray(pred, np.float32)
            if pred.shape[:2] != warped.shape[:2]:
                import cv2
                pred = cv2.resize(pred, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
            hid = col.pool_min(col.hide(pred, seen_m))
        hid = np.asarray(hid, np.float32)
        corr = fit.correct_thin(hid, CFG["alpha"], CFG["lo"], CFG["hi"], int(CFG["max_run"]))
        cols = fit.thin_columns(hid, CFG["lo"], CFG["hi"], int(CFG["max_run"]))
        state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, np.deg2rad(yaw)], np.float64)
        goal = state.copy()
        goal[0] += 8.0
        straight = np.zeros((1, 10, 4), np.float32)
        straight[0, :, 0] = 0.8
        s_raw, _ = gap.score([model], hid, state, goal, straight, np.zeros(4, np.float32))
        s_corr, _ = gap.score([model], corr, state, goal, straight, np.zeros(4, np.float32))
        print(
            name,
            "gt", np.round(gt, 2),
            "raw", np.round(col.cones3(hid), 2),
            "corr", np.round(col.cones3(corr), 2),
            "thin", int(cols.sum()),
            "surv_raw", round(float(s_raw[0]), 3),
            "surv_corr", round(float(s_corr[0]), 3),
            flush=True,
        )
        print(" cols", " ".join(f"{v:.2f}" for v in column_mins(hid)), flush=True)


if __name__ == "__main__":
    main()
