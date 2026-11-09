#!/usr/bin/env python3
"""Match the collection exposure and remeasure the gap depth."""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
spec = importlib.util.spec_from_file_location("ft", "/data/simworld/run/finetune_office_da3_cone.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

import collect_office_train as col
import sim_verify_gap as gap
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

CKPT = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt"
CONS = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/indoor_office_mix_angle_f_20261102/c_s218/consequence_best.pt"


def read_at(model, ucv, cam, offset: float):
    col.req(ucv, "vrun r.DefaultFeature.AutoExposure 0")
    col.req(ucv, f"vrun r.ExposureOffset {offset}")
    col.place(ucv, -160.0, -200.0, 0.0)
    col.look(ucv, cam, -160.0, -200.0, 0.0)
    rgb, depth = col.grab(ucv, cam)
    image, seen, warped = col.warp_pair(rgb, depth)
    gt = col.cones3(col.pool_min(col.hide(warped, seen)))
    with torch.no_grad():
        pred = ft.forward_maps(model, image[None]).detach().float().cpu().numpy()[0]
    da = ft.cones_np(pred)
    return image, pred, gt, da


def main() -> None:
    wrapper = DepthMinPredictor.from_checkpoint(CKPT, device="cuda")
    model = wrapper._model
    model.eval()
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    col.req(ucv, "vrun r.Fog 0")
    col.req(ucv, "vrun r.VolumetricFog 0")
    col.req(ucv, "vset /object/ExponentialHeightFog_1/hide")
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    rows = []
    for offset in (0, 6, 8, 12):
        image, pred, gt, da = read_at(model, ucv, cam, offset)
        print(
            f"offset {offset} mean {float(image.mean()):.1f} "
            f"gt {float(gt[0]):.2f},{float(gt[1]):.2f},{float(gt[2]):.2f} "
            f"da {float(da[0]):.2f},{float(da[1]):.2f},{float(da[2]):.2f}",
            flush=True,
        )
        rows.append((offset, image, pred, da))
    # Survival at the collection exposure, straight ahead.
    cons = gap.load_checkpoint(CONS, "cuda")
    _off, _image, pred, da = rows[1]
    state = np.array([-1.6, -2.0, 0.52, 0, 0, 0, 0.0])
    goal = np.array([0.2, -2.0, 0.52])
    straight = np.zeros((1, 10, 4), np.float32)
    straight[0, :, 0] = 0.8
    surv, _prog = gap.score([cons], pred, state, goal, straight, np.zeros(4, np.float32))
    print(f"offset6 straight surv {float(surv[0]):.3f} da_f {float(da[0]):.2f}", flush=True)


if __name__ == "__main__":
    main()
