#!/usr/bin/env python3
"""Restore the collection exposure and compare the live gap image to the stored jpeg."""
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


def stats(image: np.ndarray, pred: np.ndarray, label: str) -> None:
    center = image[72:152, 72:152]
    nz = image[image.sum(-1) > 12]
    da = ft.cones_np(pred)
    print(
        f"{label} full {float(image.mean()):.1f} center {float(center.mean()):.1f} "
        f"nz {float(nz.mean()) if nz.size else -1:.1f} "
        f"da {float(da[0]):.2f},{float(da[1]):.2f},{float(da[2]):.2f}",
        flush=True,
    )


def depth_of(model, image: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return ft.forward_maps(model, image[None]).detach().float().cpu().numpy()[0]


def main() -> None:
    wrapper = DepthMinPredictor.from_checkpoint(CKPT, device="cuda")
    model = wrapper._model
    model.eval()
    z = np.load("/data/simworld/run/indoor_office_gap/episodes/episode_00558.npz")
    stored = ft.decode_one(z, 0)
    stats(stored, depth_of(model, stored), "stored")
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    for cmd in (
        "vrun r.Fog 0",
        "vrun r.VolumetricFog 0",
        "vset /object/ExponentialHeightFog_1/hide",
        "vrun r.EyeAdaptationQuality 0",
        "vrun r.DefaultFeature.AutoExposure 0",
        "vrun r.ExposureOffset 6",
    ):
        print(cmd, col.req(ucv, cmd), flush=True)
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.place(ucv, -160.0, -200.0, 0.0)
    col.look(ucv, cam, -160.0, -200.0, 0.0)
    rgb, depth = col.grab(ucv, cam)
    image, seen, warped = col.warp_pair(rgb, depth)
    gt = ft.cones_np(np.asarray(col.pool_min(col.hide(warped, seen)), np.float32))
    print(f"gt {float(gt[0]):.2f},{float(gt[1]):.2f},{float(gt[2]):.2f}", flush=True)
    pred = depth_of(model, image)
    stats(image, pred, "live_collect")
    cons = gap.load_checkpoint(CONS, "cuda")
    state = np.array([-1.6, -2.0, 0.52, 0, 0, 0, 0.0])
    goal = np.array([0.2, -2.0, 0.52])
    action = np.zeros((1, 10, 4), np.float32)
    action[0, :, 0] = 0.8
    surv, _ = gap.score([cons], pred, state, goal, action, np.zeros(4, np.float32))
    print(f"straight surv {float(surv[0]):.3f}", flush=True)


if __name__ == "__main__":
    main()
