#!/usr/bin/env python3
"""See whether matching the training brightness restores the gap depth."""
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
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

CKPT = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt"


def show(model, image: np.ndarray, label: str) -> None:
    with torch.no_grad():
        pred = ft.forward_maps(model, image[None]).detach().float().cpu().numpy()[0]
    da = ft.cones_np(pred)
    print(
        f"{label} mean {float(image.mean()):.1f} "
        f"da {float(da[0]):.2f},{float(da[1]):.2f},{float(da[2]):.2f}",
        flush=True,
    )


def main() -> None:
    wrapper = DepthMinPredictor.from_checkpoint(CKPT, device="cuda")
    model = wrapper._model
    model.eval()
    z = np.load("/data/simworld/run/indoor_office_gap/episodes/episode_00558.npz")
    stored = ft.decode_one(z, 0)
    show(model, stored, "stored")
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    col.req(ucv, "vrun r.DefaultFeature.AutoExposure 0")
    col.req(ucv, "vrun r.ExposureOffset 8")
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.place(ucv, -160.0, -200.0, 0.0)
    col.look(ucv, cam, -160.0, -200.0, 0.0)
    rgb, depth = col.grab(ucv, cam)
    live, _seen, warped = col.warp_pair(rgb, depth)
    gt = ft.cones_np(np.asarray(col.pool_min(col.hide(warped, _seen)), np.float32))
    print(f"gt {float(gt[0]):.2f},{float(gt[1]):.2f},{float(gt[2]):.2f}", flush=True)
    show(model, live, "live x1")
    target = float(stored.mean())
    base = max(float(live.mean()), 1.0)
    for scale in (2.0, 3.0, 4.0, 5.0, target / base):
        bright = np.clip(live.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        show(model, bright, f"live x{scale:.2f}")


if __name__ == "__main__":
    main()
