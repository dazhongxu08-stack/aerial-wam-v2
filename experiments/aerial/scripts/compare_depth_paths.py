#!/usr/bin/env python3
"""Compare the training forward path with the sim predict_depth path."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
spec = importlib.util.spec_from_file_location("ft", "/data/simworld/run/finetune_office_da3_cone.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

import collect_office_train as col
import sim_verify_gap as gap
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation

CKPT = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt"


def cones_of(depth56: np.ndarray) -> str:
    c = ft.cones_np(np.asarray(depth56, np.float32))
    return f"{float(c[0]):.2f},{float(c[1]):.2f},{float(c[2]):.2f}"


def via_forward(model, image: np.ndarray, n: int = 1) -> np.ndarray:
    stack = np.stack([image] * n, axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(stack)).unsqueeze(0).to("cuda")
    with torch.no_grad():
        depth, _ = model.predict_from_window(tensor)
    pred = depth.squeeze(0).detach().float()
    if pred.ndim == 3:
        pred = pred[0]
    rgb = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).to("cuda")
    return ft.hide_pool(pred.unsqueeze(0), rgb).detach().float().cpu().numpy()[0]


def via_predict(wrapper, image: np.ndarray, seen: np.ndarray) -> np.ndarray:
    wrapper.reset()
    pred = wrapper.predict_depth(Observation(
        rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False))
    pred = np.asarray(pred, np.float32)
    if pred.shape[:2] != image.shape[:2]:
        import cv2
        pred = cv2.resize(pred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.asarray(col.pool_min(col.hide(pred, seen)), np.float32)


def main() -> None:
    payload = torch.load(CKPT, map_location="cpu")
    print("n_frames", payload.get("n_frames"), "backbone", payload.get("backbone"), flush=True)
    del payload
    wrapper = DepthMinPredictor.from_checkpoint(CKPT, device="cuda")
    model = wrapper._model
    model.eval()
    print("n", wrapper.n_frames, flush=True)
    z = np.load("/data/simworld/run/indoor_office_gap/episodes/episode_00558.npz")
    image = ft.decode_one(z, 0)
    gt = np.asarray(z["depth56"][0], np.float32)
    seen = np.ones(image.shape[:2], np.uint8) > 0
    print("stored shape", image.shape, "mean", round(float(image.mean()), 1), "gt", cones_of(gt), flush=True)
    print("stored n1", cones_of(via_forward(model, image, 1)), flush=True)
    print("stored n4", cones_of(via_forward(model, image, 4)), flush=True)
    print("stored predict", cones_of(via_predict(wrapper, image, seen)), flush=True)
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.place(ucv, -160.0, -200.0, 0.0)
    col.look(ucv, cam, -160.0, -200.0, 0.0)
    rgb, depth = col.grab(ucv, cam)
    live, live_seen, warped = col.warp_pair(rgb, depth)
    gt_live = col.pool_min(col.hide(warped, live_seen))
    print("live shape", live.shape, "mean", round(float(live.mean()), 1), "gt", cones_of(gt_live), flush=True)
    print("live n1", cones_of(via_forward(model, live, 1)), flush=True)
    print("live n4", cones_of(via_forward(model, live, 4)), flush=True)
    print("live predict", cones_of(via_predict(wrapper, live, live_seen)), flush=True)


if __name__ == "__main__":
    main()
