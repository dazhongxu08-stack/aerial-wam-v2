#!/usr/bin/env python3
"""Map the live gap image onto the stored frame's tones and rescore depth."""
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


def match_tone(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    out = np.empty_like(src)
    for c in range(3):
        s = src[:, :, c].ravel()
        r = ref[:, :, c].ravel()
        s_vals, s_inv, s_cnt = np.unique(s, return_inverse=True, return_counts=True)
        r_vals, r_cnt = np.unique(r, return_counts=True)
        s_cdf = np.cumsum(s_cnt) / s.size
        r_cdf = np.cumsum(r_cnt) / r.size
        mapped = np.interp(s_cdf, r_cdf, r_vals)
        out[:, :, c] = mapped[s_inv].reshape(src.shape[:2])
    return out


def main() -> None:
    wrapper = DepthMinPredictor.from_checkpoint(CKPT, device="cuda")
    model = wrapper._model
    model.eval()
    stored = ft.decode_one(np.load("/data/simworld/run/indoor_office_gap/episodes/episode_00558.npz"), 0)
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    col.req(ucv, "vrun r.Fog 0")
    col.req(ucv, "vrun r.VolumetricFog 0")
    col.req(ucv, "vset /object/ExponentialHeightFog_1/hide")
    col.req(ucv, "vrun r.EyeAdaptationQuality 0")
    col.req(ucv, "vrun r.DefaultFeature.AutoExposure 0")
    col.req(ucv, "vrun r.ExposureOffset 6")
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.place(ucv, -160.0, -200.0, 0.0)
    col.look(ucv, cam, -160.0, -200.0, 0.0)
    rgb, depth = col.grab(ucv, cam)
    live, _seen, _w = col.warp_pair(rgb, depth)
    toned = match_tone(live, stored)
    cons = gap.load_checkpoint(CONS, "cuda")
    state = np.array([-1.6, -2.0, 0.52, 0, 0, 0, 0.0])
    goal = np.array([0.2, -2.0, 0.52])
    action = np.zeros((1, 10, 4), np.float32)
    action[0, :, 0] = 0.8

    def report(image: np.ndarray, label: str) -> None:
        with torch.no_grad():
            pred = ft.forward_maps(model, image[None]).detach().float().cpu().numpy()[0]
        da = ft.cones_np(pred)
        surv, _ = gap.score([cons], pred, state, goal, action, np.zeros(4, np.float32))
        print(
            f"{label} center {float(image[72:152, 72:152].mean()):.1f} "
            f"da {float(da[0]):.2f},{float(da[1]):.2f},{float(da[2]):.2f} "
            f"surv {float(surv[0]):.3f}",
            flush=True,
        )

    report(stored, "stored")
    report(live, "live")
    report(toned, "toned")


if __name__ == "__main__":
    main()
