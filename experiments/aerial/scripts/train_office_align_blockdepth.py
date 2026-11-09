#!/usr/bin/env python3
"""Pull the center-lane blockage in, and leave the open corridor where it is.

The aligned consequence drives the gap, then walks into the lane blockage
because that face still reads as about 1.7 m. Only those face frames are
shortened. Corridor frames are tied to the current prediction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/aerial-wam-v2/experiments/aerial/scripts")

import collect_office_train as col
import run_office_cons_loop as loop
import train_office_da3_far as far
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from office_bands import cones3_forward

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "da3_office_align_20261104/depth_office_cone_align.pt"
OUT = ART / "da3_office_align_20261104/depth_office_cone_align_block.pt"
GAP_CACHE = Path("/data/simworld/run/office_align_depthb_20261104/poses_v2.npz")
STEPS = 5


def grab(ucv, cam, x, y, yaw):
    col.place(ucv, x, y, yaw)
    col.look(ucv, cam, x, y, yaw)
    packed = col.grab(ucv, cam)
    if packed is None:
        return None
    rgb, depth = packed
    image, seen, warped = col.warp_pair(rgb, depth)
    gt56 = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
    return np.ascontiguousarray(image), gt56


def cones_of(model, rgb):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rgb), 4):
            pred = far.forward_maps(model, rgb[i:i + 4]).detach().float().cpu().numpy()
            out.append(np.stack([cones3_forward(p) for p in pred]))
    model.decoder.train()
    return np.concatenate(out)


def main() -> int:
    gap = np.load(GAP_CACHE)
    keep = np.isin(gap["kind"], ["gap", "offset"])
    gap_rgb = gap["rgb"][keep]
    gap_gt = gap["gt56"][keep]
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.ensure_dog(ucv)
    loop.boxes.save_restore(ucv)
    loop.restore_boxes(ucv)
    face_rgb, face_gt = [], []
    for x in (-320.0, -310.0, -300.0, -290.0, -275.0):
        got = grab(ucv, cam, x, -200.0, 0.0)
        if got is None:
            continue
        image, gt = got
        c = cones3_forward(gt)
        print(f"block x={x:.0f} gt={c[0]:.2f},{c[1]:.2f},{c[2]:.2f}", flush=True)
        if c[0] < 1.3:
            face_rgb.append(image)
            face_gt.append(gt)
    if len(face_rgb) < 2:
        print("BLOCK_TOO_FEW", flush=True)
        return 2
    face_rgb = np.stack(face_rgb)
    face_gt = np.stack(face_gt).astype(np.float32)
    wrap = DepthMinPredictor.from_checkpoint(str(INIT), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    model = wrap._model
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    src = torch.load(str(INIT), map_location="cpu", weights_only=False)
    teacher = cones_of(model, gap_rgb)
    before = cones_of(model, face_rgb)
    print(f"before block da {[round(float(v), 2) for v in before[:, 0]]}", flush=True)
    opt = torch.optim.AdamW((p for p in model.decoder.parameters() if p.requires_grad), lr=1e-6)
    rng = np.random.default_rng(5)
    saved = False
    for step in range(1, STEPS + 1):
        model.decoder.train()
        fi = rng.choice(len(face_rgb), size=min(2, len(face_rgb)), replace=False)
        gi = rng.choice(len(gap_rgb), size=4, replace=False)
        pred = far.forward_maps(model, np.concatenate([face_rgb[fi], gap_rgb[gi]]))
        gt_t = torch.as_tensor(np.concatenate([face_gt[fi], gap_gt[gi]]), dtype=torch.float32, device="cuda")
        # Teacher is the frozen forward cone, broadcast as a log penalty on the gap half.
        loss = 8.0 * far.far_near_loss(pred[: len(fi)], gt_t[: len(fi)])
        gap_pred = pred[len(fi):]
        teach = torch.as_tensor(teacher[gi, 0], dtype=torch.float32, device="cuda").clamp(0.05, 30)
        now = far.p20_fwd_t(gap_pred).clamp(0.05, 30)
        loss = loss + torch.relu((torch.log(now) - torch.log(teach)).abs() - math_log_tol()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 5 != 0 and step != STEPS:
            continue
        da = cones_of(model, face_rgb)
        gap_now = cones_of(model, gap_rgb)
        drop = float(np.median(teacher[:, 0] - gap_now[:, 0]))
        print(
            f"step {step} block {[round(float(v), 2) for v in da[:, 0]]} gap_drop {drop:.2f}",
            flush=True,
        )
        closer = float(np.mean(before[:, 0] - da[:, 0])) > 0.15
        if closer and drop < 0.45:
            far.save_ckpt(model, src, OUT, {"step": step, "gap_drop": drop})
            saved = True
            print("BLOCKDEPTH_SAVE", step, flush=True)
    if not saved:
        print("BLOCKDEPTH_NO", flush=True)
        return 2
    print("BLOCKDEPTH_OUT", OUT, flush=True)
    return 0


def math_log_tol() -> torch.Tensor:
    return torch.tensor(0.05, device="cuda")


if __name__ == "__main__":
    raise SystemExit(main())
