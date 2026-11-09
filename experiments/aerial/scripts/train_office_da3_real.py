#!/usr/bin/env python3
"""Fine-tune the live office depth decoder on RealSense pairs.

Encoder stays frozen. Init is the chassis yaw head, not the sim far head.
The live robot file is not replaced. A spread of pairs is held out.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/simworld/run")

from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402


def _detach(feats):
    if torch.is_tensor(feats):
        return feats.detach()
    if isinstance(feats, dict):
        return {k: _detach(v) for k, v in feats.items()}
    if isinstance(feats, (list, tuple)):
        return type(feats)(_detach(v) for v in feats)
    return feats


def _predict(model, feats):
    out = model.decoder(feats, model.image_size, model.image_size, patch_start_idx=0)
    depth = out["depth"][:, 0]
    if depth.shape[-2:] != (model.image_size, model.image_size):
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(model.image_size, model.image_size),
            mode="bilinear", align_corners=False,
        )[:, 0]
    return depth[0]

ROOT = Path("/data/simworld/run/da3_pairs")
CKPT = Path(
    "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/"
    "da3_office_cone_yaw_live_20261103/depth_office_cone_yaw.pt"
)
OUT = Path(
    "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/"
    "da3_office_real_20261103/depth_office_cone_real.pt"
)
HOLDOUT = {5, 15, 25, 35, 45, 55, 60, 61, 62, 63}
STEPS = 180
LR = 3e-6


def load_frame(path: Path) -> dict:
    z = np.load(path)
    rgb = np.ascontiguousarray(z["rgb"])
    depth = z["depth"].astype(np.float32)
    seen = z["seen"].astype(bool)
    pose = [round(float(v), 3) for v in z["pose"]]
    z.close()
    valid = seen & np.isfinite(depth) & (depth > 0.35) & (depth < 8.0)
    return {"name": path.name, "rgb": rgb, "depth": depth, "valid": valid, "pose": pose}


def to_cpu(feats):
    if torch.is_tensor(feats):
        return feats.detach().cpu()
    if isinstance(feats, dict):
        return {k: to_cpu(v) for k, v in feats.items()}
    if isinstance(feats, (list, tuple)):
        return type(feats)(to_cpu(v) for v in feats)
    return feats


def to_cuda(feats):
    if torch.is_tensor(feats):
        return feats.to("cuda")
    if isinstance(feats, dict):
        return {k: to_cuda(v) for k, v in feats.items()}
    if isinstance(feats, (list, tuple)):
        return type(feats)(to_cuda(v) for v in feats)
    return feats


def encode(model, frames):
    encoded = []
    for frame in frames:
        rgb = torch.from_numpy(frame["rgb"]).view(1, 1, *frame["rgb"].shape).to("cuda")
        with torch.no_grad():
            feats, _aux = model.encoder(
                model._prep_last_frame(rgb).unsqueeze(1), cam_token=None, export_feat_layers=[]
            )
        encoded.append(to_cpu(_detach(feats)))
    return encoded


def pixel_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    p = pred.clamp(0.05, 12.0)
    g = gt.clamp(0.05, 12.0)
    pix = torch.nn.functional.smooth_l1_loss(torch.log(p[valid]), torch.log(g[valid]))
    far = valid & (gt >= 4.0)
    near = valid & (gt <= 1.2)
    extra = pred.new_zeros(())
    if int(far.sum()) >= 30:
        # Far real pixels may be a little short, not collapsed to a nearby wall.
        extra = extra + torch.relu(torch.log(g[far]) - torch.log(p[far]) - math.log(1.4)).mean()
    if int(near.sum()) >= 30:
        extra = extra + torch.relu(torch.log(p[near]) - torch.log(g[near]) - math.log(1.35)).mean()
    return pix + extra


def summarize(model, frames, feats) -> dict:
    model.decoder.eval()
    abs_err = []
    far_short = 0
    far_n = 0
    near_long = 0
    near_n = 0
    with torch.no_grad():
        for frame, feat in zip(frames, feats):
            pred = _predict(model, to_cuda(feat)).clamp(min=0.05).float().cpu().numpy()
            real = frame["depth"]
            valid = frame["valid"]
            band = valid & (real >= 0.6) & (real < 4.0)
            if int(band.sum()) >= 30:
                abs_err.append(np.abs(pred[band] - real[band]))
            far = valid & (real >= 4.0)
            far_n += int(far.sum())
            far_short += int((pred[far] < 2.0).sum()) if far.any() else 0
            near = valid & (real <= 1.2)
            near_n += int(near.sum())
            near_long += int((pred[near] > real[near] * 1.5).sum()) if near.any() else 0
    med = 1e9 if not abs_err else float(np.median(np.concatenate(abs_err)))
    return {
        "pixel_med_m": round(med, 3),
        "far_short_frac": None if far_n == 0 else round(far_short / far_n, 3),
        "near_long_frac": None if near_n == 0 else round(near_long / near_n, 3),
    }


def main() -> int:
    names = sorted(ROOT.glob("pair_*.npz"))
    frames = [load_frame(path) for path in names]
    train, hold = [], []
    for frame in frames:
        idx = int(frame["name"].split("_")[1].split(".")[0])
        (hold if idx in HOLDOUT else train).append(frame)
    print(f"frames train {len(train)} hold {len(hold)}", flush=True)
    wrap = DepthMinPredictor.from_checkpoint(str(CKPT), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    model = wrap._model
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    for param in model.decoder.parameters():
        param.requires_grad_(True)
    train_feats = encode(model, train)
    hold_feats = encode(model, hold)
    model.encoder.cpu()
    torch.cuda.empty_cache()
    gts = [torch.from_numpy(frame["depth"]).pin_memory() for frame in train]
    masks = [torch.from_numpy(frame["valid"]).pin_memory() for frame in train]
    before = summarize(model, hold, hold_feats)
    print("holdout_before", json.dumps(before), flush=True)
    best = before["pixel_med_m"]
    best_step = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    opt = torch.optim.AdamW(
        [p for p in model.decoder.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    for step in range(1, STEPS + 1):
        i = (step - 1) % len(train)
        model.decoder.train()
        pred = _predict(model, to_cuda(train_feats[i])).clamp(min=0.05)
        gt = gts[i].to("cuda", non_blocking=True)
        valid = masks[i].to("cuda", non_blocking=True)
        if int(valid.sum()) < 500:
            continue
        loss = pixel_loss(pred, gt, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 15 == 0:
            stats = summarize(model, hold, hold_feats)
            print(json.dumps({"step": step, "loss": round(float(loss), 4), **stats}), flush=True)
            if stats["pixel_med_m"] < best:
                best = stats["pixel_med_m"]
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    model.decoder.load_state_dict(best_state)
    model.decoder.to("cuda").eval()
    after = summarize(model, hold, hold_feats)
    src = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    payload = dict(src)
    payload["model"] = {k: v.detach().cpu() for k, v in src["model"].items()}
    for key, value in best_state.items():
        payload["model"]["decoder." + key] = value.detach().cpu()
    payload["office_real"] = {"best_step": best_step, "before": before, "after": after, "holdout": sorted(HOLDOUT)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(OUT))
    report = {"best_step": best_step, "before": before, "after": after, "out": str(OUT), "installed": False}
    (OUT.parent / "finetune_real.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
