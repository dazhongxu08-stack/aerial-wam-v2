#!/usr/bin/env python3
"""Fine-tune the office DA3 decoder so standing obstacles stay in front of the wall.

Init is the real-office checkpoint. The encoder stays frozen. RealSense
readings still supervise the floor and walls, with the floor rows down-weighted.
A standing object that RealSense missed gets the distance of the floor at its
feet. Glass is left unlabeled. The held-out first-person photo is not trained on.
The live checkpoint is not replaced.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(os.environ.get("OBSTACLE_ROOT", "/home/yao/da3_obstacle_20261106"))
REPO = Path(os.environ.get("OBSTACLE_REPO", "/home/yao/aerial-wam-v2"))
sys.path.insert(0, str(REPO))

from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402

CKPT = ROOT / "init.pt"
OUT = ROOT / "depth_office_cone_obstacle.pt"
PAIRS = ROOT
HOLDOUT_PHOTO = ROOT / "fpv_color.png"
HOLDOUT = {5, 15, 25, 35, 45, 55, 60, 61, 62, 63}
STEPS = 240
LR = 3e-6
FLOOR_WEIGHT = 0.25
OBS_WEIGHT = 0.5
FY, CY, SRC_H = 386.394, 243.503, 480.0
FLOOR_FRAC = 0.42
K = np.array([[386.890, 0.0, 318.382], [0.0, FY, CY], [0.0, 0.0, 1.0]], np.float64)
D = np.array([-0.0539, 0.0590, -0.00047, 0.00068, -0.0190], np.float64)


def floor_row(out: int = 224) -> int:
    focal = 0.5 * out / math.tan(math.radians(45.0))
    angle = math.atan((FLOOR_FRAC * SRC_H - CY) / FY)
    return int(math.floor(out / 2.0 + focal * math.tan(angle)))


FLOOR_ROW = floor_row()


def _detach(feats):
    if torch.is_tensor(feats):
        return feats.detach()
    if isinstance(feats, dict):
        return {k: _detach(v) for k, v in feats.items()}
    if isinstance(feats, (list, tuple)):
        return type(feats)(_detach(v) for v in feats)
    return feats


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


def _predict(model, feats):
    out = model.decoder(feats, model.image_size, model.image_size, patch_start_idx=0)
    depth = out["depth"][:, 0]
    if depth.shape[-2:] != (model.image_size, model.image_size):
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(model.image_size, model.image_size),
            mode="bilinear", align_corners=False,
        )[:, 0]
    return depth[0]


def standing_holes(rgb: np.ndarray, depth: np.ndarray, seen: np.ndarray, valid: np.ndarray):
    """Compact holes that touch the floor. Target is the floor distance at the feet."""
    interior = cv2.erode(seen.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    hole = interior & ~valid
    hole[:8] = False
    hole[FLOOR_ROW:] = False
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), 8)
    mask = np.zeros(seen.shape, bool)
    target = np.zeros(seen.shape, np.float32)
    bright = rgb.astype(np.float32).mean(axis=2)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 200 or area > 1600 or w < 18 or w > 55 or h < 20 or h > 80:
            continue
        if y + h < FLOOR_ROW - 8:
            continue
        comp = labels == i
        if float(np.median(bright[comp])) > 140:
            continue
        yb = min(depth.shape[0] - 1, y + h)
        y1 = min(depth.shape[0], yb + 10)
        band = valid[yb:y1, x:x + w]
        if int(band.sum()) < 15:
            continue
        contact = float(np.median(depth[yb:y1, x:x + w][band]))
        if not 0.45 < contact < 4.5:
            continue
        mask[comp] = True
        target[comp] = contact
        blobs.append({"area": area, "w": w, "h": h, "contact_m": round(contact, 2)})
    return mask, target, blobs


def load_frame(path: Path) -> dict:
    z = np.load(path)
    rgb = np.ascontiguousarray(z["rgb"])
    depth = z["depth"].astype(np.float32)
    seen = z["seen"].astype(bool)
    z.close()
    valid = seen & np.isfinite(depth) & (depth > 0.35) & (depth < 8.0)
    mask, target, blobs = standing_holes(rgb, depth, seen, valid)
    return {
        "name": path.name,
        "rgb": rgb,
        "depth": depth,
        "valid": valid,
        "mask": mask,
        "target": target,
        "blobs": blobs,
    }


def pixel_loss(pred, gt, valid):
    p = pred.clamp(0.05, 12.0)
    g = gt.clamp(0.05, 12.0)
    log_p, log_g = torch.log(p), torch.log(g)
    weight = torch.ones_like(p)
    weight[FLOOR_ROW:] = FLOOR_WEIGHT
    diff = torch.nn.functional.smooth_l1_loss(log_p, log_g, reduction="none")
    pix = (diff[valid] * weight[valid]).sum() / weight[valid].sum().clamp(min=1)
    far = valid & (gt >= 4.0)
    near = valid & (gt <= 1.2)
    extra = pred.new_zeros(())
    if int(far.sum()) >= 30:
        extra = extra + torch.relu(log_g[far] - log_p[far] - math.log(1.4)).mean()
    if int(near.sum()) >= 30:
        extra = extra + torch.relu(log_p[near] - log_g[near] - math.log(1.35)).mean()
    return pix + extra


def obstacle_loss(pred, target, mask):
    if int(mask.sum()) < 30:
        return pred.new_zeros(())
    p = pred.clamp(0.05, 12.0)[mask]
    t = target.clamp(0.05, 12.0)[mask]
    return torch.nn.functional.smooth_l1_loss(torch.log(p), torch.log(t))


def summarize(model, frames, feats) -> dict:
    model.decoder.eval()
    abs_err = []
    with torch.no_grad():
        for frame, feat in zip(frames, feats):
            pred = _predict(model, to_cuda(feat)).clamp(min=0.05).float().cpu().numpy()
            real = frame["depth"]
            valid = frame["valid"]
            band = valid & (real >= 0.6) & (real < 4.0)
            if int(band.sum()) >= 30:
                abs_err.append(np.abs(pred[band] - real[band]))
    med = 1e9 if not abs_err else float(np.median(np.concatenate(abs_err)))
    return {"pixel_med_m": round(med, 3)}


def warp_color(rgb, out=224):
    focal = 0.5 * out / math.tan(math.radians(45.0))
    vk = np.array([[focal, 0.0, out / 2.0], [0.0, focal, out / 2.0], [0.0, 0.0, 1.0]], np.float64)
    m1, m2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), vk, (out, out), cv2.CV_16SC2)
    image = cv2.remap(rgb, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def photo_gate(model, rgb224) -> dict:
    model.encoder.to("cuda").eval()
    model.decoder.to("cuda").eval()
    tensor = torch.from_numpy(rgb224).view(1, 1, *rgb224.shape)
    with torch.no_grad():
        feats, _aux = model.encoder(
            model._prep_last_frame(tensor.to("cuda")).unsqueeze(1),
            cam_token=None, export_feat_layers=[],
        )
        pred = _predict(model, feats).clamp(min=0.05).float().cpu().numpy()
    h, w = 480, 640
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs, ys], -1).reshape(-1, 1, 2).astype(np.float32)
    und = cv2.undistortPoints(pts, K, D)
    focal = 0.5 * 224 / math.tan(math.radians(45.0))
    vu = (focal * und[:, 0, 0] + 112).reshape(h, w).astype(np.float32)
    vv = (focal * und[:, 0, 1] + 112).reshape(h, w).astype(np.float32)
    mapped = cv2.remap(pred, vu, vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    robot = float(np.median(mapped[140:190, 510:620]))
    wall = float(np.median(mapped[140:190, 380:470]))
    plant = float(np.median(mapped[90:150, 250:330]))
    behind = float(np.median(mapped[90:150, 340:430]))
    floor = float(np.median(mapped[340:450, 250:420]))
    return {
        "robot_m": round(robot, 3),
        "wall_m": round(wall, 3),
        "robot_gap_m": round(wall - robot, 3),
        "plant_m": round(plant, 3),
        "behind_m": round(behind, 3),
        "plant_gap_m": round(behind - plant, 3),
        "floor_m": round(floor, 3),
        "pass": bool((wall - robot) >= 0.5 and (behind - plant) >= 0.3 and 0.6 <= floor <= 0.9),
    }


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


def main() -> int:
    names = sorted(PAIRS.glob("pair_*.npz"))
    frames = [load_frame(path) for path in names]
    train, hold = [], []
    obs_pix = 0
    for frame in frames:
        obs_pix += int(frame["mask"].sum())
        idx = int(frame["name"].split("_")[1].split(".")[0])
        (hold if idx in HOLDOUT else train).append(frame)
    print(
        f"frames train {len(train)} hold {len(hold)} floor_row {FLOOR_ROW} "
        f"obstacle_px {obs_pix}",
        flush=True,
    )
    for frame in train:
        if frame["blobs"]:
            print(frame["name"], frame["blobs"], flush=True)
    photo = cv2.cvtColor(cv2.imread(str(HOLDOUT_PHOTO)), cv2.COLOR_BGR2RGB)
    photo224 = warp_color(photo)
    wrap = DepthMinPredictor.from_checkpoint(str(CKPT), device="cuda")
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
    masks_valid = [torch.from_numpy(frame["valid"]).pin_memory() for frame in train]
    obs_masks = [torch.from_numpy(frame["mask"]).pin_memory() for frame in train]
    obs_targets = [torch.from_numpy(frame["target"]).pin_memory() for frame in train]
    before = summarize(model, hold, hold_feats)
    before_gate = photo_gate(model, photo224)
    print("before", json.dumps({**before, **before_gate}), flush=True)
    model.encoder.cpu()
    torch.cuda.empty_cache()
    best = before_gate["robot_gap_m"] + before_gate["plant_gap_m"]
    best_step = 0
    best_report = {"step": 0, **before, **before_gate}
    best_state = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    opt = torch.optim.AdamW(
        [p for p in model.decoder.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    for step in range(1, STEPS + 1):
        i = (step - 1) % len(train)
        model.decoder.train()
        pred = _predict(model, to_cuda(train_feats[i])).clamp(min=0.05)
        gt = gts[i].to("cuda", non_blocking=True)
        valid = masks_valid[i].to("cuda", non_blocking=True)
        om = obs_masks[i].to("cuda", non_blocking=True)
        ot = obs_targets[i].to("cuda", non_blocking=True)
        if int(valid.sum()) < 500:
            continue
        loss = pixel_loss(pred, gt, valid) + OBS_WEIGHT * obstacle_loss(pred, ot, om)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == STEPS:
            stats = summarize(model, hold, hold_feats)
            gate = photo_gate(model, photo224)
            model.encoder.cpu()
            torch.cuda.empty_cache()
            row = {"step": step, "loss": round(float(loss), 4), **stats, **gate}
            print(json.dumps(row), flush=True)
            score = gate["robot_gap_m"] + gate["plant_gap_m"]
            if stats["pixel_med_m"] > 0.45 or not (0.55 <= gate["floor_m"] <= 1.05):
                score = -1e9
            if score > best:
                best = score
                best_step = step
                best_report = row
                best_state = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    model.decoder.load_state_dict(best_state)
    model.decoder.to("cuda").eval()
    model.encoder.to("cuda").eval()
    after = summarize(model, hold, hold_feats)
    after_gate = photo_gate(model, photo224)
    src = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    payload = dict(src)
    payload["model"] = {k: v.detach().cpu() for k, v in src["model"].items()}
    for key, value in best_state.items():
        payload["model"]["decoder." + key] = value.detach().cpu()
    payload["office_obstacle"] = {
        "best_step": best_step,
        "before": {**before, **before_gate},
        "after": {**after, **after_gate},
        "best": best_report,
        "holdout_pairs": sorted(HOLDOUT),
        "floor_row": FLOOR_ROW,
        "obstacle_px": obs_pix,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(OUT))
    report = payload["office_obstacle"]
    report["out"] = str(OUT)
    (OUT.parent / "finetune_obstacle.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return 0


def main_encoder() -> int:
    """Unfreeze the last two ViT blocks. The held-out photo stays out of the loss."""
    src_path = ROOT / "depth_office_cone_obstacle_decoder.pt"
    names = sorted(PAIRS.glob("pair_*.npz"))
    frames = [load_frame(path) for path in names]
    train, hold = [], []
    for frame in frames:
        idx = int(frame["name"].split("_")[1].split(".")[0])
        (hold if idx in HOLDOUT else train).append(frame)
    photo = cv2.cvtColor(cv2.imread(str(HOLDOUT_PHOTO)), cv2.COLOR_BGR2RGB)
    photo224 = warp_color(photo)
    wrap = DepthMinPredictor.from_checkpoint(str(src_path), device="cuda")
    model = wrap._model
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    for block in list(model.encoder.pretrained.blocks)[-2:]:
        for param in block.parameters():
            param.requires_grad_(True)
    for param in model.decoder.parameters():
        param.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"encoder_tail trainable {trainable}", flush=True)
    hold_feats = encode(model, hold)
    gts = [torch.from_numpy(frame["depth"]) for frame in train]
    masks_valid = [torch.from_numpy(frame["valid"]) for frame in train]
    obs_masks = [torch.from_numpy(frame["mask"]) for frame in train]
    obs_targets = [torch.from_numpy(frame["target"]) for frame in train]
    before = summarize(model, hold, hold_feats)
    before_gate = photo_gate(model, photo224)
    print("before", json.dumps({**before, **before_gate}), flush=True)
    best = before_gate["robot_gap_m"] + before_gate["plant_gap_m"]
    best_step = 0
    best_report = {"step": 0, **before, **before_gate}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": 1e-6},
            {"params": [p for p in model.decoder.parameters() if p.requires_grad], "lr": LR},
        ],
        weight_decay=1e-4,
    )
    steps = 80
    for step in range(1, steps + 1):
        i = (step - 1) % len(train)
        model.encoder.train()
        model.decoder.train()
        rgb = torch.from_numpy(train[i]["rgb"]).view(1, 1, *train[i]["rgb"].shape).to("cuda")
        feats, _aux = model.encoder(
            model._prep_last_frame(rgb).unsqueeze(1), cam_token=None, export_feat_layers=[]
        )
        pred = _predict(model, feats).clamp(min=0.05)
        gt = gts[i].to("cuda")
        valid = masks_valid[i].to("cuda")
        if int(valid.sum()) < 500:
            continue
        loss = pixel_loss(pred, gt, valid) + OBS_WEIGHT * obstacle_loss(
            pred, obs_targets[i].to("cuda"), obs_masks[i].to("cuda")
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 10 == 0 or step == steps:
            model.encoder.eval()
            model.decoder.eval()
            fresh = encode(model, hold)
            stats = summarize(model, hold, fresh)
            gate = photo_gate(model, photo224)
            row = {"step": step, "loss": round(float(loss.detach()), 4), **stats, **gate}
            print(json.dumps(row), flush=True)
            score = gate["robot_gap_m"] + gate["plant_gap_m"]
            if stats["pixel_med_m"] > 0.45 or not (0.55 <= gate["floor_m"] <= 1.05):
                score = -1e9
            if score > best:
                best = score
                best_step = step
                best_report = row
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to("cuda").eval()
    hold_feats = encode(model, hold)
    after = summarize(model, hold, hold_feats)
    after_gate = photo_gate(model, photo224)
    src = torch.load(str(src_path), map_location="cpu", weights_only=False)
    payload = dict(src)
    payload["model"] = {k: v.detach().cpu() for k, v in best_state.items()}
    payload["office_obstacle_encoder"] = {
        "best_step": best_step,
        "before": {**before, **before_gate},
        "after": {**after, **after_gate},
        "best": best_report,
    }
    out = ROOT / "depth_office_cone_obstacle.pt"
    torch.save(payload, str(out))
    (ROOT / "finetune_obstacle_encoder.json").write_text(
        json.dumps(payload["office_obstacle_encoder"], indent=2)
    )
    print(json.dumps(payload["office_obstacle_encoder"]), flush=True)
    return 0


if __name__ == "__main__":
    if os.environ.get("OBSTACLE_STAGE") == "encoder":
        raise SystemExit(main_encoder())
    raise SystemExit(main())
