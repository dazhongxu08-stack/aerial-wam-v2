#!/usr/bin/env python3
"""Teach the office DA3 decoder that a close object stays in front of the wall.

Init is the live real-office checkpoint. The previous obstacle run is not
the init and is not overwritten. The encoder stays frozen unless the close-box
gate still fails. RealSense depth supervises the new close-box frame, including
the rows below the floor line where the crate actually sits. Old office pairs
keep a light loss so the walls and floor do not drift. The hallway photo is
reported and is not trained on. Nothing is copied onto the car.
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

ROOT = Path(os.environ.get("ORDER_ROOT", "/home/yao/da3_order_20261106"))
PAIRS = Path(os.environ.get("ORDER_PAIRS", "/home/yao/da3_obstacle_20261106"))
REPO = Path(os.environ.get("ORDER_REPO", "/home/yao/aerial-wam-v2"))
sys.path.insert(0, str(REPO))

from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402

CKPT = Path(os.environ.get("ORDER_CKPT", str(PAIRS / "init.pt")))
OUT = ROOT / "depth_office_cone_order.pt"
BOX_PHOTO = ROOT / "box_color.png"
HALL_PHOTO = PAIRS / "fpv_color.png"
HOLDOUT = {5, 15, 25, 35, 45, 55, 60, 61, 62, 63}
STEPS = 180
LR = 3e-6
FLOOR_ROW = 99
FY, CY, SRC_H = 386.394, 243.503, 480.0
K = np.array([[386.890, 0.0, 318.382], [0.0, FY, CY], [0.0, 0.0, 1.0]], np.float64)
D = np.array([-0.0539, 0.0590, -0.00047, 0.00068, -0.0190], np.float64)


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


def load_frame(path: Path, order: bool) -> dict:
    z = np.load(path)
    rgb = np.ascontiguousarray(z["rgb"])
    depth = z["depth"].astype(np.float32)
    seen = z["seen"].astype(bool)
    z.close()
    valid = seen & np.isfinite(depth) & (depth > 0.35) & (depth < 8.0)
    return {"name": path.name, "rgb": rgb, "depth": depth, "valid": valid, "order": order}


def warp_color(rgb, out=224):
    focal = 0.5 * out / math.tan(math.radians(45.0))
    vk = np.array([[focal, 0.0, out / 2.0], [0.0, focal, out / 2.0], [0.0, 0.0, 1.0]], np.float64)
    m1, m2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), vk, (out, out), cv2.CV_16SC2)
    return cv2.remap(rgb, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def pixel_loss(pred, gt, valid, order: bool):
    p = pred.clamp(0.05, 12.0)
    g = gt.clamp(0.05, 12.0)
    log_p, log_g = torch.log(p), torch.log(g)
    weight = torch.ones_like(p)
    if order:
        rows = torch.arange(p.shape[0], device=p.device)[:, None]
        close = (rows >= 104) & (rows < 142) & (gt < 0.92) & (gt > 0.40)
        weight = torch.where(close, weight * 4.0, weight)
        weight = torch.where(rows >= 150, weight * 0.5, weight)
    else:
        weight[FLOOR_ROW:] = 0.25
    diff = torch.nn.functional.smooth_l1_loss(log_p, log_g, reduction="none")
    return (diff[valid] * weight[valid]).sum() / weight[valid].sum().clamp(min=1)


def order_loss(pred, gt, valid):
    """The close crate has to stay nearer than the wall above it."""
    rows = torch.arange(pred.shape[0], device=pred.device)[:, None]
    cols = torch.arange(pred.shape[1], device=pred.device)[None, :]
    wall = valid & (rows >= 36) & (rows < 100) & (gt > 0.80) & (gt < 1.25)
    obj = (
        valid & (rows >= 108) & (rows < 140) & (cols >= 56) & (cols < 168)
        & (gt > 0.40) & (gt < 0.85)
    )
    if int(wall.sum()) < 50 or int(obj.sum()) < 30:
        return pred.new_zeros(())
    wall_log = torch.log(pred.clamp(0.05, 12.0)[wall]).mean()
    obj_log = torch.log(pred.clamp(0.05, 12.0)[obj]).mean()
    return torch.relu(obj_log - wall_log + math.log(1.20))


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


def summarize(model, frames, feats) -> dict:
    model.decoder.eval()
    abs_err = []
    with torch.no_grad():
        for frame, feat in zip(frames, feats):
            pred = _predict(model, to_cuda(feat)).clamp(min=0.05).float().cpu().numpy()
            real = frame["depth"]
            band = frame["valid"] & (real >= 0.6) & (real < 4.0)
            if int(band.sum()) >= 30:
                abs_err.append(np.abs(pred[band] - real[band]))
    med = 1e9 if not abs_err else float(np.median(np.concatenate(abs_err)))
    return {"pixel_med_m": round(med, 3)}


def _map_photo(model, rgb224):
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
    return cv2.remap(pred, vu, vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _med(mapped, y0, y1, x0, x1):
    part = mapped[y0:y1, x0:x1]
    vals = part[np.isfinite(part) & (part > 0.05) & (part < 30)]
    if vals.size < 20:
        return None
    return float(np.median(vals))


def box_gate(model, rgb224) -> dict:
    mapped = _map_photo(model, rgb224)
    wall = _med(mapped, 80, 180, 200, 480)
    box = _med(mapped, 230, 330, 200, 500)
    vacuum = _med(mapped, 250, 330, 0, 130)
    right = _med(mapped, 250, 330, 540, 639)
    floor = _med(mapped, 400, 460, 200, 480)
    left_wall = _med(mapped, 160, 230, 0, 130)
    right_wall = _med(mapped, 160, 230, 540, 639)
    ok = (
        wall is not None and box is not None and floor is not None
        and box < wall and box < 1.0 and 0.75 <= wall <= 1.35 and 0.40 <= floor <= 0.95
    )
    return {
        "box_m": None if box is None else round(box, 3),
        "wall_m": None if wall is None else round(wall, 3),
        "box_gap_m": None if box is None or wall is None else round(wall - box, 3),
        "vacuum_m": None if vacuum is None else round(vacuum, 3),
        "right_m": None if right is None else round(right, 3),
        "left_wall_m": None if left_wall is None else round(left_wall, 3),
        "right_wall_m": None if right_wall is None else round(right_wall, 3),
        "floor_m": None if floor is None else round(floor, 3),
        "pass": bool(ok),
    }


def hall_gate(model, rgb224) -> dict:
    mapped = _map_photo(model, rgb224)
    robot = _med(mapped, 140, 190, 510, 620)
    wall = _med(mapped, 140, 190, 380, 470)
    floor = _med(mapped, 340, 450, 250, 420)
    return {
        "hall_robot_m": None if robot is None else round(robot, 3),
        "hall_wall_m": None if wall is None else round(wall, 3),
        "hall_floor_m": None if floor is None else round(floor, 3),
    }


def eligible(stats, gate) -> bool:
    wall, floor = gate["wall_m"], gate["floor_m"]
    return (
        stats["pixel_med_m"] <= 0.42
        and wall is not None and floor is not None
        and 0.70 <= wall <= 1.40
        and 0.40 <= floor <= 1.00
    )


def score_of(gate) -> float:
    gap = gate["box_gap_m"] if gate["box_gap_m"] is not None else -5.0
    right_gap = 0.0
    if gate["right_m"] is not None and gate["wall_m"] is not None:
        right_gap = gate["wall_m"] - gate["right_m"]
    penalty = 0.0
    for key, limit in (("vacuum_m", 1.3), ("left_wall_m", 1.5), ("right_wall_m", 1.5)):
        value = gate[key]
        if value is not None and value > limit:
            penalty += value - limit
    return gap + 0.5 * right_gap - penalty


def evaluate(model, hold, hold_feats, box224, hall224):
    stats = summarize(model, hold, hold_feats)
    gate = box_gate(model, box224)
    hall = hall_gate(model, hall224)
    model.encoder.cpu()
    torch.cuda.empty_cache()
    return {**stats, **gate, **hall}


def pack(src_path, state, report, decoder_only: bool):
    src = torch.load(str(src_path), map_location="cpu", weights_only=False)
    payload = dict(src)
    if decoder_only:
        payload["model"] = {k: v.detach().cpu() for k, v in src["model"].items()}
        for key, value in state.items():
            payload["model"]["decoder." + key] = value.detach().cpu()
    else:
        payload["model"] = {k: v.detach().cpu() for k, v in state.items()}
    payload["office_order"] = report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(OUT))
    (ROOT / "finetune_order.json").write_text(json.dumps(report, indent=2))


def main() -> int:
    old_paths = sorted(PAIRS.glob("pair_*.npz"))
    train, hold = [], []
    for path in old_paths:
        idx = int(path.name.split("_")[1].split(".")[0])
        frame = load_frame(path, order=False)
        (hold if idx in HOLDOUT else train).append(frame)
    box_frames = [load_frame(ROOT / "box_pair.npz", True), load_frame(ROOT / "box_pair_flip.npz", True)]
    near = box_frames[0]["valid"] & (box_frames[0]["depth"] < 0.85)
    near[150:] = False
    print(
        f"old_train {len(train)} hold {len(hold)} box_near_px {int(near.sum())}",
        flush=True,
    )
    if int(near.sum()) < 200:
        print("box frame has no close object labels", flush=True)
        return 1
    box_bgr = cv2.imread(str(BOX_PHOTO))
    hall_bgr = cv2.imread(str(HALL_PHOTO))
    box224 = warp_color(cv2.cvtColor(box_bgr, cv2.COLOR_BGR2RGB))
    hall224 = warp_color(cv2.cvtColor(hall_bgr, cv2.COLOR_BGR2RGB))
    wrap = DepthMinPredictor.from_checkpoint(str(CKPT), device="cuda")
    model = wrap._model
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    for param in model.decoder.parameters():
        param.requires_grad_(True)
    train_feats = encode(model, train)
    hold_feats = encode(model, hold)
    box_feats = encode(model, box_frames)
    model.encoder.cpu()
    torch.cuda.empty_cache()
    before = evaluate(model, hold, hold_feats, box224, hall224)
    print("before", json.dumps(before), flush=True)
    best = score_of(before)
    best_step = 0
    best_report = {"step": 0, **before}
    best_state = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    best_full = None
    opt = torch.optim.AdamW(
        [p for p in model.decoder.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    gts = [torch.from_numpy(frame["depth"]) for frame in train]
    valids = [torch.from_numpy(frame["valid"]) for frame in train]
    box_gts = [torch.from_numpy(frame["depth"]) for frame in box_frames]
    box_valids = [torch.from_numpy(frame["valid"]) for frame in box_frames]
    for step in range(1, STEPS + 1):
        use_box = step % 3 == 0
        model.decoder.train()
        if use_box:
            i = (step // 3) % len(box_frames)
            pred = _predict(model, to_cuda(box_feats[i])).clamp(min=0.05)
            gt = box_gts[i].to("cuda")
            valid = box_valids[i].to("cuda")
            loss = pixel_loss(pred, gt, valid, True) + 2.0 * order_loss(pred, gt, valid)
        else:
            i = (step - 1) % len(train)
            pred = _predict(model, to_cuda(train_feats[i])).clamp(min=0.05)
            gt = gts[i].to("cuda")
            valid = valids[i].to("cuda")
            if int(valid.sum()) < 500:
                continue
            loss = 0.35 * pixel_loss(pred, gt, valid, False)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == STEPS:
            row = {"step": step, "loss": round(float(loss.detach()), 4)}
            row.update(evaluate(model, hold, hold_feats, box224, hall224))
            print(json.dumps(row), flush=True)
            if eligible(row, row) and score_of(row) > best:
                best = score_of(row)
                best_step = step
                best_report = row
                best_state = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    model.decoder.load_state_dict(best_state)
    model.decoder.to("cuda").eval()
    model.encoder.to("cuda").eval()
    after = evaluate(model, hold, hold_feats, box224, hall224)
    report = {
        "stage": "decoder",
        "best_step": best_step,
        "before": before,
        "after": after,
        "best": best_report,
        "trained_on_box_frame": True,
        "hallway_trained": False,
        "init": str(CKPT),
    }
    pack(CKPT, best_state, report, decoder_only=True)
    print(json.dumps(report), flush=True)
    if after.get("pass"):
        return 0
    return encoder_tail(model, train, hold, box_frames, box224, hall224, after)


def encoder_tail(model, train, hold, box_frames, box224, hall224, decoder_after) -> int:
    """Last two ViT blocks, only because the frozen decoder left the box behind the wall."""
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    for block in list(model.encoder.pretrained.blocks)[-2:]:
        for param in block.parameters():
            param.requires_grad_(True)
    for param in model.decoder.parameters():
        param.requires_grad_(True)
    model.to("cuda")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"encoder_tail trainable {trainable}", flush=True)
    hold_feats = encode(model, hold)
    before = evaluate(model, hold, hold_feats, box224, hall224)
    print("encoder_before", json.dumps(before), flush=True)
    best = score_of(before)
    best_step = 0
    best_report = {"step": 0, **before}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": 1e-6},
            {"params": [p for p in model.decoder.parameters() if p.requires_grad], "lr": LR},
        ],
        weight_decay=1e-4,
    )
    steps = 40
    for step in range(1, steps + 1):
        use_box = step % 2 == 0
        model.encoder.train()
        model.decoder.train()
        frame = box_frames[(step // 2) % len(box_frames)] if use_box else train[(step - 1) % len(train)]
        rgb = torch.from_numpy(frame["rgb"]).view(1, 1, *frame["rgb"].shape).to("cuda")
        feats, _aux = model.encoder(
            model._prep_last_frame(rgb).unsqueeze(1), cam_token=None, export_feat_layers=[]
        )
        pred = _predict(model, feats).clamp(min=0.05)
        gt = torch.from_numpy(frame["depth"]).to("cuda")
        valid = torch.from_numpy(frame["valid"]).to("cuda")
        if int(valid.sum()) < 500:
            continue
        if use_box:
            loss = pixel_loss(pred, gt, valid, True) + 2.0 * order_loss(pred, gt, valid)
        else:
            loss = 0.35 * pixel_loss(pred, gt, valid, False)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 10 == 0 or step == steps:
            model.encoder.eval()
            model.decoder.eval()
            fresh = encode(model, hold)
            row = {"step": step, "loss": round(float(loss.detach()), 4)}
            row.update(evaluate(model, hold, fresh, box224, hall224))
            print(json.dumps(row), flush=True)
            if eligible(row, row) and score_of(row) > best:
                best = score_of(row)
                best_step = step
                best_report = row
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to("cuda").eval()
    hold_feats = encode(model, hold)
    after = evaluate(model, hold, hold_feats, box224, hall224)
    report = {
        "stage": "encoder_tail",
        "best_step": best_step,
        "decoder_after": decoder_after,
        "before": before,
        "after": after,
        "best": best_report,
        "trained_on_box_frame": True,
        "hallway_trained": False,
        "init": str(CKPT),
    }
    src = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    payload = dict(src)
    payload["model"] = {k: v.detach().cpu() for k, v in best_state.items()}
    payload["office_order"] = report
    torch.save(payload, str(OUT))
    (ROOT / "finetune_order.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
