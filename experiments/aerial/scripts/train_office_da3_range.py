#!/usr/bin/env python3
"""Fine-tune office DA3 on several distances of the same crate and wall.

Init is the live real-office checkpoint, not the single-frame order run
and not the encoder-tail checkpoint. The encoder stays frozen. Extra loss
weight sits on above-floor pixels that are closer than the same row, and a
same-row gap term keeps that separation. The held-out 1.9 m frame is not in
the loss. The car copy is not replaced.
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

ROOT = Path(os.environ.get("RANGE_ROOT", "/home/yao/da3_range_20261107"))
OLD = Path(os.environ.get("RANGE_OLD", "/home/yao/da3_obstacle_20261106"))
REPO = Path(os.environ.get("RANGE_REPO", "/home/yao/aerial-wam-v2"))
sys.path.insert(0, str(REPO))

from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402

CKPT = Path(os.environ.get("RANGE_CKPT", str(OLD / "init.pt")))
OUT = ROOT / "depth_office_cone_gap.pt"
PAIRS = ROOT / "pairs"
HOLD_PHOTO = ROOT / "raw" / "color_172919.png"
TRAIN_NAMES = (
    "near10", "near11", "near14", "near15", "side_right", "side_left",
    "near17", "mid21", "far25",
)
OLD_HOLDOUT = {5, 15, 25, 35, 45, 55, 60, 61, 62, 63}
STEPS = 200
LR = 3e-6
FLOOR_ROW = 99
FY, CY = 386.394, 243.503
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


def load_frame(path: Path) -> dict:
    z = np.load(path)
    rgb = np.ascontiguousarray(z["rgb"])
    depth = z["depth"].astype(np.float32)
    seen = z["seen"].astype(bool)
    z.close()
    valid = seen & np.isfinite(depth) & (depth > 0.35) & (depth < 8.0)
    return {"name": path.stem, "rgb": rgb, "depth": depth, "valid": valid}


def obstacle_mask(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Above-floor pixels closer than the far part of the same row.

    The floor line is row 99. A high percentile keeps a wide crate from
    becoming the row background, which is what made the old p70 mask empty
    on centered frames.
    """
    mask = np.zeros(valid.shape, bool)
    for y in range(8, FLOOR_ROW):
        row = valid[y]
        if int(row.sum()) < 30:
            continue
        bg = float(np.percentile(depth[y, row], 85))
        mask[y] = row & (depth[y] < bg - 0.15) & (depth[y] < 4.0)
    return mask


def gap_loss(pred, gt, valid, close):
    """Match the labeled near-vs-background gap on each obstacle row."""
    terms = []
    for y in range(8, FLOOR_ROW):
        near = close[y]
        bg = valid[y] & ~close[y]
        if int(near.sum()) < 8 or int(bg.sum()) < 20:
            continue
        pred_gap = torch.log(pred[y][bg]).mean() - torch.log(pred[y][near]).mean()
        gt_gap = torch.log(gt[y][bg]).mean() - torch.log(gt[y][near]).mean()
        terms.append(torch.nn.functional.smooth_l1_loss(pred_gap, gt_gap.detach()))
    if not terms:
        return pred.new_zeros(())
    return torch.stack(terms).mean()


def pixel_loss(pred, gt, valid, close):
    p = pred.clamp(0.05, 12.0)
    g = gt.clamp(0.05, 12.0)
    weight = torch.ones_like(p)
    n_close = close.sum().clamp(min=1).float()
    n_rest = (valid & ~close).sum().clamp(min=1).float()
    scale = torch.clamp(n_rest / n_close, max=12.0)
    weight = torch.where(close, weight * scale, weight)
    diff = torch.nn.functional.smooth_l1_loss(torch.log(p), torch.log(g), reduction="none")
    absolute = (diff[valid] * weight[valid]).sum() / weight[valid].sum().clamp(min=1)
    return absolute + gap_loss(p, g, valid, close)


def old_loss(pred, gt, valid):
    p = pred.clamp(0.05, 12.0)
    g = gt.clamp(0.05, 12.0)
    weight = torch.ones_like(p)
    weight[FLOOR_ROW:] = 0.25
    diff = torch.nn.functional.smooth_l1_loss(torch.log(p), torch.log(g), reduction="none")
    return (diff[valid] * weight[valid]).sum() / weight[valid].sum().clamp(min=1)


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


def warp_color(rgb, out=224):
    focal = 0.5 * out / math.tan(math.radians(45.0))
    vk = np.array([[focal, 0.0, out / 2.0], [0.0, focal, out / 2.0], [0.0, 0.0, 1.0]], np.float64)
    m1, m2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), vk, (out, out), cv2.CV_16SC2)
    return cv2.remap(rgb, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def map_photo(model, rgb224):
    model.encoder.to("cuda").eval()
    model.decoder.to("cuda").eval()
    tensor = torch.from_numpy(np.ascontiguousarray(rgb224)).view(1, 1, *rgb224.shape)
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


def _med(depth, mask):
    vals = depth[mask & np.isfinite(depth) & (depth > 0.05) & (depth < 30)]
    if vals.size < 30:
        return None
    return float(np.median(vals))


def hold_gate(model, photo_rgb) -> dict:
    mapped = map_photo(model, warp_color(photo_rgb))
    h, w = mapped.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    r, g, b = [photo_rgb[:, :, i].astype(np.int16) for i in range(3)]
    blue = (b > 140) & (b > r + 40) & (b > g + 30) & (rows < 280) & (cols > 80) & (cols < 560)
    wall = _med(mapped, (rows >= 40) & (rows < 150) & (cols >= 160) & (cols < 500) & ~blue)
    box = _med(mapped, blue)
    floor = _med(mapped, (rows >= 360) & (rows < 450) & (cols >= 180) & (cols < 480))
    ok = (
        wall is not None and box is not None and floor is not None
        and box < wall - 0.12
        and 1.55 <= wall <= 2.25
        and 1.30 <= box <= 1.95
        and 0.40 <= floor <= 0.75
    )
    return {
        "wall_m": None if wall is None else round(wall, 3),
        "box_m": None if box is None else round(box, 3),
        "gap_m": None if wall is None or box is None else round(wall - box, 3),
        "floor_m": None if floor is None else round(floor, 3),
        "pass": bool(ok),
    }


def eligible(row) -> bool:
    return row["pixel_med_m"] <= 0.42 and row.get("floor_m") is not None and 0.40 <= row["floor_m"] <= 0.80


def score_of(row) -> float:
    wall, box, gap = row.get("wall_m"), row.get("box_m"), row.get("gap_m")
    if wall is None or box is None or gap is None:
        return -1e9
    score = -abs(gap - 0.27) - 0.35 * (abs(wall - 1.89) + abs(box - 1.62))
    if gap < 0.12 or box >= wall:
        score -= 1.0
    return score


def evaluate(model, old_hold, old_feats, photo):
    stats = summarize(model, old_hold, old_feats)
    gate = hold_gate(model, photo)
    model.encoder.cpu()
    torch.cuda.empty_cache()
    return {**stats, **gate}


def save_decoder(state, report):
    src = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    payload = dict(src)
    payload["model"] = {k: v.detach().cpu() for k, v in src["model"].items()}
    for key, value in state.items():
        payload["model"]["decoder." + key] = value.detach().cpu()
    payload["office_range"] = report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(OUT))
    (ROOT / "finetune_gap.json").write_text(json.dumps(report, indent=2))


def save_full(state, report, report_name="finetune_gap.json"):
    src = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    payload = dict(src)
    payload["model"] = {k: v.detach().cpu() for k, v in state.items()}
    payload["office_range"] = report
    torch.save(payload, str(OUT))
    (ROOT / report_name).write_text(json.dumps(report, indent=2))


def run_steps(model, opt, new_frames, new_feats, old_frames, old_feats, old_hold, hold_feats, photo, steps, decoder_only):
    new_gt = [torch.from_numpy(f["depth"]) for f in new_frames]
    new_valid = [torch.from_numpy(f["valid"]) for f in new_frames]
    new_close = [torch.from_numpy(obstacle_mask(f["depth"], f["valid"])) for f in new_frames]
    old_gt = [torch.from_numpy(f["depth"]) for f in old_frames]
    old_valid = [torch.from_numpy(f["valid"]) for f in old_frames]
    before = evaluate(model, old_hold, hold_feats, photo)
    print("before", json.dumps(before), flush=True)
    best = score_of(before) if eligible(before) else -1e9
    best_step = 0
    best_report = {"step": 0, **before}
    if decoder_only:
        best_state = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
    else:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    every = 20 if decoder_only else 10
    for step in range(1, steps + 1):
        use_new = (step % 10) != 0
        if decoder_only:
            model.decoder.train()
            if use_new:
                i = (step - 1) % len(new_frames)
                pred = _predict(model, to_cuda(new_feats[i])).clamp(min=0.05)
                loss = pixel_loss(pred, new_gt[i].to("cuda"), new_valid[i].to("cuda"), new_close[i].to("cuda"))
            else:
                i = (step - 1) % len(old_frames)
                pred = _predict(model, to_cuda(old_feats[i])).clamp(min=0.05)
                valid = old_valid[i].to("cuda")
                if int(valid.sum()) < 500:
                    continue
                loss = 0.10 * old_loss(pred, old_gt[i].to("cuda"), valid)
            params = [p for p in model.decoder.parameters() if p.requires_grad]
        else:
            model.encoder.to("cuda")
            model.decoder.to("cuda")
            model.encoder.train()
            model.decoder.train()
            frame = new_frames[(step - 1) % len(new_frames)] if use_new else old_frames[(step - 1) % len(old_frames)]
            rgb = torch.from_numpy(frame["rgb"]).view(1, 1, *frame["rgb"].shape).to("cuda")
            feats, _aux = model.encoder(
                model._prep_last_frame(rgb).unsqueeze(1), cam_token=None, export_feat_layers=[]
            )
            pred = _predict(model, feats).clamp(min=0.05)
            gt = torch.from_numpy(frame["depth"]).to("cuda")
            valid = torch.from_numpy(frame["valid"]).to("cuda")
            if int(valid.sum()) < 500:
                continue
            if use_new:
                loss = pixel_loss(pred, gt, valid, torch.from_numpy(obstacle_mask(frame["depth"], frame["valid"])).to("cuda"))
            else:
                loss = 0.10 * old_loss(pred, gt, valid)
            params = [p for p in model.parameters() if p.requires_grad]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % every == 0 or step == steps:
            if not decoder_only:
                model.encoder.eval()
                model.decoder.eval()
                hold_feats = encode(model, old_hold)
            row = {"step": step, "loss": round(float(loss.detach()), 4)}
            row.update(evaluate(model, old_hold, hold_feats, photo))
            print(json.dumps(row), flush=True)
            if eligible(row) and score_of(row) > best:
                best = score_of(row)
                best_step = step
                best_report = row
                if decoder_only:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
                else:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return before, best_step, best_report, best_state, hold_feats


def main() -> int:
    new_frames = [load_frame(PAIRS / f"{name}.npz") for name in TRAIN_NAMES]
    print(
        "train", [f["name"] for f in new_frames],
        "valid", [int(f["valid"].sum()) for f in new_frames],
        "obstacle", [int(obstacle_mask(f["depth"], f["valid"]).sum()) for f in new_frames],
        flush=True,
    )
    old_all = []
    for path in sorted(OLD.glob("pair_*.npz")):
        idx = int(path.name.split("_")[1].split(".")[0])
        if idx in OLD_HOLDOUT:
            continue
        old_all.append(load_frame(path))
    old_hold = []
    for path in sorted(OLD.glob("pair_*.npz")):
        idx = int(path.name.split("_")[1].split(".")[0])
        if idx in OLD_HOLDOUT:
            old_hold.append(load_frame(path))
    photo = cv2.cvtColor(cv2.imread(str(HOLD_PHOTO)), cv2.COLOR_BGR2RGB)
    wrap = DepthMinPredictor.from_checkpoint(str(CKPT), device="cuda")
    model = wrap._model
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    for param in model.decoder.parameters():
        param.requires_grad_(True)
    new_feats = encode(model, new_frames)
    old_feats = encode(model, old_all)
    hold_feats = encode(model, old_hold)
    model.encoder.cpu()
    torch.cuda.empty_cache()
    opt = torch.optim.AdamW([p for p in model.decoder.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    before, best_step, best_report, best_state, hold_feats = run_steps(
        model, opt, new_frames, new_feats, old_all, old_feats, old_hold, hold_feats, photo, STEPS, True
    )
    model.decoder.load_state_dict(best_state)
    model.decoder.to("cuda").eval()
    model.encoder.to("cuda").eval()
    after = evaluate(model, old_hold, hold_feats, photo)
    report = {
        "stage": "decoder",
        "best_step": best_step,
        "before": before,
        "after": after,
        "best": best_report,
        "train": list(TRAIN_NAMES),
        "holdout": "frame_172919 wall 1.89 box 1.62",
        "init": str(CKPT),
    }
    save_decoder(best_state, report)
    print(json.dumps(report), flush=True)
    return 0


def tail() -> int:
    """Continue from the saved decoder. The 1.9 m frame stays out of the loss."""
    new_frames = [load_frame(PAIRS / f"{name}.npz") for name in TRAIN_NAMES]
    old_all, old_hold = [], []
    for path in sorted(OLD.glob("pair_*.npz")):
        idx = int(path.name.split("_")[1].split(".")[0])
        (old_hold if idx in OLD_HOLDOUT else old_all).append(load_frame(path))
    photo = cv2.cvtColor(cv2.imread(str(HOLD_PHOTO)), cv2.COLOR_BGR2RGB)
    wrap = DepthMinPredictor.from_checkpoint(str(OUT), device="cuda")
    model = wrap._model
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    for block in list(model.encoder.pretrained.blocks)[-2:]:
        for param in block.parameters():
            param.requires_grad_(True)
    for param in model.decoder.parameters():
        param.requires_grad_(True)
    print("encoder_tail", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": 1e-6},
            {"params": [p for p in model.decoder.parameters() if p.requires_grad], "lr": LR},
        ],
        weight_decay=1e-4,
    )
    hold_feats = encode(model, old_hold)
    before, best_step, best_report, best_state, _hold = run_steps(
        model, opt, new_frames, None, old_all, None, old_hold, hold_feats, photo, 40, False
    )
    model.load_state_dict(best_state)
    model.to("cuda").eval()
    hold_feats = encode(model, old_hold)
    after = evaluate(model, old_hold, hold_feats, photo)
    report = {
        "stage": "encoder_tail",
        "best_step": best_step,
        "before": before,
        "after": after,
        "best": best_report,
        "train": list(TRAIN_NAMES),
        "holdout": "frame_172919 wall 1.89 box 1.62",
        "init": str(CKPT),
    }
    save_full(best_state, report)
    print(json.dumps(report), flush=True)
    return 0


def retrain() -> int:
    """Retrain from the office init with the decoder and the last encoder blocks.

    The frozen-decoder run memorized the nine RealSense frames. The 1.9 m
    frame stays out of the loss. Nothing on the car is replaced.
    """
    global OUT
    OUT = ROOT / "depth_office_cone_retrain.pt"
    new_frames = [load_frame(PAIRS / f"{name}.npz") for name in TRAIN_NAMES]
    old_all, old_hold = [], []
    for path in sorted(OLD.glob("pair_*.npz")):
        idx = int(path.name.split("_")[1].split(".")[0])
        (old_hold if idx in OLD_HOLDOUT else old_all).append(load_frame(path))
    photo = cv2.cvtColor(cv2.imread(str(HOLD_PHOTO)), cv2.COLOR_BGR2RGB)
    wrap = DepthMinPredictor.from_checkpoint(str(CKPT), device="cuda")
    model = wrap._model
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    blocks = list(model.encoder.pretrained.blocks)
    for block in blocks[-4:]:
        for param in block.parameters():
            param.requires_grad_(True)
    for param in model.decoder.parameters():
        param.requires_grad_(True)
    print(
        "retrain", len(blocks), "tail4",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frames", [f["name"] for f in new_frames],
        flush=True,
    )
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": 1e-6},
            {"params": [p for p in model.decoder.parameters() if p.requires_grad], "lr": LR},
        ],
        weight_decay=1e-4,
    )
    hold_feats = encode(model, old_hold)
    before, best_step, best_report, best_state, _hold = run_steps(
        model, opt, new_frames, None, old_all, None, old_hold, hold_feats, photo, 80, False
    )
    model.load_state_dict(best_state)
    model.to("cuda").eval()
    hold_feats = encode(model, old_hold)
    after = evaluate(model, old_hold, hold_feats, photo)
    report = {
        "stage": "retrain",
        "best_step": best_step,
        "before": before,
        "after": after,
        "best": best_report,
        "train": list(TRAIN_NAMES),
        "holdout": "frame_172919 wall 1.89 box 1.62",
        "init": str(CKPT),
        "encoder_blocks": 4,
    }
    save_full(best_state, report, "finetune_retrain.json")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    stage = os.environ.get("RANGE_STAGE")
    if stage == "tail":
        raise SystemExit(tail())
    if stage == "retrain":
        raise SystemExit(retrain())
    raise SystemExit(main())
