#!/usr/bin/env python3
"""Office stills, trained the way DA3 trains a metric head.

Teacher and student both start from the original DA3METRIC-LARGE weights.
The teacher stays frozen. RealSense is not a per-image meter target: its
holes and its absolute scale are not copied into the label. One robust
scale is shared by every frame and the shift is zero. The dense label is
that single rescaling of the teacher. DINOv2 stays frozen. The DPT is
trained with AbsRel + SILog and no NLL.

Meters follow the published DA3METRIC conversion when that conversion is
the one the pretrained head already uses (focal * network / 300). If the
raw network output is already in meters, the run keeps that convention
instead of applying the factor twice.

The 1.89 m frame is not in the loss and is not used to choose the step.
It is measured once after the checkpoint is written. Nothing is copied
to the car, and the earlier office checkpoints are not overwritten.
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

RANGE = Path(os.environ.get("RANGE_ROOT", "/home/yao/da3_range_20261107"))
OLD = Path(os.environ.get("RANGE_OLD", "/home/yao/da3_obstacle_20261106"))
REPO = Path(os.environ.get("RANGE_REPO", "/home/yao/aerial-wam-v2"))
OUT_DIR = Path(os.environ.get("DA3_TEACHER_OUT", "/home/yao/da3_teacher_20261108"))
sys.path.insert(0, str(REPO))

from experiments.aerial.rl.dynamics_torch import DA3DepthHead, depth_head_loss  # noqa: E402

WEIGHTS = Path(os.environ.get(
    "DA3_WEIGHTS",
    str(OUT_DIR / "weights" / "model.safetensors"),
))
WEIGHTS_PT = OUT_DIR / "weights" / "da3metric_large.pt"
OUT = OUT_DIR / "depth_office_da3_global.pt"
REPORT = OUT_DIR / "finetune_da3_global.json"
PAIRS = RANGE / "pairs"
HOLD_PHOTO = RANGE / "raw" / "color_172919.png"
HOLD_NPZ = PAIRS / "hold19.npz"
TRAIN_NAMES = (
    "near10", "near11", "near14", "near15", "side_right", "side_left",
    "near17", "mid21", "far25",
)
OLD_HOLDOUT = {5, 15, 25, 35, 45, 55, 60, 61, 62, 63}
STEPS = 300
EVAL_EVERY = 20
FOCAL = 112.0  # 224 px, 90 deg HFOV, the warp these pairs already use
METRIC_FACTOR = FOCAL / 300.0
TEACHER_TOL_M = 0.15
FY, CY = 386.394, 243.503
K = np.array([[386.890, 0.0, 318.382], [0.0, FY, CY], [0.0, 0.0, 1.0]], np.float64)
D = np.array([-0.0539, 0.0590, -0.00047, 0.00068, -0.0190], np.float64)
PROTECTED = {
    OLD / "init.pt",
    Path("/home/yao/da3_order_20261106/depth_office_cone_order.pt"),
    RANGE / "depth_office_cone_range.pt",
    RANGE / "depth_office_cone_gap.pt",
    RANGE / "depth_office_cone_retrain.pt",
    Path("/home/user/rt/office_wam/depth_office_cone_real.pt"),
    OUT_DIR / "depth_office_da3_teacher.pt",
}


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


def load_da3_state(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and any(str(k).startswith("model.backbone.") for k in blob):
        return blob
    if isinstance(blob, dict) and isinstance(blob.get("model"), dict):
        return blob["model"]
    return blob


def load_frame(path: Path) -> dict:
    z = np.load(path)
    rgb = np.ascontiguousarray(z["rgb"])
    depth = z["depth"].astype(np.float32)
    seen = z["seen"].astype(bool)
    z.close()
    valid = seen & np.isfinite(depth) & (depth > 0.35) & (depth < 8.0)
    return {"name": path.stem, "rgb": rgb, "depth": depth, "valid": valid}


def _predict_net(model, feats) -> torch.Tensor:
    out = model.decoder(feats, model.image_size, model.image_size, patch_start_idx=0)
    depth = out["depth"][:, 0]
    if depth.shape[-2:] != (model.image_size, model.image_size):
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(model.image_size, model.image_size),
            mode="bilinear", align_corners=False,
        )[:, 0]
    return depth[0]


def encode_all(model, frames):
    encoded = []
    nets = []
    model.encoder.eval()
    model.decoder.eval()
    for frame in frames:
        rgb = torch.from_numpy(frame["rgb"]).view(1, 1, *frame["rgb"].shape).to("cuda")
        with torch.no_grad():
            feats, _aux = model.encoder(
                model._prep_last_frame(rgb).unsqueeze(1),
                cam_token=None, export_feat_layers=[],
            )
            net = _predict_net(model, feats).float().cpu().numpy()
        encoded.append(to_cpu(_detach(feats)))
        nets.append(net)
    return encoded, nets


def fit_scale_shift(teacher: np.ndarray, frame: dict):
    valid = frame["valid"]
    t = teacher[valid].astype(np.float64)
    g = frame["depth"][valid].astype(np.float64)
    keep = np.isfinite(t) & (t > 1e-4) & np.isfinite(g)
    t, g = t[keep], g[keep]
    n = int(t.size)
    if n < 500:
        return None
    st, sg = float(t.sum()), float(g.sum())
    stt, stg = float(np.dot(t, t)), float(np.dot(t, g))
    det = stt * n - st * st
    if abs(det) < 1e-8:
        return None
    scale = (stg * n - st * sg) / det
    shift = (stt * sg - st * stg) / det
    if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0:
        return None
    aligned = scale * t + shift
    return {
        "s": float(scale),
        "b": float(shift),
        "resid_med_m": float(np.median(np.abs(aligned - g))),
        "n": n,
    }


def regions(rgb: np.ndarray):
    """Crate = blue component nearest the center. Wall = upper non-blue band."""
    red, green, blue = [rgb[:, :, i].astype(np.int16) for i in range(3)]
    mask = (blue > 140) & (blue > red + 40) & (blue > green + 30)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    h, w = mask.shape
    best, best_d = None, 1e18
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        left = int(stats[i, cv2.CC_STAT_LEFT])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        if area < 80 or height < 8:
            continue
        if left <= 1 or left + width >= w - 1:
            continue
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        dist = (cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2
        if dist < best_d:
            best, best_d = i, dist
    crate = labels == best if best is not None else np.zeros(mask.shape, bool)
    rows = np.arange(h)[:, None]
    wall = (rows >= 16) & (rows < 90) & ~mask
    return crate, wall


def _median(depth: np.ndarray, mask: np.ndarray):
    vals = depth[mask & np.isfinite(depth) & (depth > 0.05) & (depth < 30.0)]
    if vals.size < 30:
        return None
    return float(np.median(vals))


def band_errors(pred: np.ndarray, frame: dict):
    crate, wall = regions(frame["rgb"])
    out = {}
    for key, mask in (("box", crate), ("wall", wall)):
        both = mask & frame["valid"]
        gt = _median(frame["depth"], both)
        pr = _median(pred, mask)
        if gt is None or pr is None:
            return None
        out[key] = {
            "gt": round(gt, 3),
            "pred": round(pr, 3),
            "err": round(pr - gt, 3),
        }
    out["worst"] = round(max(abs(out["box"]["err"]), abs(out["wall"]["err"])), 3)
    return out


def teacher_band_residual(aligned: np.ndarray, frame: dict):
    crate, wall = regions(frame["rgb"])
    out = {}
    for key, mask in (("box", crate), ("wall", wall)):
        both = mask & frame["valid"]
        gt = _median(frame["depth"], both)
        pr = _median(aligned, both)
        if gt is None or pr is None:
            out[key] = None
            continue
        out[key] = {"gt": round(gt, 3), "aligned": round(pr, 3), "err": round(pr - gt, 3)}
    return out


def absrel_of(pred: np.ndarray, frame: dict):
    valid = frame["valid"] & np.isfinite(pred) & (pred > 0.05)
    if int(valid.sum()) < 30:
        return None
    rel = np.abs(pred[valid] - frame["depth"][valid]) / np.clip(frame["depth"][valid], 1e-3, None)
    return float(np.mean(rel))


def mean_absrel(rows):
    vals = [r for r in rows if r is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore(model, state):
    model.load_state_dict(state)


def warp_color(rgb, out=224):
    focal = 0.5 * out / math.tan(math.radians(45.0))
    vk = np.array([[focal, 0.0, out / 2.0], [0.0, focal, out / 2.0], [0.0, 0.0, 1.0]], np.float64)
    m1, m2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), vk, (out, out), cv2.CV_16SC2)
    return cv2.remap(rgb, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def map_photo(model, rgb224, factor: float):
    model.encoder.to("cuda").eval()
    model.decoder.to("cuda").eval()
    tensor = torch.from_numpy(np.ascontiguousarray(rgb224)).view(1, 1, *rgb224.shape)
    with torch.no_grad():
        feats, _aux = model.encoder(
            model._prep_last_frame(tensor.to("cuda")).unsqueeze(1),
            cam_token=None, export_feat_layers=[],
        )
        pred = (_predict_net(model, feats) * factor).clamp(min=0.05).float().cpu().numpy()
    h, w = 480, 640
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs, ys], -1).reshape(-1, 1, 2).astype(np.float32)
    und = cv2.undistortPoints(pts, K, D)
    focal = 0.5 * 224 / math.tan(math.radians(45.0))
    vu = (focal * und[:, 0, 0] + 112).reshape(h, w).astype(np.float32)
    vv = (focal * und[:, 0, 1] + 112).reshape(h, w).astype(np.float32)
    return cv2.remap(pred, vu, vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def photo_reading(model, photo_rgb, factor: float) -> dict:
    mapped = map_photo(model, warp_color(photo_rgb), factor)
    h, w = mapped.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    red, green, blue = [photo_rgb[:, :, i].astype(np.int16) for i in range(3)]
    crate = (blue > 140) & (blue > red + 40) & (blue > green + 30)
    crate = crate & (rows < 280) & (cols > 80) & (cols < 560)
    wall = _median(mapped, (rows >= 40) & (rows < 150) & (cols >= 160) & (cols < 500) & ~crate)
    box = _median(mapped, crate)
    floor = _median(mapped, (rows >= 360) & (rows < 450) & (cols >= 180) & (cols < 480))
    gap = None if wall is None or box is None else round(wall - box, 3)
    return {
        "wall_m": None if wall is None else round(wall, 3),
        "box_m": None if box is None else round(box, 3),
        "gap_m": gap,
        "floor_m": None if floor is None else round(floor, 3),
        "used_for_selection": False,
    }


def robust_scale(teacher_m: np.ndarray, frame: dict):
    """Median RealSense/teacher ratio. One number per image, no shift."""
    valid = frame["valid"]
    teacher = teacher_m[valid]
    gt = frame["depth"][valid]
    ok = np.isfinite(teacher) & (teacher > 0.05) & np.isfinite(gt) & (gt > 0.35)
    if int(ok.sum()) < 500:
        return None
    return float(np.median(gt[ok] / teacher[ok]))


def build_global_target(teacher_m: np.ndarray, scale: float) -> np.ndarray:
    aligned = float(scale) * teacher_m
    good = np.isfinite(aligned) & (aligned > 0.05) & (aligned < 8.0)
    return np.where(good, aligned, 0.0).astype(np.float32)


def absrel_target(pred: np.ndarray, target: np.ndarray):
    mask = (
        np.isfinite(pred) & np.isfinite(target)
        & (pred > 0.05) & (target > 0.05) & (target < 8.0)
    )
    if int(mask.sum()) < 30:
        return None
    rel = np.abs(pred[mask] - target[mask]) / target[mask]
    return float(np.mean(rel))


def band_vs(pred: np.ndarray, reference: np.ndarray, frame: dict):
    """Wall and crate errors against the teacher label, not RealSense."""
    crate, wall = regions(frame["rgb"])
    out = {}
    pred_box = pred_wall = None
    for key, mask in (("box", crate), ("wall", wall)):
        ref = _median(reference, mask)
        pr = _median(pred, mask)
        if ref is None or pr is None:
            return None
        out[key] = {"teacher": round(ref, 3), "pred": round(pr, 3), "err": round(pr - ref, 3)}
        if key == "box":
            pred_box = pr
        else:
            pred_wall = pr
    out["worst"] = round(max(abs(out["box"]["err"]), abs(out["wall"]["err"])), 3)
    out["order_ok"] = bool(pred_box < pred_wall)
    return out


def meters_of(net: np.ndarray, factor: float) -> np.ndarray:
    return net.astype(np.float32) * np.float32(factor)


@torch.no_grad()
def predict_meters(model, feat, factor: float) -> np.ndarray:
    model.decoder.eval()
    return (_predict_net(model, to_cuda(feat)) * factor).clamp(min=1e-3).float().cpu().numpy()


def evaluate(model, crate, crate_feats, crate_targets, hold_feats, hold_targets, factor: float):
    bands = []
    for frame, feat, target in zip(crate, crate_feats, crate_targets):
        pred = predict_meters(model, feat, factor)
        band = band_vs(pred, target, frame)
        if band is None:
            print(f"[da3] no box/wall mask on {frame['name']}", flush=True)
            band = {"box": None, "wall": None, "worst": 1e9, "order_ok": False}
        band["name"] = frame["name"]
        bands.append(band)
    rels = []
    for feat, target in zip(hold_feats, hold_targets):
        pred = predict_meters(model, feat, factor)
        rels.append(absrel_target(pred, target))
    worst = max(row["worst"] for row in bands)
    order_ok = all(bool(row.get("order_ok")) for row in bands)
    return {
        "crate": bands,
        "worst_m": round(float(worst), 3),
        "order_ok": bool(order_ok),
        "qualified": bool(order_ok and worst <= TEACHER_TOL_M),
        "old_holdout_absrel": None if mean_absrel(rels) is None else round(mean_absrel(rels), 4),
    }


def better(candidate, incumbent, qualified_only: bool) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return (not qualified_only) or bool(candidate["qualified"])
    if qualified_only:
        if candidate["qualified"] and not incumbent["qualified"]:
            return True
        if not candidate["qualified"]:
            return False
        left, right = candidate["old_holdout_absrel"], incumbent["old_holdout_absrel"]
        if left is None:
            return False
        if right is None:
            return True
        return left < right
    return candidate["worst_m"] < incumbent["worst_m"]


def save_checkpoint(model, report):
    if OUT.resolve() in {p.resolve() for p in PROTECTED}:
        raise SystemExit(f"refusing to write protected checkpoint {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "step": int(report["step"]),
        "n_frames": 1,
        "image_size": 224,
        "base": 32,
        "motion_channels": False,
        "scale_factorized": False,
        "backbone": "da3",
        "da3_arch": model.arch,
        "freeze_encoder": True,
        "metric": report["metric"],
        "office_da3_teacher": report,
    }
    torch.save(payload, str(OUT))
    REPORT.write_text(json.dumps(report, indent=2) + "\n")


def main():
    if "hold19" in TRAIN_NAMES:
        raise SystemExit("hold19 must not be a training frame")
    if not torch.cuda.is_available():
        raise SystemExit("cuda is not available")
    weight_path = WEIGHTS_PT if WEIGHTS_PT.is_file() else WEIGHTS
    if not weight_path.is_file():
        raise SystemExit(f"missing DA3METRIC-LARGE weights at {WEIGHTS}")
    print(f"[da3] loading {weight_path}", flush=True)
    state = load_da3_state(weight_path)
    model = DA3DepthHead(image_size=224, n_frames=1).to("cuda")
    loaded = model.load_da3_pretrained(state)
    print(
        f"[da3] backbone {loaded['backbone_loaded']} missing {loaded['backbone_missing']}",
        flush=True,
    )
    print(
        f"[da3] head {loaded['head_loaded']} missing {loaded['head_missing']}",
        flush=True,
    )
    if loaded["backbone_missing"] or loaded["head_missing"]:
        raise SystemExit("pretrained DA3 load is incomplete")
    for param in model.encoder.parameters():
        param.requires_grad_(False)

    crate = [load_frame(PAIRS / f"{name}.npz") for name in TRAIN_NAMES]
    old_train, old_hold = [], []
    for path in sorted(OLD.glob("pair_*.npz")):
        idx = int(path.stem.split("_")[1])
        (old_hold if idx in OLD_HOLDOUT else old_train).append(load_frame(path))
    print(
        f"[da3] crate {len(crate)} old_train {len(old_train)} old_hold {len(old_hold)}",
        flush=True,
    )
    if len(crate) != len(TRAIN_NAMES) or not old_train or not old_hold:
        raise SystemExit("training pairs are incomplete")

    print("[da3] encoding teacher", flush=True)
    crate_feats, crate_net = encode_all(model, crate)
    old_feats, old_net = encode_all(model, old_train)
    hold_feats, hold_net = encode_all(model, old_hold)
    model.encoder.cpu()
    torch.cuda.empty_cache()

    def median_ratio(nets, frames, factor):
        ratios = []
        for net, frame in zip(nets, frames):
            ratio = robust_scale(meters_of(net, factor), frame)
            if ratio is not None and ratio > 0:
                ratios.append(ratio)
        if not ratios:
            return None
        return float(np.median(ratios))

    s_raw = median_ratio(crate_net, crate, 1.0)
    s_scaled = median_ratio(crate_net, crate, METRIC_FACTOR)
    if s_raw is None or s_scaled is None:
        raise SystemExit("robust scale failed on the crate frames")
    if abs(math.log(s_scaled)) <= abs(math.log(s_raw)):
        convention, factor = "focal_over_300", METRIC_FACTOR
    else:
        convention, factor = "network_is_meters", 1.0
    crate_scales, old_scales = [], []
    for net, frame in zip(crate_net, crate):
        ratio = robust_scale(meters_of(net, factor), frame)
        if ratio is None:
            raise SystemExit(f"robust scale failed on {frame['name']}")
        crate_scales.append(ratio)
    for net, frame in zip(old_net, old_train):
        ratio = robust_scale(meters_of(net, factor), frame)
        if ratio is not None:
            old_scales.append(ratio)
    if len(old_scales) < max(1, len(old_train) // 2):
        raise SystemExit("robust scale failed on too many old frames")
    global_s = float(np.median(crate_scales + old_scales))
    metric_warm = 0.67 <= global_s <= 1.5
    lr = 3e-5 if metric_warm else 1e-4
    print(
        f"[da3] convention {convention} factor {factor:.6f} "
        f"global_s {global_s:.4f} shift 0 "
        f"crate_s {float(np.median(crate_scales)):.3f} "
        f"old_s {float(np.median(old_scales)):.3f} "
        f"metric_warm {metric_warm} lr {lr}",
        flush=True,
    )

    alignment = []
    for frame, net, ratio in zip(crate, crate_net, crate_scales):
        teacher_m = meters_of(net, factor)
        # Least-squares scale and shift are recorded and not used as the label.
        refused = fit_scale_shift(teacher_m, frame)
        aligned = global_s * teacher_m
        bands = teacher_band_residual(aligned, frame)
        row = {
            "name": frame["name"],
            "scale_only": round(ratio, 4),
            "global_s": round(global_s, 4),
            "shift": 0.0,
            "refused_s": None if refused is None else round(refused["s"], 4),
            "refused_b": None if refused is None else round(refused["b"], 4),
            "bands_vs_realsense": bands,
        }
        alignment.append(row)
        print(
            f"[da3] scale {frame['name']} image {ratio:.3f} global {global_s:.3f} "
            f"refused_s {None if refused is None else round(refused['s'], 3)} "
            f"refused_b {None if refused is None else round(refused['b'], 3)}",
            flush=True,
        )

    crate_targets = [
        build_global_target(meters_of(net, factor), global_s) for net in crate_net
    ]
    old_targets = [
        build_global_target(meters_of(net, factor), global_s) for net in old_net
    ]
    hold_targets = [
        build_global_target(meters_of(net, factor), global_s) for net in hold_net
    ]

    trainable = [p for p in model.decoder.parameters() if p.requires_grad]
    frozen = sum(int(p.requires_grad) for p in model.encoder.parameters())
    print(
        f"[da3] trainable {sum(p.numel() for p in trainable)} encoder_trainable {frozen}",
        flush=True,
    )
    opt = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95))
    model.decoder.train()

    def consider(step, chosen, effort, history):
        stats = evaluate(
            model, crate, crate_feats, crate_targets, hold_feats, hold_targets, factor,
        )
        if stats is None:
            print(f"[da3] step {step} distance mask missing", flush=True)
            return chosen, effort
        stats["step"] = step
        history.append({
            "step": step,
            "worst_m": stats["worst_m"],
            "order_ok": stats["order_ok"],
            "qualified": stats["qualified"],
            "old_holdout_absrel": stats["old_holdout_absrel"],
            "crate": [
                {
                    "name": row["name"],
                    "box": None if row["box"] is None else row["box"]["err"],
                    "wall": None if row["wall"] is None else row["wall"]["err"],
                    "order_ok": row.get("order_ok"),
                }
                for row in stats["crate"]
            ],
        })
        print(
            f"[da3] step {step} teacher_worst {stats['worst_m']:.3f} "
            f"order {stats['order_ok']} qualified {stats['qualified']} "
            f"old_absrel {stats['old_holdout_absrel']}",
            flush=True,
        )
        if better(stats, chosen, True):
            chosen = {**stats, "state": snapshot(model)}
        if better(stats, effort, False):
            effort = {**stats, "state": snapshot(model)}
        model.decoder.train()
        return chosen, effort

    history = []
    chosen, effort = None, None
    init_state = snapshot(model)
    chosen, effort = consider(0, chosen, effort, history)
    for step in range(1, STEPS + 1):
        if step % 2 == 0:
            index = (step // 2) % len(crate)
            feat, target = crate_feats[index], crate_targets[index]
        else:
            index = (step // 2) % len(old_train)
            feat, target = old_feats[index], old_targets[index]
        pred = _predict_net(model, to_cuda(feat)) * factor
        gt = torch.from_numpy(target).to(pred.device)
        loss, stats = depth_head_loss(
            pred, torch.zeros_like(pred), gt,
            absrel_weight=1.0, silog_weight=0.5, nll_weight=0.0, max_depth_m=8.0,
        )
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == STEPS:
            print(
                f"[da3] step {step} loss {stats['loss']:.4f} "
                f"absrel {stats['absrel']:.4f} silog {stats['silog']:.4f}",
                flush=True,
            )
            chosen, effort = consider(step, chosen, effort, history)

    if chosen is None and effort is None:
        raise SystemExit("no checkpoint was evaluated")
    qualified = chosen is not None
    winner = chosen if qualified else effort
    restore(model, winner["state"])
    model.decoder.eval()
    reading = None
    teacher_reading = None
    realsense_diag = []
    for frame, feat in zip(crate, crate_feats):
        pred = predict_meters(model, feat, factor)
        band = band_errors(pred, frame)
        if band is None:
            band = {"name": frame["name"], "box": None, "wall": None}
        else:
            band["name"] = frame["name"]
        realsense_diag.append(band)
        print(
            f"[da3] realsense {frame['name']} "
            f"box {None if band['box'] is None else band['box']['err']} "
            f"wall {None if band['wall'] is None else band['wall']['err']}",
            flush=True,
        )
    if HOLD_PHOTO.is_file():
        photo = cv2.imread(str(HOLD_PHOTO), cv2.IMREAD_COLOR)
        if photo is not None:
            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            reading = photo_reading(model, photo, factor)
            print(
                f"[da3] hold19 once wall {reading['wall_m']} box {reading['box_m']} "
                f"gap {reading['gap_m']} floor {reading['floor_m']}",
                flush=True,
            )
    if HOLD_NPZ.is_file():
        hold = load_frame(HOLD_NPZ)
        # Oracle alignment uses this photo's RealSense and the untouched
        # teacher. It is computed after the step was chosen.
        restore(model, init_state)
        model.encoder.to("cuda").eval()
        model.decoder.to("cuda").eval()
        _feat, nets = encode_all(model, [hold])
        fit = fit_scale_shift(meters_of(nets[0], factor), hold)
        if fit is not None:
            aligned = fit["s"] * meters_of(nets[0], factor) + fit["b"]
            teacher_reading = teacher_band_residual(aligned, hold)
            teacher_reading["s"] = round(fit["s"], 4)
            teacher_reading["b"] = round(fit["b"], 4)
            teacher_reading["note"] = "oracle scale-shift on hold19, not a prediction"
            print(f"[da3] hold19 teacher oracle {teacher_reading}", flush=True)
        restore(model, winner["state"])

    report = {
        "method": "da3_teacher_student",
        "teacher": "depth-anything/DA3METRIC-LARGE",
        "student_init": "depth-anything/DA3METRIC-LARGE",
        "encoder_frozen": True,
        "loss": "absrel + 0.5 silog on one global scale of the teacher, shift 0",
        "nll_weight": 0.0,
        "realsense_per_image_scale_shift": False,
        "metric": {
            "convention": convention,
            "focal_px": FOCAL,
            "factor": factor,
            "global_scale": round(global_s, 4),
            "shift": 0.0,
            "formula": "focal_px * network / 300" if convention == "focal_over_300" else "network output",
            "metric_warm": metric_warm,
            "median_scale_only_raw": round(s_raw, 4),
            "median_scale_only_focal": round(s_scaled, 4),
            "crate_scale_median": round(float(np.median(crate_scales)), 4),
            "old_scale_median": round(float(np.median(old_scales)), 4),
            "note": "global scale is trained into the decoder; inference meters use factor * network",
        },
        "lr": lr,
        "steps": STEPS,
        "teacher_tolerance_m": TEACHER_TOL_M,
        "train": list(TRAIN_NAMES),
        "old_train_count": len(old_train),
        "old_holdout_indices": sorted(OLD_HOLDOUT),
        "hold19_in_loss": False,
        "hold19_in_selection": False,
        "selection": "every crate wall and box within 0.15 m of the global-scaled teacher and box closer than wall, then lowest old-holdout AbsRel against that teacher",
        "step": winner["step"],
        "qualified": qualified,
        "worst_m": winner["worst_m"],
        "old_holdout_absrel": winner["old_holdout_absrel"],
        "crate": winner["crate"],
        "alignment": alignment,
        "history": history,
        "hold19_student": reading,
        "hold19_teacher_oracle": teacher_reading,
        "realsense_after_selection": realsense_diag,
        "deployed": False,
    }
    save_checkpoint(model, report)
    print(
        f"[da3] saved {OUT} step {winner['step']} qualified {qualified} "
        f"worst {winner['worst_m']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
