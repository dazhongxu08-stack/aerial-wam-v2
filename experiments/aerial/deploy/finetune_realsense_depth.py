#!/usr/bin/env python3
"""Fine-tune the DA3 decoder on this camera. The encoder stays frozen.

The RealSense depth, warped into the same 90° view, is the meter label.
Pixels outside 0.3–3 m and the right side of the image are not in the loss.
Does not command the chassis and does not write a checkpoint to disk.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("AERIAL_DA3_FP16", "1")

from pinhole90 import k_from_msg, mask_unseen, warp_color, warp_depth

WEIGHTS = os.environ.get("RT_RAM", "/dev/shm/rtw") + "/weights/m_d.bin"
HOLD_COL = 168


def _rgb(msg):
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(msg.data, np.uint8)
    img = np.ndarray((h, w, 3), np.uint8, buf, strides=(int(msg.step), 3, 1))
    if msg.encoding.lower() == "bgr8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def _depth_m(msg):
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(msg.data, np.uint16)
    raw = np.ndarray((h, w), np.uint16, buf, strides=(int(msg.step), 2))
    d = raw.astype(np.float32) * 0.001
    d[raw == 0] = np.nan
    return d


def _grab():
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = Node("da3_ft")
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
    box = {}
    node.create_subscription(Image, "/camera1/camera1/color/image_raw", lambda m: box.__setitem__("c", _rgb(m)), qos)
    node.create_subscription(Image, "/camera1/camera1/depth/image_rect_raw", lambda m: box.__setitem__("d", _depth_m(m)), qos)
    node.create_subscription(CameraInfo, "/camera1/camera1/color/camera_info", lambda m: box.__setitem__("ck", m), qos)
    node.create_subscription(CameraInfo, "/camera1/camera1/depth/camera_info", lambda m: box.__setitem__("dk", m), qos)
    node.create_subscription(Odometry, "/odom", lambda m: box.__setitem__("yaw", m.pose.pose.orientation), qos)
    t0 = time.time()
    while time.time() - t0 < 8 and not all(k in box for k in ("c", "d", "ck", "dk")):
        rclpy.spin_once(node, timeout_sec=0.1)
    ck, dk = box["ck"], box["dk"]
    hats_c, reals, seens = [], [], []
    for _ in range(4):
        t1 = time.time()
        while time.time() - t1 < 0.3:
            rclpy.spin_once(node, timeout_sec=0.05)
        color, seen = warp_color(box["c"], k_from_msg(ck), np.array(ck.d, np.float64))
        real = warp_depth(box["d"], k_from_msg(dk))
        hats_c.append(color)
        reals.append(real)
        seens.append(seen)
    q = box.get("yaw")
    yaw = None
    if q is not None:
        import math
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    node.destroy_node()
    rclpy.shutdown()
    return hats_c[-1], np.median(np.stack(reals), 0).astype(np.float32), np.all(np.stack(seens), 0), yaw


def _bins(real, pred, mask):
    out = []
    for lo, hi in ((0.3, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)):
        m = mask & (real >= lo) & (real < hi) & np.isfinite(pred)
        if int(m.sum()) < 30:
            out.append({"real_m": [lo, hi], "n": int(m.sum())})
            continue
        err = np.abs(pred[m] - real[m])
        out.append({
            "real_m": [lo, hi],
            "n": int(m.sum()),
            "med_abs_m": round(float(np.median(err)), 3),
            "p90_abs_m": round(float(np.percentile(err, 90)), 3),
        })
    return out


def _score(pred, real, mask):
    m = mask & (real >= 0.6) & (real < 2.0) & np.isfinite(pred) & (pred > 0.05)
    if int(m.sum()) < 30:
        return 1e9
    return float(np.median(np.abs(pred[m] - real[m])))


def main() -> None:
    import torch
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor

    color, real, seen, yaw = _grab()
    valid = seen & np.isfinite(real) & (real > 0.3) & (real < 3.0)
    train = valid.copy()
    train[:, HOLD_COL:] = False
    hold = valid.copy()
    hold[:, :HOLD_COL] = False
    pred = DepthMinPredictor.from_checkpoint(WEIGHTS, device="cuda")
    pred.forward_scale = 1.0
    pred.forward_exponent = 1.0
    model = pred._model
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    rgb = torch.from_numpy(np.ascontiguousarray(color)).view(1, 1, *color.shape).to("cuda")
    with torch.no_grad():
        feats, _aux = model.encoder(model._prep_last_frame(rgb).unsqueeze(1), cam_token=None, export_feat_layers=[])
    feats = _detach(feats)
    model.encoder.cpu()
    torch.cuda.empty_cache()
    gt = torch.from_numpy(real).to("cuda")
    train_m = torch.from_numpy(train).to("cuda")
    opt = torch.optim.AdamW([p for p in model.decoder.parameters() if p.requires_grad], lr=1e-5, weight_decay=1e-2)
    base = _predict(model, feats).detach()
    report = {"yaw_rad": None if yaw is None else round(float(yaw), 3), "train_px": int(train.sum()), "hold_px": int(hold.sum())}
    report["before_hold"] = _bins(real, base.cpu().numpy(), hold)
    report["before_train"] = _bins(real, base.cpu().numpy(), train)
    best = _score(base.cpu().numpy(), real, hold)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_step = 0
    h = w = color.shape[0]
    for step in range(1, 41):
        model.decoder.train()
        depth = _predict(model, feats).clamp(min=0.05)
        loss = torch.nn.functional.smooth_l1_loss(torch.log(depth[train_m]), torch.log(gt[train_m].clamp(min=0.05)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            model.decoder.eval()
            with torch.no_grad():
                now = _predict(model, feats).clamp(min=0.05).float().cpu().numpy()
            score = _score(now, real, hold)
            if score < best:
                best = score
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.decoder.to("cuda").eval()
    with torch.no_grad():
        after = _predict(model, feats).clamp(min=0.05).float().cpu().numpy()
    report["best_step"] = best_step
    report["after_hold"] = _bins(real, after, hold)
    report["after_train"] = _bins(real, after, train)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _detach(feats):
    import torch
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
        import torch
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(model.image_size, model.image_size),
            mode="bilinear", align_corners=False)[:, 0]
    return depth[0]


if __name__ == "__main__":
    main()
