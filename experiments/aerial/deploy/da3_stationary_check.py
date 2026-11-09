#!/usr/bin/env python3
"""Compare DA3 on the 90° view with the depth camera. Does not command the chassis."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("AERIAL_DA3_FP16", "1")

from pinhole90 import band_mins, ground_guard_rows, k_from_msg, mask_unseen, warp_color, warp_depth

COLOR = "/camera1/camera1/color/image_raw"
COLOR_INFO = "/camera1/camera1/color/camera_info"
DEPTH = "/camera1/camera1/depth/image_rect_raw"
DEPTH_INFO = "/camera1/camera1/depth/camera_info"
WEIGHTS = Path(os.environ.get("RT_RAM", "/dev/shm/rtw")) / "weights"
OUT = Path(os.environ.get("YHS_SCALE", "/home/user/rt/da3_scale.json"))


def _rgb(msg):
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = np.ndarray((h, w, 3), np.uint8, buf, strides=(int(msg.step), 3, 1))
    if msg.encoding.lower() == "bgr8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def _depth(msg):
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(msg.data, dtype=np.uint16)
    raw = np.ndarray((h, w), np.uint16, buf, strides=(int(msg.step), 2))
    d = raw.astype(np.float32) * 0.001
    d[raw == 0] = np.nan
    return d


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from aerial_finder.nav.consequence_model import pool_min
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation

    rclpy.init()
    node = Node("da3_check")
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
    box = {}
    node.create_subscription(Image, COLOR, lambda m: box.__setitem__("color", _rgb(m)), qos)
    node.create_subscription(Image, DEPTH, lambda m: box.__setitem__("depth", _depth(m)), qos)
    node.create_subscription(CameraInfo, COLOR_INFO, lambda m: box.__setitem__("ck", m), qos)
    node.create_subscription(CameraInfo, DEPTH_INFO, lambda m: box.__setitem__("dk", m), qos)
    t0 = time.time()
    while time.time() - t0 < 8.0 and not all(k in box for k in ("color", "depth", "ck", "dk")):
        rclpy.spin_once(node, timeout_sec=0.1)
    if not all(k in box for k in ("color", "depth", "ck", "dk")):
        raise SystemExit("camera did not arrive")
    ck, dk = box["ck"], box["dk"]
    rows = ground_guard_rows(float(ck.k[4]), float(ck.k[5]), int(ck.height))
    pred = DepthMinPredictor.from_checkpoint(WEIGHTS / "m_d.bin", device="cuda")
    pred.forward_scale = 1.0
    pred.forward_exponent = 1.0
    samples = []
    for _ in range(8):
        t1 = time.time()
        while time.time() - t1 < 1.0:
            rclpy.spin_once(node, timeout_sec=0.05)
        color, seen = warp_color(box["color"], k_from_msg(ck), np.array(ck.d, np.float64))
        depth = warp_depth(box["depth"], k_from_msg(dk))
        hat = pred.predict_depth(Observation(rgb=color, state=np.zeros(7, np.float32), collided=False))
        if hat is None:
            continue
        hat = mask_unseen(hat, seen)
        model = band_mins(pool_min(np.where(np.isfinite(hat), hat, np.inf)), *rows)
        truth = band_mins(pool_min(np.where(np.isfinite(depth), depth, np.inf)), *rows)
        samples.append((model, truth))
        time.sleep(0.15)
    node.destroy_node()
    rclpy.shutdown()
    if len(samples) < 4:
        raise SystemExit(f"only {len(samples)} paired frames")
    fit, hold = samples[:4], samples[4:]
    names = ("front", "left", "right")

    def ratios(group):
        out = []
        for model, truth in group:
            out.append([truth[i] / model[i] if model[i] > 0.2 and truth[i] < 1e3 else None for i in range(3)])
        return out

    fit_r = ratios(fit)
    cols = list(zip(*fit_r))
    scale = {}
    for name, col in zip(names, cols):
        vals = [v for v in col if v is not None and np.isfinite(v)]
        scale[name] = None if not vals else round(float(np.median(vals)), 3)
    usable = [scale[n] for n in names if scale[n] is not None]
    one = None
    if len(usable) == 3 and max(usable) / min(usable) < 1.5:
        one = round(float(np.median(usable)), 3)
    hold_err = []
    if one is not None:
        for model, truth in hold:
            scaled = [m * one for m in model]
            hold_err.append([round(s - t, 2) if t < 1e3 else None for s, t in zip(scaled, truth)])
    report = {
        "guard_rows": list(rows),
        "fit_model_m": [[round(v, 2) if v < 1e3 else None for v in m] for m, _ in fit],
        "fit_real_m": [[round(v, 2) if v < 1e3 else None for v in t] for _, t in fit],
        "ratio_real_over_model": scale,
        "scale": one,
        "holdout_scaled_minus_real_m": hold_err,
        "order_ok": all(
            (m[1] - m[2]) * (t[1] - t[2]) >= 0
            for m, t in fit
            if m[1] < 1e3 and m[2] < 1e3 and t[1] < 1e3 and t[2] < 1e3
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if one is not None:
        OUT.write_text(json.dumps({
            "scale": one,
            "guard_rows": list(rows),
            "guard_top": 0.22,
            "guard_bot": 0.47,
        }, indent=2))


if __name__ == "__main__":
    main()
