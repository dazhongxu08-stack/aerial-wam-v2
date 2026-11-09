#!/usr/bin/env python3
"""Record 90° color and RealSense depth when the robot is moved by hand.

Does not publish a velocity. A frame is kept after the chassis has been still
for a moment and the pose differs from the last kept frame.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

from pinhole90 import k_from_msg, warp_color, warp_depth

OUT = Path(os.environ.get("YHS_PAIRS", "/home/user/rt/da3_pairs"))
MIN_MOVE_M = 0.30
MIN_YAW = 0.30
STILL_S = 0.6
MAX_FRAMES = int(os.environ.get("YHS_PAIR_MAX", "48"))


def _rgb(msg):
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(msg.data, np.uint8)
    img = np.ndarray((h, w, 3), np.uint8, buf, strides=(int(msg.step), 3, 1))
    if msg.encoding.lower() == "bgr8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def _depth(msg):
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(msg.data, np.uint16)
    raw = np.ndarray((h, w), np.uint16, buf, strides=(int(msg.step), 2))
    d = raw.astype(np.float32) * 0.001
    d[raw == 0] = np.nan
    return d


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _status(body: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps(body, ensure_ascii=False))
    tmp.replace(OUT / "status.json")


def main() -> None:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = Node("da3_pairs")
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
    box = {}
    node.create_subscription(Image, "/camera1/camera1/color/image_raw", lambda m: box.__setitem__("c", _rgb(m)), qos)
    node.create_subscription(Image, "/camera1/camera1/depth/image_rect_raw", lambda m: box.__setitem__("d", _depth(m)), qos)
    node.create_subscription(CameraInfo, "/camera1/camera1/color/camera_info", lambda m: box.__setitem__("ck", m), qos)
    node.create_subscription(CameraInfo, "/camera1/camera1/depth/camera_info", lambda m: box.__setitem__("dk", m), qos)
    node.create_subscription(Odometry, "/odom", lambda m: box.__setitem__("odom", m), qos)
    poses = []
    for path in sorted(OUT.glob("pair_*.npz")):
        with np.load(path) as old:
            poses.append([round(float(v), 3) for v in old["pose"]])
    last = tuple(poses[-1]) if poses else None
    still_since = None
    still_pose = None
    OUT.mkdir(parents=True, exist_ok=True)
    while rclpy.ok() and len(poses) < MAX_FRAMES and not (OUT / "STOP").exists():
        rclpy.spin_once(node, timeout_sec=0.1)
        if not all(k in box for k in ("c", "d", "ck", "dk", "odom")):
            _status({"collecting": True, "frames": len(poses), "waiting": "sensors"})
            continue
        p = box["odom"].pose.pose.position
        pose = (float(p.x), float(p.y), float(p.z), _yaw(box["odom"].pose.pose.orientation))
        now = time.time()
        if still_pose is None or math.hypot(pose[0] - still_pose[0], pose[1] - still_pose[1]) > 0.03 or abs(_wrap(pose[3] - still_pose[3])) > 0.03:
            still_pose = pose
            still_since = now
            _status({"collecting": True, "frames": len(poses), "pose": [round(v, 3) for v in pose], "state": "moving"})
            continue
        if now - still_since < STILL_S:
            continue
        if last is not None:
            moved = math.hypot(pose[0] - last[0], pose[1] - last[1])
            turned = abs(_wrap(pose[3] - last[3]))
            if moved < MIN_MOVE_M and turned < MIN_YAW:
                _status({"collecting": True, "frames": len(poses), "pose": [round(v, 3) for v in pose], "state": "same_place"})
                continue
        ck, dk = box["ck"], box["dk"]
        color, seen = warp_color(box["c"], k_from_msg(ck), np.array(ck.d, np.float64))
        depth = warp_depth(box["d"], k_from_msg(dk))
        finite = np.isfinite(depth) & seen & (depth > 0.3) & (depth < 3.0)
        if int(finite.sum()) < 2000:
            _status({"collecting": True, "frames": len(poses), "state": "depth_sparse"})
            continue
        name = OUT / f"pair_{len(poses):02d}.npz"
        np.savez(
            name,
            rgb=color,
            depth=np.nan_to_num(depth, nan=0.0).astype(np.float16),
            seen=seen.astype(np.uint8),
            pose=np.array(pose, np.float32),
        )
        poses.append([round(v, 3) for v in pose])
        last = pose
        _status({"collecting": True, "frames": len(poses), "poses": poses, "state": "saved", "last": str(name)})
        print(json.dumps({"saved": str(name), "pose": poses[-1], "valid_px": int(finite.sum())}), flush=True)
        time.sleep(0.5)
    _status({"collecting": False, "frames": len(poses), "poses": poses})
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
