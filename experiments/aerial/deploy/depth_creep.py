#!/usr/bin/env python3
"""Creep through the room using only the depth camera. Does not use the model.

Forward motion requires the obstacle band to be open. The floor strip is not
treated as a wall. The chassis stays in agent mode when this script ends.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

from pinhole90 import band_mins, ground_guard_rows, k_from_msg, warp_depth

CMD = "/nav/cmd_vel"
DEPTH = "/camera1/camera1/depth/image_rect_raw"
DEPTH_INFO = "/camera1/camera1/depth/camera_info"
COLOR_INFO = "/camera1/camera1/color/camera_info"
ODOM = "/odom"
VX = 0.10
YAW = 0.20
FRONT_GO = 1.15
SIDE_GO = 0.80
FRONT_STOP = 0.85
RUN_S = float(os.environ.get("YHS_CREEP_S", "150"))
PATH_M = float(os.environ.get("YHS_CREEP_PATH", "8"))
RADIUS_M = float(os.environ.get("YHS_CREEP_RADIUS", "6"))
TURN_FIRST = float(os.environ.get("YHS_CREEP_TURN", "0"))
TURN_SIGN = float(os.environ.get("YHS_CREEP_TURN_SIGN", "1"))
STEP_M = 0.35
STEP_YAW = 0.40
PAUSE_S = 1.2


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


def _command(node, name: str, payload: dict) -> None:
    from std_msgs.msg import String
    pub = node.create_publisher(String, "/motion_control/command", 10)
    msg = String()
    msg.data = json.dumps(payload)
    t0 = time.time()
    while time.time() - t0 < 3.0 and pub.get_subscription_count() < 1:
        time.sleep(0.05)
    pub.publish(msg)
    time.sleep(0.3)


def _cones(depth, dk, rows):
    from aerial_finder.nav.consequence_model import pool_min
    warped = warp_depth(depth, k_from_msg(dk))
    finite = np.where(np.isfinite(warped), warped, np.inf)
    return band_mins(pool_min(finite), *rows)


def main() -> None:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = Node("depth_creep")
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
    box = {}
    node.create_subscription(Image, DEPTH, lambda m: box.__setitem__("d", (_depth(m), time.time())), qos)
    node.create_subscription(CameraInfo, DEPTH_INFO, lambda m: box.__setitem__("dk", m), qos)
    node.create_subscription(CameraInfo, COLOR_INFO, lambda m: box.__setitem__("ck", m), qos)
    node.create_subscription(Odometry, ODOM, lambda m: box.__setitem__("odom", m), qos)
    from std_msgs.msg import String
    pub = node.create_publisher(Twist, CMD, 10)
    status = {}
    node.create_subscription(String, "/motion_control/status", lambda m: status.__setitem__("raw", m.data), 10)
    _command(node, "agent", {"command": "request_control", "actor": "agent", "intent": "skill", "owner": "operator", "reason": "depth_creep"})
    t_mode = time.time()
    while time.time() - t_mode < 2.0:
        rclpy.spin_once(node, timeout_sec=0.05)
        if "raw" in status and json.loads(status["raw"]).get("mode") == "agent":
            break
    t0 = time.time()
    while time.time() - t0 < 5.0 and not all(k in box for k in ("d", "dk", "ck", "odom")):
        rclpy.spin_once(node, timeout_sec=0.1)
    if not all(k in box for k in ("d", "dk", "ck", "odom")):
        raise SystemExit("sensors missing")
    ck = box["ck"]
    rows = ground_guard_rows(float(ck.k[4]), float(ck.k[5]), int(ck.height))
    start = time.time()
    origin = None
    seg = None
    pause_until = 0.0
    path = 0.0
    last_xy = None
    boxed = 0
    drive_since = None
    drive_pose = None
    last_drive = 0.0
    try:
        while time.time() - start < RUN_S:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.time()
            depth, stamp = box["d"]
            od = box["odom"]
            p = od.pose.pose.position
            pose = (float(p.x), float(p.y), _yaw(od.pose.pose.orientation))
            if origin is None:
                origin = pose
                seg = pose
                last_xy = pose[:2]
            step = math.hypot(pose[0] - last_xy[0], pose[1] - last_xy[1])
            path += step
            last_xy = pose[:2]
            msg = Twist()
            fresh = now - stamp < 0.4
            front = left = right = float("inf")
            if fresh:
                front, left, right = _cones(depth, box["dk"], rows)
            open_fwd = fresh and front >= FRONT_GO and left >= SIDE_GO and right >= SIDE_GO
            if not fresh or front < FRONT_STOP or min(left, right) < 0.55:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                if fresh and front < FRONT_STOP and max(left, right) >= 1.0:
                    msg.angular.z = YAW if left >= right else -YAW
                if fresh and max(front, left, right) < 0.9:
                    boxed += 1
                else:
                    boxed = 0
            elif open_fwd:
                msg.linear.x = VX
                msg.angular.z = 0.0
                boxed = 0
            else:
                msg.linear.x = 0.0
                msg.angular.z = YAW if left >= right else -YAW
                if max(left, right) >= 1.0 and front >= 1.0:
                    msg.linear.x = 0.08
                if (msg.angular.z > 0 and left < 0.75) or (msg.angular.z < 0 and right < 0.75):
                    msg.angular.z = 0.0
                    msg.linear.x = 0.0
                boxed = 0
            hard = (not fresh) or front < FRONT_STOP or min(left, right) < 0.55
            if (
                not hard
                and TURN_FIRST > 0.0
                and abs(_wrap(pose[2] - origin[2])) < TURN_FIRST
                and fresh
                and front >= FRONT_STOP
            ):
                prefer = TURN_SIGN * YAW
                side = left if prefer > 0.0 else right
                if side >= 0.80:
                    msg.linear.x = 0.0
                    msg.angular.z = prefer
                elif max(left, right) >= 1.0:
                    msg.linear.x = 0.0
                    msg.angular.z = YAW if left >= right else -YAW
            moved = math.hypot(pose[0] - seg[0], pose[1] - seg[1])
            turned = abs(_wrap(pose[2] - seg[2]))
            if pause_until <= now and (msg.linear.x > 0 or abs(msg.angular.z) > 0) and (moved >= STEP_M or turned >= STEP_YAW):
                pause_until = now + PAUSE_S
                seg = pose
            if now < pause_until or path > PATH_M or boxed > 15 or math.hypot(pose[0] - origin[0], pose[1] - origin[1]) > RADIUS_M:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
            driving = msg.linear.x > 0.0 or abs(msg.angular.z) > 0.0
            if driving:
                last_drive = now
                if drive_since is None:
                    drive_since = now
                    drive_pose = pose
                elif math.hypot(pose[0] - drive_pose[0], pose[1] - drive_pose[1]) > 0.03 or abs(_wrap(pose[2] - drive_pose[2])) > 0.08:
                    drive_since = now
                    drive_pose = pose
                elif now - drive_since > 3.5:
                    msg.linear.x = 0.0
                    msg.angular.z = 0.0
                    pub.publish(msg)
                    print(json.dumps({"stop": True, "reason": "no_motion", "path_m": round(path, 2)}), flush=True)
                    break
            elif drive_since is not None and now - last_drive > 0.6:
                drive_since = None
                drive_pose = None
            if path > PATH_M or boxed > 15 or math.hypot(pose[0] - origin[0], pose[1] - origin[1]) > RADIUS_M:
                pub.publish(msg)
                print(json.dumps({"stop": True, "path_m": round(path, 2), "clear": [round(front, 2), round(left, 2), round(right, 2)]}), flush=True)
                break
            pub.publish(msg)
            if int((now - start) * 5) % 5 == 0:
                print(json.dumps({
                    "t": round(now - start, 1),
                    "vx": round(msg.linear.x, 2),
                    "yaw": round(msg.angular.z, 2),
                    "clear": [None if not math.isfinite(v) else round(v, 2) for v in (front, left, right)],
                    "path_m": round(path, 2),
                    "pose": [round(v, 2) for v in pose],
                }), flush=True)
            time.sleep(0.1)
    finally:
        zero = Twist()
        t1 = time.time()
        while time.time() - t1 < 1.5:
            pub.publish(zero)
            time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
