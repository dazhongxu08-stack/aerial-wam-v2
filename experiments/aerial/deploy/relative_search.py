#!/usr/bin/env python3
"""Find an object the way the drone mission does.

The sentence fixes one point: current pose plus the stated distance along
the current heading. The chassis drives to that point and looks on the way. A color patch does
not pull the chassis off the point. A find is recorded only when the same
region stays in view and has a depth, and the result includes that position.
A close obstacle stops the forward speed. The script does not command reverse.
"""
from __future__ import annotations

import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

VX = 0.10
ARC = 0.08
YAW = 0.20
FRONT_GO = 1.15
FRONT_STOP = 0.85
SIDE_GO = 0.80
ARRIVE_M = 0.55
STOP_FILE = "/tmp/relative_search.stop"
RESULT = "/tmp/relative_search.json"
PID_FILE = "/tmp/relative_search.pid"


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _meters(msg: Image) -> np.ndarray | None:
    if msg.encoding == "16UC1":
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)
        return raw.astype(np.float32) / 1000.0
    if msg.encoding == "32FC1":
        return np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(msg.height, msg.width)
    return None


def _cones(depth: np.ndarray) -> tuple[float, float, float]:
    h, w = depth.shape
    band = depth[int(0.22 * h):int(0.47 * h)]

    def side(x0: float, x1: float) -> float:
        part = band[:, int(x0 * w):int(x1 * w)]
        valid = part[(part > 0.30) & (part < 6.0)]
        if valid.size < 30:
            return math.inf
        return float(np.percentile(valid, 20))

    return side(0.33, 0.67), side(0.0, 0.33), side(0.67, 1.0)


def _blue_box(msg: Image) -> tuple[float, float, float, float, float, float] | None:
    """Return center fractions and the box as fractions of the image."""
    if msg.encoding != "rgb8":
        return None
    rgb = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, 3)
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (95, 80, 50), (130, 255, 255))
    h, w = mask.shape
    mask[: int(0.25 * h), :] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        if area < 3500 or bh < 50 or bw < 36 or bh > 3.0 * max(bw, 1):
            continue
        cx = (x + 0.5 * bw) / w
        cy = (y + 0.5 * bh) / h
        if not (0.20 <= cx <= 0.80 and 0.30 <= cy <= 0.85):
            continue
        if best is None or area > best[0]:
            best = (float(area), cx, cy, x / w, y / h, (x + bw) / w, (y + bh) / h)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4], best[5], best[6]


class Search(Node):
    def __init__(self) -> None:
        super().__init__("relative_search")
        self.distance_m = float(os.environ.get("REL_SEARCH_M", "5"))
        self.target = os.environ.get("REL_SEARCH_TARGET", "")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.create_subscription(Image, "/camera1/camera1/depth/image_rect_raw", self._depth, qos)
        self.create_subscription(Image, "/camera1/camera1/color/image_raw", self._color, qos)
        self.pub = self.create_publisher(Twist, "/nav/cmd_vel", 10)
        self.xy = None
        self.yaw = 0.0
        self.odom_at = 0.0
        self.front = self.left = self.right = math.inf
        self.depth_img = None
        self.depth_at = 0.0
        self.hit = None
        self.hits = 0
        self.phase = "transit"
        self.goal = None
        self.origin = None
        self.avoiding = False
        self.last_vx = 0.0
        self.found_pose = None

    def _odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.xy = (p.x, p.y)
        self.yaw = _yaw(msg.pose.pose.orientation)
        self.odom_at = time.time()
        if self.origin is None:
            self.origin = self.xy
            h = self.yaw
            self.goal = (
                p.x + self.distance_m * math.cos(h),
                p.y + self.distance_m * math.sin(h),
            )

    def _depth(self, msg: Image) -> None:
        depth = _meters(msg)
        if depth is None:
            return
        self.depth_img = depth
        self.front, self.left, self.right = _cones(depth)
        self.depth_at = time.time()

    def _box_range(self, box: tuple[float, float, float, float, float, float]) -> float | None:
        if self.depth_img is None:
            return None
        h, w = self.depth_img.shape
        x0, y0, x1, y1 = box[2], box[3], box[4], box[5]
        patch = self.depth_img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        valid = patch[(patch > 0.50) & (patch < 4.0)]
        if valid.size < 40:
            return None
        return float(np.median(valid))

    def _color(self, msg: Image) -> None:
        box = _blue_box(msg)
        rng = None if box is None else self._box_range(box)
        # The patch has to be the thing in front. A blue region elsewhere is not a find.
        if box is None or rng is None or not math.isfinite(self.front) or abs(rng - self.front) > 0.45:
            self.hits = max(0, self.hits - 1)
            if self.hits == 0:
                self.hit = None
            return
        cx = box[0]
        if self.hit is not None and abs(cx - self.hit[0]) < 0.06 and abs(rng - self.hit[1]) < 0.30:
            self.hits += 1
        else:
            self.hits = 1
        self.hit = (cx, rng)

    def remember_find(self) -> bool:
        if self.hits < 8 or self.hit is None or self.xy is None:
            return False
        cx, rng = self.hit
        bearing = (cx - 0.5) * math.radians(79.2)
        ang = self.yaw + bearing
        self.found_pose = {
            "robot_x": round(self.xy[0], 2),
            "robot_y": round(self.xy[1], 2),
            "robot_yaw": round(self.yaw, 2),
            "object_x": round(self.xy[0] + rng * math.cos(ang), 2),
            "object_y": round(self.xy[1] + rng * math.sin(ang), 2),
            "range_m": round(rng, 2),
        }
        return True

    def goal_dist(self) -> float:
        if self.xy is None or self.goal is None:
            return math.inf
        return math.hypot(self.goal[0] - self.xy[0], self.goal[1] - self.xy[1])

    def bearing(self) -> float:
        if self.xy is None or self.goal is None:
            return 0.0
        return _wrap(math.atan2(self.goal[1] - self.xy[1], self.goal[0] - self.xy[0]) - self.yaw)

    def fresh(self) -> bool:
        now = time.time()
        return now - self.odom_at < 0.4 and now - self.depth_at < 0.4 and math.isfinite(self.front)

    def command(self, vx: float, wz: float) -> None:
        twist = Twist()
        if not self.fresh():
            self.last_vx = 0.0
            self.pub.publish(twist)
            return
        front, left, right = self.front, self.left, self.right
        self.avoiding = False
        if front < FRONT_STOP:
            vx = 0.0
            wz = 0.0
        elif min(left, right) < 0.55 and abs(wz) > 0.0:
            toward = (wz > 0 and left <= right) or (wz < 0 and right < left)
            if toward:
                wz = 0.0
                vx = 0.0
        if abs(wz) > 0.0 and vx < 1e-3 and front >= FRONT_GO:
            side = left if wz > 0.0 else right
            if side >= SIDE_GO:
                vx = ARC
            else:
                wz = 0.0
        elif abs(wz) > 0.0 and vx < 1e-3:
            wz = 0.0
        vx = max(0.0, vx)
        self.last_vx = vx
        twist.linear.x = vx
        twist.angular.z = wz
        self.pub.publish(twist)

    def write(self, reason: str) -> None:
        body = {
            "reason": reason,
            "phase": self.phase,
            "target": self.target,
            "asked_m": self.distance_m,
            "goal_x": None if self.goal is None else round(self.goal[0], 2),
            "goal_y": None if self.goal is None else round(self.goal[1], 2),
            "robot_x": None if self.xy is None else round(self.xy[0], 2),
            "robot_y": None if self.xy is None else round(self.xy[1], 2),
            "robot_yaw": round(self.yaw, 2),
            "goal_dist_m": None if self.goal is None else round(self.goal_dist(), 2),
            "front_m": None if not math.isfinite(self.front) else round(self.front, 2),
            "found": reason == "found",
            "found_pose": self.found_pose,
        }
        open(RESULT, "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
        print(json.dumps(body, ensure_ascii=False), flush=True)


def main() -> None:
    open(PID_FILE, "w").write(str(os.getpid()))
    try:
        os.remove(STOP_FILE)
    except OSError:
        pass
    rclpy.init()
    node = Search()
    t0 = time.time()
    best = math.inf
    progress_at = t0
    status_at = t0
    scan_until = 0.0
    avoid_xy = None
    reason = "timeout"
    try:
        while rclpy.ok() and time.time() - t0 < 120.0:
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.time() - status_at > 1.0:
                status_at = time.time()
                print(
                    f"phase={node.phase} goal_dist={node.goal_dist():.2f} front={node.front:.2f} left={node.left:.2f} right={node.right:.2f} yaw={node.yaw:.2f} vx={node.last_vx:.2f} hits={node.hits}",
                    flush=True,
                )
            if os.path.exists(STOP_FILE):
                reason = "stopped"
                break
            if node.goal is None:
                if time.time() - t0 > 3.0:
                    reason = "no_odom"
                    break
                continue
            if not node.fresh() and time.time() - t0 > 2.0:
                reason = "no_depth"
                break
            dist = node.goal_dist()
            if dist < best - 0.15:
                best = dist
                progress_at = time.time()
            if node.avoiding:
                if avoid_xy is None:
                    avoid_xy = node.xy
                backed = 0.0 if avoid_xy is None or node.xy is None else math.hypot(
                    node.xy[0] - avoid_xy[0], node.xy[1] - avoid_xy[1]
                )
                if backed > 0.35:
                    reason = "blocked"
                    break
                progress_at = time.time()
            else:
                avoid_xy = None
                if time.time() - progress_at > 12.0 and node.phase == "transit":
                    reason = "blocked"
                    break
            if node.remember_find():
                reason = "found"
                break
            if node.phase == "transit" and dist <= ARRIVE_M:
                node.phase = "scan"
                scan_until = time.time() + 10.0
            if node.phase == "scan":
                if time.time() >= scan_until:
                    reason = "not_found"
                    break
                wz = YAW if int(time.time()) % 4 < 2 else -YAW
                node.command(0.0, wz)
                continue
            bear = node.bearing()
            if abs(bear) > 0.40:
                node.command(0.0, YAW if bear > 0.0 else -YAW)
            else:
                node.command(VX, float(np.clip(bear, -YAW, YAW)))
        node.write(reason)
    finally:
        end = time.time()
        zero = Twist()
        while time.time() - end < 0.5:
            node.pub.publish(zero)
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.shutdown()
        try:
            os.remove(PID_FILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
