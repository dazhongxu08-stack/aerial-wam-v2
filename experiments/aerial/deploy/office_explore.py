#!/usr/bin/env python3
"""Slow office search with the deployed consequence stack, front camera only.

The depth model turns the front color image into range. The consequence
navigator picks forward or turn. The depth camera is only a last-resort brake.
Speed stays low. A top-down map is drawn from the monocular ranges and
odometry, then cropped to the area the robot actually saw.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("AERIAL_DA3_FP16", "1")

COLOR_TOPIC = "/camera1/camera1/color/image_raw"
INFO_TOPIC = "/camera1/camera1/color/camera_info"
DEPTH_TOPIC = "/camera1/camera1/depth/image_rect_raw"
DEPTH_INFO = "/camera1/camera1/depth/camera_info"
ODOM_TOPIC = "/odom"
CMD_TOPIC = "/nav/cmd_vel"
WEIGHTS = Path(os.environ.get("RT_RAM", "/dev/shm/rtw")) / "weights"
SCALE_FILE = Path(os.environ.get("YHS_SCALE", "/home/user/rt/da3_scale.json"))
MAP_PNG = Path(os.environ.get("YHS_MAP", "/home/user/rt/office_map.png"))
STATUS = Path(os.environ.get("YHS_EXPLORE_STATUS", "/home/user/rt/explore_status.json"))

VX = 0.10
YAW_RATE = 0.20
SEARCH_S = float(os.environ.get("YHS_SEARCH_S", "100"))
RES = 0.05
SPAN = 24.0
RAY_MAX = 6.0


def _rgb(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding.lower()
    if enc not in ("rgb8", "bgr8"):
        raise RuntimeError(f"unsupported color encoding {msg.encoding}")
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = np.ndarray((h, w, 3), dtype=np.uint8, buffer=buf, strides=(int(msg.step), 3, 1))
    if enc == "bgr8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _depth_m(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(msg.data, dtype=np.uint16)
    raw = np.ndarray((h, w), dtype=np.uint16, buffer=buf, strides=(int(msg.step), 2))
    depth = raw.astype(np.float32) * 0.001
    depth[raw == 0] = np.nan
    return depth


def _twist(action: np.ndarray, real):
    """Turn a model action into a slow command. The depth camera can only cancel it."""
    from geometry_msgs.msg import Twist
    from pinhole90 import brake

    dx, _dy, _dz, dyaw = (float(v) for v in np.asarray(action, np.float64))
    msg = Twist()
    msg.linear.y = 0.0
    msg.linear.x = VX if dx > 1e-3 else 0.0
    msg.angular.z = 0.0 if abs(dyaw) < 1e-3 else math.copysign(YAW_RATE, -dyaw)
    if real is None:
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        return msg
    return brake(msg, real, YAW_RATE)


class Grid:
    def __init__(self) -> None:
        self.n = int(SPAN / RES)
        self.log = np.zeros((self.n, self.n), np.float32)
        self.seen = np.zeros((self.n, self.n), np.uint8)
        self.path: list[tuple[int, int]] = []
        self.origin = None

    def set_origin(self, x: float, y: float) -> None:
        self.origin = (x - SPAN * 0.5, y - SPAN * 0.5)

    def ij(self, x: float, y: float):
        ix = int((x - self.origin[0]) / RES)
        iy = int((y - self.origin[1]) / RES)
        if 0 <= ix < self.n and 0 <= iy < self.n:
            return ix, iy
        return None

    def mark_path(self, x: float, y: float) -> None:
        hit = self.ij(x, y)
        if hit is not None:
            self.path.append(hit)

    def ray(self, x: float, y: float, yaw: float, ranges: np.ndarray, hfov: float) -> None:
        cols = len(ranges)
        for i, dist in enumerate(ranges):
            dist = float(dist)
            if not math.isfinite(dist) or dist <= 0.15:
                continue
            bearing = hfov * 0.5 - (i + 0.5) / cols * hfov
            heading = yaw + bearing
            reach = min(dist, RAY_MAX)
            occupied = dist < RAY_MAX - 0.05
            steps = max(int(reach / RES), 1)
            last = None
            for s in range(steps + 1):
                t = reach * s / steps
                hit = self.ij(x + t * math.cos(heading), y + t * math.sin(heading))
                if hit is None or hit == last:
                    continue
                last = hit
                self.seen[hit[1], hit[0]] = 1
                end = occupied and s == steps
                self.log[hit[1], hit[0]] = np.clip(self.log[hit[1], hit[0]] + (0.8 if end else -0.15), -4.0, 4.0)

    def save(self, path: Path) -> dict:
        import cv2

        ys, xs = np.nonzero(self.seen)
        if xs.size == 0:
            img = np.full((64, 64, 3), 220, np.uint8)
            cv2.imwrite(str(path), img)
            return {"cells": 0}
        pad = 8
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, self.n)
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, self.n)
        sub_log = self.log[y0:y1, x0:x1]
        sub_seen = self.seen[y0:y1, x0:x1]
        img = np.full(sub_log.shape + (3,), 210, np.uint8)
        free = sub_seen.astype(bool) & (sub_log < -0.2)
        occ = sub_log > 0.6
        img[free] = (245, 245, 245)
        img[occ] = (20, 20, 20)
        for ix, iy in self.path:
            if x0 <= ix < x1 and y0 <= iy < y1:
                img[iy - y0, ix - x0] = (40, 90, 220)
        img = np.flipud(img)
        cv2.imwrite(str(path), img)
        return {
            "cells": int(xs.size),
            "occupied": int(occ.sum()),
            "width_m": round((x1 - x0) * RES, 2),
            "height_m": round((y1 - y0) * RES, 2),
            "png": str(path),
        }


class Explorer:
    def __init__(self) -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image

        self.lock = threading.Lock()
        self.color = None
        self.depth = None
        self.pose = None
        self.color_k = None
        self.color_d = None
        self.depth_k = None
        self.color_hw = (0, 0)
        self.hfov = math.radians(90.0)
        self.twist = None
        self.stop = False
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        rclpy.init()
        self.node = Node("office_explore")
        self.node.create_subscription(Image, COLOR_TOPIC, self._on_color, qos)
        self.node.create_subscription(Image, DEPTH_TOPIC, self._on_depth, qos)
        self.node.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, qos)
        self.node.create_subscription(CameraInfo, INFO_TOPIC, self._on_info, qos)
        self.node.create_subscription(CameraInfo, DEPTH_INFO, self._on_depth_info, qos)
        self.pub = self.node.create_publisher(Twist, CMD_TOPIC, 10)
        self._spin = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin.start()
        self._hold = threading.Thread(target=self._republish, daemon=True)
        self._hold.start()

    def _on_color(self, msg) -> None:
        try:
            rgb = _rgb(msg)
        except Exception:
            return
        with self.lock:
            self.color = rgb

    def _on_depth(self, msg) -> None:
        try:
            depth = _depth_m(msg)
        except Exception:
            return
        with self.lock:
            self.depth = depth

    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        with self.lock:
            self.pose = (float(p.x), float(p.y), float(p.z), _yaw(msg.pose.pose.orientation))

    def _on_info(self, msg) -> None:
        from pinhole90 import k_from_msg

        with self.lock:
            self.color_k = k_from_msg(msg)
            self.color_d = np.array(msg.d, np.float64)
            self.color_hw = (int(msg.height), int(msg.width))

    def _on_depth_info(self, msg) -> None:
        from pinhole90 import k_from_msg

        with self.lock:
            self.depth_k = k_from_msg(msg)

    def snapshot(self):
        with self.lock:
            return self.color, self.pose, self.hfov

    def geometry(self):
        with self.lock:
            return self.color_k, self.color_d, self.color_hw, self.depth, self.depth_k

    def set_twist(self, twist) -> None:
        with self.lock:
            self.twist = twist

    def _republish(self) -> None:
        import rclpy

        while rclpy.ok() and not self.stop:
            with self.lock:
                twist = self.twist
            if twist is not None:
                self.pub.publish(twist)
            time.sleep(0.05)

    def halt(self) -> None:
        from geometry_msgs.msg import Twist

        zero = Twist()
        self.set_twist(zero)
        for _ in range(20):
            self.pub.publish(zero)
            time.sleep(0.05)


def _goal(pose, front: float, left: float, right: float):
    x, y, z, yaw = pose
    if left > front + 0.4 and left >= right:
        bearing = 0.7
    elif right > front + 0.4 and right > left:
        bearing = -0.7
    else:
        bearing = 0.0
    dist = 3.0
    return np.array([
        x + dist * math.cos(yaw + bearing),
        y + dist * math.sin(yaw + bearing),
        z,
    ], np.float64), bearing


def _write(body: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False))
    tmp.replace(STATUS)


def main() -> None:
    from aerial_finder.nav.consequence_model import pool_min
    from aerial_finder.nav.navigator import ConsequenceNavigator, NavConfig
    from aerial_finder.types import VehicleState
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation
    from pinhole90 import ground_guard_rows, load_scale, mask_unseen, warp_color, warp_depth, band_mins

    bot = Explorer()
    grid = Grid()
    try:
        t0 = time.time()
        while time.time() - t0 < 20.0:
            color, pose, _hfov = bot.snapshot()
            color_k, _color_d, color_hw, sensor_depth, depth_k = bot.geometry()
            if color is not None and pose is not None and color_k is not None and sensor_depth is not None and depth_k is not None:
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("front camera or odom did not arrive")
        grid.set_origin(pose[0], pose[1])
        depth = DepthMinPredictor.from_checkpoint(WEIGHTS / "m_d.bin", device="cuda")
        scale = load_scale(str(SCALE_FILE))
        if scale is None:
            scale = load_scale(str(Path(__file__).with_name("da3_scale.json")))
        if scale is None:
            scale = 1.0
        depth.forward_scale = scale
        rows = ground_guard_rows(float(color_k[1, 1]), float(color_k[1, 2]), int(color_hw[0]))
        nav = ConsequenceNavigator(
            [WEIGHTS / f"m_n{i}.bin" for i in range(3)],
            [WEIGHTS / f"m_g{i}.bin" for i in range(3)],
            device="cuda",
            cfg=NavConfig(
                surv_min=0.75,
                fwd_dx=0.8,
                fov_max_deg=35.0,
                seed=0,
                families=("fwd", "yaw_fwd", "yaw_fwd", "yaw_fwd", "yaw_fwd", "goal_turn_fwd", "yaw_only"),
                action_scale=(0.06, 0.0, 0.0, 0.255),
                macro_set="dog",
                step_hz=5.0,
                replan_every=3,
                guard_turn_m=1.2,
                guard_stop_m=0.8,
                guard_yaw=0.06,
                guard_rows=rows,
            ),
        )
        period = 0.2
        start = time.time()
        origin = pose
        path_m = 0.0
        last_xy = pose[:2]
        boxed = 0
        last_save = 0.0
        info = {}
        while time.time() - start < SEARCH_S:
            tick = time.time()
            color, pose, _hfov = bot.snapshot()
            color_k, color_d, _hw, sensor_depth, depth_k = bot.geometry()
            if color is None or pose is None or color_k is None or sensor_depth is None or depth_k is None:
                bot.set_twist(_twist(np.zeros(4), None))
                time.sleep(0.05)
                continue
            rgb, seen = warp_color(color, color_k, color_d)
            pred = depth.predict_depth(Observation(rgb=rgb, state=np.zeros(7, np.float32), collided=False))
            if pred is None:
                time.sleep(period)
                continue
            pred = mask_unseen(pred, seen)
            frames = list(depth._hist)
            while len(frames) < depth.n_frames:
                frames.insert(0, frames[0])
            env = None
            model = depth._model
            if hasattr(model, "encode_env"):
                import torch

                stack = np.stack(frames[-depth.n_frames :], axis=0)
                tensor = torch.from_numpy(stack).unsqueeze(0).to(depth.device)
                with torch.no_grad():
                    env = model.encode_env(tensor).squeeze(0).float().cpu().numpy()
            nav.observe(pred, env, None)
            d56 = np.asarray(pool_min(pred), np.float32)
            ranges = d56[rows[0]:rows[1]].min(axis=0)
            x, y, z, yaw = pose
            grid.ray(x, y, yaw, ranges, math.radians(90.0))
            grid.mark_path(x, y)
            front, left, right = nav._front_sides()
            real_map = warp_depth(sensor_depth, depth_k)
            real = band_mins(pool_min(np.where(np.isfinite(real_map), real_map, np.inf)), *rows)
            goal, bearing = _goal(pose, front, left, right)
            action = nav.act(VehicleState(pos=np.array([x, y, z]), yaw=yaw), goal)
            bot.set_twist(_twist(action, real))
            step = math.hypot(x - last_xy[0], y - last_xy[1])
            path_m += step
            last_xy = (x, y)
            if front < 0.55 and left < 0.7 and right < 0.7:
                boxed += 1
            else:
                boxed = 0
            if time.time() - last_save > 5.0:
                info = grid.save(MAP_PNG)
                last_save = time.time()
            _write({
                "searching": True,
                "elapsed_s": round(time.time() - start, 1),
                "path_m": round(path_m, 2),
                "hfov_deg": 90.0,
                "clear_m": [round(front, 2), round(left, 2), round(right, 2)],
                "real_clear_m": [None if not math.isfinite(v) else round(float(v), 2) for v in real],
                "bearing": round(bearing, 2),
                "vx": round(bot.twist.linear.x, 3) if bot.twist is not None else 0.0,
                "vyaw": round(bot.twist.angular.z, 3) if bot.twist is not None else 0.0,
                "pose": [round(v, 3) for v in pose],
                "map": info,
            })
            if path_m > 8.0 or boxed >= 8:
                break
            if math.hypot(x - origin[0], y - origin[1]) > 6.0:
                break
            time.sleep(max(0.0, period - (time.time() - tick)))
        info = grid.save(MAP_PNG)
        _write({"searching": False, "path_m": round(path_m, 2), "map": info, "pose": [round(v, 3) for v in (pose or origin)]})
    finally:
        bot.halt()
        bot.stop = True
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
