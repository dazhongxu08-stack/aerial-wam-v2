#!/usr/bin/env python3
"""Independent office stack: align depth + nearmiss consequence, next-step survival.

Lives under /home/user/rt/office_align. It does not read or write the running
wire's goal, status, or weights, and it exits before opening the camera or the
GPU if yhs_wire is already up. Nothing is published unless ALIGN_ARM=1.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ALIGN_ROOT", "/home/user/rt/office_align"))
APP = Path(os.environ.get("ALIGN_APP", "/home/user/rt/app"))
DEPLOY = APP / "experiments" / "aerial" / "deploy"
DEPTH_PATH = Path(os.environ.get("ALIGN_DEPTH", str(ROOT / "depth_office_cone_align.pt")))
CONS_PATH = Path(os.environ.get("ALIGN_CONS", str(ROOT / "consequence_nearmiss.pt")))
STATUS_PATH = ROOT / "status.json"
GOAL_PATH = ROOT / "goal.json"
HEARTBEAT_PATH = ROOT / "watch.heartbeat"
SURV_MIN = 0.95
STOP_DIST_M = 0.40
GOAL_Z = 0.52
INVALID_M = 30.0
LIMIT = 0.3141592653589793
H = 10
FWD_DX = 0.8
YAW_CAP = 0.10
TRIM_YAW = 0.04
TRIM_RAD = math.radians(25.0)


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace")


def wire_pids() -> list[int]:
    """Processes of the other stack. This file is not one of them."""
    found = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        cmd = _cmdline(int(entry.name))
        if "yhs_align" in cmd:
            continue
        if (
            "yhs_wire.py" in cmd or "yhs_wire.pyc" in cmd or "bin/yhs_wire.sh" in cmd
            or "yhs_detour.py" in cmd or "yhs_detour.sh" in cmd
        ):
            found.append(int(entry.name))
    return found


def _check() -> int:
    print("root", ROOT)
    print("wire_pids", wire_pids())
    ok = True
    for path in (DEPTH_PATH, CONS_PATH):
        if not path.is_file():
            print("missing", path)
            ok = False
            continue
        print("file", path.name, path.stat().st_size)
    if not ok:
        return 1
    sys.path.insert(0, str(APP))
    sys.path.insert(0, str(DEPLOY))
    import torch
    from aerial_finder.nav.consequence_model import load_checkpoint

    model = load_checkpoint(CONS_PATH, device="cpu")
    blob = torch.load(str(CONS_PATH), map_location="cpu")
    print("cons_step", blob.get("step"), "near", blob.get("near"), "horizon", model.horizon)
    print("status", STATUS_PATH)
    print("goal", GOAL_PATH)
    print("check_ok")
    return 0


def goal_turn(rng, bearing: float) -> np.ndarray:
    action = np.zeros((H, 4), np.float64)
    left = float(bearing)
    for t in range(H):
        step = float(np.clip(left, -LIMIT, LIMIT))
        if abs(left) > 0.05:
            action[t, 3] = step
            action[t, 0] = float(rng.uniform(0.0, 0.25))
            left -= step
        else:
            action[t, 0] = float(rng.uniform(0.4, 1.0))
    return action


def goal_arc(bearing: float) -> np.ndarray:
    cap = YAW_CAP
    if abs(bearing) < TRIM_RAD:
        cap = min(YAW_CAP, TRIM_YAW)
    action = np.zeros((H, 4), np.float64)
    left = float(bearing)
    for t in range(H):
        step = float(np.clip(left, -cap, cap))
        action[t, 0] = FWD_DX
        action[t, 3] = step
        left -= step
    return action


def spin_in_place(sign: float, bearing: float) -> np.ndarray:
    action = np.zeros((H, 4), np.float64)
    toward = bearing * sign > 0.0
    mag = min(abs(float(bearing)), LIMIT) if toward else LIMIT
    left = mag
    for _t in range(H):
        step = min(left, LIMIT)
        action[_t, 3] = float(sign) * step
        left = max(0.0, left - step)
    return action


def candidates(rng, bearing: float):
    fwd = np.zeros((H, 4), np.float64)
    fwd[:, 0] = FWD_DX
    slow = np.zeros((H, 4), np.float64)
    slow[:, 0] = 0.35
    names = ["fwd", "slow", "goal_turn_fwd", "goal_arc", "spin_left", "spin_right"]
    stack = np.stack([
        fwd,
        slow,
        goal_turn(rng, bearing),
        goal_arc(bearing),
        spin_in_place(-1.0, bearing),
        spin_in_place(1.0, bearing),
    ])
    stack[:, :, 0] = np.maximum(stack[:, :, 0], 0.0)
    return names, stack


def pick_learned(names, cands, surv) -> tuple[np.ndarray, str, float]:
    surv = np.asarray(surv, np.float64)
    ok = np.flatnonzero(surv >= SURV_MIN)
    if ok.size:
        moving = [int(i) for i in ok if "spin" not in names[int(i)]]
        if moving:
            chosen = int(moving[int(np.argmax(cands[moving, 0, 0]))])
        else:
            chosen = int(ok[int(np.argmax(surv[ok]))])
        return cands[chosen, 0].astype(np.float64), str(names[chosen]), float(surv[chosen])
    return np.zeros(4, np.float64), "stop", float(np.max(surv))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _bearing(x: float, y: float, yaw: float, gx: float, gy: float) -> float:
    return _wrap(math.atan2(gy - y, gx - x) - yaw)


def _heartbeat_stale() -> bool:
    """A supervisor that stops updating this file has let go. Hold still."""
    if not HEARTBEAT_PATH.is_file():
        return False
    try:
        age = time.time() - HEARTBEAT_PATH.stat().st_mtime
    except OSError:
        return True
    return age > 2.0


def _read_goal(pose, frozen: dict):
    if _heartbeat_stale() or not GOAL_PATH.is_file():
        frozen.clear()
        return None
    try:
        body = json.loads(GOAL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return exc
    gid = str(body.get("id") or "")
    if "x" in body and "y" in body:
        return gid, float(body["x"]), float(body["y"])
    if "distance_m" not in body:
        return ValueError("goal needs x,y or distance_m")
    if gid not in frozen or frozen[gid][0] != "rel":
        if pose is None:
            return None
        x, y, _z, yaw = pose
        dist = float(body["distance_m"])
        bearing = float(body.get("bearing_rad", 0.0))
        gx = float(x) + dist * math.cos(float(yaw) + bearing)
        gy = float(y) + dist * math.sin(float(yaw) + bearing)
        frozen.clear()
        frozen[gid] = ("rel", gx, gy)
    return gid, frozen[gid][1], frozen[gid][2]


def _sensor_forward(depth_m):
    if depth_m is None:
        return None
    height, width = depth_m.shape[:2]
    patch = np.asarray(depth_m, np.float32)[
        int(height * 0.35):int(height * 0.55),
        int(width * 0.42):int(width * 0.58),
    ]
    finite = patch[np.isfinite(patch) & (patch > 0.15) & (patch < INVALID_M)]
    if finite.size < 20:
        return None
    return float(np.median(finite))


def _score_next(model, da56, state, goal, cands, prev, device: str):
    import torch
    from aerial_finder.nav.consequence_model import goal_features, history_inputs, survival

    count = len(cands)
    goal_feat = torch.as_tensor(np.repeat(goal_features(state, goal)[None], count, 0), device=device)
    prev_action = torch.as_tensor(np.repeat(np.asarray(prev, np.float32)[None], count, 0), device=device)
    actions = torch.as_tensor(np.asarray(cands, np.float32), device=device)
    with torch.no_grad():
        depth_k, _hist = history_inputs([da56], [], model.frame_offsets, model.hist_actions)
        if len(model.frame_offsets) == 1:
            depth_k = depth_k[0]
        depth = torch.as_tensor(np.repeat(depth_k[None], count, 0), device=device)
        out = model(depth, goal_feat, prev_action, actions, None)
        curve = survival(out["hazard_logit"]).float().cpu().numpy()
    return curve[:, 0]


def run() -> int:
    pids = wire_pids()
    if pids:
        print(f"another office stack is running ({pids}); align stack not started", file=sys.stderr)
        return 3
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["AERIAL_DA3_FP16"] = "0"
    os.environ["YHS_STATUS"] = str(STATUS_PATH)
    os.environ["YHS_SEARCH_GOAL"] = str(GOAL_PATH)
    sys.path.insert(0, str(APP))
    sys.path.insert(0, str(DEPLOY))

    import cv2
    import rclpy.node as rclpy_node
    import torch
    from aerial_finder.nav.consequence_model import load_checkpoint
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation
    from aerial_finder.nav.consequence_model import pool_min
    from pinhole90 import warp_color

    import yhs_wire

    yhs_wire.STATUS = STATUS_PATH
    node_cls = rclpy_node.Node

    class _Renamed(node_cls):
        def __init__(self, node_name, *args, **kwargs):
            if node_name == "yhs_wire":
                node_name = "yhs_align"
            super().__init__(node_name, *args, **kwargs)

    rclpy_node.Node = _Renamed
    armed = os.environ.get("ALIGN_ARM") == "1"
    wire = yhs_wire.YhsWire(armed=armed)
    depth_model = DepthMinPredictor.from_checkpoint(str(DEPTH_PATH), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    cons = load_checkpoint(CONS_PATH, device="cuda")
    period = 1.0 / yhs_wire.STEP_HZ
    prev = np.zeros(4, np.float64)
    frozen: dict = {}
    rng_key = None
    rng = np.random.default_rng(0)
    print("align_ready", "armed" if armed else "idle", flush=True)

    while True:
        if wire_pids():
            wire.publish_stop()
            print("another office stack appeared; align stack stopping", file=sys.stderr)
            return 3
        loop_start = time.time()
        color, sensor_depth, pose, color_k, color_d, _depth_k, color_at, _depth_at, odom_at = wire.snapshot()
        goal = _read_goal(pose, frozen)
        fresh = (
            color is not None and color_k is not None and pose is not None
            and time.time() - color_at < 1.0 and time.time() - odom_at < 1.0
        )
        if isinstance(goal, Exception) or goal is None or not fresh:
            prev = np.zeros(4, np.float64)
            wire.publish_stop()
            wire.inference_ok = goal is None or fresh
            wire.error = "" if not isinstance(goal, Exception) else str(goal)
            wire.write_status({
                "stack": "office_align",
                "goal_mode": "error" if isinstance(goal, Exception) else "idle",
                "publishing": False,
                "office_family": "stop",
            })
            time.sleep(period)
            continue
        gid, gx, gy = goal
        if gid != rng_key:
            rng = np.random.default_rng(0)
            rng_key = gid
            prev = np.zeros(4, np.float64)
        x, y, z, yaw = (float(v) for v in pose)
        dist = math.hypot(gx - x, gy - y)
        if dist <= STOP_DIST_M:
            prev = np.zeros(4, np.float64)
            wire.publish_stop()
            wire.inference_ok = True
            wire.error = ""
            wire.write_status({
                "stack": "office_align",
                "goal_mode": "arrived",
                "goal_dist_m": round(dist, 2),
                "publishing": False,
                "office_family": "stop",
            })
            time.sleep(period)
            continue
        try:
            rgb, seen = warp_color(color, color_k, color_d)
            pred = depth_model.predict_depth(
                Observation(rgb=rgb, state=np.zeros(7, np.float32), collided=False))
            if pred is None:
                raise RuntimeError("depth model returned no map")
            pred = np.asarray(pred, np.float32)
            if seen.shape != pred.shape[:2]:
                seen = cv2.resize(
                    seen.astype(np.uint8), (pred.shape[1], pred.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            metric = np.where(seen, pred, INVALID_M).astype(np.float32)
            fy, cy = float(color_k[1, 1]), float(color_k[1, 2])
            hidden, _row = yhs_wire.hide_ground(metric, fy, cy, color.shape[0])
            da56 = pool_min(hidden)
            state = np.zeros(7, np.float64)
            state[0], state[1], state[2], state[6] = x, y, GOAL_Z if z == 0.0 else z, yaw
            goal_xyz = np.array([gx, gy, GOAL_Z], np.float64)
            bearing = _bearing(x, y, yaw, gx, gy)
            names, cands = candidates(rng, bearing)
            surv = _score_next(cons, da56, state, goal_xyz, cands, prev, "cuda")
            action, family, surv_pick = pick_learned(names, cands, surv)
            prev = np.asarray(action, np.float64)
            if family == "stop":
                wire.publish_stop()
                publishing = False
            else:
                wire.publish_action(action)
                publishing = bool(wire.armed)
            wire.inference_ok = True
            wire.error = ""
            wire.write_status({
                "stack": "office_align",
                "goal_mode": "point",
                "goal_id": gid,
                "goal_dist_m": round(dist, 2),
                "publishing": publishing,
                "office_family": family,
                "surv_pick": round(surv_pick, 3),
                "realsense_fwd_m": None if _sensor_forward(sensor_depth) is None
                else round(_sensor_forward(sensor_depth), 2),
                "metric_scale": 1.0,
            })
        except Exception as exc:  # noqa: BLE001
            prev = np.zeros(4, np.float64)
            wire.publish_stop()
            wire.inference_ok = False
            wire.error = str(exc)
            wire.write_status({
                "stack": "office_align",
                "goal_mode": "error",
                "publishing": False,
                "office_family": "stop",
            })
        remain = period - (time.time() - loop_start)
        if remain > 0:
            time.sleep(remain)


def main() -> int:
    if "--check" in sys.argv:
        return _check()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
