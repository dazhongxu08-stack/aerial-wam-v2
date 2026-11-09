#!/usr/bin/env python3
"""Consequence episodes for the Avant-AGSim Go1, in that body's own camera and command.

``run(session)`` is the navigation entry ``Engine.sh`` loads. Each step sends
``body_velocity`` in the Go1 frame (forward m/s, left m/s, yaw rate rad/s), the
ranges the pinned Playground policy was trained to track. The image is the Go1
face camera (pinhole, 90 degree horizontal field). That camera contract is RGB
only, so ``depth56`` is left empty and marked; it is not the old street-depth
grid. Labels are the pose, the command that was sent, and whether the body fell.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

MACROS = ("fwd", "yaw_fwd", "strafe", "yaw_only", "smooth_random", "hold")
MACRO_P = np.array([0.28, 0.22, 0.14, 0.12, 0.16, 0.08])
# Playground Go1 joystick command box: vx, vy, yaw rate.
CMD_LIMIT = np.array([1.5, 0.8, 1.2], dtype=np.float64)
HOLD_S = 0.5
ROBOT = "Ground_01"
UP_MIN = 0.7


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _pose(state: dict) -> tuple:
    pose = state["rendered_body_pose"]
    matrix = np.asarray(pose["matrix"], dtype=np.float64)
    pos = matrix[:3, 3].copy()
    forward = matrix[:3, 0]
    yaw = math.atan2(float(forward[1]), float(forward[0]))
    return pos, yaw, float(matrix[2, 2]), matrix


def _rgb(image) -> np.ndarray:
    raw = np.frombuffer(bytes(image.raw_data), dtype=np.uint8)
    bgra = raw.reshape(int(image.height), int(image.width), 4)
    return bgra[:, :, :3][:, :, ::-1].copy()


def _jpeg(rgb: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _command(rng: np.random.Generator, name: str, k: int, bearing: float) -> np.ndarray:
    """Rows are forward m/s, left m/s, yaw rate rad/s."""
    c = np.zeros((k, 3), dtype=np.float64)
    lim = CMD_LIMIT
    if name == "fwd":
        c[:, 0] = rng.uniform(0.3, 1.0)
    elif name == "yaw_fwd":
        m = int(rng.integers(2, k))
        c[:m, 2] = math.copysign(float(rng.uniform(0.4, 0.9)), bearing if abs(bearing) > 0.05 else rng.choice([-1.0, 1.0]))
        c[:m, 0] = rng.uniform(0.0, 0.25)
        c[m:, 0] = rng.uniform(0.3, 1.0)
    elif name == "strafe":
        c[:, 0] = rng.uniform(0.1, 0.5)
        c[:, 1] = rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.45)
    elif name == "yaw_only":
        c[:, 2] = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 0.8)
    elif name == "smooth_random":
        u = np.zeros(3)
        for t in range(k):
            u = 0.7 * u + 0.3 * rng.normal(0.0, 1.0, size=3)
            c[t, 0] = 0.55 * (1.0 + math.tanh(u[0]))
            c[t, 1] = 0.4 * math.tanh(u[1])
            c[t, 2] = 0.8 * math.tanh(u[2])
    elif name != "hold":
        raise ValueError(f"unknown macro {name}")
    c[:, 0] = np.clip(c[:, 0], 0.0, lim[0])
    c[:, 1] = np.clip(c[:, 1], -lim[1], lim[1])
    c[:, 2] = np.clip(c[:, 2], -lim[2], lim[2])
    return c


def _step_cmd(session, forward: float, left: float, yaw_rate: float, frames: int):
    return session.step(
        [dict(robot=ROBOT, operation="body_velocity",
              parameters=dict(frame="BODY_RH", forward_m_s=float(forward),
                              left_m_s=float(left), yaw_rate_rad_s=float(yaw_rate)))],
        frames=int(frames),
    )


class _Buf:
    def __init__(self) -> None:
        self.jpeg, self.state, self.action = [], [], []
        self.cmd, self.body_vel, self.collided = [], [], []
        self.macro, self.segment = [], []
        self.hw = None

    def add(self, rgb, state, action, cmd, body_vel, collided, macro, segment) -> None:
        self.hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        self.jpeg.append(_jpeg(rgb))
        self.state.append(np.asarray(state, np.float32))
        self.action.append(np.asarray(action, np.float32))
        self.cmd.append(np.asarray(cmd, np.float32))
        self.body_vel.append(np.asarray(body_vel, np.float32))
        self.collided.append(bool(collided))
        self.macro.append(int(macro))
        self.segment.append(int(segment))

    def __len__(self) -> int:
        return len(self.state)

    def save(self, path: Path, meta: dict) -> None:
        offsets = np.cumsum([0] + [len(b) for b in self.jpeg]).astype(np.int64)
        n, h = len(self.state), 56
        empty = np.full((n, h, h), np.nan, np.float16)
        profile = np.full((n, h), np.nan, np.float16)
        np.savez(
            path,
            jpeg=np.frombuffer(b"".join(self.jpeg), dtype=np.uint8),
            jpeg_offsets=offsets,
            depth56=empty,
            cones=np.full((n, 5), np.nan, np.float32),
            profile=profile,
            state=np.stack(self.state),
            action=np.stack(self.action),
            collided=np.asarray(self.collided, dtype=bool),
            macro=np.asarray(self.macro, dtype=np.int16),
            segment=np.asarray(self.segment, dtype=np.int32),
            cmd=np.stack(self.cmd),
            body_vel=np.stack(self.body_vel),
            meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        )


def _episode(session, rng: np.random.Generator, frames: int, args: dict) -> tuple:
    session.reset()
    obs = _step_cmd(session, 0.0, 0.0, 0.0, frames)
    pos, yaw, up, _ = _pose(obs.states[ROBOT])
    goal = pos + np.array([math.cos(yaw), math.sin(yaw), 0.0]) * float(args["goal_m"])
    buf = _Buf()
    prev = (pos.copy(), yaw, obs.time_ns)
    buf.add(_rgb(obs.images[ROBOT]["front"]), [pos[0], pos[1], pos[2], 0, 0, 0, yaw, up],
            [0, 0, 0, 0], [0, 0, 0], [0, 0, 0], up < UP_MIN, -1, -1)
    end, seg = "max_steps", 0
    while len(buf) <= int(args["max_steps"]):
        bearing = _wrap(math.atan2(goal[1] - pos[1], goal[0] - pos[0]) - yaw)
        name = str(rng.choice(MACROS, p=MACRO_P))
        seq = _command(rng, name, int(rng.choice([4, 6, 8])), bearing)
        mid = MACROS.index(name)
        for row in seq:
            buf.action[-1] = np.array([row[0], row[1], 0.0, row[2]], np.float32)
            buf.macro[-1] = mid
            buf.segment[-1] = seg
            obs = _step_cmd(session, row[0], row[1], row[2], frames)
            pos, yaw, up, _ = _pose(obs.states[ROBOT])
            dt = max((obs.time_ns - prev[2]) / 1e9, 1e-6)
            moved = pos - prev[0]
            speed = float(np.linalg.norm(moved[:2]) / dt)
            yaw_rate = _wrap(yaw - prev[1]) / dt
            # Forward component of the measured planar velocity, body frame.
            fwd = float(moved[0] * math.cos(prev[1]) + moved[1] * math.sin(prev[1])) / dt
            left = float(-moved[0] * math.sin(prev[1]) + moved[1] * math.cos(prev[1])) / dt
            fell = up < UP_MIN
            buf.add(_rgb(obs.images[ROBOT]["front"]),
                    [pos[0], pos[1], pos[2], fwd, left, moved[2] / dt, yaw, up],
                    [0, 0, 0, 0], row, [fwd, left, yaw_rate], fell, -1, -1)
            prev = (pos.copy(), yaw, obs.time_ns)
            if fell:
                end = "fell"
                break
            if float(np.linalg.norm(pos[:2] - goal[:2])) <= float(args["arrive_m"]):
                end = "arrived"
                break
            if len(buf) >= 8:
                st = np.stack(buf.state[-8:])
                cmd = np.stack(buf.cmd[-7:])
                travel = float(np.linalg.norm(st[-1, :2] - st[0, :2]))
                asked = float(np.sum(np.linalg.norm(cmd[:, :2], axis=1))) * HOLD_S
                if asked > 0.8 and travel < 0.15:
                    end = "stuck"
                    break
            if len(buf) > int(args["max_steps"]):
                break
        seg += 1
        if end != "max_steps":
            break
    meta = {
        "vehicle": "unitree_go1",
        "map": "WarehousePark",
        "command_frame": "BODY_RH",
        "command_units": ["forward_m_s", "left_m_s", "unused", "yaw_rate_rad_s"],
        "command_limits": CMD_LIMIT.tolist(),
        "state_layout": ["x", "y", "z", "v_forward", "v_left", "v_up", "yaw", "up_dot"],
        "pose_frame": "ENGINE_WORLD_RH",
        "body_frame": "BODY_FLU",
        "camera": "front",
        "camera_contract": "robot_profile_camera_optical_face",
        "hfov_deg": 90.0,
        "image_hw": buf.hw,
        "depth_source": "absent_rgb_only_face_camera",
        "hold_s": HOLD_S,
        "end_reason": end,
        "steps": len(buf) - 1,
        "segments": seg,
        "goal_m": [float(x) for x in goal],
        "macros": list(MACROS),
    }
    return buf, meta


def run(session) -> dict:
    out = Path(os.environ.get("GO1_OUT", "/data/go1_consequence_20261021"))
    episodes = out / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    review_path = out / "AUTO_REVIEW.json"
    review = json.loads(review_path.read_text()) if review_path.is_file() else {
        "vehicle": "unitree_go1", "map": "WarehousePark", "episodes": [],
        "note": "WarehousePark roads, Go1 face camera, body_velocity commands. Building interiors were not entered. depth56 is empty because that camera is RGB only.",
    }
    args = {"max_steps": int(os.environ.get("GO1_MAX_STEPS", "80")),
            "goal_m": float(os.environ.get("GO1_GOAL_M", "25")),
            "arrive_m": float(os.environ.get("GO1_ARRIVE_M", "1.5")),
            "target_kept": int(os.environ.get("GO1_TARGET_KEPT", "20"))}
    frames = max(1, int(round(HOLD_S / (session.host_step_ns / 1e9))))
    rng = np.random.default_rng(int(os.environ.get("GO1_SEED", "20261021")) + len(review["episodes"]))
    stop = out / "STOP"
    kept = sum(1 for ep in review["episodes"] if ep.get("keep"))
    t0 = time.time()
    while kept < args["target_kept"]:
        if stop.exists() or session._cancelled():
            break
        idx = len(review["episodes"])
        try:
            buf, meta = _episode(session, rng, frames, args)
            travel = 0.0
            if len(buf) > 1:
                st = np.stack(buf.state)
                travel = float(np.linalg.norm(st[-1, :2] - st[0, :2]))
            keep = meta["end_reason"] in {"arrived", "max_steps", "stuck"} and travel >= 1.0 and meta["steps"] >= 8
            # A fall after real motion is still a labeled episode.
            if meta["end_reason"] == "fell" and travel >= 1.0 and meta["steps"] >= 8:
                keep = True
            meta.update(index=idx, keep=keep, travel_m=round(travel, 3),
                        reason="kept" if keep else "short_or_failed")
            if keep:
                buf.save(episodes / f"ep_{idx:05d}.npz", meta)
                kept += 1
        except Exception as exc:
            review["episodes"].append({"index": idx, "keep": False, "end_reason": f"error:{exc}"})
            review_path.write_text(json.dumps(review, indent=2))
            raise
        review["episodes"].append({k: meta[k] for k in
                                   ("index", "keep", "end_reason", "steps", "travel_m", "reason", "image_hw")})
        review["kept"] = kept
        review["wall_s"] = round(time.time() - t0, 1)
        review_path.write_text(json.dumps(review, indent=2))
        print(f"go1 ep {idx} {meta['end_reason']} travel={travel:.2f} keep={keep} kept={kept}", flush=True)
    return {"kept": kept, "episodes": len(review["episodes"]), "out": str(out)}
