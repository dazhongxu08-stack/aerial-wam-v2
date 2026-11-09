"""Resolve 起点 to the pose where the current office task started.

The map has a place named 起点. On this chassis that word means the departure
recorded when the current task latched, and the driving program needs those
coordinates rather than the map name.
"""
from __future__ import annotations

import json
import math
import os
import re

START_POINTS = frozenset({"起点", "原点", "出发点", "出发原点"})
_RETURN = re.compile(r"回到.{0,12}(原点|起点|出发)")


def point_name(parameters: dict | None) -> str:
    params = parameters or {}
    nested = params.get("skill_params")
    if isinstance(nested, dict) and nested.get("point"):
        return str(nested.get("point")).strip()
    if params.get("point"):
        return str(params.get("point")).strip()
    return ""


def is_task_start_request(parameters: dict | None, user_text: str | None) -> bool:
    if point_name(parameters) in START_POINTS:
        return True
    text = user_text or ""
    if _RETURN.search(text):
        return True
    return "出发的原点" in text or "出发的起点" in text


def current_task_departure(track_dir: str) -> tuple[float, float, float] | None:
    """Newest task that is still looking for something. Its latch origin is the departure."""
    try:
        names = [name for name in os.listdir(track_dir) if name.endswith(".jsonl")]
    except OSError:
        return None
    paths = [os.path.join(track_dir, name) for name in names]
    paths.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                line = handle.readline()
        except OSError:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") != "latch":
            continue
        if not str(row.get("target") or "").strip():
            continue
        origin = row.get("origin")
        if not isinstance(origin, (list, tuple)) or len(origin) < 2:
            continue
        try:
            x = float(origin[0])
            y = float(origin[1])
            z = float(origin[2]) if len(origin) >= 3 else 0.0
        except (TypeError, ValueError):
            continue
        return x, y, z
    return None


def chassis_pose(status_path: str) -> tuple[float, float, float] | None:
    try:
        data = json.loads(open(status_path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError):
        return None
    pose = data.get("pose")
    if not isinstance(pose, (list, tuple)) or len(pose) < 2 or pose[0] is None or pose[1] is None:
        return None
    try:
        z = float(pose[2]) if len(pose) >= 3 and pose[2] is not None else 0.0
        return float(pose[0]), float(pose[1]), z
    except (TypeError, ValueError):
        return None


def return_goal(
    session_id: str,
    text: str,
    start: tuple[float, float, float],
    pose: tuple[float, float, float],
) -> dict | str:
    """Point goal whose computed point is the departure. A string means do not write a goal."""
    sx, sy, _sz = start
    cx, cy, cz = pose
    dx, dy = sx - cx, sy - cy
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return "车已经在这一趟的出发点附近。"
    if dist > 30.0:
        return "出发点离车超过 30 米，这次没有写入目标。"
    heading = math.atan2(dy, dx)
    return {
        "id": session_id,
        "text": text,
        "target": "",
        "distance_m": dist,
        "bearing_rad": 0.0,
        "bearing": "正前方",
        "heading": heading,
        "origin": [cx, cy, cz],
        "source": "task_departure",
        "mode": "point",
    }


def goal_point(body: dict) -> tuple[float, float]:
    ox, oy = float(body["origin"][0]), float(body["origin"][1])
    dist = float(body["distance_m"])
    heading = float(body["heading"])
    return ox + dist * math.cos(heading), oy + dist * math.sin(heading)
