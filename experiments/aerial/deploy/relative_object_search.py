"""Start a relative find when the utterance has an object but no map point.

The patrol planner otherwise stops and asks for a semantic-map name. The
language model has already filled ``search_target``. This hands the distance
in the sentence to the chassis search node and does not wait for a second
confirmation.
"""
from __future__ import annotations

import json
import math
import os
import re

from app.schemas.events import EventStatus
from app.services.event_emitter import emit_tool_step
from app.services.tools.base import ToolContext, ToolResult
from app.services.tools.schedule_helpers import has_patrol_target

_PID = "/tmp/relative_search.pid"
_GOAL = "/home/user/rt/search_goal.json"
_TRACKS = "/home/user/rt/tracks"
_STATUS = "/home/user/rt/wire_status.json"
_BEARINGS = (
    ("右前方", -math.pi / 4, "右前方"),
    ("左前方", math.pi / 4, "左前方"),
    ("正前方", 0.0, "正前方"),
    ("后方", math.pi, "后方"),
    ("后面", math.pi, "后方"),
    ("右侧", -math.pi / 2, "右侧"),
    ("右边", -math.pi / 2, "右侧"),
    ("右方", -math.pi / 2, "右侧"),
    ("左侧", math.pi / 2, "左侧"),
    ("左边", math.pi / 2, "左侧"),
    ("左方", math.pi / 2, "左侧"),
    ("前方", 0.0, "正前方"),
)


_ZH_METERS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _explicit_distance(text: str) -> float | None:
    # 「3米找」里「米」和「找」都是汉字，中间没有 \\b，不能用词边界。
    match = re.search(r"(\d+(?:\.\d+)?)\s*米", text)
    if match is None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*m(?![A-Za-z])", text, re.I)
    if match is not None:
        value = float(match.group(1))
    else:
        zh = re.search(r"([一二两三四五六七八九十])\s*米", text)
        if zh is None:
            return None
        value = float(_ZH_METERS[zh.group(1)])
    # 句中写了十米就用十米。不再把超过 8 的明确米数改成 8。
    if value < 1.0 or value > 30.0:
        return None
    return value


def _bearing(text: str) -> tuple[float, str]:
    for key, rad, name in _BEARINGS:
        if key in text:
            return rad, name
    return 0.0, "正前方"


def _from_intent(params: dict, text: str) -> tuple[float | None, float, str, str]:
    """Use a distance only when the sentence or the model actually gave one."""
    named = {name: (rad, name) for _key, rad, name in _BEARINGS}
    raw_name = str(params.get("bearing") or "").strip()
    if raw_name in named:
        bearing, bearing_name = named[raw_name]
        bearing_source = "model"
    else:
        bearing, bearing_name = _bearing(text)
        bearing_source = "sentence"
    spoken = _explicit_distance(text)
    if spoken is not None:
        return spoken, bearing, bearing_name, f"sentence/{bearing_source}"
    try:
        distance = float(params.get("distance_m") or 0)
    except (TypeError, ValueError):
        distance = 0.0
    if 1.0 <= distance <= 30.0:
        return distance, bearing, bearing_name, f"model/{bearing_source}"
    return None, bearing, bearing_name, f"direction/{bearing_source}"


def _stop_rule_search() -> None:
    try:
        open("/tmp/relative_search.stop", "w", encoding="utf-8").close()
    except OSError:
        pass
    try:
        pid = int(open(_PID, encoding="utf-8").read().strip())
    except (OSError, ValueError):
        return
    try:
        os.kill(pid, 15)
    except OSError:
        pass


def _write_goal(session_id: str, text: str, target: str, distance: float | None, bearing: float, bearing_name: str, source: str) -> None:
    body = {
        "id": session_id,
        "text": text,
        "target": target,
        "distance_m": distance,
        "bearing_rad": bearing,
        "bearing": bearing_name,
        "source": source,
        "mode": "point" if distance is not None else "along",
    }
    tmp = _GOAL + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(body, handle, ensure_ascii=False)
    os.replace(tmp, _GOAL)


async def maybe_start_relative_search(ctx: ToolContext) -> ToolResult | None:
    params = dict(ctx.parameters or {})
    if str(params.get("patrol_purpose") or "") not in ("object_search", "person_search"):
        return None
    target = str(params.get("search_target") or "").strip()
    if not target or has_patrol_target(params, ctx.user_text or ""):
        return None
    text = ctx.user_text or ""
    distance, bearing, bearing_name, source = _from_intent(params, text)
    _stop_rule_search()
    _write_goal(ctx.session_id, text, target, distance, bearing, bearing_name, source)
    if distance is None:
        step = f"后果模型朝当前朝向{bearing_name}开，途中寻找{target}，靠近后停下"
        reply = (
            f"已交给后果模型：朝当前朝向{bearing_name}开，途中寻找{target}。"
            "没有规定米数，不开向一个补出来的点。靠近目标后记下坐标并停下。"
        )
    else:
        step = f"后果模型前往当前朝向{bearing_name}约 {distance:.0f} 米，寻找{target}"
        reply = (
            f"寻物点已交给后果模型：当前朝向{bearing_name}约 {distance:.0f} 米，寻找{target}。"
            "途中看到与这句话一致的目标会记下坐标并停下。前方变近时向前绕开，不原地转。"
        )
    await emit_tool_step(ctx.session_id, "relative_search", step, status=EventStatus.COMPLETED)
    return ToolResult(
        success=True,
        tool_name="plan_patrol",
        data={
            "status": "goal_handed_to_model",
            "distance_m": distance,
            "mode": "point" if distance is not None else "along",
            "bearing": bearing_name,
            "bearing_rad": bearing,
            "target": target,
            "source": source,
        },
        reply_message=reply,
    )


async def maybe_drive_to_task_departure(ctx: ToolContext) -> ToolResult | None:
    """Hand 起点 to the chassis as the current task's departure coordinates."""
    from app.services.tools.task_departure import (
        chassis_pose,
        current_task_departure,
        is_task_start_request,
        return_goal,
    )

    if not is_task_start_request(ctx.parameters, ctx.user_text):
        return None
    start = current_task_departure(_TRACKS)
    if start is None:
        return ToolResult(
            success=False,
            tool_name="task_departure",
            error_message="没有当前任务的出发坐标。",
        )
    pose = chassis_pose(_STATUS)
    if pose is None:
        return ToolResult(
            success=False,
            tool_name="task_departure",
            error_message="还没有车的当前位置，写不出回起点的坐标。",
        )
    body = return_goal(str(ctx.session_id or ""), ctx.user_text or "", start, pose)
    if isinstance(body, str):
        return ToolResult(
            success=True,
            tool_name="task_departure",
            data={"start": [round(start[0], 3), round(start[1], 3)]},
            reply_message=body,
        )
    _stop_rule_search()
    tmp = _GOAL + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(body, handle, ensure_ascii=False)
    os.replace(tmp, _GOAL)
    dist = float(body["distance_m"])
    reply = (
        f"起点是这一趟出发的坐标 ({start[0]:.2f}, {start[1]:.2f})。"
        f"已交给开车的程序，从当前位置开回去，大约 {dist:.0f} 米。"
    )
    await emit_tool_step(ctx.session_id, "task_departure", reply, status=EventStatus.COMPLETED)
    return ToolResult(
        success=True,
        tool_name="task_departure",
        data={
            "status": "goal_handed_to_model",
            "start": [round(start[0], 3), round(start[1], 3)],
            "distance_m": round(dist, 3),
            "origin": body["origin"],
            "heading": body["heading"],
        },
        reply_message=reply,
    )
