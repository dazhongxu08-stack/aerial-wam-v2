import json
import os
import time

from task_departure import (
    chassis_pose,
    current_task_departure,
    goal_point,
    is_task_start_request,
    return_goal,
)


def test_start_name_is_the_task_departure_request():
    params = {"skill_id": "navigate_to_point", "skill_params": {"point": "起点"}}
    assert is_task_start_request(params, "回到刚才出发的原点") is True
    assert is_task_start_request({"skill_params": {"point": "桥上起点"}}, "去桥上起点") is False
    assert is_task_start_request({}, "正前方11米，找一个蓝色的垃圾桶。") is False


def test_goal_lands_on_the_departure(tmp_path):
    start = (18.2826, 3.8359, 0.0)
    pose = (8.733, 0.551, 0.0)
    body = return_goal("sess_next", "回到刚才出发的原点", start, pose)
    assert isinstance(body, dict)
    landed = goal_point(body)
    assert abs(landed[0] - start[0]) < 1e-6
    assert abs(landed[1] - start[1]) < 1e-6
    assert 1.0 <= body["distance_m"] <= 30.0
    assert body["heading"] is not None
    assert body["origin"] == [pose[0], pose[1], pose[2]]


def test_departure_is_the_newest_task_that_has_a_target(tmp_path):
    older = tmp_path / "sess_old.jsonl"
    newer = tmp_path / "sess_new.jsonl"
    empty = tmp_path / "sess_back.jsonl"
    older.write_text(json.dumps({
        "kind": "latch", "target": "蓝色的垃圾桶", "origin": [1, 2, 0],
    }) + "\n", encoding="utf-8")
    time.sleep(0.02)
    newer.write_text(json.dumps({
        "kind": "latch", "target": "蓝色的垃圾桶", "origin": [18.2826, 3.8359, 0.0],
    }) + "\n", encoding="utf-8")
    time.sleep(0.02)
    empty.write_text(json.dumps({
        "kind": "latch", "target": "", "origin": [8.7, 0.5, 0.0],
    }) + "\n", encoding="utf-8")
    os.utime(empty, None)
    assert current_task_departure(str(tmp_path)) == (18.2826, 3.8359, 0.0)


def test_missing_pose_is_not_a_goal(tmp_path):
    status = tmp_path / "wire_status.json"
    status.write_text(json.dumps({"pose": None}), encoding="utf-8")
    assert chassis_pose(str(status)) is None
    assert return_goal("sess", "回起点", (18.0, 3.0, 0.0), (18.2, 3.1, 0.0)) == "车已经在这一趟的出发点附近。"
