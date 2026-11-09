from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from aerial_finder.backends.fake import FakeBackend, FakeTarget, FakeWorld, FakeYolo, GoalSeekNavigator
from aerial_finder.comm.dog import HttpReporter, MockDogServer
from aerial_finder.geo import GeoRef
from aerial_finder.mission.llm_parser import merge_rule_llm, spec_from_llm_json
from aerial_finder.mission.parser import parse_instruction
from aerial_finder.mission.runner import MissionConfig, MissionRunner
from aerial_finder.nav.depth import SensorDepth
from aerial_finder.perception.detector import ColorHints, TargetDetector, color_fraction
from aerial_finder.perception.locate import depth_scale_from_ground, estimate_ground_z, locate, world_to_pixel
from aerial_finder.perception.tracker import Tracker
from aerial_finder.types import Detection, Frame, Intrinsics, VehicleState


def test_parse_example():
    s = parse_instruction("去前方200m左右找一个红色的凳子，然后把位置发给机器狗")
    assert s.label == "stool" and s.color == "red"
    assert s.prompts == ["stool", "bench", "chair"]
    assert s.distance_m == 200 and s.tolerance_m == pytest.approx(50.0)
    assert s.body_bearing == 0.0 and s.world_heading is None
    assert s.report_to == ["dog"]


@pytest.mark.parametrize("text,label,color,dist,bearing,heading", [
    ("往左前方大约150米找蓝色的箱子，发给机器狗", "box", "blue", 150, -math.pi / 4, None),
    ("向东300米找一辆白色的车", "car", "white", 300, None, math.pi / 2),
    ("find a red stool about 200 m ahead and send it to the robot dog", "stool", "red", 200, 0.0, None),
    ("在附近找一个黄色的锥桶", "traffic cone", "yellow", None, None, None),
    ("前方1公里找绿色长椅", "bench", "green", 1000, 0.0, None),
])
def test_parse_variants(text, label, color, dist, bearing, heading):
    s = parse_instruction(text)
    assert (s.label, s.color, s.distance_m) == (label, color, dist)
    assert s.body_bearing == bearing and s.world_heading == heading


def test_parse_robot_is_not_target():
    s = parse_instruction("前方100米找一个红色的凳子，然后通知机器人")
    assert s.label == "stool"


def test_llm_json_fills_paraphrase_object():
    text = "正前方大概20米，找个能坐的红色东西，告诉机器狗"
    spec = spec_from_llm_json(text, {"label": "red stool", "prompts": ["find a red thing"], "color": "red",
                                     "distance_m": 99, "body_bearing": 0, "report_to": ["dog"]})
    assert spec is not None
    assert spec.label == "stool" and spec.prompts == ["stool", "bench", "chair"]
    assert spec.color == "red" and spec.distance_m == 20 and spec.report_to == ["dog"]
    merged = merge_rule_llm(text, spec)
    assert merged.label == "stool" and merged.distance_m == 20


def test_hybrid_falls_back_to_rules():
    text = "去前方200m左右找一个红色的凳子，然后把位置发给机器狗"
    s = merge_rule_llm(text, None)
    assert s.label == "stool" and s.distance_m == 200


def test_geo_roundtrip():
    g = GeoRef(47.641468, -122.140165, 122.0)
    lat, lon, alt = g.to_gps(200.0, -35.0, 3.0)
    assert lat - g.lat == pytest.approx(200.0 / 111_200, rel=0.01)
    n, e, u = g.to_local(lat, lon, alt)
    assert (n, e, u) == pytest.approx((200.0, -35.0, 3.0), abs=1e-6)


def _planar_ground_depth(intr: Intrinsics, st: VehicleState) -> np.ndarray:
    v = np.arange(intr.height, dtype=np.float64)[:, None]
    yn = (v - intr.cy) / intr.fy
    with np.errstate(divide="ignore"):
        z = np.where(yn > 1e-6, st.pos[2] / yn, np.inf)
    return np.repeat(z, intr.width, 1).astype(np.float32)


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0])
def test_locate_depth_and_ground(yaw):
    intr = Intrinsics.from_hfov(640, 360, 90.0)
    st = VehicleState(pos=np.array([10.0, -4.0, 6.0]), yaw=yaw)
    tgt = st.pos + 25.0 * np.array([math.cos(yaw + 0.2), math.sin(yaw + 0.2), 0.0])
    tgt[2] = 0.0
    u, v, z = world_to_pixel(tgt, st, intr)
    depth = np.full((intr.height, intr.width), np.inf, np.float32)
    depth[int(v) - 3:int(v) + 4, int(u) - 3:int(u) + 4] = z
    bbox = (u - 3, v - 3, u + 3, v + 3)
    p, r, how = locate(bbox, Frame(np.zeros((360, 640, 3), np.uint8), intr, depth, intr), st)
    assert how == "depth" and np.linalg.norm(p - tgt) < 0.2 and r == pytest.approx(np.linalg.norm(tgt - st.pos), abs=0.3)
    p2, _, how2 = locate((u - 3, v - 6, u + 3, v), Frame(np.zeros((360, 640, 3), np.uint8), intr), st, ground_z=0.0)
    assert how2 == "ground" and np.linalg.norm(p2[:2] - tgt[:2]) < 0.5


def test_locate_with_camera_offset():
    intr = Intrinsics.from_hfov(640, 360, 90.0)
    st = VehicleState(pos=np.array([0.0, 0.0, 8.0]), yaw=0.3)
    off = (0.2, 0.0, 1.0)
    tgt = np.array([20.0 * math.cos(0.3), 20.0 * math.sin(0.3), 0.0])
    u, v, _ = world_to_pixel(tgt, st, intr, cam_offset_frd=off)
    p, _, how = locate((u - 3, v - 6, u + 3, v), Frame(np.zeros((360, 640, 3), np.uint8), intr), st, ground_z=0.0,
                       cam_offset_frd=off)
    assert how == "ground" and np.linalg.norm(p[:2] - tgt[:2]) < 0.05


def test_ground_estimate():
    intr = Intrinsics.from_hfov(224, 224, 90.0)
    st = VehicleState(pos=np.array([0.0, 0.0, 7.5]), yaw=1.0)
    assert estimate_ground_z(_planar_ground_depth(intr, st), intr, st) == pytest.approx(0.0, abs=0.05)
    assert depth_scale_from_ground(0.4 * _planar_ground_depth(intr, st), intr, st, 0.0) == pytest.approx(2.5, rel=0.02)


def test_color_verify_and_hints():
    img = np.full((360, 640, 3), 120, np.uint8)
    img[100:130, 200:225] = (210, 25, 30)
    assert color_fraction(img, (200, 100, 225, 130), "red") > 0.9
    assert color_fraction(img, (300, 100, 325, 130), "red") == 0.0
    hints = ColorHints()(img, "red")
    assert len(hints) == 1 and abs(hints[0].bbox[0] - 200) < 2


def test_tracker_confirms_consistent_target():
    tr = Tracker()
    rng = np.random.default_rng(0)
    for i in range(5):
        d = Detection("stool", "red stool", (0, 0, 1, 1), 0.5, True, 0.8)
        d.world = np.array([100.0, 20.0, 0.2]) + rng.normal(0, 0.3, 3)
        d.range_m = 12.0
        tr.add(d, np.array([90.0, 15.0, 5.0]), float(i))
    far = Detection("stool", "stool", (0, 0, 1, 1), 0.5, True, 0.8)
    far.world, far.range_m = np.array([140.0, 0.0, 0.0]), 40.0
    tr.add(far, np.zeros(3), 9.0)
    c = tr.confirmed()
    assert c is not None and np.linalg.norm(c.position[:2] - [100.0, 20.0]) < 0.5
    assert len(tr.tracks) == 2


def _runner(world, tmp_path, start=(0.0, 0.0, 6.0), yaw=0.0, reporters=(), **cfg):
    be = FakeBackend(world, start=start, yaw=yaw)
    det = TargetDetector(FakeYolo(be, max_range_m=30.0), hints=None)
    mc = MissionConfig(save_frames=False, async_detect=False, **cfg)
    return be, MissionRunner(be, SensorDepth(), GoalSeekNavigator(), det, list(reporters), mc, tmp_path)


def test_mission_finds_and_reports(tmp_path):
    yaw = 0.6
    tgt = np.array([205 * math.cos(yaw) - 15 * math.sin(yaw), 205 * math.sin(yaw) + 15 * math.cos(yaw), 0.25])
    world = FakeWorld([FakeTarget(tgt)])
    dog = MockDogServer().start()
    try:
        be, run = _runner(world, tmp_path, yaw=yaw, reporters=[HttpReporter(dog.url)])
        res = run.run(parse_instruction("去前方200m左右找一个红色的凳子，然后把位置发给机器狗"))
        assert res["state"] == "DONE", res
        assert len(dog.received) == 1
        msg = dog.received[0]
        n, e, _ = be.geo_ref.to_local(msg["position"]["lat"], msg["position"]["lon"], msg["position"]["alt_m"])
        assert math.hypot(n - tgt[0], e - tgt[1]) < 1.5
        assert msg["query"]["label"] == "stool" and msg["observations"] >= 3
        with urllib.request.urlopen(dog.url + "/api/v1/targets") as r:
            assert len(json.loads(r.read())) == 1
        assert (Path(tmp_path) / res["mission_id"] / "report.json").is_file()
    finally:
        dog.stop()


def test_mission_ignores_other_class_and_reports_not_found(tmp_path):
    world = FakeWorld([FakeTarget(np.array([60.0, 0.0, 0.25]), label="barrel")])
    _, run = _runner(world, tmp_path, search_radius_max_m=20.0)
    res = run.run(parse_instruction("前方60米左右找一个红色的凳子"))
    assert res["state"] == "FAILED" and res["reason"] == "not_found"


def test_mission_abort(tmp_path):
    world = FakeWorld([])
    _, run = _runner(world, tmp_path)
    run.start(parse_instruction("前方200米找一个红色的凳子"))
    run.abort()
    assert run.wait(10)["state"] == "ABORTED"


REPO = Path(__file__).resolve().parents[5]


@pytest.mark.skipif(not (REPO / "experiments/aerial/rl/consequence_model.py").is_file(), reason="training repo absent")
def test_vendored_model_matches_training_repo(tmp_path):
    import torch
    sys.path.insert(0, str(REPO))
    from experiments.aerial.rl import consequence_model as train_cm
    from experiments.aerial.rl.collect_consequence_dataset import sample_macro as train_sample
    from experiments.aerial.rl.probe_counterfactual_airsim import FAMILIES as TRAIN_FAMILIES
    from aerial_finder.nav import consequence_model as cm

    torch.manual_seed(0)
    m = train_cm.ConsequenceModel(frame_offsets=(0, 2, 4), hist_actions=3, env_dim=32).eval()
    train_cm.save_checkpoint(tmp_path / "c.pt", m)
    v = cm.load_checkpoint(tmp_path / "c.pt")
    B = 5
    d = torch.rand(B, 3, 56, 56)
    g, pa, a = torch.rand(B, 4), torch.rand(B, 4) * 0.2, torch.rand(B, 10, 4) * 0.3
    ha, env = torch.rand(B, 3, 4) * 0.2, torch.rand(B, 32)
    with torch.no_grad():
        o1, o2 = m(d, g, pa, a, ha, env=env), v(d, g, pa, a, ha, env=env)
    for k in ("hazard_logit", "log_cones", "dgoal"):
        assert torch.allclose(o1[k], o2[k])
    assert cm.FAMILIES == tuple(TRAIN_FAMILIES)
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    for fam in cm.FAMILIES + ("descend", "hover", "spin"):
        assert np.allclose(train_sample(r1, 10, 0.4, name=fam)[1], cm.sample_macro(r2, 10, 0.4, name=fam)[1])
    st, goal = np.array([1.0, 2.0, 5.0, 0, 0, 0, 0.3]), np.array([30.0, -4.0, 5.5])
    assert np.allclose(train_cm.goal_features(st, goal), cm.goal_features(st, goal))


def test_tracker_full_keeps_new_tracks():
    from aerial_finder.perception.tracker import Tracker
    tk = Tracker(max_tracks=4)
    for i in range(10):
        d = Detection(label="x", prompt="stool", bbox=(0, 0, 1, 1), conf=0.1, verified=False, color_frac=0.0,
                      world=np.array([10.0 * i, 0.0, 0.0]), range_m=10.0)
        tk.add(d, np.zeros(3), float(i))
    assert len(tk.tracks) == 4
    assert any(np.allclose(t.position[:2], [90.0, 0.0]) for t in tk.tracks)
    tr = tk.add(Detection(label="x", prompt="stool", bbox=(0, 0, 1, 1), conf=0.1, verified=False, color_frac=0.0,
                          world=np.array([90.5, 0.0, 0.0]), range_m=10.0), np.zeros(3), 11.0)
    assert tr.n_unverified == 2 and tr in tk.tracks


def test_approach_descends_and_climbs_vertically(tmp_path):
    from aerial_finder.mission.runner import Task
    from aerial_finder.perception.tracker import Tracker
    be, run = _runner(FakeWorld([]), tmp_path, start=(0.0, 0.0, 10.0))
    run.out = tmp_path
    tk = Tracker()
    d = Detection("stool", "stool", (0, 0, 1, 1), 0.5, True, 0.8)
    d.world, d.range_m = np.array([9.0, 0.0, 0.0]), 9.0
    tr = tk.add(d, np.zeros(3), 0.0)
    task = Task("approach", track_id=tr.id, phase="go", t_start=0.0)
    phases, z0 = [], 10.0
    for _ in range(200):
        a, done = run._execute(task, be.state(), tk, z0)
        phases.append(task.phase)
        if task.phase in ("descend", "climb"):
            assert a[0] == 0.0 and a[1] == 0.0
        be.step(a)
        if done:
            break
    assert done and "descend" in phases and "climb" in phases
    assert be.state().pos[2] == pytest.approx(z0, abs=1.0)
    assert np.linalg.norm(be.state().pos[:2]) < 1e-6


def test_tracker_prefers_aimed_track():
    from aerial_finder.perception.tracker import Tracker
    tk = Tracker()
    a = Detection("stool", "stool", (0, 0, 1, 1), 0.3, True, 0.8)
    a.world, a.range_m = np.array([10.0, 0.0, 0.0]), 15.0
    t1 = tk.add(a, np.zeros(3), 0.0)
    b = Detection("stool", "stool", (0, 0, 1, 1), 0.5, True, 0.8, extra={"mode": "focus"})
    b.world, b.range_m = np.array([13.0, 0.0, 0.0]), 15.0
    assert tk.add(b, np.zeros(3), 1.0, prefer_id=t1.id, prefer_gate_m=4.0) is t1
    c = Detection("stool", "stool", (0, 0, 1, 1), 0.5, True, 0.8)
    c.world, c.range_m = np.array([13.0, 5.0, 0.0]), 15.0
    assert tk.add(c, np.zeros(3), 2.0) is not t1


def test_hint_reject_does_not_block_a_nearby_verified_target():
    from aerial_finder.perception.tracker import Tracker, reject_look

    tk = Tracker()
    hint = Detection("stool", "red blob", (0, 0, 1, 1), 0.0, False, 0.4, source="color")
    hint.world, hint.range_m = np.array([9.12, -1.62, 0.0]), 5.3
    blob = tk.add(hint, np.zeros(3), 0.0)
    tk.add(hint, np.zeros(3), 0.2)
    reject_look(tk, blob)
    assert blob.rejected and not blob.looked
    stool = Detection("stool", "chair", (0, 0, 1, 1), 0.42, True, 0.81)
    # Farther than the association gate (2 m) and inside the old 3 m blacklist.
    stool.world, stool.range_m = np.array([9.12, 0.78, 0.0]), 1.2
    found = tk.add(stool, np.array([8.1, 0.7, 0.0]), 1.0)
    assert found is not blob
    assert any(c.id == found.id for c in tk.candidates(np.array([10.0, 0.0, 0.4]), 4.0))
    # A look that did see the class still blacklists the neighbourhood.
    reject_look(tk, found)
    assert found.looked
    assert tk.candidates(np.array([10.0, 0.0, 0.4]), 4.0) == []


def test_verified_detection_revives_a_hint_rejection():
    from aerial_finder.perception.tracker import Tracker, reject_look

    tk = Tracker()
    hint = Detection("stool", "red blob", (0, 0, 1, 1), 0.0, False, 0.5, source="color")
    hint.world, hint.range_m = np.array([9.0, -2.0, 0.0]), 5.0
    blob = tk.add(hint, np.zeros(3), 0.0)
    reject_look(tk, blob)
    hit = Detection("stool", "chair", (0, 0, 1, 1), 0.26, True, 0.58)
    hit.world, hit.range_m = np.array([9.2, -1.8, 0.0]), 5.1
    assert tk.add(hit, np.zeros(3), 1.0) is blob
    assert not blob.rejected
    assert blob in tk.candidates()


def test_veto_survives_eviction():
    from aerial_finder.perception.tracker import Tracker, veto
    tk = Tracker(max_tracks=3)
    a = Detection("stool", "stool", (0, 0, 1, 1), 0.3, True, 0.8)
    a.world, a.range_m = np.array([10.0, 0.0, 0.0]), 15.0
    t1 = tk.add(a, np.zeros(3), 0.0)
    tk.add(a, np.zeros(3), 0.1)
    veto(tk, t1)
    for i in range(5):
        d = Detection("stool", "stool", (0, 0, 1, 1), 0.3, True, 0.8)
        d.world, d.range_m = np.array([50.0 + 10 * i, 0.0, 0.0]), 15.0
        tk.add(d, np.zeros(3), 1.0 + i)
    assert t1 not in tk.tracks
    for k in range(2):
        b = Detection("stool", "stool", (0, 0, 1, 1), 0.3, True, 0.8)
        b.world, b.range_m = np.array([10.5, 0.5, 0.0]), 15.0
        tk.add(b, np.zeros(3), 10.0 + k)
    assert all(np.linalg.norm(c.position[:2] - [10.0, 0.0]) > 3.0 for c in tk.candidates())


def test_fisheye_rectifier_places_tape_crosses():
    import cv2
    from aerial_finder.fisheye import FisheyeRectifier
    calib = Path(__file__).resolve().parents[1] / "configs" / "sjcam_dog_fisheye.json"
    rect = FisheyeRectifier(str(calib))
    pts = np.array([[[407.9, 344.21]], [[666.01, 338.76]]], np.float64)
    und = cv2.fisheye.undistortPoints(pts, rect.K, rect.D, R=np.eye(3), P=rect.P).reshape(-1, 2)
    assert np.allclose(und[0], (239.6, 299.5), atol=0.5)
    assert np.allclose(und[1], (342.4, 297.9), atol=0.5)
    bgr = np.full((720, 1280, 3), 255, np.uint8)
    out, valid = rect.apply(bgr)
    assert out.shape == (640, 640, 3) and valid.shape == (640, 640)
    assert valid[int(und[0, 1]), int(und[0, 0])] == 255
    assert valid[0, 0] == 0
    assert rect.intr.fx == pytest.approx(320.0)


def test_metric_da3_scales_and_masks_unseen():
    from aerial_finder.nav.depth import MetricDa3

    class Inner:
        size = 4

        def reset(self):
            pass

        def __call__(self, rgb):
            return np.full((4, 4), 10.0, np.float32), "env"

    m = MetricDa3(Inner(), 0.396, 30.0)
    valid = np.zeros((8, 8), np.uint8)
    valid[:, :4] = 255
    depth, env = m(np.zeros((8, 8, 3), np.uint8), valid)
    assert env == "env"
    assert np.allclose(depth[:, :2], 3.96)
    assert np.allclose(depth[:, 2:], 30.0)
    bare, _ = m(np.zeros((8, 8, 3), np.uint8), None)
    assert np.allclose(bare, 3.96)


def test_realize_dog_commands_uses_tape_and_imu_gains():
    from aerial_finder.nav.consequence_model import LIMITS
    from aerial_finder.nav.navigator import realize_dog_commands

    cmd = np.zeros((2, 4))
    cmd[:, 0] = 1.0
    cmd[0, 3] = float(LIMITS[3])
    cmd[1, 3] = -float(LIMITS[3])
    out = realize_dog_commands(cmd, vx_max=0.35, vyaw_max=0.5, vx_gain=0.57, yaw_gain=0.5, dt=0.2)
    assert out[:, 0] == pytest.approx(0.35 * 0.57 * 0.2)
    assert out[0, 3] == pytest.approx(0.5 * 0.5 * 0.2)
    assert out[1, 3] == pytest.approx(-0.5 * 0.5 * 0.2)
    assert np.all(out[:, 1:3] == 0.0)


def test_a_close_side_does_not_block_an_open_bypass():
    from aerial_finder.nav.navigator import blocked_ahead

    # 12 m stop: pallet on the left, aisle ahead and to the right.
    assert blocked_ahead(2.66, 0.85, 2.67, 1.0) is False
    # Same shape on the first pass, which did finish.
    assert blocked_ahead(2.68, 0.99, 2.66, 1.0) is False
    assert blocked_ahead(0.7, 2.0, 3.0, 1.0) is True
    assert blocked_ahead(2.0, 0.85, 0.9, 1.0) is True


def test_bypass_holds_the_open_side_until_the_body_clears():
    from aerial_finder.nav.consequence_model import LIMITS
    from aerial_finder.nav.navigator import bypass_command, bypass_lateral

    kw = dict(turn_m=1.8, stop_m=0.8, clear_m=2.0, pass_m=2.0, yaw_limit=float(LIMITS[3]))
    pos = np.zeros(2)
    goal = np.array([5.0, 0.0])
    cmd, state, phase = bypass_command(None, 0.73, 0.67, 1.73, pos, 0.0, goal, **kw)
    assert phase == "bypass_align" and state["side"] == 1.0
    assert cmd[3] == pytest.approx(float(LIMITS[3])) and cmd[0] == 0.0
    # The other side becoming clearer does not flip the latch, and spinning in place does not finish.
    cmd, state, phase = bypass_command(state, 0.7, 3.0, 1.0, pos, 0.4, goal, **kw)
    assert phase == "bypass_align" and state["side"] == 1.0 and cmd[3] > 0.0
    assert bypass_lateral(pos, np.array([state["x"], state["y"]]), state["yaw"], state["side"]) == pytest.approx(0.0)
    # Aisle ahead: walk, still latched, until the body is 2 m to that side.
    cmd, state, phase = bypass_command(state, 2.5, 1.0, 3.0, pos, 1.0, goal, **kw)
    assert phase == "bypass_pass" and cmd[0] == pytest.approx(1.0) and cmd[3] == 0.0
    side_pos = np.array([1.2, 2.1])
    cmd, state, phase = bypass_command(state, 3.0, 1.0, 3.0, side_pos, 1.0, goal, **kw)
    assert phase == "bypass_rejoin" and cmd[3] < 0.0 and state is not None
    # Goal yaw brings the obstacle back: turn to the latched side again and demand more shift.
    cmd, state, phase = bypass_command(state, 0.8, 0.6, 2.5, side_pos, 0.3, goal, **kw)
    assert phase == "bypass_align" and cmd[3] > 0.0 and state["need"] == pytest.approx(3.1)
    # Facing the goal with a clear front hands back one forward step.
    facing = math.atan2(-2.0, 4.0)
    clear_pos = np.array([1.0, 2.0])
    cmd, state, phase = bypass_command(
        {"side": 1.0, "x": 0.0, "y": 0.0, "yaw": 0.0, "phase": "rejoin", "need": 2.0},
        3.0, 2.0, 2.0, clear_pos, facing, goal, **kw)
    assert phase == "bypass_done" and state is None and cmd[0] == pytest.approx(1.0)


def test_direct_nav_config_realizes_motion():
    from aerial_finder.nav.navigator import NavConfig

    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "lumos_dog_direct.json").read_text())
    nc = dict(cfg["nav"])
    nc.pop("type")
    nc.pop("device")
    nc.pop("surv_ckpts")
    nav = NavConfig(**nc)
    assert nav.realize_motion and nav.macro_set == "dog" and nav.bypass
    assert nav.bypass_clear_m == pytest.approx(2.0)
    assert nav.bypass_pass_m == pytest.approx(2.0)
    assert nav.dog_vx_gain == pytest.approx(0.57)
    assert nav.dog_yaw_gain == pytest.approx(0.5)
    assert nav.step_hz == pytest.approx(5.0)
    assert cfg["backend"]["vx_gain"] == pytest.approx(0.57)
    assert cfg["backend"]["cmd_mode"] == "direct"
