"""Scene memory and schematic subgoals. No network, no flight."""

import math

import numpy as np

from experiments.aerial.rl.scene_memory import (
    GAP_M,
    SceneGuide,
    SceneMemory,
    SchematicMap,
    _Panel,
    load_schematic,
    open_guide,
    point_at_pixel,
    project_pixel,
    schematic_point,
    view_subgoal,
)


def _vec(seed: float, n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.normal(size=n)


def test_similar_view_merges_and_raises_weight():
    mem = SceneMemory()
    v = _vec(1)
    a = mem.observe(np.array([0.0, 0.0, 24.0]), 0.0, v, np.full((4, 4), 10.0, np.float32), "metric")
    b = mem.observe(np.array([2.0, 0.0, 24.0]), 0.1, v, np.full((4, 4), 12.0, np.float32), "metric")
    assert a.id == b.id
    assert b.weight == 2
    assert len(mem.frames) == 1
    assert len(mem.views) == 2


def test_yaw_or_distance_or_vector_opens_a_new_keyframe():
    mem = SceneMemory()
    v = _vec(2)
    mem.observe(np.array([0.0, 0.0, 1.0]), 0.0, v, None, "metric")
    turned = mem.observe(np.array([1.0, 0.0, 1.0]), 0.0 + np.deg2rad(35), v, None, "metric")
    assert turned.id == 1
    far = mem.observe(np.array([1.0 + GAP_M + 1.0, 0.0, 1.0]), np.deg2rad(35), v, None, "metric")
    assert far.id == 2
    other = mem.observe(far.pose + np.array([1.0, 0.0, 0.0]), far.yaw, _vec(9), None, "metric")
    assert other.id == 3
    assert len(mem.frames) == 4


def test_revisit_merges_into_the_older_place():
    mem = SceneMemory()
    v = _vec(3)
    first = mem.observe(np.array([0.0, 0.0, 1.0]), 0.0, v, None, "metric")
    mem.observe(np.array([GAP_M + 2.0, 0.0, 1.0]), 0.0, _vec(4), None, "metric")
    back = mem.observe(np.array([1.0, 0.5, 1.0]), 0.05, v, None, "metric")
    assert back.id == first.id
    assert first.weight == 2


def test_schematic_carrot_stays_inside_the_horizon():
    poly = np.array([[0.0, 0.0, 24.0], [40.0, 0.0, 24.0], [40.0, 20.0, 24.0]])
    here = np.array([0.0, 0.0, 24.0])
    goal = np.array([40.0, 20.0, 24.0])
    carrot = schematic_point(poly, here, goal, 8.0, ())
    assert carrot is not None
    assert np.linalg.norm(carrot[:2] - here[:2]) <= 8.0 + 1e-6
    assert carrot[0] > 5.0 and abs(carrot[1]) < 1.0


def test_blocked_heading_drops_the_schematic_edge():
    poly = np.array([[0.0, 0.0, 1.0], [30.0, 0.0, 1.0]])
    here = np.array([0.0, 0.0, 1.0])
    goal = np.array([30.0, 0.0, 1.0])
    assert schematic_point(poly, here, goal, 8.0, [0.0]) is None


def test_recorded_path_does_not_replace_the_schematic():
    mem = SceneMemory()
    v = _vec(5)
    kf = mem.observe(np.array([0.0, 0.0, 5.0]), 0.2, v, None, "metric")
    kf.passed.append(np.array([20.0, 6.0, 5.0]))
    guide = SceneGuide(mem, horizon_m=8.0, trust="metric", record=False, use_seen=True,
                       default_poly=np.array([[0.0, 0.0, 5.0], [20.0, 0.0, 5.0]]))
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    proposal = guide.step_view(state, np.array([20.0, 6.0, 5.0]), v, None)
    assert proposal.source == "schematic"
    assert abs(np.linalg.norm(proposal.aim[:2]) - 8.0) < 1e-6
    assert abs(proposal.aim[1]) < 1e-6


def test_rough_pose_applies_the_schematic_offset_at_the_live_pose():
    mem = SceneMemory()
    v = _vec(6)
    mem.observe(np.array([100.0, 50.0, 1.0]), 0.0, v, None, "rough")
    guide = SceneGuide(mem, horizon_m=8.0, trust="rough", use_seen=True,
                       default_poly=np.array([[100.0, 50.0, 1.0], [140.0, 50.0, 1.0]]))
    # Live odometry is 10 m west of the stored place. The schematic step is applied there.
    state = np.array([90.0, 50.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    proposal = guide.step_view(state, np.array([140.0, 50.0, 1.0]), v, None)
    assert proposal.source == "schematic"
    assert abs(proposal.aim[0] - 98.0) < 1e-6
    assert abs(proposal.aim[1] - 50.0) < 1e-6


def test_record_only_does_not_steer():
    mem = SceneMemory()
    v = _vec(7)
    kf = mem.observe(np.zeros(3), 0.0, v, None, "metric")
    kf.passed.append(np.array([10.0, 0.0, 0.0]))
    guide = SceneGuide(mem, horizon_m=8.0, record=True, use_seen=False)
    goal = np.array([30.0, 0.0, 0.0])
    proposal = guide.step_view(np.zeros(7), goal, v, None)
    assert proposal.source == "goal"
    np.testing.assert_allclose(proposal.aim, goal)


def test_a_turn_that_does_not_close_distance_is_not_a_blocked_heading():
    mem = SceneMemory()
    v = _vec(4)
    kf = mem.observe(np.zeros(3), 0.0, v, None, "metric")
    for yaw in (0.2, 0.6, 1.0):
        mem.note(kf.id, yaw=yaw, surv=0.95, d_before=20.0, d_after=20.0, collided=False,
                 surv_min=0.95, pose=np.zeros(3))
    assert kf.blocked_yaw == []


def test_progress_is_remembered_and_a_miss_blocks_that_heading():
    mem = SceneMemory()
    v = _vec(8)
    kf = mem.observe(np.zeros(3), 0.0, v, None, "metric")
    mem.note(kf.id, yaw=0.0, surv=0.99, d_before=20.0, d_after=19.0, collided=False,
             surv_min=0.95, pose=np.array([4.0, 0.0, 0.0]))
    assert len(kf.passed) == 1
    mem.note(kf.id, yaw=1.2, surv=0.2, d_before=19.0, d_after=19.0, collided=False,
             surv_min=0.95, pose=np.array([4.0, 0.0, 0.0]))
    assert any(abs(y - 1.2) < 1e-6 for y in kf.blocked_yaw)
    guide = SceneGuide(mem, horizon_m=8.0, use_seen=True,
                       default_poly=np.array([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]]))
    state = np.array([0.0, 0.0, 0.0, 0, 0, 0, 1.2])
    proposal = guide.step_view(state, np.array([0.0, 20.0, 0.0]), v, None)
    assert proposal.source == "dropped"
    assert guide.blocks_heading(state[:3], np.array([0.0, 20.0, 0.0]))
    mem.note(kf.id, yaw=1.2, surv=0.99, d_before=19.0, d_after=18.0, collided=False,
             surv_min=0.95, pose=np.array([0.0, 4.0, 0.0]))
    assert kf.blocked_yaw == []


def test_grid_carrot_goes_around_the_wall_not_through_it():
    free = np.ones((20, 16), bool)
    free[8, :11] = False  # wall at x≈17 m, open only on the north side
    grid = SchematicMap([_Panel(free, 0.0, 0.0, 2.0)])
    guide = SceneGuide(SceneMemory(), horizon_m=8.0, grid=grid)
    goal = np.array([36.0, 10.0, 24.0])
    guide.bind("r", goal)
    state = np.array([2.0, 10.0, 24.0, 0.0, 0.0, 0.0, 0.0])
    proposal = guide.step_view(state, goal, None, None)
    assert proposal.source == "schematic"
    assert proposal.aim[1] > 14.0
    assert proposal.aim[0] < 14.0
    assert np.linalg.norm(proposal.aim[:2] - state[:2]) <= 8.0 + 1e-6


def test_blocked_match_holds_until_the_place_is_left():
    mem = SceneMemory()
    v = _vec(11)
    kf = mem.observe(np.zeros(3), 0.0, v, None, "metric")
    kf.blocked_yaw.append(np.pi / 2)
    guide = SceneGuide(mem, horizon_m=8.0, use_seen=True,
                       default_poly=np.array([[0.0, 0.0, 0.0], [0.0, 30.0, 0.0]]))
    here = np.zeros(7)
    goal = np.array([0.0, 30.0, 0.0])
    assert guide.step_view(here, goal, v, None).source == "dropped"
    other = np.zeros_like(v)
    other[0], other[1] = -v[1], v[0]
    missed = guide.step_view(np.array([0.0, 2.0, 0.0, 0, 0, 0, 0.0]), goal, other, None)
    assert missed.source == "dropped"
    guide.bind("other", goal)
    left = guide.step_view(np.array([0.0, 0.0, 0.0, 0, 0, 0, 0.0]), goal, other, None)
    assert left.source == "schematic"


def test_other_schematic_edge_remains_when_one_heading_is_blocked():
    free = np.ones((16, 16), bool)
    free[6, 6:9] = False
    grid = SchematicMap([_Panel(free, 0.0, 0.0, 2.0)])
    guide = SceneGuide(SceneMemory(), horizon_m=8.0, grid=grid)
    goal = np.array([28.0, 15.0, 1.0])
    guide.bind("r", goal)
    here = np.array([2.0, 15.0, 1.0])
    plain = grid.carrot(here, 8.0, ())
    assert plain is not None
    blocked = grid.carrot(here, 8.0, [float(np.arctan2(plain[1] - here[1], plain[0] - here[0]))])
    assert blocked is not None
    heading = float(np.arctan2(blocked[1] - here[1], blocked[0] - here[0]))
    plain_h = float(np.arctan2(plain[1] - here[1], plain[0] - here[0]))
    assert abs((heading - plain_h + np.pi) % (2 * np.pi) - np.pi) > 0.5


def test_grid_ignores_a_polyline_in_the_same_file(tmp_path):
    free = np.ones((8, 4), bool)
    np.savez(tmp_path / "map.npz", east_free=free, east_x0=0.0, east_y0=0.0, east_res=2.0)
    schem = tmp_path / "schematic.json"
    schem.write_text(
        '{"grid": "map.npz", "routes": {"r": [[0, 0, 1], [30, 0, 1]]}}',
        encoding="utf-8",
    )
    guide = open_guide(str(schem), "", 8.0, "metric", "both")
    assert guide is not None and guide.grid is not None and guide.poly is None
    assert not guide.record and not guide.use_seen


def test_empty_paths_leave_the_trim_guide_off():
    assert open_guide("", "", 8.0, "metric", "both") is None


def test_roundtrip_and_route_schematic(tmp_path):
    mem = SceneMemory()
    pose = np.array([1.0, 2.0, 3.0])
    depth = np.full((8, 8), 20.0, np.float32)
    depth[4, 4] = 0.7
    kf = mem.observe(pose, 0.4, _vec(10), depth, "metric")
    mem.note(kf.id, 0.4, 0.99, 5.0, 4.0, False, 0.95, np.array([2.0, 2.0, 3.0]))
    path = tmp_path / "scene.json"
    mem.save(path)
    loaded = SceneMemory.load(path)
    assert len(loaded.frames) == 1
    assert loaded.frames[0].weight == 1
    assert len(loaded.frames[0].passed) == 1
    sample = point_at_pixel(4, 4, 0.7, pose, 0.4, depth.shape)
    assert abs(loaded.query_depth(sample) - 0.7) < 1e-3
    schem = tmp_path / "schematic.json"
    schem.write_text(
        '{"routes": {"urban_hard200_01": [[0, 0, 1], [10, 4, 1]]}}',
        encoding="utf-8",
    )
    default, routes, grid = load_schematic(schem)
    assert grid is None
    assert default is None and "urban_hard200_01" in routes
    guide = open_guide("", str(path), 8.0, "metric", "use")
    guide.bind("missing")
    assert guide.poly is None
    assert guide.use_seen and not guide.record


def test_a_thin_pixel_stays_beside_a_far_pixel():
    mem = SceneMemory()
    pose = np.array([0.0, 0.0, 24.0])
    depth = np.full((8, 8), 20.0, np.float32)
    depth[4, 3] = 0.6
    mem.observe(pose, 0.0, _vec(11), depth, "metric")
    near = point_at_pixel(4, 3, 0.6, pose, 0.0, depth.shape)
    far = point_at_pixel(4, 4, 20.0, pose, 0.0, depth.shape)
    assert abs(mem.query_depth(near) - 0.6) < 1e-3
    assert abs(mem.query_depth(far) - 20.0) < 1e-3


def test_a_newer_view_replaces_the_same_direction():
    mem = SceneMemory()
    pose = np.array([0.0, 0.0, 24.0])
    v = _vec(12)
    first = np.full((8, 8), 20.0, np.float32)
    first[4, 4] = 0.5
    mem.observe(pose, 0.0, v, first, "metric")
    second = np.full((8, 8), 20.0, np.float32)
    second[4, 4] = 15.0
    mem.observe(pose, 0.0, v, second, "metric")
    sample = point_at_pixel(4, 4, 0.5, pose, 0.0, first.shape)
    assert abs(mem.query_depth(sample) - 15.0) < 1e-3


def test_a_turned_camera_does_not_erase_the_old_direction(tmp_path):
    mem = SceneMemory()
    pose = np.array([0.0, 0.0, 24.0])
    v = _vec(13)
    first = np.full((8, 8), 20.0, np.float32)
    first[4, 4] = 0.8
    mem.observe(pose, 0.0, v, first, "metric")
    mem.observe(pose, 1.2, v, np.full((8, 8), 30.0, np.float32), "metric")
    sample = point_at_pixel(4, 4, 0.8, pose, 0.0, first.shape)
    assert abs(mem.query_depth(sample) - 0.8) < 1e-3
    mem.save(tmp_path / "scene.json")
    loaded = SceneMemory.load(tmp_path / "scene.json")
    assert abs(loaded.query_depth(sample) - 0.8) < 1e-3


def _open_depth(shape=(8, 8), far=20.0) -> np.ndarray:
    return np.full(shape, far, np.float32)


def test_an_open_view_puts_the_subgoal_eight_meters_toward_the_goal():
    mem = SceneMemory()
    pose = np.array([0.0, 0.0, 24.0])
    mem.observe(pose, 0.0, _vec(21), _open_depth(), "metric")
    guide = SceneGuide(mem, horizon_m=8.0, use_seen=True)
    proposal = guide.step_view(np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0]),
                               np.array([40.0, 0.0, 24.0]), _vec(21), None)
    assert proposal.source == "seen"
    assert abs(proposal.aim[0] - 8.0) < 1e-6
    assert abs(proposal.aim[1]) < 1e-6


def test_a_surface_ahead_moves_the_subgoal_off_that_ray():
    mem = SceneMemory()
    pose = np.array([0.0, 0.0, 24.0])
    depth = _open_depth()
    direct = np.array([8.0, 0.0, 24.0])
    pix = project_pixel(direct, pose, 0.0, depth.shape)
    assert pix is not None
    depth[pix] = 2.0
    mem.observe(pose, 0.0, _vec(22), depth, "metric")
    guide = SceneGuide(mem, horizon_m=8.0, use_seen=True)
    proposal = guide.step_view(np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0]),
                               np.array([40.0, 0.0, 24.0]), _vec(22), None)
    assert proposal.source == "seen"
    assert abs(proposal.aim[1]) > 1.0
    assert np.linalg.norm(proposal.aim[:2] - np.array([40.0, 0.0])) < 40.0 - 0.5
    assert not mem.ray_hits_surface(pose, proposal.aim)


def test_a_side_surface_does_not_move_the_forward_subgoal():
    mem = SceneMemory()
    pose = np.array([0.0, 0.0, 24.0])
    depth = _open_depth()
    direct = np.array([8.0, 0.0, 24.0])
    center = project_pixel(direct, pose, 0.0, depth.shape)
    assert center is not None
    depth[center[0], max(0, center[1] - 3)] = 2.0
    mem.observe(pose, 0.0, _vec(23), depth, "metric")
    guide = SceneGuide(mem, horizon_m=8.0, use_seen=True)
    proposal = guide.step_view(np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0]),
                               np.array([40.0, 0.0, 24.0]), _vec(23), None)
    assert proposal.source == "seen"
    assert abs(proposal.aim[0] - 8.0) < 1e-6
    assert abs(proposal.aim[1]) < 1e-6


def test_a_newer_open_view_restores_the_direct_subgoal():
    mem = SceneMemory()
    pose = np.array([0.0, 0.0, 24.0])
    v = _vec(24)
    blocked = _open_depth()
    pix = project_pixel(np.array([8.0, 0.0, 24.0]), pose, 0.0, blocked.shape)
    blocked[pix] = 2.0
    mem.observe(pose, 0.0, v, blocked, "metric")
    mem.observe(pose, 0.0, v, _open_depth(), "metric")
    guide = SceneGuide(mem, horizon_m=8.0, use_seen=True)
    proposal = guide.step_view(np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0]),
                               np.array([40.0, 0.0, 24.0]), v, None)
    assert proposal.source == "seen"
    assert abs(proposal.aim[1]) < 1e-6


def test_a_wall_ahead_keeps_a_short_subgoal_instead_of_dropping_it():
    mem = SceneMemory()
    pose = np.array([0.0, 0.0, 24.0])
    v = _vec(25)
    for yaw in (0.0, 1.2, -1.2):
        mem.observe(pose, yaw, v, _open_depth(far=2.0), "metric")
    guide = SceneGuide(mem, horizon_m=8.0, use_seen=True)
    proposal = guide.step_view(np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0]),
                               np.array([40.0, 0.0, 24.0]), v, None)
    assert proposal.source == "seen"
    assert float(np.linalg.norm(proposal.aim[:2] - pose[:2])) < 2.5
    assert not mem.ray_hits_surface(pose, proposal.aim)


def test_a_clear_goal_ray_keeps_the_subgoal_on_the_path():
    mem = SceneMemory()
    pose = np.array([0.0, 0.0, 24.0])
    depth = _open_depth(shape=(56, 56))
    left = np.array([4.0 * np.cos(np.deg2rad(40.0)), 4.0 * np.sin(np.deg2rad(40.0)), 24.0])
    pix = project_pixel(left, pose, 0.0, depth.shape)
    assert pix is not None
    depth[pix] = 2.5
    v = _vec(27)
    mem.observe(pose, 0.0, v, depth, "metric")
    guide = SceneGuide(mem, horizon_m=8.0, use_seen=True)
    state = np.array([0.0, 0.0, 24.0, 0, 0, 0, 0.0])
    goal = np.array([40.0, 0.0, 24.0])
    first = guide.step_view(state, goal, v, None)
    second = guide.step_view(state, goal, v, None)
    assert first.source == "seen" and second.source == "seen"
    assert abs(first.aim[0] - 8.0) < 1.0 and abs(first.aim[1]) < 1.0
    assert abs(second.aim[0] - 8.0) < 1.0 and abs(second.aim[1]) < 1.0


def test_a_latched_side_returns_to_the_goal_when_the_goal_ray_is_clear():
    mem = SceneMemory()
    pose = np.array([0.0, 0.0, 24.0])
    mem.observe(pose, 2.0, _vec(28), _open_depth(shape=(56, 56)), "metric")
    kind, point, sign, heading = view_subgoal(
        mem, pose, np.array([40.0, 0.0, 24.0]), 8.0, prefer_sign=-1.0, prefer_heading=-math.pi / 2)
    assert kind == "seen" and heading is None
    assert abs(point[0] - 8.0) < 1.0 and abs(point[1]) < 1.0
    assert sign == 0.0 or abs(point[1]) < 1.0


def test_a_latched_side_releases_when_the_goal_side_is_seen_clear():
    mem = SceneMemory()
    pose = np.array([0.0, 0.0, 24.0])
    mem.observe(pose, 0.0, _vec(29), _open_depth(shape=(56, 56)), "metric")
    kind, point, sign, heading = view_subgoal(
        mem, pose, np.array([40.0, 0.0, 24.0]), 8.0, prefer_sign=-1.0, prefer_heading=-math.pi / 2)
    assert kind == "seen" and sign == 0.0 and heading is None
    assert abs(point[1]) < 1e-6


def test_record_only_does_not_steer_from_a_blocking_view():
    mem = SceneMemory()
    pose = np.zeros(3)
    depth = _open_depth(far=2.0)
    mem.observe(pose, 0.0, _vec(26), depth, "metric")
    guide = SceneGuide(mem, horizon_m=8.0, record=True, use_seen=False)
    goal = np.array([30.0, 0.0, 0.0])
    proposal = guide.step_view(np.zeros(7), goal, _vec(26), None)
    assert proposal.source == "goal"
    np.testing.assert_allclose(proposal.aim, goal)
