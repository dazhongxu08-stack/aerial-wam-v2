"""Body-frame occupancy labels for occ_topo.

The grid is the training target only. It is built from simulator depth and
pose, then integrated with the recorded body actions. Those poses never enter
the network.

Axes are forward, left, up. Cell ``(i, j, k)`` covers
``[i, i+1) x [j, j+1) x [k, k+1)`` with ``i, j, k`` the integer coordinates
below. A hit inside 80 m writes occupied at the hit cell and free along the
ray before it. A miss, a hole, or a depth past 80 m writes nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

OCCUPIED = 0
FREE = 1
UNKNOWN = 2
N_CLASSES = 3

X_MIN = -20
Y_MIN = -15
Z_MIN = -2
NX = 60
NY = 30
NZ = 4
N_CELLS = NX * NY * NZ

DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 80.0
RAY_STEP_M = 0.5
LABEL_WINDOW = 20


def empty_grid() -> np.ndarray:
    return np.full((NZ, NY, NX), UNKNOWN, dtype=np.uint8)


def cell_centers() -> np.ndarray:
    """Center of every cell, shaped ``[NZ, NY, NX, 3]`` as forward, left, up."""
    xs = (np.arange(NX, dtype=np.float64) + X_MIN) + 0.5
    ys = (np.arange(NY, dtype=np.float64) + Y_MIN) + 0.5
    zs = (np.arange(NZ, dtype=np.float64) + Z_MIN) + 0.5
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1)


_CENTERS = cell_centers()


def _index(fwd: np.ndarray, left: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    i = np.floor(fwd).astype(np.int64) - X_MIN
    j = np.floor(left).astype(np.int64) - Y_MIN
    k = np.floor(up).astype(np.int64) - Z_MIN
    ok = (i >= 0) & (i < NX) & (j >= 0) & (j < NY) & (k >= 0) & (k < NZ)
    return ok, (k, j, i)


def intrinsics(width: int, height: int, hfov_deg: float = 90.0) -> Tuple[float, float, float, float]:
    fx = (float(width) / 2.0) / np.tan(np.radians(float(hfov_deg)) / 2.0)
    fy = fx if width == height else fx * (float(height) / float(width))
    return float(fx), float(fy), (width - 1) / 2.0, (height - 1) / 2.0


def hits_from_depth(
    depth: np.ndarray,
    *,
    hfov_deg: float = 90.0,
    stride: int = 2,
) -> np.ndarray:
    """Planar depth to body points ``[N, 3]`` (forward, left, up).

    Pixels outside ``(0.5, 80]`` m are omitted, so a miss cannot clear cells.
    """
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError("depth must be [H, W]")
    step = max(1, int(stride))
    small = d[::step, ::step]
    vv, uu = np.mgrid[0 : d.shape[0] : step, 0 : d.shape[1] : step]
    valid = np.isfinite(small) & (small > DEPTH_MIN_M) & (small <= DEPTH_MAX_M)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    z = small[valid].astype(np.float64)
    fx, fy, cx, cy = intrinsics(d.shape[1], d.shape[0], hfov_deg)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    right = (u - cx) / fx * z
    down = (v - cy) / fy * z
    return np.stack([z, -right, -down], axis=1)


def raster_points(grid: np.ndarray, points: np.ndarray) -> None:
    """Write hits into ``grid``. Later points overwrite earlier ones.

    A hit overwrites free cells of the same ray. A later ray overwrites an
    earlier ray. Samples outside the grid are dropped, so a hit beyond the
    grid still marks the in-grid prefix free.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return
    dist = np.linalg.norm(pts, axis=1)
    keep = dist > 1e-6
    pts = pts[keep]
    dist = dist[keep]
    if pts.shape[0] == 0:
        return
    n_rays = int(pts.shape[0])
    n = int(np.ceil(float(dist.max()) / RAY_STEP_M)) + 1
    n = max(n, 2)
    alpha = np.linspace(0.0, 1.0, n, dtype=np.float64)
    samples = alpha[:, None, None] * pts[None, :, :]
    is_hit = np.zeros((n, n_rays), dtype=bool)
    is_hit[-1, :] = True
    flat = samples.reshape(-1, 3)
    hit = is_hit.reshape(-1)
    ray = np.broadcast_to(np.arange(n_rays, dtype=np.int64), (n, n_rays)).reshape(-1)
    ok, (k, j, i) = _index(flat[:, 0], flat[:, 1], flat[:, 2])
    if not np.any(ok):
        return
    pri = ray[ok] * 2 + hit[ok].astype(np.int64)
    order = np.argsort(pri, kind="stable")
    grid[k[ok][order], j[ok][order], i[ok][order]] = np.where(
        hit[ok][order], OCCUPIED, FREE
    ).astype(np.uint8)


def _write(grid: np.ndarray, points: np.ndarray, value: int) -> None:
    ok, (k, j, i) = _index(points[:, 0], points[:, 1], points[:, 2])
    if not np.any(ok):
        return
    grid[k[ok], j[ok], i[ok]] = np.uint8(value)


def body_to_world(points: np.ndarray, pos: Sequence[float], yaw: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    fwd, left, up = p[:, 0], p[:, 1], p[:, 2]
    world = np.empty_like(p)
    world[:, 0] = float(pos[0]) + c * fwd - s * left
    world[:, 1] = float(pos[1]) + s * fwd + c * left
    world[:, 2] = float(pos[2]) + up
    return world


def world_to_body(world: np.ndarray, pos: Sequence[float], yaw: float) -> np.ndarray:
    w = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    dx = w[:, 0] - float(pos[0])
    dy = w[:, 1] - float(pos[1])
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    out = np.empty_like(w)
    out[:, 0] = c * dx + s * dy
    out[:, 1] = -s * dx + c * dy
    out[:, 2] = w[:, 2] - float(pos[2])
    return out


def advance_points(points: np.ndarray, action: Sequence[float]) -> np.ndarray:
    """Move body points by one action. Same order as ``advance_goal_rel_body``."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3).copy()
    a = np.asarray(action, dtype=np.float64).reshape(4)
    p[:, 0] -= float(a[0])
    p[:, 1] -= float(a[1])
    p[:, 2] -= float(a[2])
    dyaw = float(a[3])
    if abs(dyaw) > 1e-12:
        c, s = float(np.cos(dyaw)), float(np.sin(dyaw))
        fwd, left = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = c * fwd + s * left
        p[:, 1] = -s * fwd + c * left
    return p


def fuse_window(
    depths: np.ndarray,
    poses: np.ndarray,
    target: int,
    *,
    stride: int = 2,
    hfov_deg: float = 90.0,
) -> np.ndarray:
    """Fuse ``depths[target-19 : target+1]`` into the body frame at ``target``.

    ``poses`` is ``[T, 7]`` with ``(x, y, z_up, ..., yaw)``. Later frames
    overwrite cells their hitting rays touch.
    """
    depth = np.asarray(depths)
    pose = np.asarray(poses, dtype=np.float64)
    t = int(target)
    start = max(0, t - (LABEL_WINDOW - 1))
    grid = empty_grid()
    pos_t = pose[t, :3]
    yaw_t = float(pose[t, 6])
    for s in range(start, t + 1):
        hits = hits_from_depth(depth[s], hfov_deg=hfov_deg, stride=stride)
        if hits.shape[0] == 0:
            continue
        world = body_to_world(hits, pose[s, :3], float(pose[s, 6]))
        local = world_to_body(world, pos_t, yaw_t)
        raster_points(grid, local)
    return grid


def labels_for_episode(
    depth: np.ndarray,
    pose: np.ndarray,
    *,
    stride: int = 2,
    hfov_deg: float = 90.0,
) -> np.ndarray:
    """``[T, NZ, NY, NX]`` uint8 grids, one fused label per frame.

    Each depth frame is unprojected once. Every target frame then transforms
    the previous 20 hit sets into its own body pose.
    """
    depth_arr = np.asarray(depth)
    pose_arr = np.asarray(pose, dtype=np.float64)
    t_count = int(depth_arr.shape[0])
    world_hits = []
    for s in range(t_count):
        hits = hits_from_depth(depth_arr[s], hfov_deg=hfov_deg, stride=stride)
        if hits.shape[0] == 0:
            world_hits.append(np.zeros((0, 3), dtype=np.float64))
        else:
            world_hits.append(body_to_world(hits, pose_arr[s, :3], float(pose_arr[s, 6])))
    out = np.empty((t_count, NZ, NY, NX), dtype=np.uint8)
    for t in range(t_count):
        grid = empty_grid()
        start = max(0, t - (LABEL_WINDOW - 1))
        for s in range(start, t + 1):
            if world_hits[s].shape[0] == 0:
                continue
            local = world_to_body(world_hits[s], pose_arr[t, :3], float(pose_arr[t, 6]))
            raster_points(grid, local)
        out[t] = grid
    return out


def _invert_points(points: np.ndarray, action: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`advance_points` for one body-delta action."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3).copy()
    a = np.asarray(action, dtype=np.float64).reshape(4)
    dyaw = float(a[3])
    if abs(dyaw) > 1e-12:
        c, s = float(np.cos(dyaw)), float(np.sin(dyaw))
        fwd, left = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = c * fwd - s * left
        p[:, 1] = s * fwd + c * left
    p[:, 0] += float(a[0])
    p[:, 1] += float(a[1])
    p[:, 2] += float(a[2])
    return p


def integrate_label(grid: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Rigidly resample a fused grid through the recorded actions.

    Each new cell center is mapped back into the start frame. Cells that land
    outside the start grid stay unknown. Sampling the source, instead of
    splatting cell centers forward, keeps a turned wall from opening holes
    that a ray would treat as unknown.
    """
    src = np.asarray(grid, dtype=np.uint8)
    acts = np.asarray(actions, dtype=np.float64).reshape(-1, 4)
    pts = _CENTERS.reshape(-1, 3).copy()
    for action in acts[::-1]:
        pts = _invert_points(pts, action)
    out = empty_grid().reshape(-1)
    ok, (k, j, i) = _index(pts[:, 0], pts[:, 1], pts[:, 2])
    if np.any(ok):
        out[ok] = src[k[ok], j[ok], i[ok]]
    return out.reshape(NZ, NY, NX)


def class_balanced_ce_np(logits: np.ndarray, target: np.ndarray) -> float:
    """Sum of per-class mean NLL. A class with no cells is skipped."""
    log = np.asarray(logits, dtype=np.float64)
    log = log - log.max(axis=-1, keepdims=True)
    pred = np.exp(log)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    tgt = np.asarray(target, dtype=np.int64).reshape(-1)
    flat = pred.reshape(-1, N_CLASSES)
    total = 0.0
    used = 0
    for cls in range(N_CLASSES):
        mask = tgt == cls
        if not np.any(mask):
            continue
        picked = flat[mask, cls]
        total += float((-np.log(np.clip(picked, 1e-8, 1.0))).mean())
        used += 1
    if used == 0:
        return 0.0
    return total


def ray_stop(
    grid: np.ndarray,
    theta: float,
    *,
    k_index: int,
    step_m: float = 0.25,
) -> Tuple[float, str]:
    """Walk one horizontal ray in layer ``k_index``.

    Returns ``(meters, stop)`` with stop in ``occupied``, ``unknown``, ``boundary``.
    """
    g = np.asarray(grid, dtype=np.uint8)
    z = float(_CENTERS[int(k_index), 0, 0, 2])
    df, dl = float(np.cos(theta)), float(np.sin(theta))
    s = float(step_m)
    limit = 80.0
    while s <= limit:
        fwd, left = s * df, s * dl
        ok, (k, j, i) = _index(np.asarray([fwd]), np.asarray([left]), np.asarray([z]))
        if not bool(ok[0]):
            return s, "boundary"
        val = int(g[int(k[0]), int(j[0]), int(i[0])])
        if val == OCCUPIED:
            return s, "occupied"
        if val == UNKNOWN:
            return s, "unknown"
        s += float(step_m)
    return s, "boundary"


def ray_length(grid: np.ndarray, theta: float) -> Tuple[float, str]:
    """Shorter of the two horizontal layers whose centers are -0.5 m and +0.5 m."""
    # k=-1 -> index 1, k=0 -> index 2.
    a = ray_stop(grid, theta, k_index=1)
    b = ray_stop(grid, theta, k_index=2)
    return a if a[0] <= b[0] else b


def counted_same_sign(
    pred_m: float,
    label_m: float,
    start_m: float,
    *,
    label_stop: str,
    min_change_m: float = 1.0,
) -> Optional[bool]:
    """Whether one ray counts, and if so whether the prior matches the label sign.

    ``None`` means the ray is not in the denominator.
    """
    if label_stop != "occupied":
        return None
    delta_l = float(label_m) - float(start_m)
    if abs(delta_l) < float(min_change_m):
        return None
    delta_p = float(pred_m) - float(start_m)
    if delta_p == 0.0:
        return False
    return (delta_p > 0.0) == (delta_l > 0.0)


LEFT_ANGLES = tuple(float(np.deg2rad(a)) for a in (5, 10, 15, 20, 25, 30))
RIGHT_ANGLES = tuple(-a for a in LEFT_ANGLES)


def occupied_median(grid: np.ndarray, angles: Sequence[float]) -> Optional[float]:
    """Median length of sector rays that hit occupied. None unless half of them do."""
    hits: List[float] = []
    for theta in angles:
        meters, stop = ray_length(grid, float(theta))
        if stop == "occupied":
            hits.append(float(meters))
    if len(hits) * 2 < len(tuple(angles)):
        return None
    hits.sort()
    return float(hits[len(hits) // 2])


def occupied_ray_errors(
    label: np.ndarray,
    pred: np.ndarray,
    angles: Sequence[float],
) -> List[float]:
    """Absolute meter error on rays whose label ends on occupied."""
    errors: List[float] = []
    for theta in angles:
        label_m, label_stop = ray_length(label, float(theta))
        if label_stop != "occupied":
            continue
        pred_m, _pred_stop = ray_length(pred, float(theta))
        errors.append(abs(float(pred_m) - float(label_m)))
    return errors


def score_gate_window(
    start: np.ndarray,
    label8: np.ndarray,
    pred8: np.ndarray,
    net_dyaw: float,
) -> Dict[str, Any]:
    """One 8-step window against the action-integrated label.

    ``open`` / ``forward`` are True, False, or None. None is outside that
    check's denominator. Errors are filled only when the window counts for
    at least one check.
    """
    left0 = occupied_median(start, LEFT_ANGLES)
    right0 = occupied_median(start, RIGHT_ANGLES)
    left8 = occupied_median(label8, LEFT_ANGLES)
    right8 = occupied_median(label8, RIGHT_ANGLES)
    pred_l = occupied_median(pred8, LEFT_ANGLES)
    pred_r = occupied_median(pred8, RIGHT_ANGLES)
    open_flag: Optional[bool] = None
    if (
        left0 is not None
        and right0 is not None
        and left8 is not None
        and right8 is not None
        and pred_l is not None
        and pred_r is not None
        and abs(float(left0) - float(right0)) >= 1.0
    ):
        if float(left0) > float(right0):
            open_is_left = True
        else:
            open_is_left = False
        turn_left = float(net_dyaw) > 0.0
        if open_is_left == turn_left:
            open_s, wall_s = (left0, right0) if open_is_left else (right0, left0)
            open_l, wall_l = (left8, right8) if open_is_left else (right8, left8)
            open_p, wall_p = (pred_l, pred_r) if open_is_left else (pred_r, pred_l)
            open_ok = counted_same_sign(open_p, open_l, open_s, label_stop="occupied")
            wall_ok = counted_same_sign(wall_p, wall_l, wall_s, label_stop="occupied")
            if open_ok is None or wall_ok is None:
                open_flag = None
            else:
                open_flag = bool(open_ok and wall_ok)
    fwd0_m, fwd0_stop = ray_length(start, 0.0)
    fwd_l_m, fwd_l_stop = ray_length(label8, 0.0)
    fwd_p_m, _fwd_p_stop = ray_length(pred8, 0.0)
    forward_flag: Optional[bool] = None
    if fwd0_stop == "occupied" and fwd_l_stop == "occupied" and float(fwd_l_m) <= float(fwd0_m) - 1.0:
        forward_flag = counted_same_sign(
            fwd_p_m, fwd_l_m, fwd0_m, label_stop="occupied"
        )
    errors: List[float] = []
    if open_flag is not None or forward_flag is not None:
        errors.extend(occupied_ray_errors(label8, pred8, (0.0,) + LEFT_ANGLES + RIGHT_ANGLES))
    return {"open": open_flag, "forward": forward_flag, "errors": errors}


def gate_passes(flags: Sequence[Optional[bool]], *, min_count: int = 50, min_rate: float = 0.8) -> bool:
    counted = [bool(f) for f in flags if f is not None]
    if len(counted) < int(min_count):
        return False
    return (sum(counted) / float(len(counted))) >= float(min_rate)
