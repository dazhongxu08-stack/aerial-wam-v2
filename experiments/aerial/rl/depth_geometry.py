"""Shared depth-map geometry for forward cones and directional clearances.

Row/col semantics match :func:`min_depth_pixel_loc` (and the ④ contact dumps):
``(0, 0)`` = top-left; ``row`` increases downward; ``col`` increases rightward.
``col`` near 0/1 ⇒ lateral edges; ``row`` near 1 ⇒ ground below.

Used by the depth predictor (P0a ``predict_cones``), V0 rollout eval / gate
forensics, and depth-vs-GT diagnostics — one definition, no drift.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# Canonical keys returned by :func:`cone_clearances` / ``predict_cones``.
CONE_KEYS = ("forward", "left", "right", "up", "down")


def _min_finite_positive(region: np.ndarray) -> float:
    finite = np.asarray(region, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    return float(np.min(finite)) if finite.size else float("inf")


def forward_min_depth(depth: np.ndarray, *, center_frac: float) -> float:
    """Min finite+positive depth over the central ``center_frac`` box (forward).

    The front camera faces body-forward (the episode yaw), so the image centre
    is the flight direction. Restricting the obstacle test to a centre crop is
    what makes "is there something *ahead*" distinct from "is there ground far
    below / a wall off to the side" — a full-field min at cruise altitude is
    almost always the ground, which never triggers a 1.5 m near-collision.
    """
    h, w = depth.shape[-2], depth.shape[-1]
    cf = float(np.clip(center_frac, 0.05, 1.0))
    # Match forward_min_depth_torch: at least 1 px per axis (int(h*cf) alone can be 0).
    dh, dw = max(1, int(h * cf)), max(1, int(w * cf))
    r0, c0 = (h - dh) // 2, (w - dw) // 2
    crop = np.asarray(depth[r0 : r0 + dh, c0 : c0 + dw], dtype=np.float64)
    return _min_finite_positive(crop)


def forward_min_depth_torch(
    depth, *, center_frac: float, softmin_temperature_m: float = 0.0
):
    """Batched forward-crop min — same crop as :func:`forward_min_depth`.

    ``depth``: ``[B,H,W]`` or ``[H,W]``. Non-finite / non-positive → +inf.

    ``softmin_temperature_m<=0`` → hard min (eval ⓪d / sampling).
    ``softmin_temperature_m>0`` → softmin ``-T logΣ exp(-x/T)`` over finite
    crop pixels (declare v3 A″; T frozen at 0.05 m in recipe).
    """
    import torch

    d = depth if depth.ndim == 3 else depth.unsqueeze(0)
    b, h, w = d.shape
    cf = float(max(0.05, min(1.0, center_frac)))
    dh, dw = max(1, int(h * cf)), max(1, int(w * cf))
    r0, c0 = (h - dh) // 2, (w - dw) // 2
    crop = d[:, r0 : r0 + dh, c0 : c0 + dw]
    valid = torch.isfinite(crop) & (crop > 0)
    T = float(softmin_temperature_m)
    if T <= 0.0:
        filled = crop.masked_fill(~valid, float("inf"))
        mins = filled.reshape(b, -1).min(dim=-1).values
    else:
        neg = torch.where(valid, -crop / T, torch.full_like(crop, float("-inf")))
        lse = torch.logsumexp(neg.reshape(b, -1), dim=-1)
        mins = torch.where(
            torch.isfinite(lse),
            -T * lse,
            torch.full_like(lse, float("inf")),
        )
    return mins if depth.ndim == 3 else mins[0]


def cone_mins_torch(depth, *, center_frac: float = 0.5):
    """Forward / left / right hard mins. Same regions as :func:`cone_clearances`.

    ``depth`` is ``[B,H,W]`` or ``[H,W]`` in meters. Returns three tensors of
    shape ``[B]`` (or scalars when the input is 2-D).
    """
    import torch

    single = depth.ndim == 2
    d = depth.unsqueeze(0) if single else depth
    b, h, w = d.shape
    cf = float(max(0.05, min(1.0, center_frac)))
    dh = max(1, int(h * cf))
    r0 = (h - dh) // 2
    mid_c = max(1, w // 2)
    band = d[:, r0 : r0 + dh, :]

    def _min(crop):
        valid = torch.isfinite(crop) & (crop > 0)
        filled = crop.masked_fill(~valid, float("inf"))
        return filled.reshape(b, -1).min(dim=-1).values

    fwd = forward_min_depth_torch(d, center_frac=cf, softmin_temperature_m=0.0)
    left = _min(band[:, :, :mid_c])
    right = _min(band[:, :, mid_c:])
    if single:
        return fwd[0], left[0], right[0]
    return fwd, left, right


def full_min_depth(depth: np.ndarray) -> float:
    d = np.asarray(depth, dtype=np.float64)
    return _min_finite_positive(d)


def min_depth_pixel_loc(depth: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    """Normalised (row, col) of the nearest finite+positive depth pixel.

    Read-only. This is the forensic that separates a FRONTAL obstacle (nearest
    pixel near image centre) from a LATERAL/rear blind-spot hit (nearest pixel at
    the left/right edge) or GROUND (bottom rows). ``row``/``col`` ∈ [0,1] with
    (0,0)=top-left; ``col`` near 0/1 ⇒ side, ``row`` near 1 ⇒ below. A forward-only
    depth shield structurally cannot react to a min that is not near the centre.
    """
    if depth is None:
        return None
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        return None
    mask = np.isfinite(d) & (d > 0)
    if not mask.any():
        return None
    dd = np.where(mask, d, np.inf)
    r, c = np.unravel_index(int(np.argmin(dd)), dd.shape)
    h, w = dd.shape
    return {
        "row": round(float(r) / max(h - 1, 1), 3),
        "col": round(float(c) / max(w - 1, 1), 3),
        "val": round(float(dd[r, c]), 3),
    }


def forward_view_mass(
    depth: np.ndarray,
    *,
    near_m: float = 18.0,
    band_frac: float = 0.5,
    n_bins: int = 8,
) -> Dict[str, Any]:
    """Near-pixel share across the forward view, not a single cone minimum.

    The horizontal band matches :func:`cone_clearances`. Eight equal column
    bins cover the camera's width. The centre is the middle four bins; the
    outer pairs are left and right. ``near_*`` is the fraction of valid pixels
    at or inside ``near_m``. One near pixel cannot fill the centre.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"forward_view_mass expects 2-D depth, got shape {d.shape}")
    h, w = d.shape
    n = max(4, int(n_bins))
    dh = max(1, int(h * float(np.clip(band_frac, 0.05, 1.0))))
    r0 = (h - dh) // 2
    band = d[r0 : r0 + dh, :]
    edges = np.linspace(0, w, n + 1).astype(int)

    def _stats(reg: np.ndarray) -> tuple:
        valid = reg[np.isfinite(reg) & (reg > 0)]
        if valid.size == 0:
            return 0.0, float("inf")
        return float(np.mean(valid <= float(near_m))), float(np.percentile(valid, 50))

    near_bins: list[float] = []
    p50_bins: list[float] = []
    for i in range(n):
        c0, c1 = int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1]))
        near_i, p50_i = _stats(band[:, c0:c1])
        near_bins.append(near_i)
        p50_bins.append(p50_i)
    mid0, mid1 = n // 4, n - n // 4
    left, center, right = slice(0, mid0), slice(mid0, mid1), slice(mid1, n)

    def _mean(idxs: slice, values: list[float]) -> float:
        chunk = values[idxs]
        finite = [v for v in chunk if np.isfinite(v)]
        if not finite:
            return float("inf")
        return float(np.mean(finite))

    return {
        "near_center": _mean(center, near_bins),
        "near_left": _mean(left, near_bins),
        "near_right": _mean(right, near_bins),
        "p50_center": _mean(center, p50_bins),
        "p50_left": _mean(left, p50_bins),
        "p50_right": _mean(right, p50_bins),
        "near_bins": near_bins,
    }


def forward_view_mass(
    depth: np.ndarray, *, near_m: float = 18.0, band_frac: float = 0.5
) -> Dict[str, float]:
    """Near-pixel fraction and median depth on the forward horizontal band.

    Center is the middle half of the columns. Left and right are the outer
    quarters. Near means ``0 < depth <= near_m``. The fraction is over the
    valid pixels in that region, so one pixel cannot flip the summary.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"forward_view_mass expects 2-D depth, got shape {d.shape}")
    h, w = d.shape
    dh = max(1, int(h * float(np.clip(band_frac, 0.05, 1.0))))
    r0 = (h - dh) // 2
    band = d[r0 : r0 + dh, :]
    c0, c1 = w // 4, (3 * w) // 4
    if c1 <= c0:
        c0, c1 = 0, w

    def _region(reg: np.ndarray) -> tuple:
        valid = reg[np.isfinite(reg) & (reg > 0)]
        if valid.size == 0:
            return 0.0, float("inf")
        near = float(np.mean(valid <= float(near_m)))
        return near, float(np.median(valid))

    near_left, med_left = _region(band[:, :c0] if c0 > 0 else band[:, :1])
    near_center, med_center = _region(band[:, c0:c1])
    near_right, med_right = _region(band[:, c1:] if c1 < w else band[:, -1:])
    return {
        "near_left": near_left,
        "near_center": near_center,
        "near_right": near_right,
        "med_left": med_left,
        "med_center": med_center,
        "med_right": med_right,
    }


def azimuth_clearances(
    depth: np.ndarray,
    *,
    n_bins: int = 8,
    center_frac: float = 0.5,
    hfov_deg: float = 90.0,
) -> Dict[str, Any]:
    """Multi-sector horizontal clearance, coarser than pixels but finer than
    the 2-bucket left/right split in :func:`cone_clearances`.

    Diagnosed hard134 route 0 (2026-09-18): a "stuck escape" that only had
    left/right (2 buckets) to choose from kept re-picking essentially the same
    heading every time it triggered, because e.g. a 45° candidate and a 150°
    candidate both read the SAME "left" clearance value — no real new spatial
    information. This gives ``n_bins`` roughly-equal-angle column bins across
    the camera's horizontal FOV, each with its own min-clearance, so a wide
    escape fan can actually distinguish "open at 120°" from "open at 45°".

    Returns ``{"bearings_deg": [...], "clearances_m": [...]}`` — bin ``i``'s
    bearing is its column-bin centre mapped through ``hfov_deg`` (0 = image
    centre = forward; + = left half of the frame, matching the ENU convention
    used elsewhere in this module: positive bearing = left).
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"azimuth_clearances expects 2-D depth, got shape {d.shape}")
    h, w = d.shape
    n = max(1, int(n_bins))
    cf = float(np.clip(center_frac, 0.05, 1.0))
    dh = max(1, int(h * cf))
    r0 = (h - dh) // 2
    band = d[r0 : r0 + dh, :]
    edges = np.linspace(0, w, n + 1).astype(int)
    clearances: list[float] = []
    bearings: list[float] = []
    half_hfov = float(hfov_deg) / 2.0
    for i in range(n):
        c0, c1 = int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1]))
        clearances.append(_min_finite_positive(band[:, c0:c1]))
        col_centre = (c0 + c1) / 2.0
        frac = (col_centre / max(1, w)) - 0.5  # [-0.5, 0.5], 0 = centre
        # + = left half of frame (smaller col) matches ENU "left" convention
        # used by cone_clearances / SceneIntentPlanner bearings.
        bearings.append(float(-frac * 2.0 * half_hfov))
    return {"bearings_deg": bearings, "clearances_m": clearances}


def train_lateral_clearances(
    depth: np.ndarray, *, center_frac: float = 0.5
) -> tuple:
    """Forward / left / right mins used by the collision loss.

    Forward is the centre-crop min (:func:`forward_min_depth`). Left and right
    are the same row band but only the columns *outside* that crop, matching
    ``TorchRSSMDynamics._fwd_lateral_depths_torch``. This is not
    :func:`cone_clearances`: those left/right halves overlap the centre.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"train_lateral_clearances expects 2-D depth, got shape {d.shape}")
    h, w = d.shape
    cf = float(np.clip(center_frac, 0.05, 1.0))
    dh, dw = max(1, int(h * cf)), max(1, int(w * cf))
    r0, c0 = (h - dh) // 2, (w - dw) // 2
    d_fwd = forward_min_depth(d, center_frac=center_frac)
    band = d[r0 : r0 + dh, :]
    left_region = band[:, :c0] if c0 > 0 else band[:, : max(1, w // 4)]
    right_start = c0 + dw
    right_region = (
        band[:, right_start:] if right_start < w else band[:, max(0, 3 * w // 4) :]
    )
    return d_fwd, _min_finite_positive(left_region), _min_finite_positive(right_region)


def forward_percentile(
    depth: np.ndarray, *, center_frac: float = 0.5, percentile: float = 10.0
) -> float:
    """Low percentile of the centre crop, not the single nearest pixel.

    A one-pixel minimum jumps by metres between frames while the vehicle moves
    a fraction of a metre. The 10th percentile stays on the obstacle surface.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"forward_percentile expects 2-D depth, got shape {d.shape}")
    h, w = d.shape
    cf = float(np.clip(center_frac, 0.05, 1.0))
    dh, dw = max(1, int(h * cf)), max(1, int(w * cf))
    r0, c0 = (h - dh) // 2, (w - dw) // 2
    crop = d[r0 : r0 + dh, c0 : c0 + dw]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    if valid.size == 0:
        return float("inf")
    return float(np.percentile(valid, float(percentile)))


def approach_clearances(
    depth: np.ndarray, *, center_frac: float = 0.5, percentile: float = 10.0
) -> tuple:
    """10th-percentile forward, 50th-percentile left and right.

    Side values are the outside-crop row band. The median ignores a single near
    pixel on an otherwise open side.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"approach_clearances expects 2-D depth, got shape {d.shape}")
    d_fwd = forward_percentile(d, center_frac=center_frac, percentile=percentile)
    h, w = d.shape
    cf = float(np.clip(center_frac, 0.05, 1.0))
    dh, dw = max(1, int(h * cf)), max(1, int(w * cf))
    r0, c0 = (h - dh) // 2, (w - dw) // 2
    band = d[r0 : r0 + dh, :]
    left = band[:, :c0] if c0 > 0 else band[:, : max(1, w // 4)]
    right_start = c0 + dw
    right = band[:, right_start:] if right_start < w else band[:, max(0, 3 * w // 4) :]

    def _pct(region: np.ndarray) -> float:
        valid = region[np.isfinite(region) & (region > 0)]
        if valid.size == 0:
            return float("inf")
        return float(np.percentile(valid, 50.0))

    return d_fwd, _pct(left), _pct(right)


def frontal_obstacle_frame(
    d_fwd: float, d_left: float, d_right: float, *, early_m: float = 18.0
) -> bool:
    """Forward p10 is in the early band and at least one side is clearly opener.

    The other side may be a nearer wall. A street has walls beside the vehicle
    and still has an obstacle ahead plus one direction to yaw.
    """
    if not (np.isfinite(d_fwd) and np.isfinite(d_left) and np.isfinite(d_right)):
        return False
    longer = max(float(d_left), float(d_right))
    return bool(8.0 < float(d_fwd) <= float(early_m) and longer >= float(d_fwd) + 2.0)


def stable_approach(values, *, min_len: int = 3, min_drop_m: float = 0.5, max_jump_m: float = 2.0) -> bool:
    """A straight run that keeps getting closer without a multi-metre flicker."""
    seq = [float(v) for v in values]
    if len(seq) < int(min_len):
        return False
    if any(abs(seq[i] - seq[i - 1]) > float(max_jump_m) for i in range(1, len(seq))):
        return False
    return bool(seq[-1] <= seq[0] - float(min_drop_m))


def early_press_mask(
    d_fwd: float, d_left: float, d_right: float, *, early_m: float = 18.0
) -> bool:
    """True when a straight press should be labeled inside the early band.

    Same numbers as the collision loss: forward still beyond 8 m, inside
    ``early_m``, and the longer outside-crop side is at least 2 m clearer.
    """
    if not (np.isfinite(d_fwd) and np.isfinite(d_left) and np.isfinite(d_right)):
        return False
    side = max(float(d_left), float(d_right))
    return bool(8.0 < float(d_fwd) <= float(early_m) and side >= float(d_fwd) + 2.0)


# Collected frames: DA3 centre-nearest / GT ≈ 0.42, and ≈ 0.53 when a
# building is inside 15 m. Thresholds written in true metres (3 / 5 / 8)
# therefore fire about twice too early if they read DA3 metres directly.
DA3_TO_GT_SCALE = 2.0
# Flight forward clearance is the road strip, not the central half of the
# frame. On a street the wider crop includes the flanking facades and the
# nose reads as a wall. Sides are everything outside this strip.
DEPLOY_ROAD_CENTER_FRAC = 0.12


def deploy_cones(
    depth: np.ndarray,
    *,
    center_frac: float = DEPLOY_ROAD_CENTER_FRAC,
    scale: float = DA3_TO_GT_SCALE,
) -> Dict[str, float]:
    """Clearances for the planner and the shield, in approximate true metres.

    Forward is the road-strip 10th percentile. Left and right are the medians
    outside that strip, so a facade beside the road is a side, not the nose.
    Up and down stay the half-frame mins. Distances are multiplied by
    ``scale`` so a DA3 reading is not compared with a true-metre threshold.
    """
    d_fwd, d_left, d_right = approach_clearances(depth, center_frac=center_frac)
    base = cone_clearances(depth)

    def _s(value: float) -> float:
        if not np.isfinite(value):
            return float(value)
        return float(value) * float(scale)

    return {
        "forward": _s(d_fwd),
        "left": _s(d_left),
        "right": _s(d_right),
        "up": _s(base["up"]),
        "down": _s(base["down"]),
    }


def cone_clearances(depth: np.ndarray, *, center_frac: float = 0.5) -> Dict[str, float]:
    """Five-direction min clearances on a 2-D depth map.

    Regions (``H×W`` depth, row↓ col→):

    * **forward** — central ``center_frac`` box (same as :func:`forward_min_depth`)
    * **left**  — center-frac rows, left half columns (col < W//2)
    * **right** — center-frac rows, right half columns (col ≥ W//2)
    * **up**    — top half rows (row < H//2), all columns
    * **down**  — bottom half rows (row ≥ H//2, ground), all columns

    left/right use the same row band as *forward* so that ground pixels in
    the lower part of the image do not dominate the lateral clearance estimate
    (ground at ~47 m was causing left/right ≪ forward in AirSim forest, which
    made the SceneIntentPlanner penalise side candidates more than forward and
    prevented E1 from routing around obstacles).

    Empty / invalid regions return ``inf`` (no finite positive depth seen).
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"cone_clearances expects 2-D depth, got shape {d.shape}")
    h, w = d.shape
    mid_r, mid_c = h // 2, w // 2
    cf = float(np.clip(center_frac, 0.05, 1.0))
    dh = max(1, int(h * cf))
    r0 = (h - dh) // 2
    return {
        "forward": forward_min_depth(d, center_frac=center_frac),
        "left": _min_finite_positive(d[r0 : r0 + dh, :mid_c]),
        "right": _min_finite_positive(d[r0 : r0 + dh, mid_c:]),
        "up": _min_finite_positive(d[:mid_r, :]),
        "down": _min_finite_positive(d[mid_r:, :]),
    }


def directional_clearance_m(
    depth: np.ndarray,
    action_xyz: np.ndarray,
    *,
    percentile: float = 5.0,
    wedge_half_deg: float = 25.0,
    hfov_deg: float = 90.0,
    vfov_deg: float = 90.0,
) -> float:
    """Low-percentile depth along a body-frame action direction (offline labels only).

    ``action_xyz`` is ``(dx, dy, dz)`` in body frame (fwd, left, up). Near-zero
    displacement returns ``inf`` (no direction). Used to supervise the learned
    obstacle cost; must not be called from ``imagine`` / planner scoring.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"directional_clearance_m expects 2-D depth, got {d.shape}")
    a = np.asarray(action_xyz, dtype=np.float64).reshape(-1)[:3]
    n = float(np.linalg.norm(a))
    if n < 1e-8:
        return float("inf")
    u = a / n
    # Bearing: +az = left (matches azimuth_clearances / cone ENU).
    az = float(np.degrees(np.arctan2(u[1], u[0])))  # left vs fwd
    el = float(np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0))))
    h, w = d.shape
    half_h = float(hfov_deg) / 2.0
    half_v = float(vfov_deg) / 2.0
    wedge = float(max(1.0, wedge_half_deg))
    # Pixel grid → bearings (col 0 = left = +half_h).
    cols = (np.arange(w, dtype=np.float64) + 0.5) / max(w, 1)
    rows = (np.arange(h, dtype=np.float64) + 0.5) / max(h, 1)
    az_map = (0.5 - cols) * 2.0 * half_h  # [H] broadcast via mesh
    el_map = (0.5 - rows) * 2.0 * half_v
    az_grid, el_grid = np.meshgrid(az_map, el_map)
    mask = (
        (np.abs(az_grid - az) <= wedge)
        & (np.abs(el_grid - el) <= wedge)
        & np.isfinite(d)
        & (d > 0)
    )
    if not bool(mask.any()):
        return float("inf")
    vals = d[mask]
    p = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(vals, p))


def clearance_to_obstacle_label(
    clearance_m: float,
    *,
    d_near: float = 3.0,
    d_far: float = 22.0,
) -> float:
    """Map directional clearance (m) → [0, 1] training label (near → 1).

    Same band endpoints as the old clearance cliff, but only for **offline**
    supervision. Online reward must not re-apply ``w_collision=10`` on top.
    """
    if not np.isfinite(clearance_m):
        return 0.0
    d = float(clearance_m)
    lo, hi = float(d_near), float(d_far)
    if hi <= lo:
        return 1.0 if d <= lo else 0.0
    if d <= lo:
        return 1.0
    if d >= hi:
        return 0.0
    return float(np.clip((hi - d) / (hi - lo), 0.0, 1.0))
