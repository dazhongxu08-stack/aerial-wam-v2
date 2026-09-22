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
