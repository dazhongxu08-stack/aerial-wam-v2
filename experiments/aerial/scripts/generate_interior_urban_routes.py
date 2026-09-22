#!/usr/bin/env python3
"""Generate interior-urban routes from smoke-validated R09 template.

R09 spawn at (-1047, 11) passed FPV smoke (city blocks, not water). m1a20 only
has ~2 strict routes; this script synthesises up to N variants by:
  * sub-segmenting the R09 A* polyline (different start waypoints, same corridor)
  * translating the R09 polyline within ``interior_west_north`` (30 m grid)

All variants must pass the same filters as ``select_outdoor_complex_routes``:
urban spawn, interior zone, spawn_inland >= 100 m, path_min_inland >= 80 m,
spawn z lifted to >= 12 m for flyability.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.aerial.phase3_unified.region_geometry import (
    DEFAULT_REGIONS_PATH,
    classify_spawn_xy,
    load_regions,
    path_inland_metrics,
)
from experiments.aerial.scripts.select_outdoor_complex_routes import score_spawn

MIN_FLY_SPAWN_Z = 12.0
MIN_ROUTE_LEN_M = 80.0
MIN_SPAWN_INLAND_M = 100.0
MIN_PATH_INLAND_M = 80.0
R09_M1A20_IDX = 8


def _load_routes(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "routes" in data:
        return list(data["routes"])
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported format: {path}")


def _lift_z(pts: np.ndarray, min_z: float = MIN_FLY_SPAWN_Z) -> np.ndarray:
    out = np.asarray(pts, dtype=np.float64).copy()
    if out[0, 2] < min_z:
        out[:, 2] += min_z - out[0, 2]
    return out


def _horizontal_start_idx(pts: np.ndarray, min_xy_m: float = 1.0) -> int:
    """First waypoint with meaningful XY displacement from spawn (skip in-place climb)."""
    if len(pts) < 2:
        return 0
    origin = pts[0, :2]
    for i in range(1, len(pts)):
        if float(np.linalg.norm(pts[i, :2] - origin)) >= min_xy_m:
            return i
    return 0


def _yaws_from_path(pts: np.ndarray) -> List[float]:
    n = len(pts)
    if n == 0:
        return []
    yaws: List[float] = []
    for i in range(n):
        if i + 1 < n:
            dx = float(pts[i + 1, 0] - pts[i, 0])
            dy = float(pts[i + 1, 1] - pts[i, 1])
            if abs(dx) + abs(dy) < 1e-6 and i > 0:
                yaws.append(yaws[-1])
            else:
                yaws.append(math.atan2(dy, dx))
        else:
            yaws.append(yaws[-1] if yaws else 0.0)
    return yaws


def _normalize_flyable_path(
    pts: np.ndarray,
    yaws: Sequence[float],
    *,
    min_z: float = MIN_FLY_SPAWN_Z,
) -> Tuple[np.ndarray, List[float]]:
    """Drop vertical climb prefix, flatten to cruise z, align yaw with path tangent."""
    pts = np.asarray(pts, dtype=np.float64).copy()
    spawn_xy = pts[0, :2].copy()
    start = _horizontal_start_idx(pts)
    pts = pts[start:]
    if len(pts) < 4:
        return pts, list(yaws[start:]) if len(yaws) > start else _yaws_from_path(pts)
    pts[0, :2] = spawn_xy
    pts[:, 2] = min_z
    return pts, _yaws_from_path(pts)


def _route_len_m(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def _trim_route(
    src: Dict[str, Any],
    pts: np.ndarray,
    yaws: Sequence[float],
    *,
    variant_id: str,
    source_kind: str,
) -> Dict[str, Any]:
    n = len(pts)
    actions = list(src.get("action") or [])
    indices = list(src.get("index_list") or [])
    if len(actions) >= n:
        actions = actions[:n]
    else:
        actions = actions + [9] * (n - len(actions))
    if len(indices) >= n:
        indices = indices[:n]
    else:
        indices = indices + [f"{variant_id}_{i}" for i in range(len(indices), n)]
    return {
        "image_path": src.get("image_path", ""),
        "gpt_instruction": src.get("gpt_instruction", ""),
        "action": actions,
        "index_list": indices,
        "pos": [[float(x), float(y), float(z)] for x, y, z in pts],
        "yaw": [float(y) for y in yaws[:n]],
        "variant_id": variant_id,
        "source_kind": source_kind,
        "source_m1a20_idx": R09_M1A20_IDX,
        "source_route_label": "R09",
    }


def _score_variant(
    route: Dict[str, Any],
    *,
    regions: Dict[str, Any],
    route_idx: int,
) -> Optional[Dict[str, Any]]:
    row = score_spawn(
        route_idx,
        route,
        regions=regions,
        min_spawn_z=MIN_FLY_SPAWN_Z,
        max_spawn_z=22.0,
        min_spawn_inland_m=MIN_SPAWN_INLAND_M,
        min_path_inland_m=MIN_PATH_INLAND_M,
    )
    if row is None:
        return None
    if _route_len_m(np.asarray(route["pos"])) < MIN_ROUTE_LEN_M:
        return None
    return row


def _spawn_key(pts: np.ndarray, step_m: float = 15.0) -> Tuple[int, int]:
    return (int(round(pts[0, 0] / step_m)), int(round(pts[0, 1] / step_m)))


def generate_variants(
    src: Dict[str, Any],
    *,
    regions: Dict[str, Any],
    n_target: int = 20,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_pts, base_yaws = _normalize_flyable_path(
        np.asarray(src["pos"], dtype=np.float64),
        list(src.get("yaw") or [0.0] * len(src["pos"])),
    )
    candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    seen_spawns: set[Tuple[int, int]] = set()

    def _try_add(pts: np.ndarray, yaws: Sequence[float], variant_id: str, kind: str) -> None:
        pts, yaws = _normalize_flyable_path(pts, yaws)
        key = _spawn_key(pts)
        if key in seen_spawns:
            return
        route = _trim_route(src, pts, yaws, variant_id=variant_id, source_kind=kind)
        meta = _score_variant(route, regions=regions, route_idx=len(candidates))
        if meta is None:
            return
        seen_spawns.add(key)
        # prefer R09-native spawns, then longer paths, then higher inland
        pri = (
            1.0 if kind == "r09_subsegment" else 0.5,
            float(meta["spawn_inland_m"]),
            float(meta["path_min_inland_m"]),
            _route_len_m(pts),
        )
        candidates.append((sum(pri), route, meta))

    for i in range(len(base_pts) - 3):
        _try_add(base_pts[i:], base_yaws[i:], f"r09_sub_{i:02d}", "r09_subsegment")

    for dx in range(-120, 121, 30):
        for dy in range(-90, 91, 30):
            if dx == 0 and dy == 0:
                continue
            shifted = base_pts.copy()
            shifted[:, 0] += dx
            shifted[:, 1] += dy
            _try_add(shifted, base_yaws, f"r09_x{dx:+d}_y{dy:+d}", "r09_translate")

    candidates.sort(key=lambda x: (-x[0], -float(x[2]["spawn_inland_m"])))
    picked: List[Dict[str, Any]] = []
    scored: List[Dict[str, Any]] = []
    for _, route, meta in candidates[:n_target]:
        route = dict(route)
        route["complexity"] = meta
        picked.append(route)
        scored.append(meta)

    return picked, scored


def main() -> int:
    p = argparse.ArgumentParser(description="Generate R09-standard interior urban routes")
    p.add_argument("--input", default="artifacts/seen_airsim16_m1a20.json")
    p.add_argument("--regions", default=str(DEFAULT_REGIONS_PATH))
    p.add_argument("--out", default="artifacts/seen_airsim16_interior_routes.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument(
        "--phase3-out",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    regions = load_regions(root / args.regions)
    routes = _load_routes(root / args.input)
    src = routes[R09_M1A20_IDX]
    selected, scored = generate_variants(src, regions=regions, n_target=int(args.n))

    if len(selected) < int(args.n):
        print(
            f"WARNING: only {len(selected)}/{args.n} interior routes generated "
            f"(R09 template + filters)",
            file=sys.stderr,
        )

    for j, r in enumerate(selected):
        r["route_id"] = f"interior_route_{j + 1:02d}"
        r["route_idx"] = j
        r["base_route_idx"] = R09_M1A20_IDX
        r["category"] = "outdoor_complex"
        r["template"] = "r09_smoke_validated"

    payload = {
        "version": "airsim16_interior_r09_template_v2",
        "description": "R09 interior routes: horizontal segment only, flat cruise z, tangent yaw.",
        "template": {"m1a20_idx": R09_M1A20_IDX, "route_label": "R09", "spawn_xy": [-1047.5, 11.3]},
        "filters": {
            "min_spawn_inland_m": MIN_SPAWN_INLAND_M,
            "min_path_inland_m": MIN_PATH_INLAND_M,
            "min_fly_spawn_z": MIN_FLY_SPAWN_Z,
            "min_route_len_m": MIN_ROUTE_LEN_M,
        },
        "n_routes": len(selected),
        "routes": selected,
        "all_scores": scored,
    }

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} routes -> {out}")
    for r in selected:
        c = r["complexity"]
        sp = r["pos"][0]
        print(
            f"  {r['route_id']} {r['variant_id']} "
            f"spawn=({sp[0]:.0f},{sp[1]:.0f},z={sp[2]:.1f}) "
            f"inland={c['spawn_inland_m']:.0f}m len={c['nominal_length_m']:.0f}m"
        )

    # Phase-3 annotation wrapper
    from experiments.aerial.phase3_unified.mixed_corpus import tag_outdoor_routes

    tagged = tag_outdoor_routes(selected)
    for ep in tagged:
        ep["scene"] = "outdoor_complex"
        if "complexity" in ep:
            ep["complexity_meta"] = ep.pop("complexity")
        ep.setdefault("complexity_meta", {})["smoke_template"] = "R09"

    phase3 = {
        "protocol_version": "phase3_outdoor_complex_r09_panel_v1",
        "scene": "outdoor_complex",
        "n_episodes": len(tagged),
        "template": "R09_smoke_validated_interior",
        "episodes": tagged,
    }
    p3_out = root / args.phase3_out
    p3_out.parent.mkdir(parents=True, exist_ok=True)
    p3_out.write_text(json.dumps(phase3, indent=2), encoding="utf-8")
    print(f"Wrote Phase-3 annotation ({len(tagged)} episodes) -> {p3_out}")
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
