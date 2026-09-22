#!/usr/bin/env python3
"""Select outdoor-complex routes by **urban spawn point** on env_airsim_16.

A route is eligible only when its birth point (``pos[0][:2]``):
  * lies inside an **urban fly zone** polygon (see ``env_airsim16_regions.json``)
  * is **outside** all **water exclusion** polygons
  * has spawn z in ``[min_spawn_z, max_spawn_z]`` (default 2–22 m)
  * prefers ``low_*`` astar band at spawn

Trajectory shape is **not** used.
"""

from __future__ import annotations

import argparse
import json
import re
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

_RE_ASTAR_CAT = re.compile(r"astar_data/([^/]+)/")


def _load_routes(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "routes" in data:
        return list(data["routes"])
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported annotation format in {path}")


def _long16_base_indices(long_routes_path: Path) -> set[int]:
    if not long_routes_path.is_file():
        return set()
    routes = _load_routes(long_routes_path)
    out: set[int] = set()
    for r in routes:
        b = r.get("base_route_idx")
        if b is not None:
            out.add(int(b))
    return out


def _astar_category(image_path: str) -> str:
    m = _RE_ASTAR_CAT.search(str(image_path or ""))
    return m.group(1) if m else ""


def _spawn_band_score(cat: str) -> float:
    head = cat.split("_")[0] if cat else ""
    return {"low": 3.0, "medium": 1.0, "high": 0.0}.get(head, 0.5)


def score_spawn(
    route_idx: int,
    route: Dict[str, Any],
    *,
    regions: Dict[str, Any],
    min_spawn_z: float = 0.0,
    max_spawn_z: float = 22.0,
    min_spawn_inland_m: float = 0.0,
    min_path_inland_m: float = 0.0,
    excluded_from_long16: bool = False,
) -> Optional[Dict[str, Any]]:
    pts = np.asarray(route.get("pos", route.get("positions")), dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return None

    spawn = pts[0].copy()
    spawn_z = float(spawn[2])
    if spawn_z < min_spawn_z or spawn_z > max_spawn_z:
        return None

    region_meta = classify_spawn_xy(float(spawn[0]), float(spawn[1]), regions)
    if not region_meta["spawn_ok"]:
        return None

    inland = path_inland_metrics(pts, regions)
    spawn_inland_m = float(region_meta["spawn_inland_m"])
    path_min_inland_m = float(inland["path_min_inland_m"])
    if spawn_inland_m < min_spawn_inland_m or path_min_inland_m < min_path_inland_m:
        return None

    cat = _astar_category(route.get("image_path", ""))
    interior_bonus = 4.0 if region_meta.get("in_interior") else 0.0
    complexity = (
        spawn_inland_m * 0.04
        + path_min_inland_m * 0.02
        + float(inland["path_median_inland_m"]) * 0.01
        + interior_bonus
        + _spawn_band_score(cat) * 2.0
        + max(0.0, (max_spawn_z - spawn_z) / max(max_spawn_z, 1e-6)) * 5.0
        + (0.5 if excluded_from_long16 else 0.0)
    )

    seg_len = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))

    return {
        "m1a20_idx": int(route_idx),
        "route_label": f"R{route_idx + 1:02d}",
        "astar_category": cat,
        "spawn_pos": [round(float(x), 3) for x in spawn],
        "spawn_z_m": round(spawn_z, 2),
        "urban_region_ids": region_meta["urban_region_ids"],
        "interior_region_ids": region_meta.get("interior_region_ids", []),
        "water_region_ids": region_meta["water_region_ids"],
        "spawn_inland_m": round(spawn_inland_m, 1),
        "path_min_inland_m": path_min_inland_m,
        "path_median_inland_m": inland["path_median_inland_m"],
        "in_interior": bool(region_meta.get("in_interior")),
        "nominal_length_m": round(seg_len, 2),
        "excluded_from_long16": bool(excluded_from_long16),
        "complexity_score": round(complexity, 3),
        "selection_basis": "urban_interior_inland",
        "gpt_instruction": str(route.get("gpt_instruction", "")),
        "image_path": route.get("image_path", ""),
    }


def _score_all_spawns(
    routes: Sequence[Dict[str, Any]],
    *,
    regions: Dict[str, Any],
    long16_bases: set[int],
    min_spawn_z: float,
    max_spawn_z: float,
    min_spawn_inland_m: float,
    min_path_inland_m: float,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for i, r in enumerate(routes):
        row = score_spawn(
            i,
            r,
            regions=regions,
            min_spawn_z=min_spawn_z,
            max_spawn_z=max_spawn_z,
            min_spawn_inland_m=min_spawn_inland_m,
            min_path_inland_m=min_path_inland_m,
            excluded_from_long16=(i not in long16_bases),
        )
        if row is not None:
            scored.append(row)
    scored.sort(
        key=lambda x: (
            -int(bool(x.get("in_interior"))),
            -float(x["spawn_inland_m"]),
            -float(x["path_min_inland_m"]),
            -float(x["complexity_score"]),
        )
    )
    return scored


def select_complex_routes(
    routes: Sequence[Dict[str, Any]],
    *,
    regions: Dict[str, Any],
    n_select: int = 6,
    long16_bases: Optional[set[int]] = None,
    min_spawn_z: float = 0.0,
    max_spawn_z: float = 22.0,
    min_spawn_inland_m: float = 100.0,
    min_path_inland_m: float = 80.0,
    force_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    long16_bases = long16_bases or set()
    scored = _score_all_spawns(
        routes,
        regions=regions,
        long16_bases=long16_bases,
        min_spawn_z=min_spawn_z,
        max_spawn_z=max_spawn_z,
        min_spawn_inland_m=min_spawn_inland_m,
        min_path_inland_m=min_path_inland_m,
    )
    if len(scored) < n_select:
        for spawn_thr, path_thr in ((80.0, 50.0), (60.0, 30.0)):
            if len(scored) >= n_select:
                break
            extra = _score_all_spawns(
                routes,
                regions=regions,
                long16_bases=long16_bases,
                min_spawn_z=min_spawn_z,
                max_spawn_z=max_spawn_z,
                min_spawn_inland_m=spawn_thr,
                min_path_inland_m=path_thr,
            )
            seen = {int(s["m1a20_idx"]) for s in scored}
            for row in extra:
                idx = int(row["m1a20_idx"])
                if idx in seen:
                    continue
                # Never backfill known waterfront corridor (R14 smoke: buildings + water).
                if float(row["spawn_inland_m"]) < 100.0 and not row.get("in_interior"):
                    continue
                scored.append(row)
                seen.add(idx)
                if len(scored) >= n_select:
                    break
        scored.sort(
            key=lambda x: (
                -int(bool(x.get("in_interior"))),
                -float(x["spawn_inland_m"]),
                -float(x["path_min_inland_m"]),
                -float(x["complexity_score"]),
            )
        )

    forced = list(force_indices or [])
    picked_idx: List[int] = []
    cat_counts: Dict[str, int] = {}
    urban_counts: Dict[str, int] = {}

    def _take(idx: int) -> None:
        if idx in picked_idx:
            return
        if idx < 0 or idx >= len(routes):
            raise ValueError(f"route index out of range: {idx}")
        row = next((s for s in scored if s["m1a20_idx"] == idx), None)
        if row is None:
            raise ValueError(
                f"route R{idx + 1:02d} failed urban-interior filter "
                f"(urban spawn, inland>={min_spawn_inland_m}m, path>={min_path_inland_m}m, "
                f"z in [{min_spawn_z}, {max_spawn_z}])"
            )
        picked_idx.append(idx)
        cat = row["astar_category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        for uid in row.get("urban_region_ids", []):
            urban_counts[uid] = urban_counts.get(uid, 0) + 1

    for idx in forced:
        _take(int(idx))

    for row in scored:
        if len(picked_idx) >= n_select:
            break
        idx = int(row["m1a20_idx"])
        if idx in picked_idx:
            continue
        uids = row.get("urban_region_ids", [])
        if uids and urban_counts.get(uids[0], 0) >= 2 and len(picked_idx) < n_select - 1:
            continue
        _take(idx)

    for row in scored:
        if len(picked_idx) >= n_select:
            break
        idx = int(row["m1a20_idx"])
        if idx not in picked_idx:
            _take(idx)

    selected_routes: List[Dict[str, Any]] = []
    for j, idx in enumerate(picked_idx):
        src = dict(routes[idx])
        meta = next(s for s in scored if s["m1a20_idx"] == idx)
        src["route_id"] = f"complex_route_{j + 1:02d}"
        src["route_idx"] = j
        src["base_route_idx"] = idx
        src["category"] = "outdoor_complex"
        src["complexity"] = meta
        selected_routes.append(src)

    return selected_routes, scored


def main() -> int:
    p = argparse.ArgumentParser(description="Select urban-spawn outdoor-complex routes")
    p.add_argument("--input", default="artifacts/seen_airsim16_m1a20.json")
    p.add_argument("--long-routes", default="artifacts/seen_airsim16_long_routes.json")
    p.add_argument("--regions", default=str(DEFAULT_REGIONS_PATH.relative_to(Path(__file__).resolve().parents[3])))
    p.add_argument("--out", default="artifacts/seen_airsim16_complex_routes.json")
    p.add_argument("--n-select", type=int, default=6)
    p.add_argument("--min-spawn-z", type=float, default=0.0)
    p.add_argument("--max-spawn-z", type=float, default=22.0)
    p.add_argument("--min-spawn-inland-m", type=float, default=100.0)
    p.add_argument("--min-path-inland-m", type=float, default=80.0)
    p.add_argument(
        "--force-indices",
        default="8,4",
        help="m1a20 indices to pin first (default R09 smoke-validated, R05 backup)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    regions = load_regions(root / args.regions)
    routes = _load_routes(root / args.input)
    long_bases = _long16_base_indices(root / args.long_routes)
    forced = [int(x) for x in str(args.force_indices).split(",") if str(x).strip() != ""]

    selected, scored = select_complex_routes(
        routes,
        regions=regions,
        n_select=int(args.n_select),
        long16_bases=long_bases,
        min_spawn_z=float(args.min_spawn_z),
        max_spawn_z=float(args.max_spawn_z),
        min_spawn_inland_m=float(args.min_spawn_inland_m),
        min_path_inland_m=float(args.min_path_inland_m),
        force_indices=forced or None,
    )

    out_payload = {
        "version": "airsim16_complex_outdoor_urban_interior_v4",
        "description": "Urban-interior outdoor complex panel (inland spawns, path stays off water).",
        "source": str((root / args.input).relative_to(root)),
        "regions": str((root / args.regions).relative_to(root)),
        "n_routes": len(selected),
        "selection": {
            "basis": "urban_interior_inland",
            "n_select": int(args.n_select),
            "min_spawn_z": float(args.min_spawn_z),
            "max_spawn_z": float(args.max_spawn_z),
            "min_spawn_inland_m": float(args.min_spawn_inland_m),
            "min_path_inland_m": float(args.min_path_inland_m),
            "force_indices": forced,
            "long16_base_indices": sorted(long_bases),
        },
        "routes": selected,
        "all_spawn_scores": scored,
    }

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    print(f"Wrote {len(selected)} urban-spawn routes -> {out_path}")
    for r in selected:
        c = r["complexity"]
        print(
            f"  {r['route_id']} {c['route_label']} score={c['complexity_score']:.2f} "
            f"inland={c['spawn_inland_m']:.0f}m path_min={c['path_min_inland_m']:.0f}m "
            f"spawn_z={c['spawn_z_m']:.1f}m interior={c.get('interior_region_ids', [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
