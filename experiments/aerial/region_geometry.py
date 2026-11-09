"""Point-in-polygon helpers for env_airsim_16 urban / water regions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_REGIONS_PATH = Path(__file__).resolve().parent / "env_airsim16_regions.json"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def point_in_polygon(x: float, y: float, polygon_xy: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon (closed ring, last point may repeat first)."""
    poly = np.asarray(polygon_xy, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        return False
    px, py = float(x), float(y)
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > py) != (y2 > py)) and (
            px < (x2 - x1) * (py - y1) / max(y2 - y1, 1e-12) + x1
        ):
            inside = not inside
    return inside


def classify_spawn_xy(
    x: float,
    y: float,
    regions: Mapping[str, Any],
) -> Dict[str, Any]:
    urban_hits = [
        r["id"]
        for r in regions.get("urban_regions", [])
        if point_in_polygon(x, y, r["polygon_xy"])
    ]
    water_hits = [
        r["id"]
        for r in regions.get("water_exclusion", [])
        if point_in_polygon(x, y, r["polygon_xy"])
    ]
    in_urban = len(urban_hits) > 0
    in_water = len(water_hits) > 0
    return {
        "in_urban": in_urban,
        "in_water": in_water,
        "urban_region_ids": urban_hits,
        "water_region_ids": water_hits,
        "spawn_ok": bool(in_urban and not in_water),
    }


def load_regions(path: str | Path | None = None) -> Dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_REGIONS_PATH
    return _load_json(p)


def spawn_ok(
    spawn_xy: Sequence[float],
    *,
    regions_path: str | Path | None = None,
    regions: Mapping[str, Any] | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    data = regions if regions is not None else load_regions(regions_path)
    xy = np.asarray(spawn_xy, dtype=np.float64).reshape(-1)[:2]
    meta = classify_spawn_xy(float(xy[0]), float(xy[1]), data)
    return bool(meta["spawn_ok"]), meta
