#!/usr/bin/env python3
"""Interior-urban start/goal routes with spread headings and a region-level split.

Every earlier urban annotation is the R09 polyline translated on a 30 m grid:
one heading, one 89 m chord. This draws new start/goal pairs inside the
``urban_interior`` polygons so train and held-out routes differ in heading,
length, and (by region) in the buildings they cross.

Filters match ``generate_interior_urban_routes``: start and goal in an urban
interior polygon and out of water, start and goal inland >= 100 m, every point
on the straight segment interior and inland >= 80 m. Output is a candidate
pool; ``validate_od_routes_airsim`` decides keep/drop in the simulator.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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

MIN_SPAWN_INLAND_M = 100.0
MIN_PATH_INLAND_M = 80.0
# west_southwest and west_central render as open water; see scan_urban_mask_airsim.
DEFAULT_TRAIN_REGIONS = ("interior_west_north",)
DEFAULT_HOLDOUT_REGIONS = ("interior_east",)

XY = Tuple[float, float]


class UrbanMask:
    """Nearest-cell lookup on the rendered-scene scan from ``scan_urban_mask_airsim``."""

    def __init__(self, path: Path, open_clear_m: float) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cells = data["cells"]
        self.grid_m = float(data["grid_m"])
        self.xy = np.array([[c["x"], c["y"]] for c in cells], dtype=np.float64)
        self.urban = np.array([bool(c["urban"]) for c in cells])
        self.open_xy = self.xy[~self.urban]
        self.open_clear_m = float(open_clear_m)

    def is_urban(self, x: float, y: float) -> bool:
        d = np.linalg.norm(self.xy - np.array([x, y]), axis=1)
        i = int(np.argmin(d))
        return bool(self.urban[i]) and float(d[i]) <= self.grid_m

    def open_dist_m(self, x: float, y: float) -> float:
        if len(self.open_xy) == 0:
            return float("inf")
        return float(np.min(np.linalg.norm(self.open_xy - np.array([x, y]), axis=1)))

    def end_ok(self, x: float, y: float) -> bool:
        return self.is_urban(x, y) and self.open_dist_m(x, y) >= self.open_clear_m


_MASK: Optional["UrbanMask"] = None


def interior_cells(regions: Dict[str, Any], region_id: str, grid_m: float) -> List[XY]:
    poly = next(r for r in regions["urban_interior"] if r["id"] == region_id)
    p = np.asarray(poly["polygon_xy"], dtype=np.float64)
    cells: List[XY] = []
    for x in np.arange(p[:, 0].min(), p[:, 0].max() + 1e-6, grid_m):
        for y in np.arange(p[:, 1].min(), p[:, 1].max() + 1e-6, grid_m):
            m = classify_spawn_xy(float(x), float(y), regions)
            if (
                region_id in m["interior_region_ids"]
                and m["in_urban"]
                and not m["in_water"]
                and m["spawn_inland_m"] >= MIN_SPAWN_INLAND_M
                and (_MASK is None or _MASK.end_ok(float(x), float(y)))
            ):
                cells.append((float(x), float(y)))
    return cells


def segment_ok(s: XY, g: XY, regions: Dict[str, Any], region_id: str) -> bool:
    line = [
        (s[0] + (g[0] - s[0]) * t, s[1] + (g[1] - s[1]) * t, 0.0)
        for t in np.linspace(0.0, 1.0, 12)
    ]
    if path_inland_metrics(line, regions)["path_min_inland_m"] < MIN_PATH_INLAND_M:
        return False
    if _MASK is not None and not all(_MASK.is_urban(x, y) for x, y, _ in line):
        return False
    return all(
        region_id in classify_spawn_xy(x, y, regions)["interior_region_ids"]
        for x, y, _ in line
    )


def heading_deg(s: XY, g: XY) -> float:
    return math.degrees(math.atan2(g[1] - s[1], g[0] - s[0]))


def region_pairs(
    regions: Dict[str, Any],
    region_id: str,
    *,
    grid_m: float,
    min_chord_m: float,
    max_chord_m: float,
) -> List[Dict[str, Any]]:
    cells = interior_cells(regions, region_id, grid_m)
    out: List[Dict[str, Any]] = []
    for s in cells:
        for g in cells:
            d = math.dist(s, g)
            if not (min_chord_m <= d <= max_chord_m):
                continue
            if not segment_ok(s, g, regions, region_id):
                continue
            out.append({"start": s, "goal": g, "chord_m": d, "heading_deg": heading_deg(s, g)})
    return out


def pick_spread(
    pairs: Sequence[Dict[str, Any]],
    n: int,
    *,
    rng: random.Random,
    min_sep_m: float,
    chord_edges: Sequence[float],
) -> List[Dict[str, Any]]:
    """Round-robin over (heading octant, chord band) buckets; spawns and goals stay apart."""
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for p in pairs:
        h = int(((p["heading_deg"] + 180.0) % 360.0) // 45.0)
        c = int(np.searchsorted(chord_edges, p["chord_m"], side="right"))
        buckets.setdefault((h, c), []).append(p)
    for v in buckets.values():
        rng.shuffle(v)
    keys = sorted(buckets)
    rng.shuffle(keys)
    picked: List[Dict[str, Any]] = []

    def far(p: Dict[str, Any]) -> bool:
        return all(
            math.dist(p["start"], q["start"]) >= min_sep_m
            and math.dist(p["goal"], q["goal"]) >= min_sep_m
            for q in picked
        )

    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:
            while buckets[k]:
                p = buckets[k].pop()
                if far(p):
                    picked.append(p)
                    break
            if len(picked) >= n:
                break
    return picked


def to_episode(
    p: Dict[str, Any],
    *,
    split: str,
    region_id: str,
    idx: int,
    cruise_z: float,
    step_m: float,
    regions: Dict[str, Any],
) -> Dict[str, Any]:
    s, g = p["start"], p["goal"]
    n = max(2, int(math.ceil(p["chord_m"] / step_m)) + 1)
    pts = [
        [s[0] + (g[0] - s[0]) * t, s[1] + (g[1] - s[1]) * t, float(cruise_z)]
        for t in np.linspace(0.0, 1.0, n)
    ]
    yaw = math.radians(p["heading_deg"])
    ms = classify_spawn_xy(s[0], s[1], regions)
    mg = classify_spawn_xy(g[0], g[1], regions)
    inland = path_inland_metrics(pts, regions)
    return {
        "image_path": "",
        "gpt_instruction": "",
        "action": [],
        "index_list": [],
        "pos": [[float(a), float(b), float(c)] for a, b, c in pts],
        "yaw": [float(yaw)] * n,
        "variant_id": f"od_{region_id}_{idx:02d}",
        "source_kind": "interior_od",
        "route_id": f"od_{split}_{idx:02d}",
        "route_idx": idx,
        "base_route_idx": None,
        "category": "outdoor_complex",
        "scene": "outdoor_complex",
        "template": "interior_od_v1",
        "complexity_meta": {
            "split": split,
            "region": region_id,
            "chord_m": round(float(p["chord_m"]), 1),
            "heading_deg": round(float(p["heading_deg"]), 1),
            "spawn_inland_m": ms["spawn_inland_m"],
            "goal_inland_m": mg["spawn_inland_m"],
            "path_min_inland_m": inland["path_min_inland_m"],
            "in_urban": bool(ms["in_urban"] and mg["in_urban"]),
            "in_interior": bool(ms["in_interior"] and mg["in_interior"]),
            "in_water": bool(ms["in_water"] or mg["in_water"]),
        },
    }


def build_split(
    regions: Dict[str, Any],
    region_ids: Sequence[str],
    n_per_region: Sequence[int],
    *,
    split: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for region_id, n in zip(region_ids, n_per_region):
        pairs = region_pairs(
            regions,
            region_id,
            grid_m=float(args.grid_m),
            min_chord_m=float(args.min_chord_m),
            max_chord_m=float(args.max_chord_m),
        )
        picked = pick_spread(
            pairs,
            int(n),
            rng=rng,
            min_sep_m=float(args.min_sep_m),
            chord_edges=(80.0, 100.0),
        )
        print(f"{split} {region_id}: {len(pairs)} valid pairs, picked {len(picked)}/{n}")
        for p in picked:
            episodes.append(
                to_episode(
                    p,
                    split=split,
                    region_id=region_id,
                    idx=len(episodes),
                    cruise_z=float(args.cruise_z),
                    step_m=float(args.step_m),
                    regions=regions,
                )
            )
    return episodes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", default=str(DEFAULT_REGIONS_PATH))
    ap.add_argument("--train-regions", default=",".join(DEFAULT_TRAIN_REGIONS))
    ap.add_argument("--train-n", default="40", help="candidates per train region")
    ap.add_argument("--holdout-regions", default=",".join(DEFAULT_HOLDOUT_REGIONS))
    ap.add_argument("--holdout-n", default="24", help="candidates per held-out region")
    ap.add_argument("--grid-m", type=float, default=10.0)
    ap.add_argument("--min-chord-m", type=float, default=60.0)
    ap.add_argument("--max-chord-m", type=float, default=120.0)
    ap.add_argument("--min-sep-m", type=float, default=15.0)
    ap.add_argument(
        "--urban-mask",
        default="experiments/aerial/rl/artifacts/interior_od_routes_20260930/urban_mask.json",
        help="rendered-scene scan; empty string disables",
    )
    ap.add_argument("--mask-open-clear-m", type=float, default=60.0)
    ap.add_argument("--cruise-z", type=float, default=24.0)
    ap.add_argument("--step-m", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument(
        "--out",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_od_candidates.json",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    regions = load_regions(args.regions)
    global _MASK
    if args.urban_mask:
        _MASK = UrbanMask(root / args.urban_mask, float(args.mask_open_clear_m))
    rng = random.Random(int(args.seed))
    train_ids = [s for s in args.train_regions.split(",") if s]
    hold_ids = [s for s in args.holdout_regions.split(",") if s]
    if set(train_ids) & set(hold_ids):
        raise SystemExit("train and held-out regions overlap")
    train = build_split(
        regions, train_ids, [int(x) for x in args.train_n.split(",")],
        split="train", rng=rng, args=args,
    )
    holdout = build_split(
        regions, hold_ids, [int(x) for x in args.holdout_n.split(",")],
        split="holdout", rng=rng, args=args,
    )
    payload = {
        "protocol_version": "interior_od_candidates_v1",
        "scene": "outdoor_complex",
        "filters": {
            "min_spawn_inland_m": MIN_SPAWN_INLAND_M,
            "min_path_inland_m": MIN_PATH_INLAND_M,
            "require_interior": True,
            "chord_m": [float(args.min_chord_m), float(args.max_chord_m)],
            "cruise_z": float(args.cruise_z),
        },
        "split_regions": {"train": train_ids, "holdout": hold_ids},
        "train": train,
        "holdout": holdout,
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(train)} train + {len(holdout)} held-out candidates -> {out}")
    return 0 if train and holdout else 1


if __name__ == "__main__":
    raise SystemExit(main())
