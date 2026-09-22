#!/usr/bin/env python3
"""Review PathExpert urban-complex NPZ + emit curated train manifest."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.aerial.phase3_unified.region_geometry import (
    DEFAULT_REGIONS_PATH,
    classify_spawn_xy,
    load_regions,
    path_inland_metrics,
)

MIN_SPAWN_INLAND_M = 100.0
MIN_PATH_INLAND_M = 80.0
MIN_SPAWN_Y = 0.0
MIN_PATH_M = 28.0
MIN_STEPS = 24


def _positions(proprio: np.ndarray) -> np.ndarray:
    p = np.asarray(proprio, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return p[:, :3]


def _match_route_idx(proprio: np.ndarray, routes: dict[int, dict]) -> int:
    """Match episode spawn XY to annotation (z may be lifted for teacher collect)."""
    spawn = _positions(proprio)[0, :2]
    best_idx = -1
    best_d = float("inf")
    for route_idx, ep in routes.items():
        ref = np.asarray(ep["pos"][0], dtype=np.float64)[:2]
        d = float(np.linalg.norm(spawn - ref))
        if d < best_d:
            best_d = d
            best_idx = int(route_idx)
    if best_idx < 0 or best_d > 5.0:
        raise ValueError(f"spawn {spawn.tolist()} did not match any route (best_d={best_d:.2f}m)")
    return best_idx


def _path_m(proprio: np.ndarray) -> float:
    p = _positions(proprio)
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def review_dataset(
    src: Path,
    annotation: Path,
    *,
    min_spawn_inland_m: float = MIN_SPAWN_INLAND_M,
    min_path_inland_m: float = MIN_PATH_INLAND_M,
    min_spawn_y: float = MIN_SPAWN_Y,
    min_path_m: float = MIN_PATH_M,
    min_steps: int = MIN_STEPS,
) -> tuple[list[dict], list[dict]]:
    ann = json.loads(annotation.read_text(encoding="utf-8"))
    routes = {int(ep["route_idx"]): ep for ep in ann["episodes"]}
    regions = load_regions(DEFAULT_REGIONS_PATH)
    manifest_path = src / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("episodes", manifest) if isinstance(manifest, dict) else manifest

    reviewed: list[dict] = []
    keep: list[dict] = []
    for ent in entries:
        fn = str(ent["file"])
        data = np.load(src / fn, allow_pickle=True)
        route_idx = ent.get("route_idx")
        if route_idx is None:
            route_idx = _match_route_idx(data["proprio"], routes)
        route = routes[int(route_idx)]
        spawn = route["pos"][0]
        spawn_meta = classify_spawn_xy(float(spawn[0]), float(spawn[1]), regions)
        inland = path_inland_metrics(route["pos"], regions)
        steps = int(data["proprio"].shape[0])
        path_m = _path_m(data["proprio"])
        arrived = bool(ent.get("arrived", data.get("arrived", False)))
        geo_ok = (
            not spawn_meta["in_water"]
            and float(spawn[1]) >= min_spawn_y
            and float(spawn_meta["spawn_inland_m"]) >= min_spawn_inland_m
            and float(inland["path_min_inland_m"]) >= min_path_inland_m
        )
        data_ok = arrived or (path_m >= min_path_m and steps >= min_steps)
        row = {
            **ent,
            "route_idx": int(route_idx),
            "route_id": route.get("route_id", ""),
            "spawn_y": float(spawn[1]),
            "spawn_inland_m": float(spawn_meta["spawn_inland_m"]),
            "path_min_inland_m": float(inland["path_min_inland_m"]),
            "path_length_m": path_m,
            "steps": steps,
            "geo_ok": geo_ok,
            "data_ok": data_ok,
            "keep": bool(geo_ok and data_ok),
        }
        reviewed.append(row)
        if row["keep"]:
            keep.append(row)
    return reviewed, keep


def _row_score(row: dict) -> tuple[int, float]:
    return (int(bool(row.get("arrived"))), float(row["path_length_m"]))


def _top_k_per_route(rows: list[dict], max_per_route: int) -> list[dict]:
    """Keep up to K samples per route_idx — prefer arrived, then longest path."""
    k = max(1, int(max_per_route))
    by_route: dict[int, list[dict]] = {}
    for row in rows:
        by_route.setdefault(int(row["route_idx"]), []).append(row)
    out: list[dict] = []
    for ri in sorted(by_route):
        ranked = sorted(by_route[ri], key=_row_score, reverse=True)
        out.extend(ranked[:k])
    return out


def write_curated(
    src: Path,
    dst: Path,
    keep: list[dict],
    meta: dict,
    *,
    max_per_route: int = 1,
) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    keep = _top_k_per_route(keep, max_per_route)
    out_entries: list[dict] = []
    for i, row in enumerate(keep):
        src_file = src / row["file"]
        dst_file = dst / f"episode_{i:05d}.npz"
        shutil.copy2(src_file, dst_file)
        out_entries.append(
            {
                "file": dst_file.name,
                "route_idx": row["route_idx"],
                "route_id": row["route_id"],
                "steps": row["steps"],
                "arrived": bool(row.get("arrived")),
                "path_length_m": row["path_length_m"],
                "usable": True,
                "spawn_inland_m": row["spawn_inland_m"],
                "path_min_inland_m": row["path_min_inland_m"],
                "source": "urban_complex_curated",
                "source_file": row["file"],
            }
        )
    payload = {
        "meta": {
            **meta,
            "kind": "urban_complex_path_expert_curated",
            "n_written": len(out_entries),
            "n_source": len(keep),
            "source_dir": str(src),
        },
        "episodes": out_entries,
    }
    (dst / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (dst / "review_summary.json").write_text(
        json.dumps({"kept": out_entries}, indent=2) + "\n", encoding="utf-8"
    )
    return len(out_entries)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default="experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_20260916_v2",
    )
    p.add_argument(
        "--dst",
        default="experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_20260916_v2_curated",
    )
    p.add_argument(
        "--annotation",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json",
    )
    p.add_argument("--write-curated", action="store_true")
    p.add_argument(
        "--max-per-route",
        type=int,
        default=1,
        help="Keep up to K best samples per route_idx in curated output (default: 1).",
    )
    args = p.parse_args()
    src = Path(args.src)
    if not src.is_absolute():
        src = _REPO / src
    reviewed, keep = review_dataset(src, _REPO / args.annotation)
    print(f"[review] src={src} total={len(reviewed)} keep={len(keep)}")
    for row in reviewed:
        mark = "KEEP" if row["keep"] else "DROP"
        print(
            f"  {row['file']} route={row['route_idx']:2d} y={row['spawn_y']:.0f} "
            f"inland={row['spawn_inland_m']:.0f}/{row['path_min_inland_m']:.0f} "
            f"steps={row['steps']:3d} path={row['path_length_m']:.1f}m "
            f"arr={row.get('arrived')} -> {mark}"
        )
    if args.write_curated:
        dst = Path(args.dst)
        if not dst.is_absolute():
            dst = _REPO / dst
        n_written = write_curated(
            src,
            dst,
            keep,
            meta={
                "review_criteria": {
                    "min_spawn_inland_m": MIN_SPAWN_INLAND_M,
                    "min_path_inland_m": MIN_PATH_INLAND_M,
                    "min_spawn_y": MIN_SPAWN_Y,
                    "min_path_m": MIN_PATH_M,
                    "min_steps": MIN_STEPS,
                    "max_per_route": int(args.max_per_route),
                }
            },
            max_per_route=int(args.max_per_route),
        )
        print(f"[review] curated -> {dst} ({n_written} episodes, max_per_route={args.max_per_route})")
    return 0 if keep else 1


if __name__ == "__main__":
    raise SystemExit(main())
