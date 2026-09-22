"""Backfill per-step polyline carrot goals into PathExpert NPZ corpora.

Writes ``goals`` array [N,3] (one subgoal per frame) so micro-FT imagination
matches V11 polyline eval. Terminal ``goal`` key is preserved as route endpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.aerial.rl.subgoal_generator import (
    nearest_on_polyline,
    sample_point_along_polyline,
)


def _match_route_idx(spawn_xy: np.ndarray, routes: Dict[int, Dict[str, Any]]) -> int:
    best_idx = -1
    best_d = float("inf")
    for ri, ep in routes.items():
        ref = np.asarray(ep["pos"][0], dtype=np.float64)[:2]
        d = float(np.linalg.norm(spawn_xy - ref))
        if d < best_d:
            best_d = d
            best_idx = int(ri)
    if best_idx < 0 or best_d > 5.0:
        raise ValueError(f"spawn {spawn_xy.tolist()} unmatched (best_d={best_d:.2f}m)")
    return best_idx


def polyline_subgoals_for_episode(
    proprio: np.ndarray,
    path_pts: np.ndarray,
    *,
    r_lookahead: float = 25.0,
) -> np.ndarray:
    pts = np.asarray(path_pts, dtype=np.float64).reshape(-1, 3)
    p = np.asarray(proprio, dtype=np.float64).reshape(-1, 4)
    out = np.zeros((len(p), 3), dtype=np.float32)
    for i in range(len(p)):
        pos = p[i, :3]
        proj, seg, _s, rem = nearest_on_polyline(pos, pts)
        if rem <= 8.0:
            out[i] = pts[-1].astype(np.float32)
        else:
            out[i] = sample_point_along_polyline(pts, seg, proj, r_lookahead).astype(np.float32)
    return out


def backfill_dir(
    dataset: Path,
    annotation: Path,
    *,
    r_lookahead: float = 25.0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    ann = json.loads(annotation.read_text(encoding="utf-8"))
    routes = {int(e["route_idx"]): e for e in ann["episodes"]}
    paths = sorted(dataset.glob("episode_*.npz"))
    n_written = 0
    for path in paths:
        raw = dict(np.load(path, allow_pickle=True))
        proprio = np.asarray(raw["proprio"], dtype=np.float64)
        spawn = proprio[0, :3]
        ri = _match_route_idx(spawn[:2], routes)
        pts = np.asarray(routes[ri]["pos"], dtype=np.float64)
        goals = polyline_subgoals_for_episode(proprio, pts, r_lookahead=r_lookahead)
        terminal = pts[-1].astype(np.float32)
        if dry_run:
            n_written += 1
            continue
        raw["goals"] = goals
        raw["goal"] = terminal
        tmp = path.with_name(path.stem + ".polybackfill.npz")
        np.savez_compressed(tmp, **raw)
        tmp.replace(path)
        n_written += 1
    return {"n_files": len(paths), "n_written": n_written, "r_lookahead": r_lookahead}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--annotation",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json",
    )
    p.add_argument("--r-lookahead", type=float, default=25.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    ds = Path(args.dataset)
    if not ds.is_absolute():
        ds = _REPO / ds
    ann = Path(args.annotation)
    if not ann.is_absolute():
        ann = _REPO / ann
    rep = backfill_dir(ds, ann, r_lookahead=float(args.r_lookahead), dry_run=args.dry_run)
    print(f"[polyline-backfill] {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
