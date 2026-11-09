"""Train routes whose goal sits behind a building, on the per-route train top-down maps (offline, no sim).

Same obstacle model as hard_route_select.py (surface >= alt - 1 m plus unrendered blobs touching it). On a 3 m grid
inside each map, a (spawn, goal) pair is kept when:
  - both ends: urban interior, not water, region inland >= 100 m, measured shore >= 100 m, >= 3 m from obstacles,
    >= --holdout-sep-m from every held-out route point (valid/hard/holdout annotations);
  - chord 25-55 m, the straight spawn->goal segment crosses a rendered building, and the 1.5 m-clearance shortest path
    is >= --min-ratio x (chord - 3 m) and every 10th point of it has region inland >= 80 m and shore >= 80 m.
Endpoints stay >= --edge-m inside the map (buildings past the edge are unseen). Pairs are thinned so spawns are
>= --sep-m apart across all maps (train maps overlap); at most --per-map per map. Writes an annotation (split=train,
route_id bh_<map route>_<k>, yaw = heading to the goal) and a top dir with top_<route_id>.npz copies of the source map
plus the original maps, for train_consequence --geo-top-dir.
Positions: annotation (x, y, z up) = AirSim NED (x, y, -z).
usage: behind_route_gen.py OUT_ANN TOP_OUT_DIR --top-dir D --regions R1,R2 --water-raster W --holdout-ann a,b
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

SUCCESS_M, GRID_M, END_CLEAR_M = 3.0, 3.0, 3.0


def disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def graph(free: np.ndarray):
    nx, ny = free.shape
    idx = -np.ones(free.shape, int)
    idx[free] = np.arange(int(free.sum()))
    rows, cols, w = [], [], []
    for di, dj, c in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
        j0, j1 = max(0, -dj), ny - max(0, dj)
        ia = idx[0:nx - di, j0:j1]
        ib = idx[di:nx, j0 + dj:j1 + dj]
        m = (ia >= 0) & (ib >= 0)
        rows += [ia[m], ib[m]]
        cols += [ib[m], ia[m]]
        w += [np.full(int(m.sum()), c)] * 2
    n = int(free.sum())
    g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    return idx, g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_ann")
    ap.add_argument("top_out_dir")
    ap.add_argument("--top-dir", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--water-raster", required=True)
    ap.add_argument("--holdout-ann", required=True)
    ap.add_argument("--holdout-sep-m", type=float, default=60.0)
    ap.add_argument("--alt", type=float, default=24.0)
    ap.add_argument("--min-ratio", type=float, default=1.35)
    ap.add_argument("--chord", default="25,55")
    ap.add_argument("--sep-m", type=float, default=12.0)
    ap.add_argument("--per-map", type=int, default=6)
    ap.add_argument("--edge-m", type=float, default=15.0, help="endpoints this far inside the map edge")
    ap.add_argument("--exclude-ann", default="",
                    help="skip pairs whose spawn and goal are both within --sep-m of a route in these annotations")
    ap.add_argument("--rid-prefix", default="bh")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiments.aerial.phase3_unified.region_geometry import classify_spawn_xy, load_regions
    from water_audit import Shore

    regions = [load_regions(p) for p in args.regions.split(",")]
    shore = Shore(args.water_raster)
    held = []
    for a in args.holdout_ann.split(","):
        for e in json.loads(Path(a).read_text())["episodes"]:
            pos = np.asarray(e["pos"], float)[:, :2]
            n = max(2, int(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)) / 5.0))
            t = np.linspace(0, len(pos) - 1, n)
            held += [pos[int(math.floor(u))] + (u - math.floor(u)) * (pos[min(int(math.floor(u)) + 1, len(pos) - 1)]
                                                                         - pos[int(math.floor(u))]) for u in t]
    held = np.asarray(held, float)
    c_lo, c_hi = (float(x) for x in args.chord.split(","))
    cache: dict = {}

    def geo(x: float, y: float, inland: float, need_interior: bool) -> bool:
        key = (round(x), round(y), inland, need_interior)
        if key not in cache:
            ok = False
            for R in regions:
                s = classify_spawn_xy(float(x), float(y), R)
                if s["in_urban"] and not s["in_water"] and s["spawn_inland_m"] >= inland and (
                        s["in_interior"] or not need_interior):
                    ok = True
                    break
            cache[key] = ok and shore(float(x), float(y)) >= inland
        return cache[key]

    top_out = Path(args.top_out_dir)
    top_out.mkdir(parents=True, exist_ok=True)
    episodes, review, spawns = [], [], []
    excl = [np.asarray([e["pos"][0][:2], e["pos"][-1][:2]], float)
            for a in filter(None, args.exclude_ann.split(",")) for e in json.loads(Path(a).read_text())["episodes"]]
    for tp in sorted(Path(args.top_dir).glob("top_*.npz")):
        shutil.copy(tp, top_out / tp.name)
        src_rid = tp.stem[len("top_"):]
        z = np.load(tp)
        top, x0, y0, res = z["top_ned_z"], float(z["x0"]), float(z["y0"]), float(z["res"])
        surf = -top
        seen = np.nan_to_num(surf, nan=-1e9) >= args.alt - 1.0
        lab, _ = label(~np.isfinite(top))
        touch = np.unique(lab[binary_dilation(seen, disk(1)) & (lab > 0)])
        obst = seen | np.isin(lab, touch[touch > 0])
        seen_lab = label(seen)[0]
        ii, jj = np.mgrid[0:top.shape[0], 0:top.shape[1]]
        wx, wy = x0 + (ii + 0.5) * res, y0 + (jj + 0.5) * res
        free = ~binary_dilation(obst, disk(int(round(1.5 / res))))
        endok = ~binary_dilation(obst, disk(int(round(END_CLEAR_M / res))))
        cell = lambda p: (int((p[0] - x0) / res), int((p[1] - y0) / res))
        pts, drop = [], {"holdout_near": 0, "geography": 0}
        for x in np.arange(math.ceil(x0 / GRID_M) * GRID_M, x0 + top.shape[0] * res, GRID_M):
            for y in np.arange(math.ceil(y0 / GRID_M) * GRID_M, y0 + top.shape[1] * res, GRID_M):
                c = cell((x, y))
                if not (0 <= c[0] < top.shape[0] and 0 <= c[1] < top.shape[1]) or not endok[c]:
                    continue
                if min(x - x0, x0 + top.shape[0] * res - x, y - y0, y0 + top.shape[1] * res - y) < args.edge_m:
                    continue
                if len(held) and np.min(np.hypot(*(held - [x, y]).T)) < args.holdout_sep_m:
                    drop["holdout_near"] += 1
                    continue
                if not geo(x, y, 100.0, True):
                    drop["geography"] += 1
                    continue
                pts.append((float(x), float(y)))
        kept = []
        if len(pts) >= 2:
            P = np.asarray(pts)
            idx, g = graph(free)
            inv = np.argwhere(idx >= 0)
            cands = []
            for b in range(len(P)):
                gm = (np.hypot(wx - P[b][0], wy - P[b][1]) <= SUCCESS_M) & free
                gc = idx[gm]
                if not len(gc):
                    continue
                D, PR = dijkstra(g, indices=gc, min_only=True, return_predecessors=True)[:2]
                for a in range(len(P)):
                    chord = float(np.hypot(*(P[b] - P[a])))
                    if a == b or not (c_lo <= chord <= c_hi):
                        continue
                    k = idx[cell(P[a])]
                    if k < 0 or not math.isfinite(D[k]):
                        continue
                    L = float(D[k]) * res
                    ratio = L / max(chord - SUCCESS_M, 1.0)
                    if ratio < args.min_ratio:
                        continue
                    line = np.linspace(P[a], P[b], 200)
                    lc = np.array([cell(q) for q in line])
                    ok = (lc[:, 0] >= 0) & (lc[:, 0] < top.shape[0]) & (lc[:, 1] >= 0) & (lc[:, 1] < top.shape[1])
                    if not (seen_lab[lc[ok, 0], lc[ok, 1]] > 0).any():
                        continue
                    cands.append((ratio, a, b, chord, L, int(k), PR))
            cands.sort(key=lambda t: -t[0])
            for ratio, a, b, chord, L, k, PR in cands:
                if len(kept) >= args.per_map:
                    break
                if any(np.hypot(*(P[a] - s)) < args.sep_m for s in spawns):
                    continue
                if any(np.hypot(*(P[a] - q[0])) < args.sep_m and np.hypot(*(P[b] - q[1])) < args.sep_m for q in excl):
                    continue
                path, q = [], k
                while q >= 0:
                    path.append(inv[q])
                    q = int(PR[q])
                pw = np.stack([x0 + (np.asarray(path)[:, 0] + 0.5) * res, y0 + (np.asarray(path)[:, 1] + 0.5) * res], 1)
                if not all(geo(x, y, 80.0, False) for x, y in pw[::10]):
                    continue
                kept.append((a, b))
                spawns.append(P[a].copy())
                rid = f"{args.rid_prefix}_{src_rid}_{len(kept) - 1}"
                yaw = math.atan2(P[b][1] - P[a][1], P[b][0] - P[a][0])
                episodes.append({"route_id": rid, "pos": [[*P[a].tolist(), args.alt], [*P[b].tolist(), args.alt]],
                                 "yaw": [yaw, yaw], "gpt_instruction": "",
                                 "complexity_meta": {"split": "train", "source_map": tp.name, "chord_m": round(chord, 1),
                                                     "geodesic_1p5_m": round(L, 1), "detour_ratio": round(ratio, 2),
                                                     "in_urban": True, "in_interior": True, "in_water": False}})
                shutil.copy(tp, top_out / f"top_{rid}.npz")
        review.append({"map": tp.name, "endpoints": len(pts), "kept": len(kept), **drop})
        print(tp.name, "endpoints", len(pts), "kept", len(kept), drop, flush=True)
    Path(args.out_ann).write_text(json.dumps({"episodes": episodes, "review": review, "args": vars(args)}, indent=1))
    print("total routes", len(episodes), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
