"""Rank candidate hard held-out routes on top-down height maps (offline, no sim).

Obstacles at flight altitude = surface >= alt - 1 m, plus unrendered (NaN) blobs that touch such surfaces. For start
and goal points on a 5 m grid inside the allowed interior regions, the shortest path to the 3 m goal circle is solved
on an 8-connected 0.5 m grid with 1.5 m and 3 m obstacle clearance. A pair is a candidate when:
  - both ends: urban interior, not water, region inland >= 100 m, measured shore >= 100 m, >= 4 m from obstacles,
    >= 30 m from every train goal and train spawn;
  - chord 40-130 m, the 3 m-clearance path exists, and every point of it has region inland >= 80 m, shore >= 80 m;
  - hard: detour ratio (1.5 m path / (chord - 3 m)) >= --min-ratio, or trap: the 3 m-clearance path has to move
    >= --trap-m back away from the goal at some point (U-shapes, dead-end pockets).
Invisible collision meshes are not in these maps unless probed (--extra-obst, invisible_probe.py);
hard_route_verify.py flies every kept route in the sim.
Positions: annotation (x, y, z up) = AirSim NED (x, y, -z).
usage: hard_route_select.py OUT_JSON --tops a.npz,b.npz --regions R1.json,R2.json --water-raster W.npz
                            --train-ann a.json,b.json [--alt 24]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

SUCCESS_M, GRID_M, END_CLEAR_M = 3.0, 5.0, 4.0


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
    ap.add_argument("out_json")
    ap.add_argument("--tops", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--water-raster", required=True)
    ap.add_argument("--train-ann", required=True)
    ap.add_argument("--alt", type=float, default=24.0)
    ap.add_argument("--min-ratio", type=float, default=1.3)
    ap.add_argument("--train-sep-m", type=float, default=30.0)
    ap.add_argument("--trap-m", type=float, default=5.0)
    ap.add_argument("--extra-obst", default="", help="invisible_probe.py npz files; hit cells become obstacles")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiments.aerial.phase3_unified.region_geometry import classify_spawn_xy, load_regions
    from water_audit import Shore

    regions = [load_regions(p) for p in args.regions.split(",")]
    shore = Shore(args.water_raster)
    train = []
    for a in args.train_ann.split(","):
        for e in json.loads(Path(a).read_text())["episodes"]:
            if e.get("complexity_meta", {}).get("split", "train") == "train":
                train += [e["pos"][0][:2], e["pos"][-1][:2]]
    train = np.asarray(train, float)

    cache: dict = {}

    def geo(x: float, y: float, inland: float, need_interior: bool) -> bool:
        key = (round(x), round(y), inland, need_interior)
        if key not in cache:
            cache[key] = geo_raw(x, y, inland, need_interior)
        return cache[key]

    def geo_raw(x: float, y: float, inland: float, need_interior: bool) -> bool:
        ok = False
        for R in regions:
            s = classify_spawn_xy(float(x), float(y), R)
            if s["in_urban"] and not s["in_water"] and s["spawn_inland_m"] >= inland and (s["in_interior"] or not need_interior):
                ok = True
                break
        return ok and shore(float(x), float(y)) >= inland

    out = []
    for tp in args.tops.split(","):
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
        for ep in filter(None, args.extra_obst.split(",")):
            pz = np.load(ep)
            pi = np.rint((wx - float(pz["x0"])) / float(pz["res"])).astype(int)
            pj = np.rint((wy - float(pz["y0"])) / float(pz["res"])).astype(int)
            inb = (pi >= 0) & (pi < pz["hit"].shape[0]) & (pj >= 0) & (pj < pz["hit"].shape[1])
            extra = np.zeros_like(obst)
            extra[inb] = pz["hit"][pi[inb], pj[inb]]
            obst |= extra
            seen = seen | extra
            seen_lab = label(seen)[0]
        free = {c: ~binary_dilation(obst, disk(int(round(c / res)))) for c in (1.5, 3.0)}
        endok = ~binary_dilation(obst, disk(int(round(END_CLEAR_M / res))))
        cell = lambda p: (int((p[0] - x0) / res), int((p[1] - y0) / res))
        pts = []
        for x in np.arange(math.ceil(x0 / GRID_M) * GRID_M, x0 + top.shape[0] * res, GRID_M):
            for y in np.arange(math.ceil(y0 / GRID_M) * GRID_M, y0 + top.shape[1] * res, GRID_M):
                c = cell((x, y))
                if not (0 <= c[0] < top.shape[0] and 0 <= c[1] < top.shape[1]) or not endok[c]:
                    continue
                if len(train) and np.min(np.hypot(*(train - [x, y]).T)) < args.train_sep_m:
                    continue
                if geo(x, y, 100.0, True):
                    pts.append((float(x), float(y)))
        print(Path(tp).name, "endpoints", len(pts), flush=True)
        if len(pts) < 2:
            continue
        P = np.asarray(pts)
        gr = {c: graph(free[c]) for c in (1.5, 3.0)}
        inv = np.argwhere(gr[3.0][0] >= 0)
        goal_cells = []
        for g in pts:
            m = (np.hypot(wx - g[0], wy - g[1]) <= SUCCESS_M)
            goal_cells.append({c: gr[c][0][m & free[c]] for c in gr})
        for a in range(len(pts)):
            D15 = dijkstra(gr[1.5][1], indices=gr[1.5][0][cell(pts[a])])
            D3, PR3 = dijkstra(gr[3.0][1], indices=gr[3.0][0][cell(pts[a])], return_predecessors=True)
            for b in range(len(pts)):
                chord = float(np.hypot(*(P[b] - P[a])))
                if a == b or not (40.0 <= chord <= 130.0):
                    continue
                gc15 = goal_cells[b][1.5][goal_cells[b][1.5] >= 0]
                gc3 = goal_cells[b][3.0][goal_cells[b][3.0] >= 0]
                if not len(gc15) or not len(gc3):
                    continue
                L15 = float(np.min(D15[gc15])) * res
                k3 = gc3[np.argmin(D3[gc3])]
                L3 = float(D3[k3]) * res
                if not (math.isfinite(L15) and math.isfinite(L3)):
                    continue
                ratio = L15 / max(chord - SUCCESS_M, 1.0)
                if ratio < min(args.min_ratio, 1.1):
                    continue
                path, k = [], int(k3)
                while k >= 0:
                    path.append(inv[k])
                    k = int(PR3[k])
                path = np.asarray(path[::-1])
                pw = np.stack([x0 + (path[:, 0] + 0.5) * res, y0 + (path[:, 1] + 0.5) * res], 1)
                dg = np.hypot(*(pw - P[b]).T)
                backtrack = float(np.max(dg - np.minimum.accumulate(dg)))
                trapped = backtrack >= args.trap_m
                if ratio < args.min_ratio and not trapped:
                    continue
                if not all(geo(x, y, 80.0, False) for x, y in pw[::10]):
                    continue
                line = np.linspace(P[a], P[b], 200)
                lc = np.array([cell(q) for q in line])
                ok = (lc[:, 0] >= 0) & (lc[:, 0] < top.shape[0]) & (lc[:, 1] >= 0) & (lc[:, 1] < top.shape[1])
                blocks = len(set(seen_lab[lc[ok, 0], lc[ok, 1]].tolist()) - {0})
                out.append({"map": Path(tp).name, "spawn": P[a].tolist(), "goal": P[b].tolist(), "chord": round(chord, 1),
                            "astar_1p5": round(L15, 1), "astar_3": round(L3, 1), "ratio": round(ratio, 2),
                            "trap": trapped, "backtrack_m": round(backtrack, 1), "line_blocks": blocks, "path3": pw[::4].round(1).tolist()})
        print(Path(tp).name, "candidates", sum(1 for o in out if o["map"] == Path(tp).name), flush=True)
    out.sort(key=lambda o: (-int(o["trap"]), -o["ratio"]))
    Path(args.out_json).write_text(json.dumps({"args": vars(args), "candidates": out}))
    print("total", len(out), "trap", sum(o["trap"] for o in out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
