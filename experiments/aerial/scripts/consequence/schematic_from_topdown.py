"""Street polylines at flight altitude from an existing top-down sim scan.

Obstacles are roofs at or above the flight altitude (the rendered scene) plus
collision shells measured at that altitude. Those shells do not appear in the
depth scan. The polyline is the shortest path that stays clear of both.
It is a schematic for scene_memory.load_schematic, not a hard occupancy veto.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def graph(free: np.ndarray):
    nx, ny = free.shape
    idx = -np.ones(free.shape, np.int32)
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


def shells_at(path: Path, z_ned: float) -> dict:
    raw = json.loads(path.read_text())
    out = {}
    for name, obj in raw.items():
        boxes = []
        for lv in obj.get("levels") or []:
            if abs(float(lv.get("z", 99)) - z_ned) > 0.1:
                continue
            for b in lv.get("boxes") or []:
                xs = [b["x_lo"][0], b["x_hi"][0]]
                ys = [b["y_lo"][1], b["y_hi"][1]]
                boxes.append((min(xs), max(xs), min(ys), max(ys)))
        if boxes:
            out[name] = boxes
    return out


def paint(shape, x0, y0, res, boxes) -> np.ndarray:
    ii, jj = np.mgrid[0:shape[0], 0:shape[1]]
    wx = x0 + (ii + 0.5) * res
    wy = y0 + (jj + 0.5) * res
    hit = np.zeros(shape, bool)
    for xlo, xhi, ylo, yhi in boxes:
        hit |= (wx >= xlo) & (wx <= xhi) & (wy >= ylo) & (wy <= yhi)
    return hit


def cell(xy, x0, y0, res, shape):
    i = int((xy[0] - x0) / res)
    j = int((xy[1] - y0) / res)
    if not (0 <= i < shape[0] and 0 <= j < shape[1]):
        return None
    return i, j


def nearest_free(free, ij, rad):
    if ij is None:
        return None
    i, j = ij
    if free[i, j]:
        return ij
    best = None
    for di in range(-rad, rad + 1):
        for dj in range(-rad, rad + 1):
            a, b = i + di, j + dj
            if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and free[a, b]:
                d = di * di + dj * dj
                if best is None or d < best[0]:
                    best = (d, a, b)
    return None if best is None else (best[1], best[2])


def segment_clear(free, x0, y0, res, a, b) -> bool:
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / res) + 1)
    for t in np.linspace(0.0, 1.0, n):
        p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        ij = cell(p, x0, y0, res, free.shape)
        if ij is None or not free[ij]:
            return False
    return True


def thin(world, spacing):
    if len(world) < 2:
        return world
    kept = [world[0]]
    acc = 0.0
    for p, q in zip(world[:-1], world[1:]):
        acc += float(np.hypot(q[0] - p[0], q[1] - p[1]))
        if acc >= spacing:
            kept.append(q)
            acc = 0.0
    if np.hypot(kept[-1][0] - world[-1][0], kept[-1][1] - world[-1][1]) > 1.0:
        kept.append(world[-1])
    return kept


def solve(free, idx_g, x0, y0, res, start, goal):
    rad = int(round(8.0 / res))
    s = nearest_free(free, cell(start, x0, y0, res, free.shape), rad)
    goal_cells = []
    gi, gj = cell(goal, x0, y0, res, free.shape) or (-1, -1)
    r = int(round(3.0 / res))
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj > r * r:
                continue
            a, b = gi + di, gj + dj
            if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and free[a, b] and idx_g[0][a, b] >= 0:
                goal_cells.append(idx_g[0][a, b])
    if s is None or not goal_cells:
        return None
    src = int(idx_g[0][s])
    dist, pred = dijkstra(idx_g[1], indices=src, return_predecessors=True)
    goal_cells = np.asarray(goal_cells)
    finite = goal_cells[np.isfinite(dist[goal_cells])]
    if not len(finite):
        return None
    k = int(finite[np.argmin(dist[finite])])
    flat = []
    seen = set()
    while k >= 0 and k not in seen:
        seen.add(k)
        flat.append(k)
        k = int(pred[k])
    flat.reverse()
    inv = np.argwhere(idx_g[0] >= 0)
    world = [(x0 + (inv[k][0] + 0.5) * res, y0 + (inv[k][1] + 0.5) * res) for k in flat]
    if segment_clear(free, x0, y0, res, start, world[0]):
        world = [tuple(start)] + world
    if segment_clear(free, x0, y0, res, world[-1], goal):
        world = world + [tuple(goal)]
    return world, float(dist[finite].min()) * res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--east", required=True)
    ap.add_argument("--west", required=True)
    ap.add_argument("--shells", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alt", type=float, default=24.0)
    ap.add_argument("--clear", type=float, default=1.0)
    ap.add_argument("--spacing", type=float, default=6.0)
    ap.add_argument("--grid-out", default="", help="write a 2 m free-space raster, with no polylines")
    args = ap.parse_args()
    shells = shells_at(Path(args.shells), -args.alt)
    print("shells at z_up", args.alt, {k: v for k, v in shells.items()}, flush=True)
    episodes = json.loads(Path(args.routes).read_text())["episodes"]
    maps = {}
    for key, path in (("east", args.east), ("west", args.west)):
        z = np.load(path)
        top = z["top_ned_z"]
        x0, y0, res = float(z["x0"]), float(z["y0"]), float(z["res"])
        height = -np.asarray(top, np.float64)
        seen = np.isfinite(height) & (height >= args.alt - args.clear)
        lab, _ = label(~np.isfinite(top))
        touch = np.unique(lab[binary_dilation(seen, disk(1)) & (lab > 0)])
        roofs = seen | np.isin(lab, touch[touch > 0])
        shell = paint(top.shape, x0, y0, res, [b for boxes in shells.values() for b in boxes])
        obst = roofs | shell
        built = {}
        for inflate in (3.0, 1.5):
            free = ~binary_dilation(obst, disk(int(round(inflate / res))))
            built[inflate] = (free, graph(free))
        maps[key] = {"x0": x0, "y0": y0, "res": res, "obst": obst, "roofs": roofs, "shell": shell, "built": built,
                     "x1": x0 + top.shape[0] * res, "y1": y0 + top.shape[1] * res}
        print(key, "roofs", int(roofs.sum()), "shell cells", int(shell.sum()), flush=True)

    if args.grid_out:
        payload = {}
        for key, m in maps.items():
            free = m["built"][3.0][0]
            factor = int(round(2.0 / m["res"]))
            nx, ny = free.shape[0] // factor, free.shape[1] // factor
            coarse = free[:nx * factor, :ny * factor].reshape(nx, factor, ny, factor).all(axis=(1, 3))
            payload[f"{key}_free"] = coarse
            payload[f"{key}_x0"] = m["x0"]
            payload[f"{key}_y0"] = m["y0"]
            payload[f"{key}_res"] = 2.0
            print(key, "grid", coarse.shape, "free", float(coarse.mean()), flush=True)
        np.savez_compressed(args.grid_out, **payload)
        print("wrote grid", args.grid_out, flush=True)

    routes, meta = {}, {}
    for ep in episodes:
        rid = ep["route_id"]
        start = (float(ep["pos"][0][0]), float(ep["pos"][0][1]))
        goal = (float(ep["pos"][-1][0]), float(ep["pos"][-1][1]))
        z_up = float(ep["pos"][0][2])
        key = "east" if start[0] > -400 else "west"
        m = maps[key]
        chosen = None
        for inflate in (3.0, 1.5):
            got = solve(m["built"][inflate][0], m["built"][inflate][1], m["x0"], m["y0"], m["res"], start, goal)
            if got is not None:
                chosen = (inflate, got)
                break
        if chosen is None:
            meta[rid] = {"ok": False, "map": key}
            print(rid, "NO PATH", flush=True)
            continue
        inflate, (world, length) = chosen
        sparse = thin(world, args.spacing)
        poly = [[round(p[0], 1), round(p[1], 1), round(z_up, 1)] for p in sparse]
        chord = float(math.hypot(goal[0] - start[0], goal[1] - start[1]))
        # Side of the chord: positive y is north of the straight line.
        ab = np.array([goal[0] - start[0], goal[1] - start[1]])
        lab = float(np.dot(ab, ab)) or 1.0
        offs = []
        for p in world[::2]:
            t = float(np.clip(np.dot(np.array(p) - start, ab) / lab, 0, 1))
            chord_y = start[1] + t * ab[1]
            offs.append(p[1] - chord_y)
        in_shell = []
        for name, boxes in shells.items():
            hit = False
            for p in world[::4]:
                for xlo, xhi, ylo, yhi in boxes:
                    if xlo <= p[0] <= xhi and ylo <= p[1] <= yhi:
                        hit = True
            if hit:
                in_shell.append(name)
        routes[rid] = poly
        meta[rid] = {
            "ok": True, "map": key, "inflate_m": inflate, "n_points": len(poly),
            "length_m": round(length, 1), "chord_m": round(chord, 1),
            "north_m": [round(float(min(offs)), 1), round(float(max(offs)), 1)],
            "crosses_shell": in_shell,
        }
        print(rid, "inflate", inflate, "len", round(length, 1), "chord", round(chord, 1),
              "north", meta[rid]["north_m"], "shell", in_shell or "-", "pts", len(poly), flush=True)

    out = {
        "note": "Street polylines from the 2026-10-14 top-down scan at 24 m, plus collision shells measured at that altitude. Roofs at or above 23 m block. Preferred clearance 3 m, else 1.5 m. Points are about 6 m apart.",
        "source": {
            "east": args.east, "west": args.west, "shells": args.shells,
            "alt_m": args.alt, "clear_m": args.clear, "inflate_prefer_m": 3.0, "inflate_fallback_m": 1.5,
            "shell_names": sorted(shells),
        },
        "meta": meta,
        "routes": routes,
    }
    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out, "routes", len(routes), "missing", [k for k, v in meta.items() if not v["ok"]], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
