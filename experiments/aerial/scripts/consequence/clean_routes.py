"""Clean held-out routes: reject candidates that cross a measured invisible collision volume or
duplicate an existing route (prefilter), and validated routes whose eval-style reset touches
geometry (spawncheck). Kept routes are appended to a pool annotation with ids od_clean_NN.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def load_boxes(paths: str, margin_m: float):
    ext = {}
    for path in [ROOT / p for p in paths.split(",") if p]:
        for name, rec in json.loads(path.read_text()).items():
            for lv in rec["levels"]:
                for f in lv.get("boxes", []):
                    e = ext.setdefault(name, [[], [], [], []])
                    e[0].append(f["x_lo"][0]); e[1].append(f["x_hi"][0])
                    e[2].append(f["y_lo"][1]); e[3].append(f["y_hi"][1])
    return {n: (min(e[0]) - margin_m, max(e[1]) + margin_m, min(e[2]) - margin_m, max(e[3]) + margin_m)
            for n, e in ext.items()}


def crosses(pos, box) -> bool:
    x0, x1, y0, y1 = box
    p = np.asarray(pos, float)[:, :2]
    for a, b in zip(p[:-1], p[1:]):
        for t in np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(b - a)) + 1)):
            q = a * (1 - t) + b * t
            if x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                return True
    return False


def episodes(path: Path):
    d = json.loads(path.read_text())
    return d["episodes"] if isinstance(d, dict) and "episodes" in d else d.get("holdout", [])


def prefilter(args) -> int:
    cand = json.loads((ROOT / args.cand).read_text())
    boxes = load_boxes(args.boxes, args.margin_m)
    existing = []
    for p in [p for p in args.existing.split(",") if p]:
        f = ROOT / p
        if f.is_file():
            existing += [np.asarray(e["pos"], float) for e in episodes(f)]
    shore = None
    if args.water_raster:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from water_audit import PATH_MIN_M, SPAWN_MIN_M, Shore
        shore = Shore(str(ROOT / args.water_raster))
    keep, rows = [], []
    for ep in cand["holdout"]:
        pos = np.asarray(ep["pos"], float)
        reasons = [f"crosses_{n}" for n, bx in boxes.items() if crosses(pos, bx)]
        if shore is not None:
            d = [shore(p[0], p[1]) for p in pos]
            if d[0] < SPAWN_MIN_M or min(d) < PATH_MIN_M:
                reasons.append(f"measured_shore_spawn{d[0]:.0f}_path{min(d):.0f}m")
        for q in existing:
            if (np.linalg.norm(q[0, :2] - pos[0, :2]) < args.dup_m and np.linalg.norm(q[-1, :2] - pos[-1, :2]) < args.dup_m) or \
               (np.linalg.norm(q[0, :2] - pos[-1, :2]) < args.dup_m and np.linalg.norm(q[-1, :2] - pos[0, :2]) < args.dup_m):
                reasons.append("duplicate_of_existing")
                break
        rows.append({"route_id": ep["route_id"], "region": ep["complexity_meta"]["region"], "keep": not reasons, "drop_reasons": reasons})
        if not reasons:
            keep.append(ep)
    if args.seen:
        seen_path = ROOT / args.seen
        seen = json.loads(seen_path.read_text()) if seen_path.is_file() else {"episodes": []}
        seen["episodes"] += keep
        seen_path.write_text(json.dumps(seen))
    cand["holdout"] = keep
    cand["train"] = []
    (ROOT / args.out).write_text(json.dumps(cand, indent=1))
    (ROOT / args.review).write_text(json.dumps({"kept": len(keep), "rows": rows}, indent=1))
    print(f"PREFILTER kept {len(keep)}/{len(rows)}", flush=True)
    return 0


def spawncheck(args) -> int:
    import yaml

    sys.path.insert(0, str(ROOT))
    from experiments.aerial.rl.train_rl import _build_env

    cfg = yaml.safe_load((ROOT / args.config).read_text()) or {}
    env_cfg = dict(cfg.get("env") or {})
    env_cfg.update({"backend": "airsim", "step_hz": 5.0, "grab_depth": True})
    env = _build_env(env_cfg)
    pool_path = ROOT / args.pool
    pool = json.loads(pool_path.read_text()) if pool_path.is_file() else {
        "protocol_version": "interior_od_clean_v1", "scene": "outdoor_complex", "template": "interior_od_v1", "episodes": []}
    review_path = ROOT / args.review
    review = json.loads(review_path.read_text()) if review_path.is_file() else {"rows": []}
    pool_pos = [np.asarray(e["pos"], float) for e in pool["episodes"]]
    for ep in episodes(ROOT / args.ann):
        pos = np.asarray(ep["pos"], float)
        if any(np.allclose(q[[0, -1], :2], pos[[0, -1], :2], atol=0.5) for q in pool_pos):
            continue
        start, goal = pos[0].copy(), pos[-1].copy()
        z_ok = ep.get("complexity_meta", {}).get("spawn_z_ok")
        if z_ok is not None:
            start[2] = goal[2] = float(z_ok)
        touched = []
        for rep in range(3):
            c = env._connect()
            t0 = int(c.simGetCollisionInfo(**env._vk).time_stamp)
            env.reset({"pos": [start.tolist(), goal.tolist()], "yaw": [float(ep["yaw"][0])] * 2})
            a = c.simGetCollisionInfo(**env._vk)
            env.step(np.zeros(4))
            h = c.simGetCollisionInfo(**env._vk)
            checks = [("reset", a), ("hover", h)]
            for k in range(args.first_steps):
                env.step(np.array([0.8, 0.0, 0.0, 0.0]))
                checks.append((f"fwd{k + 1}", c.simGetCollisionInfo(**env._vk)))
            for tag, ci in checks:
                if ci.has_collided and int(ci.time_stamp) != t0:
                    touched.append({"rep": rep, "at": tag, "object": str(ci.object_name), "penetration_m": round(float(ci.penetration_depth), 2)})
                    break
        row = {"source_route_id": ep["route_id"], "region": ep["complexity_meta"]["region"],
               "blocked": ep["complexity_meta"].get("blocked"), "spawn_contacts": touched, "keep": not touched,
               "drop_reasons": ["spawn_contact"] if touched else []}
        if not touched:
            e = dict(ep)
            e["complexity_meta"] = {**ep["complexity_meta"], "source_route_id": ep["route_id"], "source_ann": args.ann}
            e["route_id"] = f"{args.id_prefix}_{len(pool['episodes']):02d}"
            row["route_id"] = e["route_id"]
            pool["episodes"].append(e)
            pool_pos.append(pos)
        review["rows"].append(row)
        print(json.dumps(row), flush=True)
    pool["n_episodes"] = len(pool["episodes"])
    for i, e in enumerate(pool["episodes"]):
        e["route_idx"] = i
    pool_path.write_text(json.dumps(pool, indent=1))
    n_block = sum(1 for e in pool["episodes"] if e["complexity_meta"].get("blocked"))
    review["summary"] = {"pool_routes": len(pool["episodes"]), "pool_blocked": n_block, "gate_blocked": args.gate_blocked,
                         "gate_pass": n_block >= args.gate_blocked}
    review_path.write_text(json.dumps(review, indent=1))
    print("POOL", json.dumps(review["summary"]), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prefilter")
    p.add_argument("--cand", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--boxes", default="experiments/aerial/rl/artifacts/invisible_boxes_flight_20261012.json",
                   help="comma list of measured box files; extents of the same object are merged")
    p.add_argument("--margin-m", type=float, default=3.0)
    p.add_argument("--existing", default="")
    p.add_argument("--dup-m", type=float, default=15.0)
    p.add_argument("--water-raster", default="experiments/aerial/rl/artifacts/water_map_measured_20261012.npz",
                   help="measured open-water raster; empty disables the shoreline check")
    p.add_argument("--seen", default="", help="candidates already sent to validation; extended with this round's keeps")
    s = sub.add_parser("spawncheck")
    s.add_argument("--ann", required=True)
    s.add_argument("--pool", required=True)
    s.add_argument("--review", required=True)
    s.add_argument("--config", default="configs/_tmp_feasible_side_h125.yaml")
    s.add_argument("--gate-blocked", type=int, default=20)
    s.add_argument("--first-steps", type=int, default=0,
                   help="after the hover step, also take this many 0.8 m forward steps along the start yaw")
    s.add_argument("--id-prefix", default="od_clean")
    args = ap.parse_args()
    return prefilter(args) if args.cmd == "prefilter" else spawncheck(args)


if __name__ == "__main__":
    raise SystemExit(main())
