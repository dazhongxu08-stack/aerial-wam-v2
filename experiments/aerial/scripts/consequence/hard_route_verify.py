"""Fly-check ranked hard-route candidates (hard_route_select.py) in the sim and write the held-out annotation.

Candidates are taken round-robin over categories (trap, then detour per map, each by hardness) with every endpoint
>= --sep-m from kept endpoints and no near-duplicate (both ends within 15 m) of an existing eval route. Keep only if:
no fresh contact at the spawn and at the goal over 8 yaws, nothing in the full frame closer than --frame-min-m;
following the 3 m-clearance shortest path at 2 m/s reaches the 3 m goal circle without contact; and flying the
straight line does NOT arrive (otherwise the map missed an opening and the route is not hard). Every decision goes to
AUTO_REVIEW.json; collision object names are recorded. Every contact position found is remembered, and later
candidates with an endpoint within 5 m or a path within 2 m of one are skipped. Exits 3 if fewer than --target
routes are kept.
Positions: annotation (x, y, z up) = AirSim NED (x, y, -z).
usage: hard_route_verify.py OUT_DIR --cands cands.json --template clean_pool.json --existing a.json,b.json [--target 12]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import airsim
import numpy as np

CAM = "front_custom"
SUCCESS_M, SPEED, DT = 3.0, 2.0, 0.2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--cands", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--existing", default="")
    ap.add_argument("--kept", default="", help="hard_eval.json files already kept; their endpoints count for --sep-m")
    ap.add_argument("--target", type=int, default=12)
    ap.add_argument("--sep-m", type=float, default=12.0)
    ap.add_argument("--frame-min-m", type=float, default=1.5)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cj = json.loads(Path(args.cands).read_text())
    alt = float(cj["args"]["alt"])
    cands = cj["candidates"]
    tmpl = json.loads(Path(args.template).read_text())
    olds = [(np.asarray(e["pos"][0][:2], float), np.asarray(e["pos"][-1][:2], float))
            for a in args.existing.split(",") if a for e in json.loads(Path(a).read_text())["episodes"]]
    cats = {"trap": sorted([c for c in cands if c["trap"]], key=lambda c: -c["backtrack_m"])}
    for m in sorted({c["map"] for c in cands}):
        cats[m] = sorted([c for c in cands if not c["trap"] and c["map"] == m], key=lambda c: -c["ratio"])

    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.enableApiControl(True)

    def ts() -> int:
        return int(c.simGetCollisionInfo().time_stamp)

    def fresh(prev: int) -> str:
        ci = c.simGetCollisionInfo()
        return str(ci.object_name) if ci.has_collided and int(ci.time_stamp) != prev else ""

    def place(p, yaw: float) -> str:
        prev = ts()
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*map(float, p)), airsim.to_quaternion(0, 0, yaw)), True)
        c.moveByVelocityAsync(0, 0, 0, 0.6).join()
        return fresh(prev)

    def frame_min() -> float:
        r = c.simGetImages([airsim.ImageRequest(CAM, airsim.ImageType.DepthPlanar, True, False)])[0]
        return float(np.array(r.image_data_float, np.float32).min())

    def hover_check(p) -> dict:
        contacts, frames = [], []
        for j in range(8):
            place(p + [0, 0, -8], j * math.pi / 4)
            hit = place(p, j * math.pi / 4)
            if hit:
                contacts.append(hit)
            frames.append(round(frame_min(), 2))
        return {"contacts": contacts, "frame_min_8yaw": frames}

    def follow(start: np.ndarray, pts: list, goal: np.ndarray, record: bool = True) -> str:
        yaw0 = math.atan2(pts[0][1] - start[1], pts[0][0] - start[0])
        place(start + [0, 0, -8], yaw0)
        if place(start, yaw0):
            return "start_embedded"
        prev, k = ts(), 0
        length = float(np.sum(np.linalg.norm(np.diff(np.vstack([start, *pts]), axis=0), axis=1)))
        for _ in range(int(1.5 * length / (SPEED * DT)) + 50):
            q = c.getMultirotorState().kinematics_estimated.position
            p = np.array([q.x_val, q.y_val, q.z_val])
            if float(np.linalg.norm(goal - p)) <= SUCCESS_M:
                return "arrived"
            while k < len(pts) - 1 and float(np.linalg.norm(pts[k][:2] - p[:2])) < 1.5:
                k += 1
            v = pts[k] - p
            v = v / max(float(np.linalg.norm(v)), 1e-6) * SPEED
            c.moveByVelocityAsync(float(v[0]), float(v[1]), float(v[2]), DT,
                                  yaw_mode=airsim.YawMode(False, math.degrees(math.atan2(v[1], v[0])))).join()
            obj = fresh(prev)
            if obj:
                if record:
                    q = c.getMultirotorState().kinematics_estimated.position
                    bad.append(np.array([q.x_val, q.y_val]))
                return f"collision {obj}"
        return "max_steps"

    review_path = out / "AUTO_REVIEW.json"
    review = {"args": vars(args), "alt_m": alt, "checked": [], "kept": []}
    ends: list = [np.asarray(e["pos"][i][:2], float) for a in args.kept.split(",") if a
                  for e in json.loads(Path(a).read_text())["episodes"] for i in (0, -1)]
    bad: list = []

    def free_of(cand) -> bool:
        s, g = np.asarray(cand["spawn"]), np.asarray(cand["goal"])
        if any(np.linalg.norm(s - a) < args.sep_m or np.linalg.norm(g - a) < args.sep_m for a in ends):
            return False
        if bad:
            B = np.asarray(bad)
            if np.min(np.hypot(*(B - s).T)) < 5.0 or np.min(np.hypot(*(B - g).T)) < 5.0:
                return False
            pp = np.asarray(cand["path3"])
            if np.min(np.hypot(pp[:, None, 0] - B[None, :, 0], pp[:, None, 1] - B[None, :, 1])) < 2.0:
                return False
        return not any(np.linalg.norm(s - a) < 15 and np.linalg.norm(g - b) < 15 for a, b in olds)

    def save() -> None:
        review_path.write_text(json.dumps(review, indent=1))
        base = tmpl["episodes"][0]
        eps = []
        for n, k in enumerate(review["kept"]):
            s = np.array([*k["spawn"], alt])
            g = np.array([*k["goal"], alt])
            npt = max(2, int(math.ceil(k["chord"] / 10.0)) + 1)
            pos = [list(map(float, s + (g - s) * t)) for t in np.linspace(0, 1, npt)]
            yaw = math.atan2(g[1] - s[1], g[0] - s[0])
            meta = {"split": "holdout", "region": k["region"], "chord_m": k["chord"], "spawn_z_ok": alt,
                    "astar_1p5_m": k["astar_1p5"], "astar_3_m": k["astar_3"], "detour_ratio": k["ratio"],
                    "trap": k["trap"], "backtrack_m": k["backtrack_m"], "blocked": True, "in_urban": True,
                    "in_interior": True, "in_water": False, "source": "hard_route_select.py + hard_route_verify.py"}
            eps.append(dict(base, pos=pos, yaw=[yaw] * npt, route_id=f"od_hard_{n:02d}", route_idx=n,
                            variant_id=f"od_hard_{n:02d}", source_kind="hard_od", template="hard_od_v1",
                            complexity_meta=meta))
        ann = {kk: v for kk, v in tmpl.items() if kk != "episodes"}
        ann.update(template="hard_od_v1", episodes=eps, n_episodes=len(eps),
                   note=f"held-out hard routes at {alt} m (detour ratio >= 1.3 or trap), fly-verified")
        (out / "hard_eval.json").write_text(json.dumps(ann, indent=1))

    its = {k: iter(v) for k, v in cats.items()}
    while len(review["kept"]) < args.target:
        progressed = False
        for cat, it in its.items():
            if len(review["kept"]) >= args.target:
                break
            cand = next((x for x in it if free_of(x)), None)
            if cand is None:
                continue
            progressed = True
            s = np.array([*cand["spawn"], -alt])
            g = np.array([*cand["goal"], -alt])
            rec = {"cat": cat, **{k: cand[k] for k in ("map", "spawn", "goal", "chord", "astar_1p5", "astar_3", "ratio",
                                                       "trap", "backtrack_m", "line_blocks")}}
            review["checked"].append(rec)
            for name, p in (("spawn", s), ("goal", g)):
                rec[name + "_check"] = hover_check(p)
                if rec[name + "_check"]["contacts"]:
                    bad.append(p[:2].copy())
            drop = ""
            for name in ("spawn", "goal"):
                h = rec[name + "_check"]
                if h["contacts"]:
                    drop = drop or f"{name}_contact {h['contacts'][0]}"
                elif min(h["frame_min_8yaw"]) < args.frame_min_m:
                    drop = drop or f"{name}_frame_min {min(h['frame_min_8yaw'])}"
            if not drop:
                path = [np.array([x, y, -alt]) for x, y in cand["path3"][1:]] + [g]
                rec["astar_flight"] = follow(s, path, g)
                if rec["astar_flight"] != "arrived":
                    drop = f"astar_flight {rec['astar_flight']}"
            if not drop:
                rec["straight_flight"] = follow(s, [g], g, record=False)
                if rec["straight_flight"] == "arrived":
                    drop = "straight_line_arrives (map missed an opening)"
            if drop:
                rec["drop"] = drop
            else:
                rec["keep"] = True
                region = "interior_east" if "east" in cand["map"] else "interior_west_northeast"
                review["kept"].append({**{k: rec[k] for k in ("spawn", "goal", "chord", "astar_1p5", "astar_3", "ratio",
                                                              "trap", "backtrack_m")}, "region": region, "cat": cat})
                ends.extend([np.asarray(cand["spawn"]), np.asarray(cand["goal"])])
            save()
            print("checked", cat, cand["spawn"], cand["goal"], "ratio", cand["ratio"], rec.get("drop", "KEEP"),
                  "kept", len(review["kept"]), flush=True)
        if not progressed:
            break
    status = "GATE_PASS" if len(review["kept"]) >= args.target else "GATE_FAIL"
    review["gate"] = {"target": args.target, "kept": len(review["kept"]), "status": status}
    save()
    print(status, len(review["kept"]), flush=True)
    return 0 if status == "GATE_PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
