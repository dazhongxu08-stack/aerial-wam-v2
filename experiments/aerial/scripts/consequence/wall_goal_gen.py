"""Make train goals that sit 2-3 m in front of a visible wall, for near-goal collection.

From each path point of the train routes (round-robin over routes), read DepthPlanar at 16 yaws. A heading whose
centre window is a flat surface 4-40 m away is a wall candidate; the new goal is placed WALL_M in front of it.
Keep a goal only if: urban interior, >= 100 m from measured water; >= 30 m from held-out goals and >= 15 m from
other kept goals; no fresh contact at the goal over 8 yaws; the closest centre-window depth is 1.8-3.3 m and
nothing in the full frame is closer than 1.0 m; a straight 1 m/s flight from 8 m out on the open side arrives
(3 m) without contact. Writes the annotation (split=train) and AUTO_REVIEW.json with every keep/drop reason, and
keeps going over more path points until --target goals are kept or the points run out.
Positions: annotation (x, y, z up) = AirSim NED (x, y, -z).
usage: wall_goal_gen.py OUT_DIR --annotation A.json --exclude-goals a.json,b.json --water-raster W.npz [--target 24]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import airsim
import numpy as np

CAM = "front_custom"
WALL_M = 2.5
CLEAR_LO, CLEAR_HI, FRAME_MIN = 1.8, 3.3, 1.0
SUCCESS_M, SPEED, DT = 3.0, 1.0, 0.2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--exclude-goals", default="")
    ap.add_argument("--exclude-goal-m", type=float, default=30.0)
    ap.add_argument("--water-raster", required=True)
    ap.add_argument("--min-inland-m", type=float, default=100.0)
    ap.add_argument("--sep-m", type=float, default=15.0)
    ap.add_argument("--target", type=int, default=24)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiments.aerial.phase3_unified.region_geometry import classify_spawn_xy, load_regions
    from water_audit import Shore

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = json.loads(Path(args.annotation).read_text())
    routes = [r for r in src["episodes"] if r.get("complexity_meta", {}).get("split") == "train"]
    regions, shore = load_regions(), Shore(args.water_raster)
    held = [np.asarray(e["pos"][-1][:2], float) for a in args.exclude_goals.split(",") if a
            for e in json.loads(Path(a).read_text())["episodes"]]

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

    def depth():
        r = c.simGetImages([airsim.ImageRequest(CAM, airsim.ImageType.DepthPlanar, True, False)])[0]
        d = np.array(r.image_data_float, np.float32).reshape(r.height, r.width)
        h, w = d.shape
        win = d[h // 2 - h // 8:h // 2 + h // 8, w // 2 - w // 8:w // 2 + w // 8]
        return float(win.min()), float(np.median(win)), float(d.min())

    def inland(x: float, y: float) -> bool:
        s = classify_spawn_xy(float(x), float(y), regions)
        return bool(s["in_urban"] and s["in_interior"] and not s["in_water"] and s["spawn_inland_m"] >= 100.0
                    and shore(float(x), float(y)) >= args.min_inland_m)

    def approach(start: np.ndarray, goal: np.ndarray) -> str:
        yaw0 = math.atan2(goal[1] - start[1], goal[0] - start[0])
        place(start + [0, 0, -8], yaw0)
        if place(start, yaw0):
            return "start_embedded"
        prev = ts()
        for _ in range(120):
            p = c.getMultirotorState().kinematics_estimated.position
            v = goal - np.array([p.x_val, p.y_val, p.z_val])
            dist = float(np.linalg.norm(v))
            if dist <= SUCCESS_M:
                return "arrived"
            v = v / max(dist, 1e-6) * SPEED
            c.moveByVelocityAsync(float(v[0]), float(v[1]), float(v[2]), DT,
                                  yaw_mode=airsim.YawMode(False, math.degrees(math.atan2(v[1], v[0])))).join()
            obj = fresh(prev)
            if obj:
                return f"collision {obj}"
        return "max_steps"

    review_path = out / "AUTO_REVIEW.json"
    review = json.loads(review_path.read_text()) if review_path.is_file() else {"args": vars(args), "checked": [], "kept": []}
    done = {(x["route_id"], x["point"]) for x in review["checked"]}
    kept = [np.asarray(k["goal_ned"][:2], float) for k in review["kept"]]
    order = [(r, i) for i in range(max(len(r["pos"]) for r in routes) - 1, -1, -1) for r in routes if i < len(r["pos"])]

    def save() -> None:
        review_path.write_text(json.dumps(review, indent=1))
        eps = []
        for n, k in enumerate(review["kept"]):
            r = next(x for x in routes if x["route_id"] == k["route_id"])
            g = [k["goal_ned"][0], k["goal_ned"][1], -k["goal_ned"][2]]
            yaw = math.atan2(g[1] - r["pos"][0][1], g[0] - r["pos"][0][0])
            meta = dict(r["complexity_meta"], split="train", wall_goal={kk: k[kk] for kk in ("point", "wall_dir_deg", "clearance_m")})
            eps.append(dict(r, pos=[r["pos"][0], g], yaw=[yaw, yaw], route_id=f"od_trainwall_{n:03d}",
                            variant_id=f"{r['variant_id']}_wall{n:03d}", source_route_id=r["route_id"], complexity_meta=meta))
        ann = {k: v for k, v in src.items() if k != "episodes"}
        ann.update(note="goals 2-3 m in front of visible walls (wall_goal_gen.py)", n_episodes=len(eps), episodes=eps)
        (out / "train_wall.json").write_text(json.dumps(ann, indent=1))

    for r, i in order:
        if len(review["kept"]) >= args.target:
            break
        if (r["route_id"], i) in done:
            continue
        o = np.array([r["pos"][i][0], r["pos"][i][1], -r["pos"][i][2]], float)
        rec = {"route_id": r["route_id"], "point": i, "origin_ned": o.round(2).tolist(), "candidates": []}
        review["checked"].append(rec)
        place(o + [0, 0, -8], 0.0)
        if place(o, 0.0):
            rec["drop"] = "origin_contact"
            save()
            continue
        for k in range(16):
            yaw = k * math.pi / 8
            place(o, yaw)
            cmin, cmed, _ = depth()
            if not (4.0 <= cmin <= 40.0 and cmed - cmin <= 3.0):
                continue
            u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            g = o + (cmin - WALL_M) * u
            cand = {"wall_dir_deg": k * 22.5, "wall_m": round(cmin, 2), "goal_ned": g.round(2).tolist()}
            rec["candidates"].append(cand)
            if not inland(g[0], g[1]):
                cand["drop"] = "geography"
                continue
            if any(float(np.linalg.norm(g[:2] - h)) < args.exclude_goal_m for h in held):
                cand["drop"] = "near_held_out_goal"
                continue
            if any(float(np.linalg.norm(g[:2] - h)) < args.sep_m for h in kept):
                cand["drop"] = "near_kept_goal"
                continue
            contacts, clears, frame = [], [], []
            for j in range(8):
                place(g + [0, 0, -8], j * math.pi / 4)
                hit = place(g, j * math.pi / 4)
                if hit:
                    contacts.append(hit)
                cm, _, fm = depth()
                clears.append(round(cm, 2))
                frame.append(round(fm, 2))
            cand.update(contacts=contacts, clear_8yaw=clears, frame_min_8yaw=frame)
            if contacts:
                cand["drop"] = f"goal_contact {contacts[0]}"
                continue
            if not (CLEAR_LO <= min(clears) <= CLEAR_HI):
                cand["drop"] = f"clearance {min(clears)}"
                continue
            if min(frame) < FRAME_MIN:
                cand["drop"] = f"frame_min {min(frame)}"
                continue
            res = approach(g - 8.0 * u, g)
            cand["approach"] = res
            if res != "arrived":
                cand["drop"] = f"approach {res}"
                continue
            cand["keep"] = True
            kept.append(g[:2])
            review["kept"].append({"route_id": r["route_id"], "point": i, "goal_ned": g.round(2).tolist(),
                                   "wall_dir_deg": k * 22.5, "clearance_m": min(clears)})
            print("KEEP", r["route_id"], i, cand["goal_ned"], "clear", min(clears), flush=True)
            break
        save()
        print("checked", r["route_id"], i, "kept", len(review["kept"]),
              [cc.get("drop", "keep") for cc in rec["candidates"]], flush=True)
    save()
    status = "GATE_PASS" if len(review["kept"]) >= args.target else "GATE_FAIL"
    review["gate"] = {"target": args.target, "kept": len(review["kept"]), "status": status}
    review_path.write_text(json.dumps(review, indent=1))
    print(status, len(review["kept"]), flush=True)
    return 0 if status == "GATE_PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
