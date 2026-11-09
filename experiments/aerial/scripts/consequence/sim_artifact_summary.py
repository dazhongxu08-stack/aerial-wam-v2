"""MPC results with simulator artifacts listed separately (not counted, not deleted).

Artifacts: collisions within 2 steps of the spawn (spawn embedded in a building), and collisions with objects whose
collision volume the camera cannot see (found by probe_collision_geometry: no visible surface within 3 m at contact).
The collision object comes from the episode's own ``collision_contact`` or from a replay RESULTS.json.

usage: sim_artifact_summary.py --replays a.json,b.json name=glob [name=glob ...]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

INVISIBLE = ("SM_NYCB_578", "SM_NYCB_22", "SM_NYCB_1-5_2406", "SM_NYCB_869", "SM_NYCB_867")
SKIP = ("od_holdout_06", "od_holdout_13")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="")
    ap.add_argument("--art", default="experiments/aerial/rl/artifacts")
    ap.add_argument("--keep-skip", action="store_true", help="keep od_holdout_06/13 (east set) in the counts")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    obj = {}
    for p in [p for p in args.replays.split(",") if p]:
        for x in json.loads(Path(p).read_text()):
            c = x["case"]
            if c.get("kind") == "contact" and x.get("hit"):
                obj[(str(Path(c["dir"]).name if "train_seed" not in c["dir"] else "/".join(Path(c["dir"]).parts[-2:])),
                     c["route_id"])] = x["hit"]["object_name"]
    for spec in args.runs:
        name, pat = spec.split("=", 1)
        rows = collections.Counter()
        per_route = collections.defaultdict(list)
        objects = collections.Counter()
        for d in sorted(glob.glob(f"{args.art}/{pat}")):
            key_dir = Path(d).name if "train_seed" not in d else "/".join(Path(d).parts[-2:])
            fresh = "fresh" in d
            for e in json.load(open(f"{d}/MPC_EVAL.json"))["episodes"]:
                if "arrived" not in e or (not fresh and not args.keep_skip and e["route_id"] in SKIP):
                    continue
                rows["flights"] += 1
                kind = "ok"
                if e["end_reason"] == "collision":
                    o = (e.get("collision_contact") or {}).get("object_name") or obj.get((key_dir, e["route_id"]))
                    if e["steps"] <= 2:
                        kind = "spawn"
                    elif o is None:
                        kind = "coll_unknown_object"
                    elif o in INVISIBLE:
                        kind = "invisible"
                    else:
                        kind = "coll_visible"
                    if o:
                        objects[o] += 1
                rows[kind] += 1
                if kind in ("spawn", "invisible"):
                    continue
                rows["counted"] += 1
                rows["arrived"] += int(e["arrived"])
                rows["collision"] += int(e["severe_collision"])
                rows["timeout_or_stuck"] += int(e["end_reason"] in ("max_steps", "stuck"))
                per_route[e["route_id"]].append(e)
        strict = sum(1 for r, es in per_route.items()
                     if sum(x["arrived"] for x in es) >= 1 and not any(x["severe_collision"] for x in es)
                     and sum(x["stuck"] for x in es) <= 1
                     and sum(x["progress_ratio"] for x in es) / len(es) >= 0.5)
        print(f"{name:14s} flights={rows['flights']} | not counted: spawn={rows['spawn']} invisible_object={rows['invisible']} "
              f"| counted={rows['counted']} arrived={rows['arrived']} collisions={rows['collision']} "
              f"(object unknown {rows['coll_unknown_object']}) timeout/stuck={rows['timeout_or_stuck']} "
              f"arrived-coll={rows['arrived'] - rows['collision']} strict_routes={strict}/{len(per_route)}")
        if objects:
            print(f"{'':14s} collision objects: {dict(objects.most_common(8))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
