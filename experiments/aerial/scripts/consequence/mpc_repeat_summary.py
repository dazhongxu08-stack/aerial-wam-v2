"""usage: mpc_repeat_summary.py name=dir1,dir2,dir3 name2=dir...  (dirs under artifacts/)"""
import json, sys
from pathlib import Path

ART = Path("/home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts")
INVALID = {"od_holdout_06", "od_holdout_13"}
args = sys.argv[1:] or [
    "model_v5=consequence_mpc_v5_20261006,consequence_mpc_v5_s1_20261006,consequence_mpc_v5_s2_20261006",
    "goal_line=consequence_mpc_goalline_20261005,consequence_mpc_goalline_s1_20261006,consequence_mpc_goalline_s2_20261006",
]
groups = {a.split("=", 1)[0]: a.split("=", 1)[1].split(",") for a in args}
per = {}
for g, dirs in groups.items():
    for d in dirs:
        p = ART / d / "MPC_EVAL.json"
        if not p.is_file():
            print("missing", d); continue
        for e in json.loads(p.read_text())["episodes"]:
            if "arrived" not in e or e["route_id"] in INVALID:
                continue
            per.setdefault(e["route_id"], {}).setdefault(g, []).append(e)
print("%-14s " % "route" + " ".join("%-24s" % (g + " arr/coll prog") for g in groups))
tot = {g: [0, 0, 0, 0.0, 0] for g in groups}
for rid in sorted(per):
    cells = []
    for g in groups:
        es = per[rid].get(g, [])
        a = sum(e["arrived"] for e in es); c = sum(e["severe_collision"] for e in es)
        st = sum(1 for e in es if not e["arrived"] and not e["severe_collision"])
        tot[g][0] += a; tot[g][1] += c; tot[g][2] += len(es); tot[g][3] += sum(e["progress_ratio"] for e in es); tot[g][4] += st
        cells.append("%d/%d of %d  %.2f" % (a, c, len(es), sum(e["progress_ratio"] for e in es) / max(1, len(es))))
    print("%-14s " % rid + " ".join("%-24s" % c for c in cells))
for g, (a, c, n, ps, st) in tot.items():
    print("%-10s flights=%d arrived=%d (%.0f%%) collisions=%d stuck_or_timeout=%d mean_progress=%.3f"
          % (g, n, a, 100 * a / max(1, n), c, st, ps / max(1, n)))
