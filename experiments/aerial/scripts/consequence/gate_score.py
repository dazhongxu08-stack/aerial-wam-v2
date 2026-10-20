"""Per-route strict gate over several runs of one recipe.

A route passes when, over its counted runs: arrived >= 1, mean progress >= 0.5, no severe collision,
and at most one stuck/timeout. Runs whose collision is a spawn artifact (step <= 2 or contact at reset)
or hits a measured invisible volume are listed separately and not counted.
usage: gate_score.py BASE ROUTES LABEL=dir1,dir2,dir3 [LABEL=...]
env: GATE_ROOF_579=1 also leaves out SM_NYCB_579 / SM_NYCB_856 contacts inside their invisible bands above the rendered roof;
     GATE_STUCK_RATIO=0.1 allows max(1, ceil(ratio * counted runs)) stuck runs; GATE_OUT=path writes the summary.
"""
import json
import math
import os
import sys

INVIS = {"SM_NYCB_578", "SM_NYCB_587", "SM_NYCB_869", "SM_NYCB_867", "SM_NYCB_22", "SM_NYCB_1-5_2406"}
# SM_NYCB_579 renders up to ~25.1 m on its south face but collides up to ~32 m (probed 2026-10-19)
ROOF_579_M = 25.15
# SM_NYCB_856 north face renders up to ~28.5 m but collides at y~17.1 for z 29-30 m, clear from 31 m (probed 2026-10-19)
ROOF_BANDS = {"SM_NYCB_579": (ROOF_579_M, math.inf), "SM_NYCB_856": (28.7, 30.6)}
ROOF_RULE = os.environ.get("GATE_ROOF_579", "0") == "1"
STUCK_RATIO = float(os.environ.get("GATE_STUCK_RATIO", "0"))
INVALID_ROUTES = {"od_clean_03": "goal inside SM_NYCB_588 collision box (measured 2026-10-13)",
                  "od_clean_35": "dropped by user 2026-10-13: SM_NYCB_588 contact near spawn",
                  "od_clean_04": "goal within 0.5 m of invisible SM_NYCB_588; only 3/8 straight approaches reach it (goal_ring_04_20261013)",
                  "od_clean_31": "goal touches invisible SM_NYCB_869 at 7/8 yaws; 3/8 straight approaches reach it (goal_ring_31_20261013)",
                  "od_clean_02": "goal touches visible SM_NYCB_579 at 7/8 yaws; 1/8 straight approaches reach it (goal_ring_02_20261013)",
                  "od_clean_41": "dropped by user 2026-10-14: goal 2.1 m from visible SM_NYCB_868 (near-wall goal, handled outside the gate)"}
INVALID_ROUTES.update({rid: "endpoint within 1.5 m of a map obstacle at 24 m; spawn collisions in fresh_eval_20261019"
                       for rid in ("od_holdout_03", "od_holdout_04", "od_holdout_07", "od_holdout_09", "od_holdout_10", "od_holdout_11",
                                   "od_holdout_13", "od_holdout_23", "od_holdout_26", "od_holdout_28", "od_holdout_31")})


def kind(e):
    if e.get("end_reason") == "reset_failed":
        return "spawn"
    if e.get("end_reason") == "collision":
        if (e.get("collided_at_step") or 99) <= 2 or (e.get("reset_contact") or {}).get("has_collided"):
            return "spawn"
        cc = e.get("collision_contact") or {}
        if (cc.get("object_name") or "") in INVIS:
            return "invis"
        band = ROOF_BANDS.get(cc.get("object_name"))
        if ROOF_RULE and band and band[0] < -float((cc.get("impact_ned") or [0, 0, 0])[2]) < band[1]:
            return "invis"
        return "collision"
    if e.get("arrived"):
        return "arrived"
    if e.get("end_reason") in ("max_steps", "stuck"):
        return "stuck"
    return e.get("end_reason") or "other"


base, routes = sys.argv[1], sys.argv[2].split(",")
summary = {}
for spec in sys.argv[3:]:
    label, dirs = spec.split("=", 1)
    runs = []
    for d in dirs.split(","):
        p = os.path.join(base, d, "MPC_EVAL.json")
        runs.append({e["route_id"]: e for e in json.load(open(p))["episodes"]} if os.path.isfile(p) else None)
    have = [r for r in runs if r is not None]
    per, npass, nfail, nunk, excluded = {}, 0, 0, 0, []
    ep_arr = ep_cnt = 0
    for rid in routes:
        if rid in INVALID_ROUTES:
            excluded.append((rid, "invalid_route", INVALID_ROUTES[rid]))
            continue
        ks, progs = [], []
        for r in have:
            if rid not in r:
                continue
            k = kind(r[rid])
            if k in ("spawn", "invis"):
                excluded.append((rid, k, (r[rid].get("collision_contact") or {}).get("object_name")))
                continue
            ks.append(k)
            progs.append(float(r[rid].get("progress_ratio", 0)))
        if not ks:
            per[rid] = "no_counted_run"
            nunk += 1
            continue
        ep_cnt += len(ks)
        ep_arr += ks.count("arrived")
        ok = (ks.count("arrived") >= 1 and sum(progs) / len(progs) >= 0.5
              and ks.count("collision") == 0 and ks.count("stuck") <= max(1, math.ceil(STUCK_RATIO * len(ks))))
        per[rid] = ("PASS " if ok else "FAIL ") + "/".join(k[:3] for k in ks)
        npass += ok
        nfail += not ok
    summary[label] = {"runs": len(have), "pass": npass, "fail": nfail, "no_counted": nunk,
                      "episode_arrival": round(ep_arr / max(1, ep_cnt), 3),
                      "fails": {k: v for k, v in per.items() if not v.startswith("PASS")},
                      "excluded": excluded}
    print(f"{label}: runs={len(have)} PASS={npass} FAIL={nfail} no_counted={nunk} episode_arrival={ep_arr}/{ep_cnt}")
    print("   fails:", {k: v for k, v in per.items() if not v.startswith("PASS")})
    print("   excluded:", excluded)
if os.environ.get("GATE_OUT"):
    json.dump(summary, open(os.environ["GATE_OUT"], "w"), indent=1)
