import json, sys, glob, os, numpy as np
from collections import Counter
ART = "/home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts"
for name, pat in [("gt", "consequence_mpc_gtdiag_s?_20261006"), ("da3", "consequence_mpc_v5_altfix_s?_20261006"), ("v7", "consequence_mpc_v7_altfix_s?_20261006")]:
    rows = []
    for d in sorted(glob.glob(f"{ART}/{pat}")):
        for e in json.load(open(f"{d}/MPC_EVAL.json"))["episodes"]:
            if e["route_id"] in ("od_holdout_06", "od_holdout_13") or e["end_reason"] != "collision":
                continue
            rid = e["route_id"]; st = json.load(open(f"{d}/steps_{rid}.json")); last = st[-1]
            f = np.load(f"{d}/last_frames_{rid}.npz"); g = f["gt_depth56"][-1]
            t = np.load(f"{d}/traj_{rid}.npz"); tr = t["traj"]; goal = t["goal"]
            ci = int(np.argmax(last["cand_surv"]))
            rows.append(dict(seed=d[-10], rid=rid[-2:], lag=e["collided_at_step"] - last["step"], fam=last["family"],
                surv=last["surv"], npass=last["n_pass"], maxs=max(last["cand_surv"]), dgoal=e["d_final_m"],
                hz=tr[-1, 2] - goal[2] if False else float(-tr[-1, 2] + goal[2]),
                gmin=float(g.min()), gctr=float(g[18:38, 18:38].min()), gbot=float(g[38:, :].min()),
                gtop=float(g[:18, :].min()), gl=float(g[:, :14].min()), gr=float(g[:, 42:].min()),
                prev_fams=[s["family"] for s in st[-4:]]))
    print(f"===== {name}: {len(rows)} collisions")
    print("rid s lag fam           surv npass maxs dgoal  gmin  ctr  bot  top   L    R   last4")
    for r in sorted(rows, key=lambda r: r["rid"]):
        print(f"{r['rid']} {r['seed']} {r['lag']:3d} {r['fam']:13s} {r['surv']:.2f} {r['npass']:2d}  {r['maxs']:.2f} {r['dgoal']:5.1f} {r['gmin']:5.1f} {r['gctr']:4.1f} {r['gbot']:4.1f} {r['gtop']:4.1f} {r['gl']:4.1f} {r['gr']:4.1f} {r['prev_fams']}")
    R = rows
    print("near goal (d<=8):", sum(r["dgoal"] <= 8 for r in R))
    print("confident pick (surv>=0.7):", sum(r["surv"] >= 0.7 for r in R), " no survivor (npass==0):", sum(r["npass"] == 0 for r in R))
    print("lag<=3 (collided within executed prefix):", sum(r["lag"] <= 3 for r in R))
    print("nearest gt obstacle at last frame: <2m", sum(r["gmin"] < 2 for r in R), " 2-5m", sum(2 <= r["gmin"] < 5 for r in R), " >=5m (not visible ahead)", sum(r["gmin"] >= 5 for r in R))
    print("argmin region:", Counter(min(("ctr", r["gctr"]), ("bot", r["gbot"]), ("top", r["gtop"]), ("L", r["gl"]), ("R", r["gr"]), key=lambda x: x[1])[0] for r in R))
