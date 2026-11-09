import json, glob, numpy as np
ART = "/home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts"
for name, pat in [("v5_altfix", "consequence_mpc_v5_altfix_s?_20261006"), ("v7_altfix", "consequence_mpc_v7_altfix_s?_20261006"), ("gt", "consequence_mpc_gtdiag_s?_20261006"), ("tune_train", "consequence_tune_v7_20261007/r3_s70_seed?")]:
    rows = []
    for d in glob.glob(f"{ART}/{pat}"):
        for e in json.load(open(f"{d}/MPC_EVAL.json"))["episodes"]:
            if "steps" not in e or e["route_id"] in ("od_holdout_06", "od_holdout_13"):
                continue
            cs = e.get("collided_at_step") if e["end_reason"] == "collision" else None
            for s in json.load(open(f"{d}/steps_{e['route_id']}.json")):
                if e["steps"] - s["step"] < 10 and cs is None and e["end_reason"] != "arrived":
                    continue
                hit = cs is not None and 0 <= cs - s["step"] <= 10
                rows.append((s["surv"], hit))
    if not rows:
        continue
    r = np.array(rows, float)
    out = []
    for lo, hi in [(0, .7), (.7, .8), (.8, .9), (.9, .95), (.95, 1.01)]:
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.sum():
            out.append(f"surv[{lo:.2f},{hi:.2f}) n={int(m.sum())} pred_coll={1 - r[m, 0].mean():.2f} real_coll={r[m, 1].mean():.2f}")
    print(name, "decisions", len(r)); [print("   ", o) for o in out]
