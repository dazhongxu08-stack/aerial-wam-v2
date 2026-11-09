#!/bin/bash
set -uo pipefail
cd /home/yao/aerial-wam-v2
export PYTHONPATH=$PWD
PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts
SRC=$ART/consequence_data_20261004
WORK=/tmp/consequence_scaling
rm -rf $WORK; mkdir -p $WORK
# Freeze a snapshot of the episodes that already have depth sidecars.
$PY - <<'EOF'
import json, os, re, random
src = "experiments/aerial/rl/artifacts/consequence_data_20261004"
r = json.load(open(f"{src}/AUTO_REVIEW.json"))
val = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
eps = [e for e in r["episodes"] if e.get("keep") and e.get("file")
       and os.path.exists(f"{src}/episodes/{e['file'][:-4]}.da3.npz")]
tr = [e for e in eps if e["route_id"] not in val]
va = [e for e in eps if e["route_id"] in val]
random.Random(0).shuffle(tr)
for frac in (0.25, 0.5, 1.0):
    d = f"/tmp/consequence_scaling/f{int(frac*100)}"
    os.makedirs(f"{d}/episodes", exist_ok=True)
    pick = tr[: int(len(tr) * frac)] + va
    for e in pick:
        for suf in (".npz", ".da3.npz"):
            n = e["file"][:-4] + suf
            os.symlink(os.path.abspath(f"{src}/episodes/{n}"), f"{d}/episodes/{n}")
    json.dump({"episodes": pick}, open(f"{d}/AUTO_REVIEW.json", "w"))
    print(frac, "train eps", int(len(tr) * frac), "steps", sum(e["steps"] for e in tr[: int(len(tr) * frac)]),
          "collisions", sum(e["end_reason"] == "collision" for e in tr[: int(len(tr) * frac)]), "val eps", len(va))
EOF
for f in 25 50 100; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $WORK/f$f --out-dir $WORK/m$f \
    --val-routes od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21 \
    --steps 6000 --eval-every 500 > $WORK/train_f$f.log 2>&1
  $PY - <<EOF
import json
h = json.load(open("$WORK/m$f/train_history.json"))["history"]
best = min(h, key=lambda r: 2 * r["val"]["hazard"] + r["val"]["cones"] + r["val"]["dgoal"])
print("f$f best step", best["step"], "val hazard %.3f collide_err %.3f dgoal %.4f cones %.3f" % (
    best["val"]["hazard"], best["val"]["collide_abs_err"], best["val"]["dgoal"], best["val"]["cones"]))
EOF
done
