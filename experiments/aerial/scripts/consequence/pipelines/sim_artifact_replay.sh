#!/bin/bash
# After mem_long: replay every compared collision that has no recorded object yet (probe_collision_geometry), then
# summarise all runs with spawn collisions and invisible-collision-volume contacts listed separately.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; OUT=$ART/collision_probe_hist_20261010; MEMLONG=$ART/consequence_memlong_20261010
mkdir -p $OUT; exec >>$OUT/run.log 2>&1
echo "REPLAY WAIT for memlong $(date -Is)"
while ! grep -q "MEMLONG_END" $MEMLONG/run.log 2>/dev/null; do sleep 120; done
echo "REPLAY START $(date -Is)"
RUNS=(
  "v8_east=consequence_mpc_v8_band_s?_20261007" "v8_train=consequence_tune_band_20261007/ret0.0_seed?"
  "v8_fresh=consequence_mpc_fresh_spz_v8_ret0.0_s?_20261008"
  "fov_east=consequence_mpc_fov_band_s?_20261009" "fov_train=consequence_fov_20261009/train_seed?"
  "fov_fresh=consequence_mpc_fresh_fov_s?_20261009"
  "esc_east=consequence_mpc_escape_band_s?_20261009" "esc_train=consequence_escape_20261009/train_seed?"
  "mem8_east=consequence_mpc_mem8_band_s?_20261009" "mem8_train=consequence_mem8_20261009/train_seed?"
  "long_east=consequence_mpc_memlong_band_s?_20261010" "long_train=consequence_memlong_20261010/train_seed?"
  "long_fresh=consequence_mpc_fresh_memlong_s?_20261010"
  "ctrl_east=consequence_mpc_memlong_ctrl_band_s?_20261010" "ctrl_train=consequence_memlong_20261010/ctrl_train_seed?"
)
$PY - "${RUNS[@]}" <<'EOF'
import glob, json, sys
from pathlib import Path
art = "experiments/aerial/rl/artifacts"
done = set()
for x in json.load(open(f"{art}/collision_probe_20261010/RESULTS.json")):
    c = x["case"]
    if c.get("kind") == "contact" and x.get("hit"):
        done.add((c["dir"], c["route_id"]))
cases = []
for spec in sys.argv[1:]:
    name, pat = spec.split("=", 1)
    for d in sorted(glob.glob(f"{art}/{pat}")):
        for e in json.load(open(f"{d}/MPC_EVAL.json"))["episodes"]:
            if e.get("end_reason") != "collision" or e["steps"] <= 2 or e.get("collision_contact"):
                continue
            if (d, e["route_id"]) in done or not Path(f"{d}/last_frames_{e['route_id']}.npz").exists():
                continue
            cases.append({"kind": "contact", "group": name, "dir": d, "route_id": e["route_id"], "seed": Path(d).name[-12:]})
json.dump(cases, open("/tmp/collision_cases_hist.json", "w"), indent=1)
print("replay cases", len(cases))
EOF
rm -f /tmp/airsim_watchdog.stop
pgrep -f "^bash /tmp/airsim_watchdog.sh" >/dev/null || (setsid nohup bash /tmp/airsim_watchdog.sh >/dev/null 2>&1 </dev/null &)
for i in $(seq 1 40); do
  timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && break
  sleep 30
done
timeout 14400 $PY -u experiments/aerial/scripts/consequence/probe_collision_geometry.py --cases /tmp/collision_cases_hist.json \
  --out $OUT > $OUT/probe.log 2>&1
echo "probe exit $? $(date -Is)"
touch /tmp/airsim_watchdog.stop
$PY experiments/aerial/scripts/consequence/sim_artifact_summary.py \
  --replays $ART/collision_probe_20261010/RESULTS.json,$OUT/RESULTS.json "${RUNS[@]}"
echo "REPLAY_END $(date -Is)"
