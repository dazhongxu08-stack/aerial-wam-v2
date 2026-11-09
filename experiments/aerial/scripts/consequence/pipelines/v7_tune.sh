#!/bin/bash
# Pick decision settings (replan-every, surv-min) for the v7 ensemble on TRAIN routes only, then fly the winner 3x on east.
# Pre-declared rule: most arrivals over 2 seeds x 16 train routes; ties -> fewer collisions -> higher mean progress.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; V7=$ART/consequence_v7_20261006
CK7=$V7/m_full_s0/consequence_best.pt,$V7/m_full_s1/consequence_best.pt,$V7/m_full_s2/consequence_best.pt
TRAIN_ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json
TR=od_train_00,od_train_02,od_train_05,od_train_07,od_train_09,od_train_13,od_train_16,od_train_18,od_train_20,od_train_24,od_train_27,od_train_30,od_train_32,od_train_37,od_train_21,od_train_29
T=$ART/consequence_tune_v7_20261007; mkdir -p $T
exec >>$T/run.log 2>&1
echo "TUNE START $(date -Is)"
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; echo "AirSim not healthy $(date -Is)"; return 1; }
fly() {  # out_dir expected_n extra-args...
  local D=$1 N=$2; shift 2; local t=0
  while [ "$(n_eps $D)" -lt $N ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK7 --policy model --alt-fix --out-dir $D "$@" \
      >> $ART/logs/consequence_tune_v7_20261007.log 2>&1
    echo "fly $(basename $D) attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
}
for CFG in r3_s70 r1_s70 r3_s85 r1_s85 r3_s95 r1_s95; do
  RE=${CFG:1:1}; SM=0.${CFG:4:2}
  for S in 0 1; do
    fly $T/${CFG}_seed$S 16 --tune-on-train --annotation $TRAIN_ANN --routes $TR --replan-every $RE --surv-min $SM --seed $S
  done
done
WIN=$($PY - <<EOF
import json, glob
rows = []
for cfg in ["r3_s70", "r1_s70", "r3_s85", "r1_s85", "r3_s95", "r1_s95"]:
    eps = [e for d in sorted(glob.glob("$T/" + cfg + "_seed?")) for e in json.load(open(d + "/MPC_EVAL.json"))["episodes"] if "arrived" in e]
    arr = sum(e["arrived"] for e in eps); col = sum(e["severe_collision"] for e in eps)
    prog = sum(e["progress_ratio"] for e in eps) / max(1, len(eps))
    rows.append((arr, -col, prog, cfg, len(eps)))
    print(f"TRAIN {cfg}: flights={len(eps)} arrived={arr} collisions={col} mean_progress={prog:.3f}", flush=True)
best = max(rows)
print("WINNER", best[3])
EOF
)
echo "$WIN"
CFG=$(echo "$WIN" | sed -n 's/^WINNER //p'); RE=${CFG:1:1}; SM=0.${CFG:4:2}
echo "east confirm with $CFG (replan-every $RE surv-min $SM) $(date -Is)"
if [ "$CFG" = "r3_s70" ]; then
  echo "winner equals the current setting; east result is v7_altfix already"
else
  for S in 0 1 2; do
    fly $ART/consequence_mpc_v7_${CFG}_s${S}_20261007 18 --replan-every $RE --surv-min $SM --seed $S
  done
  python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
    "v7_$CFG=consequence_mpc_v7_${CFG}_s0_20261007,consequence_mpc_v7_${CFG}_s1_20261007,consequence_mpc_v7_${CFG}_s2_20261007" \
    "v7_altfix=consequence_mpc_v7_altfix_s0_20261006,consequence_mpc_v7_altfix_s1_20261006,consequence_mpc_v7_altfix_s2_20261006" \
    "v5_altfix=consequence_mpc_v5_altfix_s0_20261006,consequence_mpc_v5_altfix_s1_20261006,consequence_mpc_v5_altfix_s2_20261006" \
    "goal_line=consequence_mpc_goalline_20261005,consequence_mpc_goalline_s1_20261006,consequence_mpc_goalline_s2_20261006"
fi
touch /tmp/airsim_watchdog.stop
echo "TUNE_END $(date -Is)"
