#!/bin/bash
# History-input consequence model (depth frames t, t-2, t-5 + last 5 actions), same data as v7.
# GPU training + offline gates now; east flights (same rule as v7_band / v8_band) after V8B_END.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004; ONP1=$ART/consequence_onpolicy_20261006
LOW=$ART/consequence_lowclear_20261006
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
M=$ART/consequence_mem_20261007; mkdir -p $M
exec >>$M/run.log 2>&1
echo "MEM START $(date -Is)"
for S in 0 1 2; do
  [ -f $M/m_s$S/consequence_best.pt ] && [ -f $M/m_s$S/train_history.json ] && continue
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $M/m_s$S \
    --branch-dirs $BRANCH,$ONP1,$LOW --val-routes $VAL --input-depth da3 --seed $S \
    --frame-offsets 0,2,5 --hist-actions 5 --steps 10000 --eval-every 500 > $M/train_s$S.log 2>&1
  echo "train mem s$S exit $? $(date -Is)"
done
$PY - <<EOF
import json
for name, d in [("v7", "$ART/consequence_v7_20261006/m_full_s"), ("mem", "$M/m_s")]:
    for s in range(3):
        h = json.load(open(f"{d}{s}/train_history.json"))["history"]
        b = min(h, key=lambda r: r["val"]["hazard"] * 2 + r["val"]["cones"] + r["val"]["dgoal"])
        print(f"VAL {name} s{s} best_step={b['step']} hazard={b['val']['hazard']:.4f} cones={b['val']['cones']:.4f} dgoal={b['val']['dgoal']:.4f} collide_abs_err={b['val']['collide_abs_err']:.4f}")
EOF
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  F=""
  for S in 0 1 2; do
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $M/m_s$S/consequence_best.pt \
      --select filter --surv-min 0.7 --out $M/rows_east_${P}_s$S.json > /dev/null 2>&1
    F="$F $M/rows_east_${P}_s$S.json"
  done
  echo "== ENSEMBLE mem east_$P"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
done
echo "MEM OFFLINE DONE $(date -Is)"
while ! grep -q V8B_END $ART/consequence_v8_20261007/run.log 2>/dev/null; do sleep 120; done
rm -f /tmp/airsim_watchdog.stop; sleep 70
pgrep -f "^bash experiments/aerial/scripts/consequence/airsim_watchdog.sh" >/dev/null || (setsid nohup bash experiments/aerial/scripts/consequence/airsim_watchdog.sh >/dev/null 2>&1 < /dev/null &)
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; echo "AirSim not healthy $(date -Is)"; return 1; }
CKM=$M/m_s0/consequence_best.pt,$M/m_s1/consequence_best.pt,$M/m_s2/consequence_best.pt
for S in 0 1 2; do
  D=$ART/consequence_mpc_mem_band_s${S}_20261007; t=0
  while [ "$(n_eps $D)" -lt 18 ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CKM --policy model --seed $S --alt-fix --alt-band \
      --replan-every 1 --surv-min 0.95 --out-dir $D >> $ART/logs/consequence_mpc_mem_20261007.log 2>&1
    echo "mem mpc s$S attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
  "mem_band=consequence_mpc_mem_band_s0_20261007,consequence_mpc_mem_band_s1_20261007,consequence_mpc_mem_band_s2_20261007" \
  "v7_band=consequence_mpc_v7_band_s0_20261007,consequence_mpc_v7_band_s1_20261007,consequence_mpc_v7_band_s2_20261007" \
  "v8_band=consequence_mpc_v8_band_s0_20261007,consequence_mpc_v8_band_s1_20261007,consequence_mpc_v8_band_s2_20261007" \
  "goal_line=consequence_mpc_goalline_20261005,consequence_mpc_goalline_s1_20261006,consequence_mpc_goalline_s2_20261006"
touch /tmp/airsim_watchdog.stop
echo "MEM_END $(date -Is)"
