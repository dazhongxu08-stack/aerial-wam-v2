#!/bin/bash
# on-policy failure-state branching with the conservative rule (replan 1, surv 0.95) -> v8 ensemble -> gate -> 3x east (same rule).
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004; ONP1=$ART/consequence_onpolicy_20261006
LOW=$ART/consequence_lowclear_20261006; FAIL=$ART/consequence_fail_20261007
V7=$ART/consequence_v7_20261006; V8=$ART/consequence_v8_20261007
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
CK7=$V7/m_full_s0/consequence_best.pt,$V7/m_full_s1/consequence_best.pt,$V7/m_full_s2/consequence_best.pt
TARGET=200
DEADLINE=$(date -d "2026-10-07 16:00" +%s)
mkdir -p $V8; exec >>$V8/run.log 2>&1
echo "V8B START (alt-band flights) $(date -Is)"
n_states() { $PY -c "import json;print(len(json.load(open('$FAIL/AUTO_REVIEW.json'))['probes']))" 2>/dev/null || echo 0; }
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done
  echo "AirSim not healthy after 20 min $(date -Is)"; return 1
}
tries=0
while [ "$(n_states)" -lt $TARGET ] && [ "$(date +%s)" -lt $DEADLINE ] && [ $tries -lt 15 ]; do
  while pgrep -f collect_onpolicy_branches >/dev/null; do sleep 30; done
  wait_airsim || { sleep 300; tries=$((tries + 1)); continue; }
  left=$(( DEADLINE - $(date +%s) )); [ $left -le 60 ] && break
  timeout $left $PY -u -m experiments.aerial.rl.collect_onpolicy_branches --ckpts $CK7 --out-dir $FAIL --n-states $TARGET \
    --branch-mode fail --alt-fix --replan-every 1 --surv-min 0.95 --stuck-s 40 --max-steps 500 --seed 41 \
    >> $ART/logs/consequence_fail_20261007.log 2>&1
  echo "collect attempt $tries exit $? states=$(n_states) $(date -Is)"
  tries=$((tries + 1))
done
N=$(n_states)
echo "collect done states=$N target=$TARGET $(date -Is)"
[ "$N" -lt $TARGET ] && echo "WARNING: failure states below target ($N < $TARGET); continuing with what was collected"
$PY -c "
import json, collections, numpy as np
r=json.load(open('$FAIL/AUTO_REVIEW.json')); p=r['probes']
print('fail states', len(p), 'mixed', sum(0<sum(x['collided'])<len(x['collided']) for x in p),
      'all_collide', sum(sum(x['collided'])==len(x['collided']) for x in p),
      'cand_coll_rate %.2f' % np.mean([np.mean(x['collided']) for x in p]),
      'flights', len(r['flights']), collections.Counter(f['end'] for f in r['flights']),
      collections.Counter(x['why'].split('_')[0] for x in p))"
for S in 0 1 2; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $V8/m_full_s$S \
    --branch-dirs $BRANCH,$ONP1,$LOW,$FAIL --val-routes $VAL --input-depth da3 --seed $S \
    --steps 10000 --eval-every 500 > $V8/train_full_s$S.log 2>&1
  echo "train s$S exit $? $(date -Is)"
done
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  F=""
  for S in 0 1 2; do
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $V8/m_full_s$S/consequence_best.pt \
      --select filter --surv-min 0.7 --out $V8/rows_east_${P}_s$S.json > /dev/null 2>&1
    F="$F $V8/rows_east_${P}_s$S.json"
  done
  echo "== ENSEMBLE v8 east_$P"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
done
CK8=$V8/m_full_s0/consequence_best.pt,$V8/m_full_s1/consequence_best.pt,$V8/m_full_s2/consequence_best.pt
fly() {  # ckpts out_dir seed
  local t=0
  while [ "$(n_eps $2)" -lt 18 ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $1 --policy model --seed $3 --alt-fix --alt-band \
      --replan-every 1 --surv-min 0.95 --out-dir $2 >> $ART/logs/consequence_mpc_v8_20261007.log 2>&1
    echo "mpc $(basename $2) attempt $t exit $? episodes=$(n_eps $2) $(date -Is)"; t=$((t + 1))
  done
}
for S in 0 1 2; do fly $CK8 $ART/consequence_mpc_v8_band_s${S}_20261007 $S; done
for S in 0 1 2; do fly $CK7 $ART/consequence_mpc_v7_band_s${S}_20261007 $S; done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
  "v8_band=consequence_mpc_v8_band_s0_20261007,consequence_mpc_v8_band_s1_20261007,consequence_mpc_v8_band_s2_20261007" \
  "v7_band=consequence_mpc_v7_band_s0_20261007,consequence_mpc_v7_band_s1_20261007,consequence_mpc_v7_band_s2_20261007" \
  "v7_r1_s95=consequence_mpc_v7_r1_s95_s0_20261007,consequence_mpc_v7_r1_s95_s1_20261007,consequence_mpc_v7_r1_s95_s2_20261007" \
  "goal_line=consequence_mpc_goalline_20261005,consequence_mpc_goalline_s1_20261006,consequence_mpc_goalline_s2_20261006"
touch /tmp/airsim_watchdog.stop
echo "V8B_END $(date -Is)"
