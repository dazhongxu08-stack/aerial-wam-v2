#!/bin/bash
# low-clearance on-policy branching -> v7 ensemble -> offline gate -> 3x east flights (alt-fix), resume-and-retry everywhere.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004; ONP1=$ART/consequence_onpolicy_20261006
V5=$ART/consequence_v5_20261006; V7=$ART/consequence_v7_20261006
LOW=$ART/consequence_lowclear_20261006
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
CK5=$V5/m_full_s0/consequence_best.pt,$V5/m_full_s1/consequence_best.pt,$V5/m_full_s2/consequence_best.pt
TARGET=180
DEADLINE=$(date -d "2026-10-07 03:00" +%s)
mkdir -p $V7; exec >>$V7/run.log 2>&1
echo "V7 START $(date -Is)"
n_states() { $PY -c "import json;print(len(json.load(open('$LOW/AUTO_REVIEW.json'))['probes']))" 2>/dev/null || echo 0; }
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
  timeout $left $PY -u -m experiments.aerial.rl.collect_onpolicy_branches --ckpts $CK5 --out-dir $LOW --n-states $TARGET \
    --branch-mode lowclear --alt-fix --seed 31 >> $ART/logs/consequence_lowclear_20261006.log 2>&1
  echo "collect attempt $tries exit $? states=$(n_states) $(date -Is)"
  tries=$((tries + 1))
done
N=$(n_states)
echo "collect done states=$N target=$TARGET $(date -Is)"
[ "$N" -lt $TARGET ] && echo "WARNING: low-clearance states below target ($N < $TARGET); continuing with what was collected"
$PY -c "
import json, collections, numpy as np
r=json.load(open('$LOW/AUTO_REVIEW.json')); p=r['probes']
print('lowclear states', len(p), 'mixed', sum(0<sum(x['collided'])<len(x['collided']) for x in p),
      'all_collide', sum(sum(x['collided'])==len(x['collided']) for x in p),
      'cand_coll_rate %.2f' % np.mean([np.mean(x['collided']) for x in p]),
      'flights', len(r['flights']), collections.Counter(f['end'] for f in r['flights']))"
for S in 0 1 2; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $V7/m_full_s$S \
    --branch-dirs $BRANCH,$ONP1,$LOW --val-routes $VAL --input-depth da3 --seed $S \
    --steps 10000 --eval-every 500 > $V7/train_full_s$S.log 2>&1
  echo "train s$S exit $? $(date -Is)"
done
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  F=""
  for S in 0 1 2; do
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $V7/m_full_s$S/consequence_best.pt \
      --select filter --surv-min 0.7 --out $V7/rows_east_${P}_s$S.json > /dev/null 2>&1
    F="$F $V7/rows_east_${P}_s$S.json"
  done
  echo "== ENSEMBLE v7 east_$P"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
done
CK7=$V7/m_full_s0/consequence_best.pt,$V7/m_full_s1/consequence_best.pt,$V7/m_full_s2/consequence_best.pt
for S in 0 1 2; do
  D=$ART/consequence_mpc_v7_altfix_s${S}_20261006; t=0
  while [ "$(n_eps $D)" -lt 18 ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK7 --policy model --seed $S --alt-fix --out-dir $D \
      >> $ART/logs/consequence_mpc_v7_altfix_20261006.log 2>&1
    echo "v7 mpc s$S attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
  "v7_altfix=consequence_mpc_v7_altfix_s0_20261006,consequence_mpc_v7_altfix_s1_20261006,consequence_mpc_v7_altfix_s2_20261006" \
  "v5_altfix=consequence_mpc_v5_altfix_s0_20261006,consequence_mpc_v5_altfix_s1_20261006,consequence_mpc_v5_altfix_s2_20261006" \
  "goal_line=consequence_mpc_goalline_20261005,consequence_mpc_goalline_s1_20261006,consequence_mpc_goalline_s2_20261006"
touch /tmp/airsim_watchdog.stop
echo "V7_END $(date -Is)"
