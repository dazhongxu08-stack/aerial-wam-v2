#!/bin/bash
# Round 2 on-policy: v5 ensemble flies train routes, branch where it fails, retrain v6, offline gate, east MPC x3.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004
ONP1=$ART/consequence_onpolicy_20261006
ONP=$ART/consequence_onpolicy2_20261006
V5=$ART/consequence_v5_20261006; V6=$ART/consequence_v6_20261006
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
CK5=$V5/m_full_s0/consequence_best.pt,$V5/m_full_s1/consequence_best.pt,$V5/m_full_s2/consequence_best.pt
TARGET=150
DEADLINE=$(date -d "2026-10-06 15:30" +%s)
mkdir -p $V6; exec >>$V6/run.log 2>&1
echo "START $(date -Is)"

n_states() { $PY -c "import json;print(len(json.load(open('$ONP/AUTO_REVIEW.json'))['probes']))" 2>/dev/null || echo 0; }
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done
  echo "AirSim not healthy after 20 min $(date -Is)"; return 1
}

tries=0
while [ "$(n_states)" -lt $TARGET ] && [ "$(date +%s)" -lt $DEADLINE ] && [ $tries -lt 12 ]; do
  while pgrep -f collect_onpolicy_branches >/dev/null; do sleep 30; done
  [ "$(n_states)" -ge $TARGET ] && break
  wait_airsim || break
  left=$(( DEADLINE - $(date +%s) )); [ $left -le 60 ] && break
  timeout $left $PY -u -m experiments.aerial.rl.collect_onpolicy_branches --ckpts $CK5 --out-dir $ONP --n-states $TARGET --seed 23 \
    >> $ART/logs/consequence_onpolicy2_20261006.log 2>&1
  echo "collect attempt $tries exit $? states=$(n_states) $(date -Is)"
  tries=$((tries + 1))
done
N=$(n_states)
echo "collect done states=$N target=$TARGET $(date -Is)"
[ "$N" -lt $TARGET ] && echo "WARNING: on-policy states below target ($N < $TARGET); continuing with what was collected"
$PY -c "
import json, collections
r=json.load(open('$ONP/AUTO_REVIEW.json')); p=r['probes']
print('onpolicy2 states', len(p), 'mixed', sum(0<sum(x['collided'])<len(x['collided']) for x in p),
      'flights', len(r['flights']), collections.Counter(f['end'] for f in r['flights']), collections.Counter(x['why'] for x in p))"

for S in 0 1 2; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $V6/m_full_s$S \
    --branch-dirs $BRANCH,$ONP1,$ONP --val-routes $VAL --input-depth da3 --seed $S \
    --steps 10000 --eval-every 500 > $V6/train_full_s$S.log 2>&1
  echo "train s$S exit $? $(date -Is)"
done
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  F=""
  for S in 0 1 2; do
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $V6/m_full_s$S/consequence_best.pt \
      --select filter --surv-min 0.7 --out $V6/rows_east_${P}_s$S.json > /dev/null 2>&1
    F="$F $V6/rows_east_${P}_s$S.json"
  done
  echo "== ENSEMBLE v6 east_$P"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
done

CK6=$V6/m_full_s0/consequence_best.pt,$V6/m_full_s1/consequence_best.pt,$V6/m_full_s2/consequence_best.pt
run_mpc() {  # seed out_dir log
  local t=0
  while [ "$(n_eps $2)" -lt 18 ] && [ $t -lt 6 ]; do
    wait_airsim || return 1
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK6 --policy model --seed $1 \
      --out-dir $2 >> $3 2>&1
    echo "mpc model s$1 attempt $t exit $? episodes=$(n_eps $2) $(date -Is)"
    t=$((t + 1))
  done
}
for S in 0 1 2; do
  run_mpc $S $ART/consequence_mpc_v6_s${S}_20261006 $ART/logs/consequence_mpc_v6_s${S}_20261006.log
done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
  "model_v6=consequence_mpc_v6_s0_20261006,consequence_mpc_v6_s1_20261006,consequence_mpc_v6_s2_20261006" \
  "model_v5=consequence_mpc_v5_20261006,consequence_mpc_v5_s1_20261006,consequence_mpc_v5_s2_20261006" \
  "goal_line=consequence_mpc_goalline_20261005,consequence_mpc_goalline_s1_20261006,consequence_mpc_goalline_s2_20261006"
touch /tmp/airsim_watchdog.stop
echo "V6_END $(date -Is)"
