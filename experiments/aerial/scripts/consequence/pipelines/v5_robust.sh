#!/bin/bash
# v5 pipeline with resume-and-retry on every AirSim step; airsim_watchdog.sh restarts the renderer.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004
V3=$ART/consequence_v3_20261005; V5=$ART/consequence_v5_20261006
ONP=$ART/consequence_onpolicy_20261006
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
CK3=$V3/m_full/consequence_best.pt,$V3/m_full_s1/consequence_best.pt,$V3/m_full_s2/consequence_best.pt
TARGET=150
DEADLINE=$(date -d "2026-10-06 11:30" +%s)
mkdir -p $V5; exec >>$V5/run.log 2>&1
echo "ROBUST START $(date -Is)"

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
  timeout $left $PY -u -m experiments.aerial.rl.collect_onpolicy_branches --ckpts $CK3 --out-dir $ONP --n-states $TARGET \
    >> $ART/logs/consequence_onpolicy_20261006.log 2>&1
  echo "collect attempt $tries exit $? states=$(n_states) $(date -Is)"
  tries=$((tries + 1))
done
N=$(n_states)
echo "collect done states=$N target=$TARGET $(date -Is)"
[ "$N" -lt $TARGET ] && echo "WARNING: on-policy states below target ($N < $TARGET); continuing with what was collected"
$PY -c "
import json, collections
r=json.load(open('$ONP/AUTO_REVIEW.json')); p=r['probes']
print('onpolicy states', len(p), 'mixed', sum(0<sum(x['collided'])<len(x['collided']) for x in p),
      'flights', len(r['flights']), collections.Counter(f['end'] for f in r['flights']), collections.Counter(x['why'] for x in p))"

for S in 0 1 2; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $V5/m_full_s$S \
    --branch-dirs $BRANCH,$ONP --val-routes $VAL --input-depth da3 --seed $S \
    --steps 10000 --eval-every 500 > $V5/train_full_s$S.log 2>&1
  echo "train s$S exit $? $(date -Is)"
done
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  F=""
  for S in 0 1 2; do
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $V5/m_full_s$S/consequence_best.pt \
      --select filter --surv-min 0.7 --out $V5/rows_east_${P}_s$S.json > /dev/null 2>&1
    F="$F $V5/rows_east_${P}_s$S.json"
  done
  echo "== ENSEMBLE v5 east_$P"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
done

CK5=$V5/m_full_s0/consequence_best.pt,$V5/m_full_s1/consequence_best.pt,$V5/m_full_s2/consequence_best.pt
run_mpc() {  # policy seed out_dir log
  local t=0
  while [ "$(n_eps $3)" -lt 18 ] && [ $t -lt 6 ]; do
    wait_airsim || return 1
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK5 --policy $1 --seed $2 \
      --out-dir $3 >> $4 2>&1
    echo "mpc $1 s$2 attempt $t exit $? episodes=$(n_eps $3) $(date -Is)"
    t=$((t + 1))
  done
}
run_mpc model 0 $ART/consequence_mpc_v5_20261006 $ART/logs/consequence_mpc_v5_20261006.log
for S in 1 2; do
  run_mpc model $S $ART/consequence_mpc_v5_s${S}_20261006 $ART/logs/consequence_mpc_v5_s${S}_20261006.log
  run_mpc goal_line $S $ART/consequence_mpc_goalline_s${S}_20261006 $ART/logs/consequence_mpc_goalline_s${S}_20261006.log
done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py
touch /tmp/airsim_watchdog.stop
echo "V5_END $(date -Is)"
