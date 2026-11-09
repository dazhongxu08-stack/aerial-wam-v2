#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004
V3=$ART/consequence_v3_20261005; V4=$ART/consequence_v4_20261006
ONP=$ART/consequence_onpolicy_20261006
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
CK3=$V3/m_full/consequence_best.pt,$V3/m_full_s1/consequence_best.pt,$V3/m_full_s2/consequence_best.pt
mkdir -p $V4; exec >>$V4/run.log 2>&1
echo "START $(date -Is)"
timeout 18000 $PY -u -m experiments.aerial.rl.collect_onpolicy_branches --ckpts $CK3 --out-dir $ONP --n-states 150 \
  >> $ART/logs/consequence_onpolicy_20261006.log 2>&1
echo "collect exit $? $(date -Is)"
$PY -c "
import json
r=json.load(open('$ONP/AUTO_REVIEW.json')); p=r['probes']
import collections
print('onpolicy states', len(p), 'mixed', sum(0<sum(x['collided'])<len(x['collided']) for x in p),
      'flights', len(r['flights']), collections.Counter(f['end'] for f in r['flights']), collections.Counter(x['why'] for x in p))"
for S in 0 1 2; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $V4/m_full_s$S \
    --branch-dirs $BRANCH,$ONP --val-routes $VAL --input-depth da3 --seed $S \
    --steps 10000 --eval-every 500 > $V4/train_full_s$S.log 2>&1
  echo "train s$S exit $? $(date -Is)"
done
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  F=""
  for S in 0 1 2; do
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $V4/m_full_s$S/consequence_best.pt \
      --select filter --surv-min 0.7 --out $V4/rows_east_${P}_s$S.json > /dev/null 2>&1
    F="$F $V4/rows_east_${P}_s$S.json"
  done
  echo "== ENSEMBLE v4 east_$P"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
done
CK4=$V4/m_full_s0/consequence_best.pt,$V4/m_full_s1/consequence_best.pt,$V4/m_full_s2/consequence_best.pt
$PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK4 --policy model \
  --out-dir $ART/consequence_mpc_v4_20261006 > $ART/logs/consequence_mpc_v4_20261006.log 2>&1
echo "mpc exit $? $(date -Is)"
grep -E "end=|SUMMARY" $ART/logs/consequence_mpc_v4_20261006.log | cut -c27-200
echo "V4_END $(date -Is)"
