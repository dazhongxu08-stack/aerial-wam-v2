#!/bin/bash
set -uo pipefail
cd /home/yao/aerial-wam-v2
export PYTHONPATH=$PWD
PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts
DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004
PROBE=$ART/consequence_probe_holdout_20261004
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
OUT=$ART/consequence_diag_20261005
mkdir -p $OUT
TRAIN_ROUTES=$($PY -c "
import json
p=json.load(open('$BRANCH/AUTO_REVIEW.json'))['probes']; v=set('$VAL'.split(','))
print(','.join(sorted({x['route_id'] for x in p if x['route_id'] not in v})))")
$PY -c "
import json
p=json.load(open('$BRANCH/AUTO_REVIEW.json'))['probes']; v=set('$VAL'.split(','))
h=[x for x in p if x['route_id'] in v]
print('branch states on val routes', len(h), 'mixed', sum(0<sum(x['collided'])<len(x['collided']) for x in h))"
summ() { $PY -c "
import json,sys
g=json.load(open(sys.argv[1])); m=g['restricted']['model']; r=g['restricted']['random_expected']
print('%-34s mixed=%2d pairwise=%.3f top1_safe=%.3f (rand %.3f) top1_prog=%.3f (rand %.3f, oracle %.3f)'%(sys.argv[2],m['n_mixed_probes'],m['pairwise'],m['top1_safe'],r['top1_safe'],m['top1_progress'],r['top1_progress'],g['restricted']['oracle']['top1_progress']))" "$1" "$2"; }
for IN in da3 gt; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $OUT/m_$IN \
    --branch-dirs $BRANCH --branch-exclude-val --val-routes $VAL --input-depth $IN \
    --steps 10000 --eval-every 500 > $OUT/train_$IN.log 2>&1
  for SET in inregion_heldout insample east; do
    case $SET in
      inregion_heldout) PD=$BRANCH; R="--routes $VAL";;
      insample) PD=$BRANCH; R="--routes $TRAIN_ROUTES";;
      east) PD=$PROBE; R="";;
    esac
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $OUT/m_$IN/consequence_best.pt \
      --input-depth $IN $R --out $OUT/gate_${IN}_${SET}.json > /dev/null 2>&1
    summ $OUT/gate_${IN}_${SET}.json "input=$IN $SET"
  done
done
