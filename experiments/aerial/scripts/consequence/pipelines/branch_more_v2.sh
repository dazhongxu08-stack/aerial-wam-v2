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
TARGET=${TARGET:-300}
OUT=$ART/consequence_v2_20261005
mkdir -p $OUT
exec >>"$OUT/run.log" 2>&1
echo "START $(date -Is) target=$TARGET"
cp $BRANCH/AUTO_REVIEW.json $OUT/AUTO_REVIEW_before.json
timeout 14400 $PY -u -m experiments.aerial.rl.probe_counterfactual_airsim \
  --split train --annotation experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json \
  --out-dir $BRANCH --n-probes $TARGET --seed 101 >> $ART/logs/consequence_branch_20261004.log 2>&1
echo "branch exit $? $(date -Is)"
$PY -c "
import json
p=json.load(open('$BRANCH/AUTO_REVIEW.json'))['probes']; v=set('$VAL'.split(','))
h=[x for x in p if x['route_id'] in v]
mx=lambda q: sum(0<sum(x['collided'])<len(x['collided']) for x in q)
print('branch states total', len(p), 'mixed', mx(p), '| on val routes', len(h), 'mixed', mx(h))"
summ() { $PY -c "
import json,sys
g=json.load(open(sys.argv[1])); m=g['restricted']['model']; r=g['restricted']['random_expected']
print('%-34s mixed=%2d pairwise=%.3f top1_safe=%.3f (rand %.3f) top1_prog=%.3f (rand %.3f, oracle %.3f) gate=%s'%(sys.argv[2],m['n_mixed_probes'],m['pairwise'],m['top1_safe'],r['top1_safe'],m['top1_progress'],r['top1_progress'],g['restricted']['oracle']['top1_progress'],g.get('gate_pass')))" "$1" "$2"; }
# heldout_val: branch states on VAL routes excluded from training (in-region generalization check)
# full: all branch states used for training; only the east set is a valid test for it
for V in heldout_val full; do
  EX=""; [ $V = heldout_val ] && EX="--branch-exclude-val"
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $OUT/m_$V \
    --branch-dirs $BRANCH $EX --val-routes $VAL --input-depth da3 \
    --steps 10000 --eval-every 500 > $OUT/train_$V.log 2>&1
  echo "train $V exit $? $(date -Is)"
  SETS="east"; [ $V = heldout_val ] && SETS="inregion_heldout east"
  for SET in $SETS; do
    case $SET in
      inregion_heldout) PD=$BRANCH; R="--routes $VAL";;
      east) PD=$PROBE; R="";;
    esac
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $OUT/m_$V/consequence_best.pt \
      --input-depth da3 $R --out $OUT/gate_${V}_${SET}.json > /dev/null 2>&1
    summ $OUT/gate_${V}_${SET}.json "model=$V $SET"
  done
done
echo "END $(date -Is)"
