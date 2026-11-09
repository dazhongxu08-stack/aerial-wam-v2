#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004; BRANCH=$ART/consequence_branch_train_20261004
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21; V3=$ART/consequence_v3_20261005
exec >>$V3/final.log 2>&1
echo "START $(date -Is)"
while ! grep -q "^END" $V3/run.log; do sleep 60; done
$PY -c "
import json
p=json.load(open('$BRANCH/AUTO_REVIEW.json'))['probes']
print('branch states', len(p), 'mixed', sum(0<sum(x['collided'])<len(x['collided']) for x in p))"
$PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $V3/m_full \
  --branch-dirs $BRANCH --val-routes $VAL --input-depth da3 --seed 0 \
  --steps 10000 --eval-every 500 > $V3/train_full.log 2>&1
echo "train exit $? $(date -Is)"
summ() { $PY -c "
import json,sys
g=json.load(open(sys.argv[1])); m=g['restricted']['model']; r=g['restricted']['random_expected']
print('%-14s mixed=%2d pairwise=%.3f top1_safe=%.3f (rand %.3f) top1_prog=%.3f (rand %.3f, need %.3f, oracle %.3f) gate=%s %s'%(sys.argv[2],m['n_mixed_probes'],m['pairwise'],m['top1_safe'],r['top1_safe'],m['top1_progress'],r['top1_progress'],r['top1_progress']+0.15,g['restricted']['oracle']['top1_progress'],g['gate_pass'],m['picked_families']))" "$1" "$2"; }
for P in old new; do
  PD=$ART/consequence_probe_holdout_20261004; [ $P = new ] && PD=$ART/consequence_probe_holdout_corridor_20261005
  $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $V3/m_full/consequence_best.pt \
    --select filter --surv-min 0.7 --out $V3/gate_filter_east_${P}_m_full.json > /dev/null 2>&1
  summ $V3/gate_filter_east_${P}_m_full.json "east_$P"
done
echo "FINAL_END $(date -Is)"
