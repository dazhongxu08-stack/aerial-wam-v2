#!/bin/bash
set -uo pipefail
cd /home/yao/aerial-wam-v2
export PYTHONPATH=$PWD
PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts
V2=$ART/consequence_v2_20261005
PROBE_OLD=$ART/consequence_probe_holdout_20261004
PROBE_NEW=$ART/consequence_probe_holdout_corridor_20261005
exec >>"$V2/east_corridor.log" 2>&1
echo "START $(date -Is)"
while pgrep -f "probe_counterfactual_airsim --split train" >/dev/null; do sleep 30; done
echo "branch collection finished $(date -Is)"
timeout 7200 $PY -u -m experiments.aerial.rl.probe_counterfactual_airsim \
  --split holdout --fwd-test corridor --out-dir $PROBE_NEW --n-probes 40 --seed 2027 \
  > $ART/logs/consequence_probe_holdout_corridor_20261005.log 2>&1
echo "east corridor probes exit $? $(date -Is)"
while ! grep -q "^END" $V2/run.log 2>/dev/null; do sleep 30; done
summ() { $PY -c "
import json,sys
g=json.load(open(sys.argv[1])); m=g['restricted']['model']; r=g['restricted']['random_expected']
print('%-34s mixed=%2d pairwise=%.3f top1_safe=%.3f (rand %.3f) top1_prog=%.3f (rand %.3f, oracle %.3f) gate=%s'%(sys.argv[2],m['n_mixed_probes'],m['pairwise'],m['top1_safe'],r['top1_safe'],m['top1_progress'],r['top1_progress'],g['restricted']['oracle']['top1_progress'],g.get('gate_pass')))" "$1" "$2"; }
for M in $ART/consequence_model_v1_20261005 $V2/m_heldout_val $V2/m_full; do
  N=$(basename $M)
  for P in old new; do
    PD=$PROBE_OLD; [ $P = new ] && PD=$PROBE_NEW
    $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $PD --ckpt $M/consequence_best.pt \
      --input-depth da3 --out $V2/gate_east_${P}_${N}.json > /dev/null 2>&1
    summ $V2/gate_east_${P}_${N}.json "$N east_$P"
  done
done
echo "END $(date -Is)"
