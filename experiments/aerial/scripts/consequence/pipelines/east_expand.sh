#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; PN=$ART/consequence_probe_holdout_corridor_20261005
cp $PN/AUTO_REVIEW.json $ART/consequence_v3_20261005/east_new_AUTO_REVIEW_40.json
echo "START $(date -Is)" >> $ART/consequence_v3_20261005/east_expand.log
timeout 12600 $PY -u -m experiments.aerial.rl.probe_counterfactual_airsim \
  --split holdout --fwd-test corridor --out-dir $PN --n-probes 120 --seed 2027 \
  >> $ART/logs/consequence_probe_holdout_corridor_20261005.log 2>&1
echo "exit $? $(date -Is)" >> $ART/consequence_v3_20261005/east_expand.log
echo EXPAND_END >> $ART/consequence_v3_20261005/east_expand.log
