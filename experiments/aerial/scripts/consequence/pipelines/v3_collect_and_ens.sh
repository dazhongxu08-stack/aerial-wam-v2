#!/bin/bash
set -uo pipefail
cd /home/yao/aerial-wam-v2
export PYTHONPATH=$PWD
PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts
DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
V2=$ART/consequence_v2_20261005
V3=$ART/consequence_v3_20261005
mkdir -p $V3
exec >>"$V3/run.log" 2>&1
echo "START $(date -Is)"
cp $BRANCH/AUTO_REVIEW.json $V3/AUTO_REVIEW_before.json
# AirSim: more branch states, same distribution as before
(timeout 16200 $PY -u -m experiments.aerial.rl.probe_counterfactual_airsim \
  --split train --annotation experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json \
  --out-dir $BRANCH --n-probes 450 --seed 101 >> $ART/logs/consequence_branch_20261004.log 2>&1;
 echo "branch exit $? $(date -Is)") &
# GPU: ensemble members on the 300-state snapshot, val routes held out, for calibration selection
cp $BRANCH/AUTO_REVIEW.json $V3/AUTO_REVIEW_300.json
for S in 1 2; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $V3/m_heldout_val_s$S \
    --branch-dirs $BRANCH --branch-exclude-val --val-routes $VAL --input-depth da3 --seed $S \
    --steps 10000 --eval-every 500 > $V3/train_heldout_val_s$S.log 2>&1
  echo "train s$S exit $? $(date -Is)"
done
for M in $V2/m_heldout_val $V3/m_heldout_val_s1 $V3/m_heldout_val_s2; do
  $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $BRANCH --ckpt $M/consequence_best.pt \
    --routes $VAL --out $V3/rows_inregion_$(basename $M).json > /dev/null 2>&1
  echo "rows $(basename $M) exit $?"
done
echo "GPU_DONE $(date -Is)"
wait
echo "END $(date -Is)"
