#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004; BRANCH=$ART/consequence_branch_train_20261004
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21; V3=$ART/consequence_v3_20261005
exec >>$V3/ens_full.log 2>&1
for S in 1 2; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $V3/m_full_s$S \
    --branch-dirs $BRANCH --val-routes $VAL --input-depth da3 --seed $S \
    --steps 10000 --eval-every 500 > $V3/train_full_s$S.log 2>&1
  echo "train s$S exit $? $(date -Is)"
done
echo ENS_TRAIN_END
