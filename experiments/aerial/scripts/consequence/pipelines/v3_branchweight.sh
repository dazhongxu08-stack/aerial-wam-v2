#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004; BRANCH=$ART/consequence_branch_train_20261004
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21; V3=$ART/consequence_v3_20261005
exec >>$V3/branchweight.log 2>&1
for CFG in "g48_r1:48:1" "g16_r3:16:3" "g48_r3:48:3"; do
  N=${CFG%%:*}; G=$(echo $CFG|cut -d: -f2); R=$(echo $CFG|cut -d: -f3)
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $V3/m_hv_$N \
    --branch-dirs $BRANCH --branch-exclude-val --val-routes $VAL --input-depth da3 --seed 0 \
    --branch-groups $G --rank-weight $R --steps 10000 --eval-every 500 > $V3/train_hv_$N.log 2>&1
  $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $BRANCH --ckpt $V3/m_hv_$N/consequence_best.pt \
    --routes $VAL --out $V3/rows_inregion_m_hv_$N.json > /dev/null 2>&1
  echo "$N done $(date -Is)"
done
echo BW_END
