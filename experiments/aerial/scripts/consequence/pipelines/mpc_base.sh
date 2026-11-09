#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; V3=$ART/consequence_v3_20261005
CK=$V3/m_full/consequence_best.pt,$V3/m_full_s1/consequence_best.pt,$V3/m_full_s2/consequence_best.pt
L=$ART/logs/consequence_mpc_base_20261005.log
$PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --policy model --routes od_holdout_00,od_holdout_06,od_holdout_13 \
  --out-dir $ART/consequence_mpc_v3_recheck_20261005 > $L 2>&1
echo "RECHECK_END $?" >> $L
$PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --policy goal_line \
  --out-dir $ART/consequence_mpc_goalline_20261005 >> $L 2>&1
echo "BASE_END $?" >> $L
