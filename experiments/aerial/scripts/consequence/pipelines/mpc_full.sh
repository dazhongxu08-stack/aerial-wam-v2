#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
V3=experiments/aerial/rl/artifacts/consequence_v3_20261005
CK=$V3/m_full/consequence_best.pt,$V3/m_full_s1/consequence_best.pt,$V3/m_full_s2/consequence_best.pt
L=experiments/aerial/rl/artifacts/logs/consequence_mpc_v3_20261005.log
$PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK \
  --out-dir experiments/aerial/rl/artifacts/consequence_mpc_v3_20261005 > $L 2>&1
echo "MPC_END $?" >> $L
