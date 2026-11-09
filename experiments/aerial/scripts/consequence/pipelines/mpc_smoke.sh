#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
V3=experiments/aerial/rl/artifacts/consequence_v3_20261005
CK=$V3/m_full/consequence_best.pt,$V3/m_full_s1/consequence_best.pt,$V3/m_full_s2/consequence_best.pt
$PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --routes od_holdout_00 \
  --out-dir experiments/aerial/rl/artifacts/consequence_mpc_smoke_20261005 > experiments/aerial/rl/artifacts/logs/consequence_mpc_smoke_20261005.log 2>&1
echo "SMOKE_END $?" >> experiments/aerial/rl/artifacts/logs/consequence_mpc_smoke_20261005.log
