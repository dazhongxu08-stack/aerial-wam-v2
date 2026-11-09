#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; V5=$ART/consequence_v5_20261006
L=$ART/logs/consequence_mpc_repeats_20261006.log
exec >>$L 2>&1
echo "START $(date -Is)"
while ! grep -q "V5_END" $V5/run.log 2>/dev/null; do sleep 60; done
CK=$V5/m_full_s0/consequence_best.pt,$V5/m_full_s1/consequence_best.pt,$V5/m_full_s2/consequence_best.pt
for S in 1 2; do
  $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --policy model --seed $S \
    --out-dir $ART/consequence_mpc_v5_s${S}_20261006 > $ART/logs/consequence_mpc_v5_s${S}_20261006.log 2>&1
  echo "model s$S exit $? $(date -Is)"
  $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --policy goal_line --seed $S \
    --out-dir $ART/consequence_mpc_goalline_s${S}_20261006 > $ART/logs/consequence_mpc_goalline_s${S}_20261006.log 2>&1
  echo "goal_line s$S exit $? $(date -Is)"
done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py
echo "REPEATS_END $(date -Is)"
