#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; V5=$ART/consequence_v5_20261006
CK5=$V5/m_full_s0/consequence_best.pt,$V5/m_full_s1/consequence_best.pt,$V5/m_full_s2/consequence_best.pt
L=$ART/logs/consequence_mpc_v5_altfix_20261006.log
exec >>$L 2>&1
echo "START $(date -Is)"
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; return 1; }
for S in 0 1 2; do
  D=$ART/consequence_mpc_v5_altfix_s${S}_20261006; t=0
  while [ "$(n_eps $D)" -lt 18 ] && [ $t -lt 6 ]; do
    wait_airsim || break
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK5 --policy model --seed $S --alt-fix --out-dir $D
    echo "altfix s$S attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
  "v5_altfix=consequence_mpc_v5_altfix_s0_20261006,consequence_mpc_v5_altfix_s1_20261006,consequence_mpc_v5_altfix_s2_20261006" \
  "model_v5=consequence_mpc_v5_20261006,consequence_mpc_v5_s1_20261006,consequence_mpc_v5_s2_20261006" \
  "goal_line=consequence_mpc_goalline_20261005,consequence_mpc_goalline_s1_20261006,consequence_mpc_goalline_s2_20261006"
touch /tmp/airsim_watchdog.stop
echo "ALTFIX_END $(date -Is)"
