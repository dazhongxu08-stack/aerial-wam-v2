#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004; ONP1=$ART/consequence_onpolicy_20261006
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
G=$ART/consequence_gtdiag_20261006; mkdir -p $G
L=$ART/logs/consequence_gtdiag_20261006.log
exec >>$L 2>&1
echo "START $(date -Is)"
for S in 0 1 2; do
  $PY -u -m experiments.aerial.rl.train_consequence --data-dir $DATA --out-dir $G/m_gt_s$S \
    --branch-dirs $BRANCH,$ONP1 --val-routes $VAL --input-depth gt --seed $S \
    --steps 10000 --eval-every 500 > $G/train_gt_s$S.log 2>&1
  echo "train gt s$S exit $? $(date -Is)"
done
F=""
for S in 0 1 2; do
  $PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir $ART/consequence_probe_holdout_corridor_20261005 \
    --ckpt $G/m_gt_s$S/consequence_best.pt --input-depth gt --select filter --surv-min 0.7 --out $G/rows_east_new_s$S.json > /dev/null 2>&1
  F="$F $G/rows_east_new_s$S.json"
done
echo "== GT ENSEMBLE east_new (offline, diagnostic)"; $PY experiments/aerial/scripts/consequence/ens_gate.py $F
while ! grep -q ALTFIX_END $ART/logs/consequence_mpc_v5_altfix_20261006.log; do sleep 60; done
rm -f /tmp/airsim_watchdog.stop
pgrep -f "bash experiments/aerial/scripts/consequence/airsim_watchdog.sh" >/dev/null || (setsid nohup bash experiments/aerial/scripts/consequence/airsim_watchdog.sh >/dev/null 2>&1 < /dev/null &)
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() { for i in $(seq 1 40); do timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0; sleep 30; done; return 1; }
CK=$G/m_gt_s0/consequence_best.pt,$G/m_gt_s1/consequence_best.pt,$G/m_gt_s2/consequence_best.pt
for S in 0 1 2; do
  D=$ART/consequence_mpc_gtdiag_s${S}_20261006; t=0
  while [ "$(n_eps $D)" -lt 18 ] && [ $t -lt 6 ]; do
    wait_airsim || break
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK --policy model --seed $S --alt-fix --input-depth gt --out-dir $D
    echo "gt mpc s$S attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
done
python3 experiments/aerial/scripts/consequence/mpc_repeat_summary.py \
  "gt_input=consequence_mpc_gtdiag_s0_20261006,consequence_mpc_gtdiag_s1_20261006,consequence_mpc_gtdiag_s2_20261006" \
  "v5_altfix=consequence_mpc_v5_altfix_s0_20261006,consequence_mpc_v5_altfix_s1_20261006,consequence_mpc_v5_altfix_s2_20261006" \
  "goal_line=consequence_mpc_goalline_20261005,consequence_mpc_goalline_s1_20261006,consequence_mpc_goalline_s2_20261006"
touch /tmp/airsim_watchdog.stop
echo "GTDIAG_END $(date -Is)"
