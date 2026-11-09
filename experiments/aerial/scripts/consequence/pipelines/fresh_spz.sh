#!/bin/bash
# Re-fly the fresh routes whose validated spawn height differs from the annotation z, with --use-spawn-z (method unchanged).
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; V8=$ART/consequence_v8_20261007
CK8=$V8/m_full_s0/consequence_best.pt,$V8/m_full_s1/consequence_best.pt,$V8/m_full_s2/consequence_best.pt
FR=$ART/interior_od_fresh_20261007; FRESH_ANN=$FR/fresh_holdout.json
RT=od_holdout_05,od_holdout_10,od_holdout_14,od_holdout_17,od_holdout_21,od_holdout_28,od_holdout_32
exec >>$FR/run_spz.log 2>&1
echo "SPZ START $(date -Is)"
rm -f /tmp/airsim_watchdog.stop
pgrep -f "^bash /tmp/airsim_watchdog.sh" >/dev/null || (setsid nohup bash /tmp/airsim_watchdog.sh >/dev/null 2>&1 </dev/null &)
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; echo "AirSim not healthy $(date -Is)"; return 1; }
fly() {
  local D=$1; shift; local t=0
  while [ "$(n_eps $D)" -lt 7 ] && [ $t -lt 6 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $CK8 --out-dir $D --annotation $FRESH_ANN --routes $RT --use-spawn-z "$@" \
      >> $ART/logs/consequence_fresh_spz_20261008.log 2>&1
    echo "fly $(basename $D) attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
}
RULE="--policy model --alt-fix --alt-band --replan-every 1 --surv-min 0.95"
for S in 0 1 2; do fly $ART/consequence_mpc_fresh_spz_v8_ret0.0_s${S}_20261008 $RULE --alt-band-return 0.0 --seed $S; done
echo "SPZ_FROZEN_DONE $(date -Is)"
for S in 0 1 2; do fly $ART/consequence_mpc_fresh_spz_goalline_s${S}_20261008 --policy goal_line --seed $S; done
for S in 0 1 2; do fly $ART/consequence_mpc_fresh_spz_v8_ret0.05_s${S}_20261008 $RULE --alt-band-return 0.05 --seed $S; done
touch /tmp/airsim_watchdog.stop
echo "SPZ_END $(date -Is)"
