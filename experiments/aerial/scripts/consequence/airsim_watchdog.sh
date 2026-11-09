#!/bin/bash
# Restart AirSim when it dies or a sim job stops making progress; sim jobs resume from their own review files.
cd /home/yao/aerial-wam-v2
PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts
LOG=$ART/logs/airsim_watchdog_altfix_20261006.log
STOP=/tmp/airsim_watchdog.stop
SIM_JOBS="collect_onpolicy_branches|mpc_consequence_eval|probe_counterfactual_airsim"
STALE_S=900
exec >>"$LOG" 2>&1
echo "START $(date -Is)"
bad=0
while [ ! -f "$STOP" ]; do
  sleep 60
  alive=1
  pgrep -f AirVLN-Linux-Shipping >/dev/null || alive=0
  if [ $alive = 1 ]; then
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 || alive=0
  fi
  stale=0
  if pgrep -f "$SIM_JOBS" >/dev/null; then
    newest=$(ls -t $ART/logs/consequence_*.log 2>/dev/null | head -1)
    age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
    [ "$age" -gt "$STALE_S" ] && stale=1
  fi
  if [ $alive = 1 ] && [ $stale = 0 ]; then bad=0; continue; fi
  bad=$((bad + 1))
  echo "$(date -Is) unhealthy alive=$alive stale=$stale bad=$bad"
  [ $bad -lt 2 ] && continue
  echo "$(date -Is) RECOVER: stopping sim jobs and restarting AirSim"
  pkill -f "$SIM_JOBS"; sleep 5; pkill -9 -f "$SIM_JOBS"
  bash /home/yao/aerial_airsim_persistent/recover_renderer.sh || echo "recover script failed"
  sleep 60
  timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); print('ping', c.ping())" 2>&1 | tail -1
  bad=0
done
echo "STOP $(date -Is)"
