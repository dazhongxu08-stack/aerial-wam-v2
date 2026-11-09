#!/bin/bash
# After the v10 pipeline: re-fly representative east routes with whole-flight frames for v8, v9 (and v10 if trained), then render videos.
# Visualization only: same frozen rule, no selection or tuning on these flights.
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
ART=experiments/aerial/rl/artifacts; V10=$ART/consequence_v10_20261009
ROUTES=od_holdout_10,od_holdout_15,od_holdout_23,od_holdout_12,od_holdout_04
RULE="--policy model --alt-fix --alt-band --alt-band-return 0.0 --replan-every 1 --surv-min 0.95 --keep-frames 0"
LOG=$ART/consequence_video_20261009.log
exec >>$LOG 2>&1
echo "VIDEO WAIT for v10 $(date -Is)"
while ! grep -q "V10_END" $V10/run.log 2>/dev/null; do sleep 60; done
echo "VIDEO START $(date -Is)"
rm -f /tmp/airsim_watchdog.stop
pgrep -f "^bash /tmp/airsim_watchdog.sh" >/dev/null || (setsid nohup bash /tmp/airsim_watchdog.sh >/dev/null 2>&1 </dev/null &)
n_eps() { $PY -c "import json;print(len(json.load(open('$1/MPC_EVAL.json'))['episodes']))" 2>/dev/null || echo 0; }
wait_airsim() {
  for i in $(seq 1 40); do
    timeout 25 $PY -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); assert c.ping()" >/dev/null 2>&1 && return 0
    sleep 30
  done; echo "AirSim not healthy $(date -Is)"; return 1; }
ck() { echo "$ART/$1/m_full_s0/consequence_best.pt,$ART/$1/m_full_s1/consequence_best.pt,$ART/$1/m_full_s2/consequence_best.pt"; }
fly() {  # name model_dir extra-args...
  local NAME=$1 M=$2; shift 2; local D=$ART/consequence_mpc_video_${NAME}_s0_20261009 t=0
  while [ "$(n_eps $D)" -lt 5 ] && [ $t -lt 4 ]; do
    wait_airsim || { sleep 300; t=$((t + 1)); continue; }
    $PY -u -m experiments.aerial.rl.mpc_consequence_eval --ckpts $(ck $M) --out-dir $D $RULE --routes $ROUTES --seed 0 "$@"
    echo "fly $NAME attempt $t exit $? episodes=$(n_eps $D) $(date -Is)"; t=$((t + 1))
  done
  $PY experiments/aerial/scripts/consequence/vis_mpc_runs.py /tmp/video_$NAME $NAME "$D" "" ""
  echo "rendered /tmp/video_$NAME $(date -Is)"
}
fly v8 consequence_v8_20261007
fly v9 consequence_v9_20261008
[ -f $V10/m_full_s2/consequence_best.pt ] && fly v10 consequence_v10_20261009 --extra-families hover,spin
touch /tmp/airsim_watchdog.stop
echo "VIDEO_END $(date -Is)"
