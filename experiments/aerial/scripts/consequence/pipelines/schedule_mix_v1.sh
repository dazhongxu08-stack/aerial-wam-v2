#!/bin/bash
set -uo pipefail
ROOT=/home/yao/aerial-wam-v2
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="$HOME/sim_verify/.venv/bin/python"
ART=experiments/aerial/rl/artifacts
LOGD=$ART/logs
DATA=$ART/consequence_data_20261004
BRANCH=$ART/consequence_branch_train_20261004
PROBE=$ART/consequence_probe_holdout_20261004
V1=$ART/consequence_model_v1_20261005
VAL=od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21
END=$(date -d "2026-10-05 04:00" +%s)
SLICE=2400
exec >>"$LOGD/consequence_schedule_v1_20261004.log" 2>&1
echo "START $(date -Is)"

collector_running() { pgrep -f "collect_consequence_dataset" >/dev/null; }
pause_collector() {
  touch "$DATA/STOP"
  while collector_running; do sleep 10; done
  rm -f "$DATA/STOP"
}
n_branch() { "$PY" -c "import json;print(len(json.load(open('$BRANCH/AUTO_REVIEW.json'))['probes']))" 2>/dev/null || echo 0; }

while [ "$(date +%s)" -lt "$END" ]; do
  collector_running || setsid nohup bash experiments/aerial/scripts/consequence/pipelines/run_consequence_collect.sh >/dev/null 2>&1 < /dev/null &
  left=$(( END - $(date +%s) )); s=$(( left < SLICE ? left : SLICE ))
  echo "random slice ${s}s $(date -Is)"; sleep "$s"
  pause_collector
  echo "collector paused $(date -Is)"
  left=$(( END - $(date +%s) ))
  [ "$left" -le 300 ] && break
  target=$(( $(n_branch) + 25 ))
  echo "branch slice target=$target $(date -Is)"
  timeout "$(( left < SLICE + 300 ? left : SLICE + 300 ))" "$PY" -u -m experiments.aerial.rl.probe_counterfactual_airsim \
    --split train --annotation experiments/aerial/phase3_unified/annotations/outdoor_complex_od_train.json \
    --out-dir "$BRANCH" --n-probes "$target" --seed 101 >> "$LOGD/consequence_branch_20261004.log" 2>&1
  echo "branch exit $? probes=$(n_branch) $(date -Is)"
done
collector_running && pause_collector
echo "collection done $(date -Is) branch=$(n_branch)"

sleep 150  # depth sidecars catch up
"$PY" -u -m experiments.aerial.rl.train_consequence --data-dir "$DATA" --out-dir "$V1" \
  --branch-dirs "$BRANCH" --val-routes "$VAL" --steps 20000 --eval-every 1000 \
  > "$LOGD/consequence_train_v1_20261005.log" 2>&1
echo "train exit $? $(date -Is)"
"$PY" -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir "$PROBE" --ckpt "$V1/consequence_best.pt" \
  > "$LOGD/consequence_gate_v1_20261005.log" 2>&1
echo "gate exit $? $(date -Is)"
echo "END $(date -Is)"
