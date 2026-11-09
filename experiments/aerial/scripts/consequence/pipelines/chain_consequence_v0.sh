#!/bin/bash
set -uo pipefail
ROOT=/home/yao/aerial-wam-v2
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="$HOME/sim_verify/.venv/bin/python"
ART=experiments/aerial/rl/artifacts
LOGD=$ART/logs
LOG=$LOGD/consequence_chain_v0_20261004.log
DATA=$ART/consequence_data_20261004
PROBE=$ART/consequence_probe_holdout_20261004
V0=$ART/consequence_model_v0_20261004
exec >>"$LOG" 2>&1
echo "START $(date -Is)"

kept() { "$PY" -c "import json;print(json.load(open('$DATA/AUTO_REVIEW.json'))['gate']['kept_steps'])" 2>/dev/null || echo 0; }

while [ "$(kept)" -lt 10000 ]; do sleep 120; done
echo "kept=$(kept) pausing collector $(date -Is)"
touch "$DATA/STOP"
while pgrep -f "collect_consequence_dataset" >/dev/null; do sleep 10; done
rm -f "$DATA/STOP"
echo "collector paused $(date -Is)"

# Let the depth sidecars catch up before training.
sleep 150
"$PY" -u -m experiments.aerial.rl.train_consequence --data-dir "$DATA" --out-dir "$V0" --steps 15000 \
  > "$LOGD/consequence_train_v0_20261004.log" 2>&1 &
TRAIN=$!

"$PY" -u -m experiments.aerial.rl.probe_counterfactual_airsim --out-dir "$PROBE" --n-probes 40 \
  > "$LOGD/consequence_probe_20261004.log" 2>&1
echo "probe exit $? $(date -Is)"

setsid nohup bash experiments/aerial/scripts/consequence/pipelines/run_consequence_collect.sh >/dev/null 2>&1 < /dev/null &
echo "collector resumed $(date -Is)"

wait "$TRAIN"
echo "train exit $? $(date -Is)"
"$PY" -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir "$PROBE" --ckpt "$V0/consequence_best.pt" \
  > "$LOGD/consequence_gate_v0_20261004.log" 2>&1
echo "gate exit $? $(date -Is)"
echo "END $(date -Is)"
