#!/bin/bash
set -uo pipefail
cd /home/yao/aerial-wam-v2
export PYTHONPATH=$PWD
PY=$HOME/sim_verify/.venv/bin/python
DATA=experiments/aerial/rl/artifacts/consequence_data_20261004
touch $DATA/STOP
while pgrep -f "collect_consequence_dataset" >/dev/null; do sleep 5; done
rm -f $DATA/STOP
echo "collector paused $(date -Is)"
timeout 900 $PY -u -m experiments.aerial.rl.probe_counterfactual_airsim --out-dir /tmp/consequence_probe_dry --n-probes 1 2>&1 | grep -v -E "spawn|Warn" | tail -4
setsid nohup bash experiments/aerial/scripts/consequence/pipelines/run_consequence_collect.sh >/dev/null 2>&1 < /dev/null &
echo "collector resumed $(date -Is)"
$PY -u -m experiments.aerial.rl.eval_consequence_gate --probe-dir /tmp/consequence_probe_dry --ckpt /tmp/consequence_dry/consequence_best.pt --out /tmp/consequence_probe_dry/GATE.json 2>&1 | grep -v -E "Warn|INFO" | tail -40
