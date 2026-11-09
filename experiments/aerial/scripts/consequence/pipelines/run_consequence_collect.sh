#!/bin/bash
set -uo pipefail
ROOT=/home/yao/aerial-wam-v2
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="$HOME/sim_verify/.venv/bin/python"
LOG="$ROOT/experiments/aerial/rl/artifacts/logs/consequence_collect_20261004.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Is)"
"$PY" -u -m experiments.aerial.rl.collect_consequence_dataset \
  --out-dir experiments/aerial/rl/artifacts/consequence_data_20261004
echo "EXIT $? $(date -Is)"
