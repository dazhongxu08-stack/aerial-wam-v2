#!/usr/bin/env bash
# Queue shield ablation on 125 until r1s2r (or any urban p2c) train exits.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
LOG="${LOG:-artifacts/urban_shield_ablation_queue.log}"
STAMP="${STAMP:-$(date +%Y%m%d)}"

echo "[$(date '+%F %T')] waiting for urban p2c train to finish..." | tee "$LOG"
while pgrep -f "train_v4_ac.*urban_complex" >/dev/null 2>&1; do
  sleep 300
  echo "[$(date '+%F %T')] still training..." | tee -a "$LOG"
done
echo "[$(date '+%F %T')] train done — hand off to posttrain auto pipeline" | tee -a "$LOG"
BASE_STAMP="${BASE_STAMP:-${STAMP}}" TRAIN_STAMP="${TRAIN_STAMP:-${BASE_STAMP}_r1s2r}" \
  bash experiments/aerial/scripts/run_urban_complex_posttrain_auto.sh 2>&1 | tee -a "$LOG"
