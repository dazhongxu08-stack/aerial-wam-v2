#!/usr/bin/env bash
# Sequential launcher: R15 validate (3 reps) then R02 forensics pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
LOG="${LOG:-artifacts/wam_phase2_v12_r15_r02_launcher_20260916/run.log}"
mkdir -p "$(dirname "$LOG")"
echo "[$(date '+%H:%M:%S')] start R15 validate" | tee "$LOG"
bash experiments/aerial/scripts/wam_phase2_v12_route15_validate.sh 2>&1 | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] start R02 pipeline" | tee -a "$LOG"
bash experiments/aerial/scripts/wam_phase2_v12_route02_pipeline.sh 2>&1 | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] done" | tee -a "$LOG"
