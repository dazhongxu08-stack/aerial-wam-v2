#!/usr/bin/env bash
# E3: GT depth upper bound vs predicted D̂ (Phase-2 full16, V11 stack).
# Runs two arms: predicted depth (mainline) already known from E1/E2 baseline;
# this script runs GT-depth arm for the contract-cost delta.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
export ANNO="${ANNO:-artifacts/seen_airsim16_long_routes.json}"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v11_mainline.inc.sh"

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-artifacts/e3_gt_depth_${STAMP}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer_h11.sh}"
BATCHES=("0,1,2,3" "4,5,6,7" "8,9,10,11" "12,13,14,15")
ARM_NAME="${ARM_NAME:-gt_depth}"

mkdir -p "$OUT_BASE/chunks" "$OUT_BASE/traj" "$ROOT/artifacts"
echo "$OUT_BASE" > "$ROOT/artifacts/e3_gt_depth_LATEST.txt"
LOG="$OUT_BASE/run.log"
echo "=== E3 $ARM_NAME full16 ANNO=$ANNO OUT=$OUT_BASE ===" | tee "$LOG"

recover_renderer() {
  echo "[$(date '+%H:%M:%S')] recover_renderer" | tee -a "$LOG"
  if [[ -x "$RECOVER" ]]; then bash "$RECOVER" >>"$LOG" 2>&1 || true; sleep 5; fi
}

for i in "${!BATCHES[@]}"; do
  routes="${BATCHES[$i]}"
  echo "[$(date '+%H:%M:%S')] e3_${ARM_NAME} batch $i routes=[$routes]" | tee -a "$LOG"
  recover_renderer
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" \
    "${WAM_PHASE2_V11_STACK[@]}" \
    --use-gt-depth \
    --routes "$routes" \
    --traj-out "$OUT_BASE/traj" \
    --out "$OUT_BASE/chunks/chunk_${i}.json" 2>&1 | tee -a "$LOG"
  echo "batch $i exit=${PIPESTATUS[0]}" | tee -a "$LOG"
  set -e
done

"$PY" -m experiments.aerial.scripts.merge_phase2_split_eval \
  --out "$OUT_BASE/full16.json" \
  "$OUT_BASE/chunks"/chunk_*.json 2>&1 | tee -a "$LOG"

echo "$OUT_BASE" > "$OUT_BASE/DONE.txt"
echo "ALL DONE E3 -> $OUT_BASE" | tee -a "$LOG"
