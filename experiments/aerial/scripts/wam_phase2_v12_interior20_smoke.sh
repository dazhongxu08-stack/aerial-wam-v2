#!/usr/bin/env bash
# Smoke-record dual-view for all 20 R09-template interior routes (spawn FPV check).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v12_mainline.inc.sh"

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
OUT_ROOT="${OUT_ROOT:-artifacts/wam_phase2_v12_interior20_smoke_20260916}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
ROUTES="${ROUTES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"

mkdir -p "$OUT_ROOT/traj" "$OUT_ROOT/videos"
LOG="$OUT_ROOT/run.log"
echo "=== interior20 smoke routes=[$ROUTES] -> $OUT_ROOT ===" | tee "$LOG"

IFS=',' read -r -a ROUTE_ARR <<< "$ROUTES"
for idx in "${ROUTE_ARR[@]}"; do
  idx="${idx// /}"
  [[ -z "$idx" ]] && continue
  label="interior$(printf '%02d' "$idx")"
  traj_dir="$OUT_ROOT/traj/${label}"
  mkdir -p "$traj_dir"

  echo "" | tee -a "$LOG"
  echo "[$(date '+%H:%M:%S')] === route_idx=${idx} ${label} ===" | tee -a "$LOG"

  if [[ -x "$RECOVER" ]]; then
    bash "$RECOVER" >>"$LOG" 2>&1 || true
    sleep 5
  fi

  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" \
    --annotation "$ANNO" \
    --routes "$idx" \
    --traj-out "$traj_dir" \
    --out "$OUT_ROOT/eval_${label}.json" \
    "${WAM_PHASE2_V12_STACK[@]}" 2>&1 | tee -a "$LOG"
  set -e

  traj_jsonl="$traj_dir/route$(printf '%02d' "$idx").jsonl"
  [[ -f "$traj_jsonl" ]] || { echo "MISSING $traj_jsonl" | tee -a "$LOG"; continue; }

  "$PY" -m experiments.aerial.scripts.wam_phase2_dual_view_from_traj \
    --traj-jsonl "$traj_jsonl" \
    --ref-polyline-json "$ANNO" \
    --route-idx "$idx" \
    --plan-mode polyline \
    --route-label "V12 interior ${label}" \
    --out-dir "$OUT_ROOT/videos" \
    --out-prefix "V12_interior_${label}" \
    --fps 5 --source-hz 5 2>&1 | tee -a "$LOG"
done

echo "[$(date '+%H:%M:%S')] interior20 smoke batch done" | tee -a "$LOG"
