#!/usr/bin/env bash
# Render dual-view videos from baseline20 arrived-route trajs (replay in AirSim).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
BASE="${BASE:-artifacts/urban_complex_baseline20_20260917_z38base}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
OUT="${OUT:-${BASE}/videos}"
ARRIVED_ROUTES="${ARRIVED_ROUTES:-12,14,15,16,18}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"

mkdir -p "$OUT"
LOG="${OUT}/render.log"
echo "=== render arrived videos routes=[$ARRIVED_ROUTES] -> $OUT ===" | tee "$LOG"

IFS=',' read -r -a RIDS <<< "$ARRIVED_ROUTES"
for idx in "${RIDS[@]}"; do
  idx="${idx// /}"
  label="interior$(printf '%02d' "$idx")"
  traj_jsonl="${BASE}/traj/route$(printf '%02d' "$idx").jsonl"
  [[ -f "$traj_jsonl" ]] || { echo "MISSING $traj_jsonl" | tee -a "$LOG"; continue; }

  echo "[$(date '+%H:%M:%S')] route_idx=$idx $label" | tee -a "$LOG"
  if [[ -x "$RECOVER" ]]; then
    bash "$RECOVER" >>"$LOG" 2>&1 || true
    sleep 5
  fi

  "$PY" -m experiments.aerial.scripts.wam_phase2_dual_view_from_traj \
    --traj-jsonl "$traj_jsonl" \
    --ref-polyline-json "$ANNO" \
    --route-idx "$idx" \
    --plan-mode toward_g \
    --route-label "baseline20 z38 ${label} ARRIVED" \
    --out-dir "$OUT" \
    --out-prefix "baseline20_${label}" \
    --fps 5 --source-hz 5 2>&1 | tee -a "$LOG"
done

echo "[$(date '+%H:%M:%S')] done" | tee -a "$LOG"
