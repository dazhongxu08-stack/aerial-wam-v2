#!/usr/bin/env bash
# V12 stack on high-complexity outdoor subset (6 routes by default).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v12_mainline.inc.sh"

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
OUT_ROOT="${OUT_ROOT:-artifacts/wam_phase2_v12_complex_outdoor_20260916}"
ANNO_COMPLEX="${ANNO_COMPLEX:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
BASELINE_JSON="${BASELINE_JSON:-}"

mkdir -p "$OUT_ROOT/traj"
LOG="$OUT_ROOT/run.log"

echo "=== V12 outdoor-complex eval -> $OUT_ROOT ===" | tee "$LOG"

"$PY" -m experiments.aerial.scripts.build_phase3_outdoor_complex_annotation \
  --regenerate \
  --out "$ANNO_COMPLEX" 2>&1 | tee -a "$LOG"

if [[ -x "$RECOVER" ]]; then
  echo "[$(date '+%H:%M:%S')] recover_renderer" | tee -a "$LOG"
  bash "$RECOVER" >>"$LOG" 2>&1
  sleep 5
fi

set +e
ANNO="$ANNO_COMPLEX" "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
  "${WAM_PHASE2_CKPTS[@]}" \
  --annotation "$ANNO_COMPLEX" \
  --traj-out "$OUT_ROOT/traj" \
  --out "$OUT_ROOT/complex_eval.json" \
  "${WAM_PHASE2_V12_STACK[@]}" 2>&1 | tee -a "$LOG"
EVAL_RC=$?
set -e

REPORT_ARGS=(
  --eval-json "$OUT_ROOT/complex_eval.json"
  --traj-dir "$OUT_ROOT/traj"
  --annotation "$ANNO_COMPLEX"
  --out-json "$OUT_ROOT/complex_report.json"
  --out-md "$OUT_ROOT/COMPLEX_REPORT.md"
)
if [[ -n "$BASELINE_JSON" && -f "$BASELINE_JSON" ]]; then
  REPORT_ARGS+=(--baseline-json "$BASELINE_JSON")
fi

"$PY" -m experiments.aerial.scripts.wam_phase2_complex_outdoor_report \
  "${REPORT_ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done (eval_rc=$EVAL_RC) ===" | tee -a "$LOG"
echo "Report: $OUT_ROOT/COMPLEX_REPORT.md" | tee -a "$LOG"
exit "$EVAL_RC"
