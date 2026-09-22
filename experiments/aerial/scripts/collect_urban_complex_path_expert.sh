#!/usr/bin/env bash
# PathExpert teacher densify on 20 urban-outdoor-complex routes (same panel as V11 smoke).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
export PYTHONUNBUFFERED=1

STAMP="${STAMP:-20260916}"
N="${N:-20}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
OUT="${OUT:-experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}}"
LOG="${LOG:-artifacts/collect_urban_complex_path_expert_${STAMP}.log}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
MAX_STEPS="${MAX_STEPS:-600}"
MIN_SPAWN_Z="${MIN_SPAWN_Z:-24.0}"
SPAWN_RETRY_M="${SPAWN_RETRY_M:-5.0}"
SPAWN_Z_MAX_RETRIES="${SPAWN_Z_MAX_RETRIES:-4}"
MIN_USABLE_PATH_M="${MIN_USABLE_PATH_M:-35.0}"
MIN_USABLE_STEPS="${MIN_USABLE_STEPS:-30}"
MIN_SPAWN_INLAND_M="${MIN_SPAWN_INLAND_M:-100.0}"
MIN_PATH_INLAND_M="${MIN_PATH_INLAND_M:-80.0}"
ROUTE_INDICES="${ROUTE_INDICES:-}"
APPEND="${APPEND:-0}"
SKIP_MIN_OK_GATE="${SKIP_MIN_OK_GATE:-0}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[urban-pe] $(date -Is) PathExpert teacher N=${N} ann=${ANNO} out=${OUT} min_z=${MIN_SPAWN_Z} retry=${SPAWN_RETRY_M}x${SPAWN_Z_MAX_RETRIES}"

if [[ -x "$RECOVER" ]]; then
  bash "$RECOVER" || true
  sleep 8
fi

EXTRA_ARGS=()
if [[ -n "$ROUTE_INDICES" ]]; then
  EXTRA_ARGS+=(--route-indices "$ROUTE_INDICES")
fi
if [[ "$APPEND" == "1" ]]; then
  EXTRA_ARGS+=(--append)
fi
if [[ "$SKIP_MIN_OK_GATE" == "1" ]]; then
  EXTRA_ARGS+=(--skip-min-ok-gate)
fi

"$AERIAL_PY" -m experiments.aerial.rl.collect_path_expert_dataset \
  --backend airsim \
  --host 127.0.0.1 \
  --step-hz 5.0 \
  --grab-depth \
  --episodes "$N" \
  --max-steps "$MAX_STEPS" \
  --success-dist-m 3.0 \
  --min-spawn-z "$MIN_SPAWN_Z" \
  --spawn-z-retry-m "$SPAWN_RETRY_M" \
  --spawn-z-max-retries "$SPAWN_Z_MAX_RETRIES" \
  --keep-failed \
  --min-usable-path-m "$MIN_USABLE_PATH_M" \
  --min-usable-steps "$MIN_USABLE_STEPS" \
  --min-spawn-inland-m "$MIN_SPAWN_INLAND_M" \
  --min-path-inland-m "$MIN_PATH_INLAND_M" \
  --annotation "$ANNO" \
  --out "$OUT" \
  "${EXTRA_ARGS[@]}"

echo "[urban-pe] $(date -Is) DONE out=${OUT}"
