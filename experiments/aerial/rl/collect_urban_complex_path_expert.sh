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
MIN_SPAWN_Z="${MIN_SPAWN_Z:-18.0}"
SPAWN_RETRY_M="${SPAWN_RETRY_M:-6.0}"
MIN_USABLE_PATH_M="${MIN_USABLE_PATH_M:-40.0}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[urban-pe] $(date -Is) PathExpert teacher N=${N} ann=${ANNO} out=${OUT} min_z=${MIN_SPAWN_Z}"

if [[ -x "$RECOVER" ]]; then
  bash "$RECOVER" || true
  sleep 8
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
  --keep-failed \
  --min-usable-path-m "$MIN_USABLE_PATH_M" \
  --annotation "$ANNO" \
  --out "$OUT"

echo "[urban-pe] $(date -Is) DONE out=${OUT}"
