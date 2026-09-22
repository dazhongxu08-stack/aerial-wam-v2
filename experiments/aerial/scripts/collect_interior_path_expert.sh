#!/usr/bin/env bash
# PathExpert densify on 20 R09-template interior urban routes (scene-validated polylines).
# Teacher follows annotated polyline exactly — labels for interior-complex micro-FT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
export PYTHONUNBUFFERED=1

STAMP="${STAMP:-20260916}"
N="${N:-20}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
OUT="${OUT:-experiments/aerial/rl/artifacts/dataset_interior_path_expert_${STAMP}}"
LOG="${LOG:-artifacts/collect_interior_path_expert_${STAMP}.log}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
MAX_STEPS="${MAX_STEPS:-300}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[interior-path-expert] $(date -Is) start N=${N} ann=${ANNO} out=${OUT}"

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
  --annotation "$ANNO" \
  --out "$OUT"

echo "[interior-path-expert] $(date -Is) DONE out=${OUT}"
