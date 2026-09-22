#!/usr/bin/env bash
# Collect depth-scene expert OA demos — urban INTERIOR inland only.
# Auto-review each episode; keep collecting until quality gate passes.
# Quality gate: arrived + path_inflate <= MAX_INFLATE + geography OK.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
OUT="${OUT:-experiments/aerial/rl/artifacts/dataset_urban_depth_scene_expert_${STAMP}}"
# Interior inland routes (no waterfront / open flats).
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
# Empty ROUTES = all annotation episodes that pass geography filter.
ROUTES="${ROUTES:-}"
MIN_SPAWN_Z="${MIN_SPAWN_Z:-38}"
MAX_INFLATE="${MAX_INFLATE:-1.8}"
MIN_KEPT="${MIN_KEPT:-12}"
MAX_ROUNDS="${MAX_ROUNDS:-40}"
MIN_SPAWN_INLAND="${MIN_SPAWN_INLAND:-100}"
MIN_PATH_INLAND="${MIN_PATH_INLAND:-80}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"

if [[ -x "$RECOVER" && "$RECOVER" != /bin/true && "$RECOVER" != /usr/bin/true ]]; then
  bash "$RECOVER" outdoor || true
  sleep 15
fi

ROUTE_ARGS=()
if [[ -n "$ROUTES" ]]; then
  ROUTE_ARGS=(--routes "$ROUTES")
fi

echo "[depth-scene-expert] ANNO=$ANNO out=$OUT min_kept=$MIN_KEPT until-gate interior inland"
"$PYTHON_BIN" -m experiments.aerial.rl.collect_depth_scene_expert_dataset \
  --backend airsim \
  --device cuda \
  --annotation "$ANNO" \
  "${ROUTE_ARGS[@]}" \
  --out "$OUT" \
  --min-spawn-z "$MIN_SPAWN_Z" \
  --spawn-z-retry-m 14 \
  --spawn-z-max-retries 3 \
  --max-inflate "$MAX_INFLATE" \
  --require-arrived \
  --require-interior \
  --min-spawn-inland-m "$MIN_SPAWN_INLAND" \
  --min-path-inland-m "$MIN_PATH_INLAND" \
  --min-kept "$MIN_KEPT" \
  --until-gate \
  --max-rounds "$MAX_ROUNDS" \
  --shield-fwd-only \
  --tti-coeff 2.5 \
  --r-m 100 \
  --cruise-speed 10 \
  --max-steps 600 \
  --episodes 20 \
  2>&1 | tee "artifacts/collect_urban_depth_scene_expert_${STAMP}.log"

echo "[depth-scene-expert] done -> $OUT (see AUTO_REVIEW.json / QUALITY_SUMMARY.json)"
