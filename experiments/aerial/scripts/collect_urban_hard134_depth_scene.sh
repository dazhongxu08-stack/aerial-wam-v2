#!/usr/bin/env bash
# Hard-route DepthScene expert collect — gate routes ONLY (0,1,3,4).
# Honest breakthrough: no soft eval on easy 8/9/10; until every hard route has keeps.
#
# Geography: urban inland (water rejected). Interior polygon NOT required —
# official gate routes 0/4 spawn outside the tight interior mask but are the hard set.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)_hard134}"
OUT="${OUT:-experiments/aerial/rl/artifacts/dataset_urban_hard134_depth_scene_${STAMP}}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
ROUTES="${ROUTES:-0,1,3,4}"
MIN_SPAWN_Z="${MIN_SPAWN_Z:-45}"
MAX_INFLATE="${MAX_INFLATE:-1.8}"
MIN_KEPT="${MIN_KEPT:-12}"
MIN_KEPT_PER_ROUTE="${MIN_KEPT_PER_ROUTE:-3}"
MAX_ROUNDS="${MAX_ROUNDS:-999}"
CRUISE="${CRUISE:-8}"
MAX_STEPS="${MAX_STEPS:-900}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
APPEND_FLAG=()
if [[ -f "$OUT/manifest.json" || -f "$OUT/AUTO_REVIEW.json" ]]; then
  APPEND_FLAG=(--append)
fi

if [[ -x "$RECOVER" && "$RECOVER" != /bin/true && "$RECOVER" != /usr/bin/true ]]; then
  bash "$RECOVER" outdoor || true
  sleep 15
fi

echo "[hard134-expert] routes=[$ROUTES] out=$OUT min_kept=$MIN_KEPT per_route=$MIN_KEPT_PER_ROUTE rounds=$MAX_ROUNDS append=${APPEND_FLAG[*]:-no}"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m experiments.aerial.rl.collect_depth_scene_expert_dataset \
  --backend airsim \
  --device cuda \
  --annotation "$ANNO" \
  --routes "$ROUTES" \
  --out "$OUT" \
  --min-spawn-z "$MIN_SPAWN_Z" \
  --spawn-z-retry-m 16 \
  --spawn-z-max-retries 4 \
  --max-inflate "$MAX_INFLATE" \
  --require-arrived \
  --no-require-interior \
  --min-spawn-inland-m 100 \
  --min-path-inland-m 80 \
  --min-kept "$MIN_KEPT" \
  --min-kept-per-route "$MIN_KEPT_PER_ROUTE" \
  --until-gate \
  --max-rounds "$MAX_ROUNDS" \
  --shield-fwd-only \
  --tti-coeff 2.5 \
  --r-m 80 \
  --cruise-speed "$CRUISE" \
  --max-steps "$MAX_STEPS" \
  --d-clear 35 \
  --d-danger 4 \
  --max-dyaw 0.45 \
  --episodes 4 \
  "${APPEND_FLAG[@]}" \
  2>&1 | tee -a "artifacts/collect_urban_hard134_${STAMP}.log"

echo "[hard134-expert] done -> $OUT"
