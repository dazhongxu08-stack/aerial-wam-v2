#!/usr/bin/env bash
# E2c: ablate multi-scale polyline subgoals → single-scale direct_g (full16).
# Phase-2 annotation only. Run on ablation-4090 (AirSim :41451).
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
OUT_BASE="${OUT_BASE:-artifacts/e2c_direct_g_${STAMP}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer_h11.sh}"
BATCHES=("0,1,2,3" "4,5,6,7" "8,9,10,11" "12,13,14,15")

mkdir -p "$OUT_BASE/chunks" "$OUT_BASE/traj" "$ROOT/artifacts"
echo "$OUT_BASE" > "$ROOT/artifacts/e2c_ablation_LATEST.txt"
LOG="$OUT_BASE/run.log"
echo "=== E2c direct_g full16 ANNO=$ANNO OUT=$OUT_BASE ===" | tee "$LOG"

# Same V11 motion/planner knobs, but single-scale direct_g (no polyline / rolling-global).
E2C_STACK=(
  --subgoal-source direct_g
  --heading-assist
  --heading-assist-cte-max-m 8.0
  --heading-assist-cos-thr 0.7
  --planner --planner-horizon 1
  --tti-coeff 2.5
  --heading-reentry-cos 0.5
  --cte-reentry-m 1.5
  --cruise-speed 10.0
  --max-steps 600
)

recover_renderer() {
  echo "[$(date '+%H:%M:%S')] recover_renderer" | tee -a "$LOG"
  if [[ -x "$RECOVER" ]]; then
    bash "$RECOVER" >>"$LOG" 2>&1 || true
    sleep 5
  else
    echo "WARN: missing $RECOVER" | tee -a "$LOG"
  fi
}

for i in "${!BATCHES[@]}"; do
  routes="${BATCHES[$i]}"
  chunk="$OUT_BASE/chunks/chunk_${i}.json"
  echo "[$(date '+%H:%M:%S')] e2c_direct_g batch $i routes=[$routes]" | tee -a "$LOG"
  recover_renderer
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" \
    "${E2C_STACK[@]}" \
    --routes "$routes" \
    --traj-out "$OUT_BASE/traj" \
    --out "$chunk" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  echo "batch $i exit=$rc" | tee -a "$LOG"
done

echo "[$(date '+%H:%M:%S')] merge" | tee -a "$LOG"
"$PY" -m experiments.aerial.scripts.merge_phase2_split_eval \
  --out "$OUT_BASE/full16.json" \
  "$OUT_BASE/chunks"/chunk_*.json 2>&1 | tee -a "$LOG"

echo "$OUT_BASE" > "$OUT_BASE/DONE.txt"
echo "ALL DONE E2c -> $OUT_BASE" | tee -a "$LOG"
