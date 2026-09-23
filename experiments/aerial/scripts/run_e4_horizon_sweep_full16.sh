#!/usr/bin/env bash
# E4: planner horizon sweep H in {1,3,5,10} on Phase-2 16 routes (batched).
# Run after E2c frees the ablation-4090 AirSim slot.
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
OUT_ROOT="${OUT_ROOT:-artifacts/e4_horizon_sweep_${STAMP}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer_h11.sh}"
HORIZONS=(${HORIZONS:-1 3 5 10})
BATCHES=("0,1,2,3" "4,5,6,7" "8,9,10,11" "12,13,14,15")

mkdir -p "$OUT_ROOT" "$ROOT/artifacts"
echo "$OUT_ROOT" > "$ROOT/artifacts/e4_horizon_LATEST.txt"

recover_renderer() {
  if [[ -x "$RECOVER" ]]; then bash "$RECOVER" || true; sleep 5; fi
}

for H in "${HORIZONS[@]}"; do
  ARM="$OUT_ROOT/H${H}"
  mkdir -p "$ARM/chunks" "$ARM/traj"
  LOG="$ARM/run.log"
  echo "=== E4 horizon H=$H ===" | tee "$LOG"
  STACK=(
    --subgoal-source polyline
    --rolling-global
    --heading-assist
    --global-horizon-m 60
    --global-replan-period-s 1.0
    --heading-assist-cte-max-m 8.0
    --heading-assist-cos-thr 0.7
    --planner --planner-horizon "$H"
    --tti-coeff 2.5
    --heading-reentry-cos 0.5
    --cte-reentry-m 1.5
    --cruise-speed 10.0
    --max-steps 600
  )
  for i in "${!BATCHES[@]}"; do
    routes="${BATCHES[$i]}"
    echo "[$(date '+%H:%M:%S')] H=$H batch $i routes=[$routes]" | tee -a "$LOG"
    recover_renderer
    set +e
    "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
      "${WAM_PHASE2_CKPTS[@]}" \
      "${STACK[@]}" \
      --routes "$routes" \
      --traj-out "$ARM/traj" \
      --out "$ARM/chunks/chunk_${i}.json" 2>&1 | tee -a "$LOG"
    set -e
  done
  "$PY" -m experiments.aerial.scripts.merge_phase2_split_eval \
    --out "$ARM/full16.json" \
    "$ARM/chunks"/chunk_*.json 2>&1 | tee -a "$LOG"
done

echo "$OUT_ROOT" > "$OUT_ROOT/DONE.txt"
echo "ALL DONE E4 -> $OUT_ROOT"
