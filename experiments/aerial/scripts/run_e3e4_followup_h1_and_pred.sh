#!/usr/bin/env bash
# Step-2 follow-ups after E3/E4:
#   (A) Re-run E4 H=1 missing batch routes 4-7 → remesh H1 full16
#   (B) V11 predicted-D̂ full16 (fair pair to E3 GT-depth)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
export ANNO="${ANNO:-artifacts/seen_airsim16_long_routes.json}"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v11_mainline.inc.sh"

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer_h11.sh}"
E4_ROOT="${E4_ROOT:-artifacts/e4_horizon_sweep_phase2_20260924_010441}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PRED_OUT="${PRED_OUT:-artifacts/e3_pred_depth_${STAMP}}"
CHAIN_LOG="artifacts/e3e4_followup_${STAMP}_nohup.log"
BATCHES=("0,1,2,3" "4,5,6,7" "8,9,10,11" "12,13,14,15")

mkdir -p "$ROOT/artifacts" "$E4_ROOT/H1/chunks" "$E4_ROOT/H1/traj"
exec > >(tee -a "$CHAIN_LOG") 2>&1
echo "=== followup start $(date -Is) E4_ROOT=$E4_ROOT PRED_OUT=$PRED_OUT ==="
echo "$PRED_OUT" > "$ROOT/artifacts/e3_pred_depth_LATEST.txt"

recover_renderer() {
  echo "[$(date '+%H:%M:%S')] recover_renderer"
  if [[ -x "$RECOVER" ]]; then bash "$RECOVER" || true; sleep 5; fi
}

# ----- A) E4 H1 missing chunk_1 -----
echo "=== A) E4 H1 chunk_1 routes 4,5,6,7 ==="
recover_renderer
set +e
"$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
  "${WAM_PHASE2_CKPTS[@]}" \
  "${WAM_PHASE2_V11_STACK[@]}" \
  --planner-horizon 1 \
  --routes "4,5,6,7" \
  --traj-out "$E4_ROOT/H1/traj" \
  --out "$E4_ROOT/H1/chunks/chunk_1.json"
echo "H1 chunk_1 exit=$?"
set -e

echo "=== A) remesh H1 full16 ==="
"$PY" -m experiments.aerial.scripts.merge_phase2_split_eval \
  --out "$E4_ROOT/H1/full16.json" \
  "$E4_ROOT/H1/chunks"/chunk_*.json
echo "H1_FULL16_OK $(date -Is)"

# ----- B) V11 predicted depth full16 (E3 pair) -----
echo "=== B) V11 predicted-D̂ full16 OUT=$PRED_OUT ==="
mkdir -p "$PRED_OUT/chunks" "$PRED_OUT/traj"
for i in "${!BATCHES[@]}"; do
  routes="${BATCHES[$i]}"
  echo "[$(date '+%H:%M:%S')] pred batch $i routes=[$routes]"
  recover_renderer
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" \
    "${WAM_PHASE2_V11_STACK[@]}" \
    --routes "$routes" \
    --traj-out "$PRED_OUT/traj" \
    --out "$PRED_OUT/chunks/chunk_${i}.json"
  echo "pred batch $i exit=$?"
  set -e
done

"$PY" -m experiments.aerial.scripts.merge_phase2_split_eval \
  --out "$PRED_OUT/full16.json" \
  "$PRED_OUT/chunks"/chunk_*.json

echo "$PRED_OUT" > "$PRED_OUT/DONE.txt"
echo "ALL DONE followup -> H1 remeshed + $PRED_OUT $(date -Is)"
