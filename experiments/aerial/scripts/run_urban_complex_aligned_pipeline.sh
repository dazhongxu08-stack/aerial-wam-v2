#!/usr/bin/env bash
# Aligned collect FT → gate eval → shield_fwd_only (if shield_off exists) → analyze.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh

BASE_STAMP="${BASE_STAMP:-20260917}"
TRAIN_STAMP="${TRAIN_STAMP:-${BASE_STAMP}_aligned}"
LOG="${LOG:-artifacts/urban_complex_aligned_pipeline_${TRAIN_STAMP}.log}"

exec > >(tee -a "$LOG") 2>&1
echo "[aligned] $(date -Is) wait train_stamp=$TRAIN_STAMP"

while pgrep -f "train_v4_ac.*${TRAIN_STAMP}" >/dev/null 2>&1; do
  ITER=$(grep -c "wrote ckpt iter" "artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log" 2>/dev/null || echo 0)
  echo "[aligned] $(date -Is) training... iters=$ITER"
  sleep 300
done
echo "[aligned] $(date -Is) training finished"

CKPT="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${TRAIN_STAMP}/v4_ac_latest.pt"
[[ -f "$CKPT" ]] || { echo "[aligned] ERROR missing $CKPT"; exit 1; }

SHIELD_OUT="artifacts/urban_complex_shield_ablation_${BASE_STAMP}"
if [[ -f "$SHIELD_OUT/eval_shield_off.json" && ! -f "$SHIELD_OUT/eval_shield_fwd_only.json" ]]; then
  echo "[aligned] $(date -Is) === shield_fwd_only only (shield_off exists) ==="
  PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
  RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
  export ACTOR="${ACTOR:-experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt}"
  # shellcheck disable=SC1091
  source experiments/aerial/scripts/wam_phase2_v11_mainline.inc.sh
  if [[ -x "$RECOVER" ]]; then bash "$RECOVER" >>"$SHIELD_OUT/run_fwd_only.log" 2>&1 || true; sleep 8; fi
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" \
    --annotation experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json \
    --routes 0,1,3,4 \
    --subgoal-source toward_g --r-m-intent 100 \
    --planner --planner-horizon 1 --tti-coeff 2.5 \
    --heading-reentry-cos 0.5 --cte-reentry-m 1.5 \
    --cruise-speed 10.0 --max-steps 600 --success-dist 3.0 \
    --min-spawn-z 38.0 --spawn-z-retry-m 14.0 --spawn-z-max-retries 3 \
    --shield-exclusion-forward-only \
    --out "$SHIELD_OUT/eval_shield_fwd_only.json" \
    --traj-out "$SHIELD_OUT/traj_shield_fwd_only" 2>&1 | tee -a "$SHIELD_OUT/run_fwd_only.log"
fi

GATE_STAMP="${TRAIN_STAMP}_gate"
GATE_OUT="artifacts/urban_complex_toward_g_gate_${GATE_STAMP}"
if [[ ! -f "$GATE_OUT/eval_all.json" ]]; then
  echo "[aligned] $(date -Is) === gate eval $CKPT ==="
  STAMP="$GATE_STAMP" ACTOR="$CKPT" ANNO="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json" \
    ROUTES="0,1,3,4" MIN_SPAWN_Z=38.0 SPAWN_RETRY_M=14.0 SPAWN_Z_MAX_RETRIES=3 \
    SHIELD_EXCLUSION_FORWARD_ONLY=1 \
    bash experiments/aerial/scripts/eval_urban_complex_toward_g_gate.sh
fi

ANALYSIS="artifacts/urban_complex_posttrain_analysis_${TRAIN_STAMP}.json"
echo "[aligned] $(date -Is) === analyze ==="
"${AERIAL_PY:-${PYTHON_BIN:-python3}}" experiments/aerial/scripts/analyze_urban_complex_posttrain.py \
  --baseline20 "artifacts/urban_complex_baseline20_${BASE_STAMP}_z38base/eval_all.json" \
  --gate-ckpt "$GATE_OUT/eval_all.json" \
  --shield-dir "$SHIELD_OUT" \
  --out "$ANALYSIS" \
  --train-stamp "$TRAIN_STAMP" \
  --base-stamp "$BASE_STAMP" \
  $(if [[ "${AUTO_CONTINUE:-0}" == "1" ]]; then echo --run-next; else echo --dry-run-next; fi)

echo "[aligned] $(date -Is) PIPELINE DONE analysis=$ANALYSIS"
