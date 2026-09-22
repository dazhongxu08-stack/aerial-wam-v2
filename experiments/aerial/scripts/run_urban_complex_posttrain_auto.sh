#!/usr/bin/env bash
# After r1s2r train: shield ablation → gate eval → analyze → auto next step.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
BASE_STAMP="${BASE_STAMP:-20260917}"
TRAIN_STAMP="${TRAIN_STAMP:-${BASE_STAMP}_r1s2r}"
LOG="${LOG:-artifacts/urban_complex_posttrain_auto_${BASE_STAMP}.log}"
STAMP="${STAMP:-${BASE_STAMP}}"

exec > >(tee -a "$LOG") 2>&1
echo "[posttrain] $(date -Is) start train_stamp=$TRAIN_STAMP"

# --- wait for training ---
while pgrep -f "train_v4_ac.*urban_complex" >/dev/null 2>&1; do
  ITER=$(grep -c "wrote ckpt iter" "artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log" 2>/dev/null || echo 0)
  echo "[posttrain] $(date -Is) training... iters=$ITER/64"
  sleep 300
done
echo "[posttrain] $(date -Is) training finished"

CKPT="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${TRAIN_STAMP}/v4_ac_latest.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "[posttrain] ERROR missing ckpt $CKPT"
  exit 1
fi

# --- shield ablation (frozen phase2 actor) ---
SHIELD_OUT="artifacts/urban_complex_shield_ablation_${STAMP}"
if [[ ! -f "$SHIELD_OUT/eval_shield_off.json" ]] || [[ ! -f "$SHIELD_OUT/eval_shield_fwd_only.json" ]]; then
  echo "[posttrain] $(date -Is) === shield ablation focus 0,1,3,4 ==="
  STAMP="$STAMP" bash experiments/aerial/scripts/eval_urban_complex_shield_ablation_focus.sh
else
  echo "[posttrain] $(date -Is) shield ablation complete — skip"
fi

# --- gate: FT ckpt on focus routes z38 ---
GATE_STAMP="${BASE_STAMP}_r1s2r_gate"
GATE_OUT="artifacts/urban_complex_toward_g_gate_${GATE_STAMP}"
if [[ ! -f "$GATE_OUT/eval_all.json" ]]; then
  echo "[posttrain] $(date -Is) === gate eval ckpt=$CKPT routes=0,1,3,4 z>=38 ==="
  STAMP="$GATE_STAMP" \
    ACTOR="$CKPT" \
    ANNO="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json" \
    ROUTES="0,1,3,4" \
    MIN_SPAWN_Z=38.0 \
    SPAWN_RETRY_M=14.0 \
    SPAWN_Z_MAX_RETRIES=3 \
    bash experiments/aerial/scripts/eval_urban_complex_toward_g_gate.sh
else
  echo "[posttrain] $(date -Is) gate eval exists — skip"
fi

# --- analyze + auto next ---
ANALYSIS="artifacts/urban_complex_posttrain_analysis_${BASE_STAMP}.json"
echo "[posttrain] $(date -Is) === analyze ==="
"$PY" experiments/aerial/scripts/analyze_urban_complex_posttrain.py \
  --baseline20 "artifacts/urban_complex_baseline20_${BASE_STAMP}_z38base/eval_all.json" \
  --gate-ckpt "$GATE_OUT/eval_all.json" \
  --shield-dir "$SHIELD_OUT" \
  --out "$ANALYSIS" \
  --train-stamp "$TRAIN_STAMP" \
  --base-stamp "$BASE_STAMP" \
  --run-next

echo "[posttrain] $(date -Is) PIPELINE DONE analysis=$ANALYSIS"
