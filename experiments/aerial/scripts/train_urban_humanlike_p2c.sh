#!/usr/bin/env bash
# Human-like urban FT: aligned collect + efficiency rewards + optional expert buffer.
#
# Changes vs plain toward_g FT:
#   - Expert = DepthScene (scene intent + D̂) not PathExpert polyline
#   - w_eff_strafe/heading/idle > 0 → discourage side-fly / idle (more human-like)
#   - Optional preload of quality-filtered expert npz into imagination buffer
#   - Same deploy-align stack: planner H=1, tti=2.5, forward-only shield
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh

STAMP="${STAMP:-$(date +%Y%m%d)_humanlike}"
FOCUS_ROUTES="${FOCUS_ROUTES:-0,1,3,4}"
ITERS="${ITERS:-32}"
MIN_SPAWN_Z="${MIN_SPAWN_Z:-38}"
EXPERT_DS="${EXPERT_DS:-}"
COLLECT_EXPERT="${COLLECT_EXPERT:-0}"
RESUME_CKPT="${RESUME_CKPT:-experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt}"
SCENE_SH="${SCENE_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"

W_EFF_STRAFE="${W_EFF_STRAFE:-0.08}"
W_EFF_HEADING="${W_EFF_HEADING:-0.08}"
W_EFF_IDLE="${W_EFF_IDLE:-0.03}"

CKPT_REL="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${STAMP}"
LOG_REL="artifacts/train_urban_complex_p2c_${STAMP}.log"
ANNO_REL="experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134.json"

if [[ "$COLLECT_EXPERT" == "1" ]]; then
  EXPERT_DS="${EXPERT_DS:-experiments/aerial/rl/artifacts/dataset_urban_depth_scene_expert_${STAMP}}"
  if [[ ! -f "$EXPERT_DS/QUALITY_SUMMARY.json" ]]; then
    echo "[humanlike] collecting depth-scene expert -> $EXPERT_DS"
    STAMP="$STAMP" OUT="$EXPERT_DS" \
      ROUTES="${EXPERT_ROUTES:-0,1,3,4,8,12,14,15,16,18}" \
      bash experiments/aerial/scripts/collect_urban_depth_scene_expert.sh
  fi
fi

EXTRA=()
if [[ -n "$EXPERT_DS" && -d "$EXPERT_DS" ]]; then
  EXTRA+=(--dataset "$EXPERT_DS")
  echo "[humanlike] preload expert buffer: $EXPERT_DS"
fi

echo "[humanlike] build focus annotation routes=[$FOCUS_ROUTES]"
python3 -m experiments.aerial.scripts.build_outdoor_complex_focus_annotation \
  --routes "$FOCUS_ROUTES" --out "$ANNO_REL"

pkill -f wam_phase2_long_eval 2>/dev/null || true
pkill -f "train_v4_ac.*urban_complex_p2c" 2>/dev/null || true
sleep 2

if [[ -x "$SCENE_SH" ]]; then
  bash "$SCENE_SH" outdoor >>"$LOG_REL" 2>&1 || true
  sleep 20
fi

echo "[humanlike] FT stamp=$STAMP iters=$ITERS eff=($W_EFF_STRAFE,$W_EFF_HEADING,$W_EFF_IDLE)"
mkdir -p "$(dirname "$CKPT_REL")" artifacts

nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m experiments.aerial.rl.train_v4_ac \
  --config configs/aerial_rl_urban_complex_p2c.yaml \
  --backend airsim --device cuda --dynamics torch --phase2 --r-m 100 \
  --wm-ckpt experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt \
  --annotation "$ANNO_REL" \
  --init-actor-ckpt "$RESUME_CKPT" \
  --iters "$ITERS" --start-iter 0 --save-every-iter \
  --episodes-per-iter 1 --imagine-batch 16 --imagine-horizon 15 \
  --near-goal-frac "${NEAR_FRAC:-0.5}" --near-goal-dist-min 5 --near-goal-dist-max 25 \
  --min-spawn-z "$MIN_SPAWN_Z" --spawn-z-retry-m 14 --spawn-z-max-retries 3 \
  --renderer-restart-every 8 \
  --renderer-restart-script "$SCENE_SH" \
  --renderer-restart-scene outdoor \
  --w-collision 1.0 \
  --w-eff-strafe "$W_EFF_STRAFE" \
  --w-eff-heading "$W_EFF_HEADING" \
  --w-eff-idle "$W_EFF_IDLE" \
  --tti-coeff 2.5 --planner --planner-horizon 1 --shield-exclusion-forward-only \
  "${EXTRA[@]}" \
  --ckpt-dir "$CKPT_REL" \
  >> "$LOG_REL" 2>&1 &

echo "TRAIN_PID=$!"
echo "[humanlike] log=$ROOT/$LOG_REL"
echo "[humanlike] ckpt=$ROOT/$CKPT_REL/v4_ac_latest.pt"
echo "[humanlike] gate: ACTOR=$CKPT_REL/v4_ac_latest.pt MIN_SPAWN_Z=38 SHIELD_EXCLUSION_FORWARD_ONLY=1 \\"
echo "  bash experiments/aerial/scripts/eval_urban_complex_toward_g_gate.sh"
