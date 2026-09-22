#!/usr/bin/env bash
# Directional OA autopilot on 125: GT collect → pack → train head → gate → policy.
# Usage (on 125, from repo root):
#   bash experiments/aerial/scripts/run_directional_oa_125_autopilot.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ART="$ROOT/experiments/aerial/rl/artifacts"
GT_DIR="$ART/obstacle_cost_gt_depth"
mkdir -p "$GT_DIR" "$ART/logs"
LOG="$ART/logs/directional_oa_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-41451}"
WM_CKPT="${WM_CKPT:-$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
ACTOR_INIT="${ACTOR_INIT:-$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear/v4_ac_latest.pt}"
if [[ ! -f "$ACTOR_INIT" ]]; then
  ACTOR_INIT="$(ls -1 "$ART"/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear/v4_ac_iter_*.pt 2>/dev/null | sort | tail -1 || true)"
fi
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
if [[ ! -f "$ROOT/$ANN" ]]; then
  ANN="experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134.json"
fi
FRAMES="$GT_DIR/frames.npz"
LABELS="$GT_DIR/labels.npz"
OBS_CKPT="$ART/wm_ckpt_obstacle_cost_${STAMP}/wm_obs.pt"
GATE_JSON="$GT_DIR/gate_${STAMP}.json"
POLICY_DIR="$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}"

echo "=== directional OA autopilot stamp=$STAMP ==="
echo "WM=$WM_CKPT ACTOR=$ACTOR_INIT ANN=$ANN"

if [[ ! -f "$WM_CKPT" ]]; then
  echo "FATAL: missing WM ckpt $WM_CKPT"
  exit 1
fi

# --- 1) GT collect ---
if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
  echo "=== [1/4] collect GT depth frames ==="
  "$PYTHON_BIN" -m experiments.aerial.rl.collect_obstacle_cost_gt_frames \
    --config configs/aerial_rl_urban_complex_p2c.yaml \
    --annotation "$ANN" \
    --out "$FRAMES" \
    --host "$HOST" --port "$PORT" \
    --episodes "${EPISODES:-20}" \
    --steps-per-ep "${STEPS_PER_EP:-12}" \
    --min-per-group "${MIN_PER_GROUP:-30}" \
    --max-frames "${MAX_FRAMES:-600}" \
    || COLLECT_RC=$?
  COLLECT_RC="${COLLECT_RC:-0}"
  if [[ ! -f "$FRAMES" ]]; then
    echo "FATAL: frames.npz missing after collect (rc=$COLLECT_RC)"
    exit 1
  fi
else
  echo "=== [1/4] SKIP_COLLECT=1 using existing $FRAMES ==="
fi

# --- 2) pack ---
echo "=== [2/4] pack RGB→feature ==="
"$PYTHON_BIN" -m experiments.aerial.rl.train_obstacle_cost_labels pack \
  --wm-ckpt "$WM_CKPT" \
  --frames "$FRAMES" \
  --out "$LABELS" \
  --device cuda

# --- 3) train head ---
echo "=== [3/4] train obstacle_cost_head ==="
mkdir -p "$(dirname "$OBS_CKPT")"
"$PYTHON_BIN" -m experiments.aerial.rl.train_obstacle_cost_labels train \
  --wm-ckpt "$WM_CKPT" \
  --labels "$LABELS" \
  --out-ckpt "$OBS_CKPT" \
  --steps "${HEAD_STEPS:-800}" \
  --batch 64 \
  --device cuda

# --- 4) gate ---
echo "=== [4/4] task-5 gate ==="
set +e
"$PYTHON_BIN" -m experiments.aerial.rl.train_obstacle_cost_labels gate \
  --wm-ckpt "$OBS_CKPT" \
  --labels "$LABELS" \
  --report "$GATE_JSON" \
  --device cuda
GATE_RC=$?
set -e
if [[ "$GATE_RC" -ne 0 ]]; then
  echo "GATE FAILED (rc=$GATE_RC) — refuse policy train. See $GATE_JSON"
  exit 2
fi
echo "GATE PASSED → start policy train"

# --- 5) policy ---
mkdir -p "$POLICY_DIR"
"$PYTHON_BIN" -m experiments.aerial.rl.train_v4_ac \
  --config configs/aerial_rl_urban_complex_p2c.yaml \
  --config-overlay configs/aerial_rl_urban_complex_directional_oa.yaml \
  --backend airsim \
  --dynamics torch \
  --wm-ckpt "$OBS_CKPT" \
  --init-actor-ckpt "$ACTOR_INIT" \
  --no-planner \
  --no-shield \
  --iters "${ITERS:-40}" \
  --episodes-per-iter 1 \
  --imagine-batch 16 \
  --imagine-horizon 15 \
  --device cuda \
  --annotation "$ANN" \
  --ckpt-dir "$POLICY_DIR" \
  --save-every-iter \
  2>&1 | tee -a "$ART/logs/directional_oa_policy_${STAMP}.log"

echo "=== DONE stamp=$STAMP policy=$POLICY_DIR gate=$GATE_JSON log=$LOG ==="
