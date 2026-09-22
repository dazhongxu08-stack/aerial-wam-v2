#!/usr/bin/env bash
# DEPRECATED for OA FT (2026-09-21): yaml defaults are depth-aux + w_intervention,
# but this launcher historically hard-coded Aug-28 WM + --w-collision 1.0, which
# silently disables clearance_risk and under-penalizes collisions.
#
# Default behavior: exec the depthaux recipe (correct OA path).
# Escape hatch for A/B against the old recipe:
#   ALLOW_LEGACY_AUG28=1 bash experiments/aerial/scripts/train_urban_complex_p2c.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

say() { echo "[urban-p2c] $*"; }

if [[ "${ALLOW_LEGACY_AUG28:-0}" != "1" ]]; then
  say "redirect → train_urban_complex_p2c_depthaux.sh (set ALLOW_LEGACY_AUG28=1 to keep Aug-28 / w_collision=1.0)"
  exec bash "$ROOT/experiments/aerial/scripts/train_urban_complex_p2c_depthaux.sh" "$@"
fi

say "WARN: ALLOW_LEGACY_AUG28=1 — using Aug-28 WM + w_collision=1.0 (OA clearance gated off)"

# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh

STAMP="${STAMP:-$(date +%Y%m%d)_legacy_aug28}"
CONFIG_REL="configs/aerial_rl_urban_complex_p2c.yaml"
WM_REL="experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828"
INIT_REL="experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt"
CKPT_REL="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${STAMP}"
LOG_REL="artifacts/train_urban_complex_p2c_${STAMP}.log"
SCENE_SH="${SCENE_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
FOCUS_ROUTES="${FOCUS_ROUTES:-0,1,3,4}"

ITERS="${ITERS:-32}"
START_ITER="${START_ITER:-0}"
EP_PER_ITER="${EP_PER_ITER:-1}"
NEAR_FRAC="${NEAR_FRAC:-0.5}"
NEAR_MIN="${NEAR_MIN:-5.0}"
NEAR_MAX="${NEAR_MAX:-25.0}"
RESUME_CKPT="${RESUME_CKPT:-}"
RENDERER_RESTART_EVERY="${RENDERER_RESTART_EVERY:-8}"
MIN_SPAWN_Z="${MIN_SPAWN_Z:-24.0}"
SPAWN_RETRY_M="${SPAWN_RETRY_M:-5.0}"
SPAWN_Z_MAX_RETRIES="${SPAWN_Z_MAX_RETRIES:-4}"

# Stage 2 full panel: FOCUS_ROUTES=all or 0-19 → 20-route annotation slice.
if [[ "${FOCUS_ROUTES}" == *"*"* || "${FOCUS_ROUTES}" == "all" || "${FOCUS_ROUTES}" == "0-19" ]]; then
  FOCUS_ROUTES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
  ANNO_REL="${ANNO_REL:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
else
  ANNO_REL="${ANNO_REL:-experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134.json}"
fi

if [[ -n "$RESUME_CKPT" ]]; then
  INIT_REL="$RESUME_CKPT"
  say "resume warm-start: $INIT_REL (start_iter=$START_ITER)"
elif [[ -f "${CKPT_REL}/v4_ac_latest.pt" && "${START_ITER:-0}" -gt 0 ]]; then
  INIT_REL="${CKPT_REL}/v4_ac_latest.pt"
  say "resume from existing ckpt: $INIT_REL (start_iter=$START_ITER)"
fi

say "=== build focus annotation routes=[$FOCUS_ROUTES] ==="
python3 -m experiments.aerial.scripts.build_outdoor_complex_focus_annotation \
  --routes "$FOCUS_ROUTES" \
  --out "$ANNO_REL"

say "=== stop stray eval / path-expert jobs ==="
pkill -f wam_phase2_long_eval 2>/dev/null || true
pkill -f route0_poly_tune 2>/dev/null || true
pkill -f focus134_goal_sweep 2>/dev/null || true
sleep 2

say "=== recover outdoor renderer ==="
if [[ -x "$SCENE_SH" ]]; then
  bash "$SCENE_SH" >>"$LOG_REL" 2>&1 || bash "$SCENE_SH" outdoor >>"$LOG_REL" 2>&1 || true
  sleep 30
else
  say "WARN: missing $SCENE_SH -- assuming outdoor already up"
fi

for i in $(seq 1 36); do
  if python3 -c "import socket;socket.create_connection(('127.0.0.1',41451),3).close()" 2>/dev/null; then
    say "airsim port open (try $i)"
    break
  fi
  sleep 5
done

if ! python3 -c "import socket;socket.create_connection(('127.0.0.1',41451),3).close()" 2>/dev/null; then
  say "ERROR: AirSim :41451 not reachable"
  exit 1
fi

pkill -f "train_v4_ac.*urban_complex_p2c" 2>/dev/null || true
sleep 2

SHIELD_EXCLUSION_FORWARD_ONLY="${SHIELD_EXCLUSION_FORWARD_ONLY:-1}"
PLANNER_ARGS=()
if [[ "${ENABLE_PLANNER:-1}" == "1" ]]; then
  PLANNER_ARGS+=(--planner --planner-horizon "${PLANNER_HORIZON:-1}")
fi
SHIELD_ARGS=()
if [[ "$SHIELD_EXCLUSION_FORWARD_ONLY" == "1" ]]; then
  SHIELD_ARGS+=(--shield-exclusion-forward-only)
fi

say "=== LEGACY online Phase-2 train (Aug-28 WM w_collision=1.0 routes=[$FOCUS_ROUTES]) ==="
mkdir -p "$(dirname "$CKPT_REL")" artifacts

nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m experiments.aerial.rl.train_v4_ac \
  --config "$CONFIG_REL" \
  --backend airsim \
  --device cuda \
  --dynamics torch \
  --phase2 \
  --r-m 100 \
  --wm-ckpt "${WM_REL}/wm_step_3500.pt" \
  --annotation "$ANNO_REL" \
  --init-actor-ckpt "$INIT_REL" \
  --iters "$ITERS" \
  --start-iter "$START_ITER" \
  --save-every-iter \
  --episodes-per-iter "$EP_PER_ITER" \
  --imagine-batch 16 \
  --imagine-horizon 15 \
  --near-goal-frac "$NEAR_FRAC" \
  --near-goal-dist-min "$NEAR_MIN" \
  --near-goal-dist-max "$NEAR_MAX" \
  --min-spawn-z "$MIN_SPAWN_Z" \
  --spawn-z-retry-m "$SPAWN_RETRY_M" \
  --spawn-z-max-retries "$SPAWN_Z_MAX_RETRIES" \
  --renderer-restart-every "$RENDERER_RESTART_EVERY" \
  --renderer-restart-script "$SCENE_SH" \
  --renderer-restart-scene outdoor \
  --w-collision 1.0 \
  --w-intervention 0.0 \
  --tti-coeff "${TTI_COEFF:-2.5}" \
  "${PLANNER_ARGS[@]}" \
  "${SHIELD_ARGS[@]}" \
  --ckpt-dir "$CKPT_REL" \
  >> "$LOG_REL" 2>&1 &

echo "TRAIN_PID=$!"
say "log: $ROOT/$LOG_REL"
say "ckpt: $ROOT/$CKPT_REL/v4_ac_latest.pt"
say "post-train: WM=$ROOT/${WM_REL}/wm_step_3500.pt ACTOR=$CKPT_REL/v4_ac_latest.pt bash experiments/aerial/scripts/eval_urban_complex_toward_g_gate.sh"
