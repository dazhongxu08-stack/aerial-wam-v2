#!/usr/bin/env bash
# e5 (2026-09-20 end-to-end-avoidance follow-up): SAME recipe as
# train_urban_complex_p2c.sh, but (a) warm-starts the WM from
# wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt (depth-reconstruction aux
# loss added -- see dynamics_torch.py loss_depth_aux; multi-seed latent probe
# showed encode_window r2_holdout 0.375->0.585 vs the un-aux'd wm_step_3500),
# and (b) turns on --w-intervention so the shield's hard_retreat/TTI-cap steps
# cost the actor something instead of being free. Isolates "does fixing the
# two diagnosed root causes (weak geometry + free shield) move hard-route SR"
# from everything else (routes, iters, warm-start actor, shield params all
# held IDENTICAL to the original run for a clean A/B).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh

STAMP="${STAMP:-$(date +%Y%m%d)_depthaux}"
CONFIG_REL="configs/aerial_rl_urban_complex_p2c.yaml"
WM_REL="${WM_REL:-experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920}"
WM_STEP="${WM_STEP:-8500}"
INIT_REL="experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt"
CKPT_REL="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${STAMP}"
LOG_REL="artifacts/train_urban_complex_p2c_${STAMP}.log"
SCENE_SH="${SCENE_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
FOCUS_ROUTES="${FOCUS_ROUTES:-0,1,3,4}"
W_INTERVENTION="${W_INTERVENTION:-0.1}"
# Default 10 matches RewardConfig / yaml and the clearance-risk usability bar
# (test_clearance_risk_from_depth_piecewise: w*risk > 5 at d≈5m). The old
# --w-collision 1.0 left imag avoidance weaker than one step of progress.
W_COLLISION="${W_COLLISION:-10.0}"

ITERS="${ITERS:-32}"
START_ITER="${START_ITER:-0}"
EP_PER_ITER="${EP_PER_ITER:-1}"
IMAGINE_BATCH="${IMAGINE_BATCH:-16}"
IMAGINE_HORIZON="${IMAGINE_HORIZON:-15}"
NEAR_FRAC="${NEAR_FRAC:-0.5}"
NEAR_MIN="${NEAR_MIN:-5.0}"
NEAR_MAX="${NEAR_MAX:-25.0}"
RESUME_CKPT="${RESUME_CKPT:-}"
RENDERER_RESTART_EVERY="${RENDERER_RESTART_EVERY:-8}"
MIN_SPAWN_Z="${MIN_SPAWN_Z:-24.0}"
SPAWN_RETRY_M="${SPAWN_RETRY_M:-5.0}"
SPAWN_Z_MAX_RETRIES="${SPAWN_Z_MAX_RETRIES:-4}"

say() { echo "[urban-p2c-depthaux] $*"; }

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
else
  # Actor collects its own actions. Imagination RL is the only learner.
  PLANNER_ARGS+=(--no-planner)
fi
SHIELD_ARGS=()
if [[ "$SHIELD_EXCLUSION_FORWARD_ONLY" == "1" ]]; then
  SHIELD_ARGS+=(--shield-exclusion-forward-only)
fi

say "=== online Phase-2 train (routes=[$FOCUS_ROUTES] near_frac=$NEAR_FRAC iters=$ITERS min_z=$MIN_SPAWN_Z planner=${ENABLE_PLANNER:-1} fwd_shield=$SHIELD_EXCLUSION_FORWARD_ONLY wm=${WM_REL}/wm_step_${WM_STEP}.pt w_collision=$W_COLLISION w_intervention=$W_INTERVENTION) ==="
mkdir -p "$(dirname "$CKPT_REL")" artifacts

nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m experiments.aerial.rl.train_v4_ac \
  --config "$CONFIG_REL" \
  --backend airsim \
  --device cuda \
  --dynamics torch \
  --phase2 \
  --r-m 100 \
  --wm-ckpt "${WM_REL}/wm_step_${WM_STEP}.pt" \
  --annotation "$ANNO_REL" \
  --init-actor-ckpt "$INIT_REL" \
  --iters "$ITERS" \
  --start-iter "$START_ITER" \
  --save-every-iter \
  --episodes-per-iter "$EP_PER_ITER" \
  --imagine-batch "$IMAGINE_BATCH" \
  --imagine-horizon "$IMAGINE_HORIZON" \
  --near-goal-frac "$NEAR_FRAC" \
  --near-goal-dist-min "$NEAR_MIN" \
  --near-goal-dist-max "$NEAR_MAX" \
  --min-spawn-z "$MIN_SPAWN_Z" \
  --spawn-z-retry-m "$SPAWN_RETRY_M" \
  --spawn-z-max-retries "$SPAWN_Z_MAX_RETRIES" \
  --renderer-restart-every "$RENDERER_RESTART_EVERY" \
  --renderer-restart-script "$SCENE_SH" \
  --renderer-restart-scene outdoor \
  --w-collision "$W_COLLISION" \
  --w-intervention "$W_INTERVENTION" \
  --tti-coeff "${TTI_COEFF:-2.5}" \
  "${PLANNER_ARGS[@]}" \
  "${SHIELD_ARGS[@]}" \
  --ckpt-dir "$CKPT_REL" \
  >> "$LOG_REL" 2>&1 &

echo "TRAIN_PID=$!"
say "log: $ROOT/$LOG_REL"
say "ckpt: $ROOT/$CKPT_REL/v4_ac_latest.pt"
say "post-train: WM=$ROOT/${WM_REL}/wm_step_${WM_STEP}.pt ACTOR=$CKPT_REL/v4_ac_latest.pt bash experiments/aerial/scripts/eval_urban_complex_toward_g_gate.sh"
