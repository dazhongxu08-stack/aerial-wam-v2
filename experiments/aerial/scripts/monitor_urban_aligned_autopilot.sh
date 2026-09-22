#!/usr/bin/env bash
# 125-side autopilot: resume train/pipeline on interrupt; analyze + auto-continue.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh

INTERVAL="${INTERVAL_SEC:-180}"
BASE_STAMP="${BASE_STAMP:-20260917}"
TRAIN_STAMP="${TRAIN_STAMP:-${BASE_STAMP}_aligned}"
ITERS_TARGET="${ITERS_TARGET:-32}"
AUTO_CONTINUE="${AUTO_CONTINUE:-1}"
MAX_CHAIN="${MAX_CHAIN_DEPTH:-2}"
STATE="${STATE:-artifacts/urban_aligned_autopilot_state.env}"
LOG="${LOG:-artifacts/urban_aligned_autopilot.log}"

mkdir -p artifacts
[[ -f "$STATE" ]] || cat >"$STATE" <<EOF
BASE_STAMP=$BASE_STAMP
TRAIN_STAMP=$TRAIN_STAMP
ITERS_TARGET=$ITERS_TARGET
AUTO_CONTINUE=$AUTO_CONTINUE
CHAIN_DEPTH=0
MAX_CHAIN_DEPTH=$MAX_CHAIN
NEXT_RAN=0
EOF
# shellcheck disable=SC1090
source "$STATE"

log() { echo "[autopilot $(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

count_iters() {
  local logf="artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log"
  grep -c "wrote ckpt iter" "$logf" 2>/dev/null || echo 0
}

train_running() {
  pgrep -f "train_v4_ac.*${TRAIN_STAMP}" >/dev/null 2>&1
}

pipeline_running() {
  pgrep -f "run_urban_complex_aligned_pipeline" >/dev/null 2>&1
}

eval_running() {
  pgrep -f "wam_phase2_long_eval" >/dev/null 2>&1
}

start_train() {
  local start_iter="$1"
  log "START train stamp=$TRAIN_STAMP start_iter=$start_iter/$ITERS_TARGET"
  pkill -f "wam_phase2_long_eval" 2>/dev/null || true
  sleep 2
  nohup env \
    STAMP="$TRAIN_STAMP" \
    FOCUS_ROUTES="${FOCUS_ROUTES:-0,1,3,4}" \
    ITERS="$ITERS_TARGET" \
    START_ITER="$start_iter" \
    MIN_SPAWN_Z="${MIN_SPAWN_Z:-38}" \
    SPAWN_RETRY_M="${SPAWN_RETRY_M:-14}" \
    SPAWN_Z_MAX_RETRIES="${SPAWN_Z_MAX_RETRIES:-3}" \
    NEAR_FRAC="${NEAR_FRAC:-0.7}" \
    NEAR_MIN="${NEAR_MIN:-5}" \
    NEAR_MAX="${NEAR_MAX:-15}" \
    ENABLE_PLANNER=1 \
    SHIELD_EXCLUSION_FORWARD_ONLY=1 \
    TTI_COEFF=2.5 \
    RESUME_CKPT="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${TRAIN_STAMP}/v4_ac_latest.pt" \
    bash experiments/aerial/scripts/train_urban_complex_p2c.sh \
    >>"artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log" 2>&1 &
}

ensure_train() {
  local iter logf age
  iter="$(count_iters)"
  logf="artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log"
  if train_running; then
    return 0
  fi
  if [[ "$iter" -ge "$ITERS_TARGET" ]]; then
    return 0
  fi
  # Stale log: no train process but recent log activity — wait one cycle.
  if [[ -f "$logf" ]]; then
    age=$(( $(date +%s) - $(stat -c %Y "$logf" 2>/dev/null || echo 0) ))
    if [[ "$age" -lt 120 ]]; then
      log "train absent but log fresh (${age}s) — wait"
      return 0
    fi
  fi
  local ckpt="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${TRAIN_STAMP}/v4_ac_latest.pt"
  if [[ "$iter" -gt 0 && ! -f "$ckpt" ]]; then
    log "WARN iter=$iter but missing ckpt — cold start from z24"
    iter=0
    nohup env \
      STAMP="$TRAIN_STAMP" FOCUS_ROUTES="${FOCUS_ROUTES:-0,1,3,4}" ITERS="$ITERS_TARGET" \
      MIN_SPAWN_Z=38 SPAWN_RETRY_M=14 SPAWN_Z_MAX_RETRIES=3 NEAR_FRAC=0.7 \
      ENABLE_PLANNER=1 SHIELD_EXCLUSION_FORWARD_ONLY=1 TTI_COEFF=2.5 \
      RESUME_CKPT=experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_20260917_z24/v4_ac_latest.pt \
      bash experiments/aerial/scripts/train_urban_complex_p2c.sh \
      >>"$logf" 2>&1 &
    return 0
  fi
  start_train "$iter"
}

ensure_pipeline() {
  local iter gate analysis
  iter="$(count_iters)"
  [[ "$iter" -ge "$ITERS_TARGET" ]] || return 0
  gate="artifacts/urban_complex_toward_g_gate_${TRAIN_STAMP}_gate/eval_all.json"
  analysis="artifacts/urban_complex_posttrain_analysis_${TRAIN_STAMP}.json"
  if [[ -f "$analysis" ]]; then
    return 0
  fi
  if pipeline_running || eval_running; then
    return 0
  fi
  log "START pipeline train_stamp=$TRAIN_STAMP"
  nohup env BASE_STAMP="$BASE_STAMP" TRAIN_STAMP="$TRAIN_STAMP" AUTO_CONTINUE=0 \
    bash experiments/aerial/scripts/run_urban_complex_aligned_pipeline.sh \
    >>"artifacts/urban_complex_aligned_pipeline_${TRAIN_STAMP}.log" 2>&1 &
}

run_auto_next() {
  local analysis gate
  analysis="artifacts/urban_complex_posttrain_analysis_${TRAIN_STAMP}.json"
  gate="artifacts/urban_complex_toward_g_gate_${TRAIN_STAMP}_gate/eval_all.json"
  [[ -f "$analysis" ]] || return 0
  [[ -f "$gate" ]] || return 0
  [[ "${NEXT_RAN:-0}" == "1" ]] && return 0
  [[ "${AUTO_CONTINUE:-0}" != "1" ]] && return 0
  [[ "${CHAIN_DEPTH:-0}" -ge "${MAX_CHAIN_DEPTH:-2}" ]] && return 0
  if train_running || pipeline_running || eval_running; then
    return 0
  fi

  log "ANALYZE + auto-next for $TRAIN_STAMP"
  PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
  "$PY" experiments/aerial/scripts/analyze_urban_complex_posttrain.py \
    --baseline20 "artifacts/urban_complex_baseline20_${BASE_STAMP}_z38base/eval_all.json" \
    --gate-ckpt "$gate" \
    --shield-dir "artifacts/urban_complex_shield_ablation_${BASE_STAMP}" \
    --out "$analysis" \
    --train-stamp "$TRAIN_STAMP" \
    --base-stamp "$BASE_STAMP" \
    --run-next 2>&1 | tee -a "$LOG" || true

  NEXT_RAN=1
  CHAIN_DEPTH=$((CHAIN_DEPTH + 1))
  # Pick up new train stamp from latest analysis next_action suffix train.
  local new_stamp
  new_stamp="$("$PY" - <<'PY' "$analysis" "$BASE_STAMP"
import json, sys
d = json.load(open(sys.argv[1]))
act = d.get("next_action", {})
base = sys.argv[2]
suffix = {
    "continue_train_route1": "r1s3",
    "train_near_miss_route9": "r9s1",
    "train_focus_partial": "focus_s1",
}.get(act.get("action"), "")
print(f"{base}_{suffix}" if suffix else "")
PY
)"
  if [[ -n "$new_stamp" && "$new_stamp" != "$TRAIN_STAMP" ]]; then
    TRAIN_STAMP="$new_stamp"
    ITERS_TARGET="${NEXT_ITERS:-32}"
    NEXT_RAN=0
    log "CHAIN -> new TRAIN_STAMP=$TRAIN_STAMP iters=$ITERS_TARGET depth=$CHAIN_DEPTH"
  fi
  # Persist state
  cat >"$STATE" <<EOF
BASE_STAMP=$BASE_STAMP
TRAIN_STAMP=$TRAIN_STAMP
ITERS_TARGET=$ITERS_TARGET
AUTO_CONTINUE=$AUTO_CONTINUE
CHAIN_DEPTH=$CHAIN_DEPTH
MAX_CHAIN_DEPTH=$MAX_CHAIN
NEXT_RAN=$NEXT_RAN
EOF
}

tick() {
  local iter
  iter="$(count_iters)"
  log "TICK stamp=$TRAIN_STAMP iter=$iter/$ITERS_TARGET train=$(train_running && echo 1 || echo 0) pipeline=$(pipeline_running && echo 1 || echo 0) eval=$(eval_running && echo 1 || echo 0) chain=$CHAIN_DEPTH"
  ensure_train
  ensure_pipeline
  run_auto_next
}

log "autopilot start stamp=$TRAIN_STAMP iters=$ITERS_TARGET interval=${INTERVAL}s"
while true; do
  tick
  sleep "$INTERVAL"
done
