#!/usr/bin/env bash
# Keep the blocked-progress Soft run moving.
# If AirSim dies or the trainer stops writing its log, restart the renderer
# and continue from v4_ac_latest.pt until 80 updates, then run the no-shield gate.
set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PYTHON_BIN:-/data/venvs/sim_verify/bin/python}"
ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="20261006_blockedprog"
OUT="$ART/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84"
TRAIN_LOG="$ART/logs/mainchannel_soft_${STAMP}_h84.log"
WATCH_LOG="$ART/logs/blockedprog_watch.log"
STATE="$ART/logs/blockedprog_watch.state"
CFG="$ROOT/configs/_tmp_mainchannel_soft_h84.yaml"
WM="$ART/wm_ckpt_soft_init_antisat_cost/wm_soft_init.pt"
ANN="experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
DEPTH="$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"
TAU="$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt"
EVAL_ANN="experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json"
TARGET_ITERS=80
POLL_SEC=60
TRAIN_HANG_SEC=1500
EVAL_HANG_SEC=9000
LOCK="$ART/logs/blockedprog_watch.lock"

mkdir -p "$ART/logs" "$OUT"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "watch already running" >&2
  exit 0
fi

log() { echo "[watch $(date -Is)] $*" | tee -a "$WATCH_LOG"; }

port_up() { ss -ltn | grep -q ':41451 '; }

train_pids() { pgrep -f '[t]rain_v4_ac' || true; }
parent_pids() { pgrep -f '[r]un_mainchannel_soft_noshield.sh' || true; }
eval_pids() { pgrep -f '[w]am_phase2_long_eval' || true; }

count_completed() {
  if [[ -f "$STATE" ]] && grep -q '^=== RESUME ' "$TRAIN_LOG" 2>/dev/null; then
    # shellcheck disable=SC1090
    local base extra
    base="$(awk -F= '/^BASE_COMPLETED=/{print $2}' "$STATE" | tail -1)"
    base="${base:-0}"
    extra="$(awk 'BEGIN{c=0} /^=== RESUME /{c=0} /corrector:iter /{c++} END{print c+0}' "$TRAIN_LOG")"
    echo $((base + extra))
  else
    if [[ -f "$TRAIN_LOG" ]]; then
      grep -c 'corrector:iter ' "$TRAIN_LOG" || true
    else
      echo 0
    fi
  fi
}

log_age() {
  [[ -f "$1" ]] || { echo 999999; return; }
  echo $(( $(date +%s) - $(stat -c %Y "$1") ))
}

stop_train() {
  local p
  for p in $(parent_pids); do
    kill "$p" 2>/dev/null || true
  done
  for p in $(train_pids); do
    kill "$p" 2>/dev/null || true
  done
  sleep 3
  for p in $(parent_pids) $(train_pids); do
    kill -9 "$p" 2>/dev/null || true
  done
  sleep 1
}

recover_sim() {
  log "recover renderer"
  if [[ ! -x "$RECOVER" ]]; then
    log "FATAL recover script missing: $RECOVER"
    return 1
  fi
  bash "$RECOVER" outdoor >>"$WATCH_LOG" 2>&1 || log "recover script returned $?"
  local i
  for i in $(seq 1 36); do
    if port_up; then
      log "AirSim port 41451 up (try $i)"
      return 0
    fi
    sleep 5
  done
  log "AirSim port still down"
  return 1
}

start_train() {
  local done="$1"
  local remain=$((TARGET_ITERS - done))
  if [[ "$remain" -le 0 ]]; then
    return 0
  fi
  local keep="$OUT/kept_through_${done}"
  mkdir -p "$keep"
  cp -n "$OUT"/v4_ac_iter_*.pt "$OUT"/v4_ac_latest.pt "$OUT"/v4_ac_best.pt "$keep"/ 2>/dev/null || true
  printf 'BASE_COMPLETED=%s\nTARGET=%s\n' "$done" "$TARGET_ITERS" >"$STATE"
  echo "=== RESUME completed=$done remain=$remain $(date -Is) ===" >>"$TRAIN_LOG"
  log "start train remain=$remain init=$OUT/v4_ac_latest.pt"
  PYTHONUNBUFFERED=1 nohup "$PY" -m experiments.aerial.rl.train_v4_ac \
    --config "$CFG" \
    --backend airsim --dynamics torch \
    --wm-ckpt "$WM" \
    --init-actor-ckpt "$OUT/v4_ac_latest.pt" \
    --no-planner --no-shield \
    --iters "$remain" \
    --episodes-per-iter 4 \
    --imagine-batch 16 --imagine-horizon 15 \
    --device cuda \
    --annotation "$ANN" \
    --ckpt-dir "$OUT" \
    --save-every-iter \
    --min-steps-for-best 20 \
    --near-goal-frac 0.25 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 6 \
    --min-spawn-clear-m 5.0 \
    >>"$TRAIN_LOG" 2>&1 &
  log "TRAIN_PID=$!"
}

start_eval() {
  if [[ -f "$ART/logs/blockedprog_eval.done" ]]; then
    return 0
  fi
  local eout="$ART/eval_mainchannel_soft_${STAMP}_h84_gate_noshield"
  mkdir -p "$eout/traj"
  echo "=== EVAL gate $(date -Is) ===" >>"$TRAIN_LOG"
  log "start no-shield gate eval"
  PYTHONUNBUFFERED=1 nohup "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" \
    --annotation "$EVAL_ANN" \
    --wm-ckpt "$WM" \
    --actor-ckpt "$OUT/v4_ac_latest.pt" \
    --depth-ckpt "$DEPTH" \
    --tau-ckpt "$TAU" \
    --goal-feat-mode meter \
    --routes 0,1,3,4 \
    --traj-out "$eout/traj" \
    --out "$eout/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10 --max-steps 600 --success-dist 3 --terminal-pin-rem-m 20 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5 \
    --no-shield \
    >>"$ART/logs/mainchannel_soft_${STAMP}_h84_gate_noshield.log" 2>&1 &
  log "EVAL_PID=$!"
  echo "$!" >"$ART/logs/blockedprog_eval.pid"
}

eval_finished() {
  local elog="$ART/logs/mainchannel_soft_${STAMP}_h84_gate_noshield.log"
  [[ -f "$ART/logs/blockedprog_eval.done" ]] && return 0
  if [[ -f "$elog" ]] && grep -q 'Phase 2 mainline complete' "$elog"; then
    date -Is >"$ART/logs/blockedprog_eval.done"
    log "eval complete"
    return 0
  fi
  return 1
}

log "watch start target=$TARGET_ITERS hang=${TRAIN_HANG_SEC}s"
while true; do
  if eval_finished; then
    log "done — exit"
    exit 0
  fi
  done_n="$(count_completed)"
  train_n="$(train_pids | wc -w)"
  eval_n="$(eval_pids | wc -w)"
  age="$(log_age "$TRAIN_LOG")"
  if port_up; then port=up; else port=down; fi
  log "completed=$done_n/$TARGET_ITERS train=$train_n eval=$eval_n port=$port log_age=${age}s"

  if [[ "$done_n" -ge "$TARGET_ITERS" && "$train_n" -eq 0 ]]; then
    if [[ "$eval_n" -eq 0 ]]; then
      if ! port_up; then
        recover_sim || { sleep "$POLL_SEC"; continue; }
      fi
      start_eval
    else
      eage="$(log_age "$ART/logs/mainchannel_soft_${STAMP}_h84_gate_noshield.log")"
      if [[ "$eage" -gt "$EVAL_HANG_SEC" ]] || ! port_up; then
        log "eval stalled age=${eage}s port=$port — restart eval"
        for p in $(eval_pids); do kill "$p" 2>/dev/null || true; done
        sleep 2
        recover_sim || { sleep "$POLL_SEC"; continue; }
        start_eval
      fi
    fi
    sleep "$POLL_SEC"
    continue
  fi

  need=0
  reason=""
  if ! port_up; then
    need=1
    reason="sim down"
  elif [[ "$train_n" -eq 0 ]]; then
    need=1
    reason="train exited at $done_n"
  elif [[ "$age" -gt "$TRAIN_HANG_SEC" ]]; then
    need=1
    reason="log silent ${age}s"
  fi
  if [[ "$need" -eq 1 ]]; then
    log "resume because $reason"
    stop_train
    if recover_sim; then
      done_n="$(count_completed)"
      start_train "$done_n"
    fi
  fi
  sleep "$POLL_SEC"
done
