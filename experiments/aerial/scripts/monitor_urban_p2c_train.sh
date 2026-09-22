#!/usr/bin/env bash
# Progress monitor for urban-complex online P2c training on 125.
set -euo pipefail

SSH_HOST="${SSH_HOST:-cursor-125-public}"
STAMP="${STAMP:-20260917_z24}"
LOG_REL="artifacts/train_urban_complex_p2c_${STAMP}.log"
CKPT_REL="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${STAMP}"
ITERS="${ITERS:-32}"

fetch_status() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$SSH_HOST" bash -s "$STAMP" "$ITERS" <<'REMOTE'
STAMP="$1"
ITERS="$2"
LOG=~/aerial-wam-v2/artifacts/train_urban_complex_p2c_${STAMP}.log
CKPT=~/aerial-wam-v2/${CKPT_REL:-experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${STAMP}}
CKPT=~/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${STAMP}
running=$(pgrep -af "train_v4_ac.*urban_complex_p2c_${STAMP}" | head -1 | grep -o 'iter [0-9]*' || true)
last_iter=$(grep -E "corrector:iter [0-9]+:" "$LOG" 2>/dev/null | tail -1 | sed -n 's/.*iter \([0-9]*\): \([0-9]*\) steps.*/iter=\1 steps=\2/p')
spawn_ok=$(grep -c "spawn retry.*ok at z" "$LOG" 2>/dev/null || echo 0)
spawn_skip=$(grep -c "reset spawned in collision" "$LOG" 2>/dev/null || echo 0)
done_json=$(test -f "$LOG" && grep -c '"iters":' "$LOG" 2>/dev/null || echo 0)
train_pid=$(pgrep -f "train_v4_ac.*urban_complex_p2c_${STAMP}" | head -1)
if [[ -z "$train_pid" ]]; then
  state=DONE
else
  state=RUNNING
fi
tail_line=$(tail -1 "$LOG" 2>/dev/null | sed 's/.*INFO://;s/.*WARNING://' | cut -c1-90)
echo "STATE=${state} TARGET=${ITERS} ${last_iter:-iter=?} SPAWN_OK=${spawn_ok} SPAWN_COLL=${spawn_skip} CKPT=$([[ -f ${CKPT}/v4_ac_latest.pt ]] && echo yes || echo no) TAIL=${tail_line:-none}"
REMOTE
}

notify_tick() {
  local msg="$1"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    osascript -e "display notification \"${msg//\"/\\\"}\" with title \"Urban P2c 125\"" 2>/dev/null || true
  fi
  echo "AGENT_LOOP_TICK_urban_p2c {\"prompt\":\"追进度 urban P2c 重训 z24\",\"status\":\"${msg//\"/\\\"}\"}"
}

line="$(fetch_status 2>/dev/null || echo "SSH_FAIL")"
notify_tick "$line"
