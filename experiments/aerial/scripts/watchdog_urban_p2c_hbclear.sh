#!/usr/bin/env bash
# Auto-resume watchdog for shield_contract_v2_hbclear OA FT on 125.
#
# Polls every CHECK_S. If train_v4_ac for this stamp is gone and the last
# saved iter is still < ITERS-1, recover the outdoor renderer and resume from
# the next iter using v4_ac_latest.pt. Progress is read from ckpt filenames
# (not only the train log), so crash mid-renderer-restart cannot lose the
# resume cursor.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-20260921_shield_contract_v2_hbclear}"
ITERS="${ITERS:-32}"
CKPT_REL="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${STAMP}"
LOG_REL="artifacts/train_urban_complex_p2c_${STAMP}.log"
WATCHLOG="artifacts/watchdog_urban_p2c_${STAMP}.log"
CHECK_S="${CHECK_S:-60}"
MAX_RESTARTS="${MAX_RESTARTS:-16}"
SCENE_SH="${SCENE_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
# Wait after detecting death before treating as crash (log flush / brief gaps).
DEAD_GRACE_S="${DEAD_GRACE_S:-45}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHLOG"; }

last_ckpt_iter() {
  local n
  n="$(ls -1 "${CKPT_REL}"/v4_ac_iter_*.pt 2>/dev/null \
    | sed -n 's/.*v4_ac_iter_0*\([0-9][0-9]*\)\.pt/\1/p' \
    | sort -n | tail -1)"
  if [[ -n "$n" ]]; then
    # Force decimal (filenames are zero-padded; bash 0008 == octal error).
    echo "$((10#$n))"
    return 0
  fi
  echo "-1"
}

train_alive() {
  pgrep -f "train_v4_ac.*${STAMP}" >/dev/null 2>&1
}

start_train() {
  local start_iter="$1"
  mkdir -p artifacts "$(dirname "$CKPT_REL")"
  if [[ ! -f "${CKPT_REL}/v4_ac_latest.pt" ]]; then
    log "ERROR: missing ${CKPT_REL}/v4_ac_latest.pt — cannot resume"
    return 1
  fi
  if [[ -x "$SCENE_SH" ]]; then
    log "recover renderer before resume (scene=outdoor)"
    bash "$SCENE_SH" outdoor >>"$LOG_REL" 2>&1 || bash "$SCENE_SH" >>"$LOG_REL" 2>&1 || true
    sleep 25
  fi
  log "launch train start_iter=$start_iter/$ITERS resume=${CKPT_REL}/v4_ac_latest.pt"
  # Reuse depthaux recipe (WM + w_intervention) with this stamp / resume cursor.
  STAMP="$STAMP" \
  ITERS="$ITERS" \
  START_ITER="$start_iter" \
  RESUME_CKPT="${CKPT_REL}/v4_ac_latest.pt" \
  W_COLLISION="${W_COLLISION:-10.0}" \
  W_INTERVENTION="${W_INTERVENTION:-0.1}" \
  RENDERER_RESTART_EVERY="${RENDERER_RESTART_EVERY:-8}" \
  SCENE_SCRIPT="$SCENE_SH" \
  FOCUS_ROUTES="${FOCUS_ROUTES:-0,1,3,4}" \
  ANNO_REL="${ANNO_REL:-experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134.json}" \
    setsid bash experiments/aerial/scripts/train_urban_complex_p2c_depthaux.sh \
    >>"$LOG_REL" 2>&1 &
  disown
  sleep 20
  if train_alive; then
    log "train relaunched OK (pid $(pgrep -f "train_v4_ac.*${STAMP}" | head -1))"
    return 0
  fi
  log "WARN: train not visible yet after launch (will recheck next poll)"
  return 0
}

mkdir -p artifacts
restarts=0
log "watchdog start: stamp=$STAMP iters=$ITERS check=${CHECK_S}s max_restarts=$MAX_RESTARTS ckpt=$CKPT_REL"

# Immediate kick if already dead.
if ! train_alive; then
  last="$(last_ckpt_iter)"
  if [[ "$last" -ge $((ITERS - 1)) ]]; then
    log "already complete: last_iter=$last — exit"
    exit 0
  fi
  next=$((last + 1))
  if [[ "$next" -lt 0 ]]; then next=0; fi
  log "boot: process dead, last_ckpt_iter=$last → start_iter=$next"
  start_train "$next" || true
  restarts=$((restarts + 1))
fi

while true; do
  sleep "$CHECK_S"
  last="$(last_ckpt_iter)"
  if [[ "$last" -ge $((ITERS - 1)) ]]; then
    log "target reached: last_iter=$last >= $((ITERS - 1)) — done"
    break
  fi
  if train_alive; then
    log "alive: last_iter=$last/$((ITERS - 1))"
    continue
  fi
  # Grace: brief gap during renderer restart inside the train process.
  sleep "$DEAD_GRACE_S"
  if train_alive; then
    log "alive after grace: last_iter=$last"
    continue
  fi
  if [[ "$restarts" -ge "$MAX_RESTARTS" ]]; then
    log "DEAD last_iter=$last — restarts exhausted ($restarts/$MAX_RESTARTS), giving up"
    break
  fi
  restarts=$((restarts + 1))
  next=$((last + 1))
  if [[ "$next" -lt 0 ]]; then next=0; fi
  log "DEAD last_iter=$last — restart #$restarts from start_iter=$next"
  start_train "$next" || true
done

log "watchdog exit"
