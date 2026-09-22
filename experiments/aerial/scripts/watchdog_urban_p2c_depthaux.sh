#!/usr/bin/env bash
# Auto-restart watchdog for train_urban_complex_p2c_depthaux.sh (e5 A/B run).
# This project's overnight AC trainings have a well-documented history of
# silent crashes / SSH-session deaths mid-run (RUNBOOK notes, 2026-09-17..19
# logs show 5 restarts in one night). Poll every CHECK_S; if the process is
# gone and the target iter count hasn't been reached, resume from the last
# saved v4_ac_latest.pt at the next iter instead of losing the run.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-20260920_depthaux}"
ITERS="${ITERS:-32}"
CKPT_REL="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${STAMP}"
LOG_REL="artifacts/train_urban_complex_p2c_${STAMP}.log"
WATCHLOG="artifacts/watchdog_urban_p2c_${STAMP}.log"
CHECK_S="${CHECK_S:-120}"
MAX_RESTARTS="${MAX_RESTARTS:-8}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHLOG"; }

restarts=0
log "watchdog start: stamp=$STAMP iters=$ITERS check_every=${CHECK_S}s max_restarts=$MAX_RESTARTS"

while true; do
  sleep "$CHECK_S"
  last_iter=$(grep -oE "corrector:iter [0-9]+:" "$LOG_REL" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  last_iter="${last_iter:--1}"
  if [[ "$last_iter" -ge $((ITERS - 1)) ]]; then
    log "target reached: last_iter=$last_iter >= $((ITERS - 1)) -- done"
    break
  fi
  if pgrep -f "train_v4_ac.*urban_complex_p2c_${STAMP}" >/dev/null 2>&1; then
    log "alive: last_iter=$last_iter"
    continue
  fi
  # process is gone but target not reached -> crashed/killed.
  if [[ "$restarts" -ge "$MAX_RESTARTS" ]]; then
    log "DEAD, last_iter=$last_iter, restarts exhausted ($restarts/$MAX_RESTARTS) -- giving up"
    break
  fi
  restarts=$((restarts + 1))
  next_iter=$((last_iter + 1))
  if [[ ! -f "${CKPT_REL}/v4_ac_latest.pt" ]]; then
    log "DEAD, last_iter=$last_iter, but no checkpoint at ${CKPT_REL}/v4_ac_latest.pt -- cannot resume, giving up"
    break
  fi
  log "DEAD, last_iter=$last_iter -- restart #$restarts from iter=$next_iter (resume ${CKPT_REL}/v4_ac_latest.pt)"
  STAMP="$STAMP" RESUME_CKPT="${CKPT_REL}/v4_ac_latest.pt" START_ITER="$next_iter" \
    setsid bash experiments/aerial/scripts/train_urban_complex_p2c_depthaux.sh \
    >> "artifacts/train_urban_complex_p2c_${STAMP}_restart${restarts}.log" 2>&1 &
  disown
  sleep 15
done

log "watchdog exit"
