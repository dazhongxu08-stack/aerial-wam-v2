#!/usr/bin/env bash
# Host-side watchdog: if JOB dies before DONE, auto-resume. Survives SSH disconnect.
# Usage on host:
#   JOB=actor_ft_125 STAMP=20260922_162118 INTERVAL=90 \
#     nohup bash experiments/aerial/scripts/watch_directional_oa_job.sh &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
JOB="${JOB:?}"
STAMP="${STAMP:?}"
INTERVAL="${INTERVAL:-90}"
ART="$ROOT/experiments/aerial/rl/artifacts"
mkdir -p "$ART/logs"
WLOG="$ART/logs/watch_${JOB}_${STAMP}.log"
PIDFILE="$ART/logs/watch_${JOB}_${STAMP}.pid"
echo $$ >"$PIDFILE"

case "$JOB" in
  actor_ft_125) DONE_MARK="ACTOR-FT-125 DONE"; ALIVE_RE="actor_ft_${STAMP}|train_v4_ac.*${STAMP}|long_eval.*${STAMP}|resume_directional_oa_job" ;;
  cl_ft_125)    DONE_MARK="CL-FT-125 DONE";    ALIVE_RE="cl_ft_${STAMP}|train_v4_ac.*${STAMP}|long_eval.*${STAMP}|resume_directional_oa_job" ;;
  fix3_84)      DONE_MARK="FIX3-84 DONE";      ALIVE_RE="fix3_84_${STAMP}|train_v4_ac.*${STAMP}|long_eval.*${STAMP}|resume_directional_oa_job" ;;
  *) echo "unknown JOB=$JOB" | tee -a "$WLOG"; exit 2 ;;
esac

MASTER=""
case "$JOB" in
  actor_ft_125) MASTER="$ART/logs/directional_oa_actor_ft_${STAMP}.log" ;;
  cl_ft_125)    MASTER="$ART/logs/directional_oa_cl_ft_${STAMP}.log" ;;
  fix3_84)      MASTER="$ART/logs/directional_oa_fix3_84_${STAMP}.log" ;;
esac

echo "[$(date -Is)] watch start job=$JOB stamp=$STAMP interval=${INTERVAL}s" | tee -a "$WLOG"

while true; do
  if [[ -f "$MASTER" ]] && grep -q "$DONE_MARK" "$MASTER"; then
    echo "[$(date -Is)] DONE detected — watch exit" | tee -a "$WLOG"
    rm -f "$PIDFILE"
    exit 0
  fi
  if pgrep -af "$ALIVE_RE" | grep -vE "pgrep|watch_directional_oa_job" >/dev/null; then
    echo "[$(date -Is)] alive ok" >>"$WLOG"
  else
    echo "[$(date -Is)] DEAD — launching resume" | tee -a "$WLOG"
    JOB="$JOB" STAMP="$STAMP" \
      nohup bash "$ROOT/experiments/aerial/scripts/resume_directional_oa_job.sh" \
      >>"$ART/logs/watch_${JOB}_${STAMP}_resume_nohup.log" 2>&1 &
    echo "[$(date -Is)] resume pid=$!" | tee -a "$WLOG"
    sleep 30
  fi
  sleep "$INTERVAL"
done
