#!/usr/bin/env bash
# Watchdog for monitor_urban_hard134_overnight_autopilot.sh.
#
# Problem this fixes: on 2026-09-20 the overnight autopilot loop died
# (process exited / a one-shot helper finished without re-entering the loop)
# at 07:44 and nobody noticed until ~16:00+ — the state file said
# GATE_STARTED=0 and the log just stopped, silently, for 8+ hours.
#
# This script is meant to run from cron every 5 minutes. It trusts ONLY the
# heartbeat file (rewritten every tick by the monitor, independent of tick
# success/failure) plus a liveness check on the monitor process itself.
# If either is missing/stale beyond STALE_SEC, it logs a loud ALERT line and
# (unless WATCHDOG_NO_RESTART=1) relaunches the monitor with nohup+setsid+disown
# so it survives the calling shell/SSH session closing.
#
# Usage (install once):
#   crontab -e
#   */5 * * * * /home/yao/aerial-wam-v2/experiments/aerial/scripts/watchdog_urban_hard134.sh >> /home/yao/aerial-wam-v2/artifacts/urban_hard134_watchdog.log 2>&1
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

HEARTBEAT="${HEARTBEAT:-artifacts/urban_hard134_overnight_heartbeat.txt}"
STALE_SEC="${STALE_SEC:-600}"          # heartbeat older than 10 min ⇒ considered dead
MONITOR_SCRIPT="experiments/aerial/scripts/monitor_urban_hard134_overnight_autopilot.sh"
WATCHDOG_LOG="${WATCHDOG_LOG:-artifacts/urban_hard134_watchdog.log}"
DISABLE_FLAG="${DISABLE_FLAG:-artifacts/urban_hard134_watchdog.disabled}"

wlog() { echo "[watchdog $(date '+%F %T')] $*" | tee -a "$WATCHDOG_LOG"; }

if [[ -f "$DISABLE_FLAG" ]]; then
  wlog "watchdog disabled via $DISABLE_FLAG — not checking/restarting (manual pause, e.g. baseline comparison in progress)"
  exit 0
fi

monitor_running() {
  pgrep -f "monitor_urban_hard134_overnight_autopilot.sh" >/dev/null 2>&1
}

heartbeat_age_sec() {
  [[ -f "$HEARTBEAT" ]] || { echo 999999; return; }
  local ts
  ts="$(grep -o 'TS=[0-9]*' "$HEARTBEAT" 2>/dev/null | head -1 | cut -d= -f2)"
  [[ -n "${ts:-}" ]] || { echo 999999; return; }
  echo $(( $(date +%s) - ts ))
}

start_monitor() {
  wlog "RESTART: launching $MONITOR_SCRIPT (nohup+setsid+disown)"
  nohup setsid bash "$MONITOR_SCRIPT" >>"artifacts/urban_hard134_overnight_autopilot.log" 2>&1 &
  disown
  sleep 3
  if monitor_running; then
    wlog "RESTART OK — pid(s): $(pgrep -f "$MONITOR_SCRIPT" | tr '\n' ' ')"
  else
    wlog "RESTART FAILED — monitor process not found after launch attempt"
  fi
}

age="$(heartbeat_age_sec)"
running="$(monitor_running && echo 1 || echo 0)"

if [[ "$running" == "1" && "$age" -lt "$STALE_SEC" ]]; then
  wlog "OK monitor alive, heartbeat age=${age}s"
  exit 0
fi

wlog "ALERT monitor unhealthy — running=$running heartbeat_age=${age}s (threshold=${STALE_SEC}s)"
if [[ "$running" == "1" && "$age" -ge "$STALE_SEC" ]]; then
  wlog "process is running but heartbeat stale — killing before restart (likely wedged)"
  pkill -f "monitor_urban_hard134_overnight_autopilot.sh" 2>/dev/null || true
  sleep 2
fi
start_monitor
