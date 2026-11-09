#!/usr/bin/env bash
# Launch + supervise the 4-host mainchannel monitor (auto-restart if it dies).
#
#   nohup bash experiments/aerial/scripts/monitor_mainchannel_quad.sh &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
ART="$ROOT/artifacts"
mkdir -p "$ART"
PY="${PYTHON_BIN:-python3}"
INTERVAL="${INTERVAL:-90}"
PIDFILE="$ART/monitor_mainchannel_quad.pid"
SUP_PIDFILE="$ART/monitor_mainchannel_quad_supervisor.pid"
LOG="$ART/monitor_mainchannel_quad_supervisor.log"
echo $$ >"$SUP_PIDFILE"

ts() { date +%Y-%m-%dT%H:%M:%S%z; }

# Stop older dual / collect-results monitors (replaced by quad)
for old in monitor_mainchannel_collect_results.pid monitor_mainchannel_fullfix_dual.pid; do
  if [[ -f "$ART/$old" ]]; then
    opid=$(tr -d ' \n' <"$ART/$old" || true)
    if [[ -n "${opid:-}" ]] && kill -0 "$opid" 2>/dev/null; then
      kill "$opid" 2>/dev/null || true
      echo "[$(ts)] stopped old monitor pid=$opid ($old)" >>"$LOG"
    fi
    rm -f "$ART/$old"
  fi
done
pkill -f 'monitor_mainchannel_collect_results.py' 2>/dev/null || true
pkill -f 'monitor_mainchannel_fullfix_dual.sh' 2>/dev/null || true

echo "[$(ts)] quad supervisor start INTERVAL=${INTERVAL}s" >>"$LOG"

while true; do
  alive=0
  if [[ -f "$PIDFILE" ]]; then
    mpid=$(tr -d ' \n' <"$PIDFILE" || true)
    if [[ -n "${mpid:-}" ]] && kill -0 "$mpid" 2>/dev/null; then
      alive=1
    fi
  fi
  if [[ "$alive" -eq 0 ]]; then
    # also match by cmdline
    if pgrep -f 'monitor_mainchannel_quad.py' >/dev/null 2>&1; then
      alive=1
    fi
  fi
  if [[ "$alive" -eq 0 ]]; then
    echo "[$(ts)] monitor DEAD — restarting" >>"$LOG"
    nohup env INTERVAL="$INTERVAL" \
      "$PY" "$ROOT/experiments/aerial/scripts/monitor_mainchannel_quad.py" \
      >>"$ART/monitor_mainchannel_quad.log" 2>&1 &
    echo "[$(ts)] started monitor pid=$!" >>"$LOG"
  fi
  sleep 60
done
