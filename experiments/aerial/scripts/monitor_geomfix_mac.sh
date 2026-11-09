#!/usr/bin/env bash
# Mac ticker: pull geomfix STATUS / PHASE0_VERDICT from 125 every INTERVAL.
#   STAMP=... SSH125=cursor-125-public bash experiments/aerial/scripts/monitor_geomfix_mac.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STAMP="${STAMP:?}"
SSH125="${SSH125:-cursor-125-public}"
INTERVAL="${INTERVAL:-120}"
LOCAL="$ROOT/artifacts/geomfix_${STAMP}"
LOG="$ROOT/artifacts/geomfix_monitor.log"
mkdir -p "$LOCAL"
REMOTE_OUT="~/aerial-wam-v2/experiments/aerial/rl/artifacts/geomfix_${STAMP}"

echo "monitor_geomfix start stamp=$STAMP → $LOG" | tee -a "$LOG"
while true; do
  ts=$(date -Is)
  scp -o ConnectTimeout=40 -o BatchMode=yes \
    "$SSH125:$REMOTE_OUT/STATUS.md" \
    "$SSH125:$REMOTE_OUT/STATUS_LIVE.md" \
    "$SSH125:$REMOTE_OUT/PHASE0_VERDICT.json" \
    "$SSH125:$REMOTE_OUT/PHASE0_VERDICT.md" \
    "$SSH125:$REMOTE_OUT/READY_SOFT.txt" \
    "$LOCAL/" 2>/dev/null || true
  {
    echo "===== $ts stamp=$STAMP ====="
    if [[ -f "$LOCAL/PHASE0_VERDICT.json" ]]; then
      cat "$LOCAL/PHASE0_VERDICT.json"
    else
      echo "(no PHASE0_VERDICT yet)"
    fi
    echo
    [[ -f "$LOCAL/STATUS.md" ]] && cat "$LOCAL/STATUS.md"
    [[ -f "$LOCAL/READY_SOFT.txt" ]] && echo ">>> READY_SOFT <<<"
  } | tee -a "$LOG" | tee "$ROOT/artifacts/geomfix_STATUS.md" >/dev/null
  # also mirror brief into STATUS for quick glance
  {
    echo "# geomfix keep-alive · $(date '+%Y-%m-%d %H:%M') +08"
    echo
    if [[ -f "$LOCAL/PHASE0_VERDICT.json" ]]; then
      python3 - <<PY
import json
from pathlib import Path
j=json.loads(Path("$LOCAL/PHASE0_VERDICT.json").read_text())
print(f"- **READY_SOFT**: {'yes' if j.get('READY_SOFT') else 'no'} (gate={'✓' if j.get('gate') else '✗'} cone={'✓' if j.get('cone') else '✗'} probe={'✓' if j.get('probe_hard') else '✗'} imagine={'✓' if j.get('imagine') else '✗'})")
print(f"- **stamp**: \`$STAMP\`")
print(f"- Soft not started until READY_SOFT.")
PY
    else
      echo "- **READY_SOFT**: pending (no verdict yet)"
      echo "- **stamp**: \`$STAMP\`"
    fi
    echo
    [[ -f "$LOCAL/STATUS.md" ]] && cat "$LOCAL/STATUS.md"
  } >"$ROOT/artifacts/geomfix_STATUS.md"
  if [[ -f "$LOCAL/READY_SOFT.txt" ]]; then
    echo "[$ts] READY_SOFT — monitor idle" | tee -a "$LOG"
    sleep 600
    continue
  fi
  sleep "$INTERVAL"
done
