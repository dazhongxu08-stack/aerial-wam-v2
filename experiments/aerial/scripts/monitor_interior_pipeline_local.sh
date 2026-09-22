#!/usr/bin/env bash
# Local loop: upload monitor + tick status + trigger remote resume on idle.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Offsite only — do not use LAN cursor-125 (10.229.20.125).
SSH_HOST="${SSH_HOST:-cursor-125-public}"
STAMP="${STAMP:-20260916_night}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"

tick_once() {
  ssh -o BatchMode=yes -o ConnectTimeout=35 "$SSH_HOST" bash -s <<REMOTE || echo SSH_FAIL
set -euo pipefail
cd ~/aerial-wam-v2
source experiments/aerial/scripts/env_4090.sh
STAMP=${STAMP}
# ensure monitor running
if ! pgrep -f monitor_interior_pipeline_resume.sh >/dev/null; then
  chmod +x experiments/aerial/scripts/monitor_interior_pipeline_resume.sh \
    experiments/aerial/scripts/run_interior_pipeline_from_checkpoint.sh 2>/dev/null || true
  nohup bash experiments/aerial/scripts/monitor_interior_pipeline_resume.sh \
    >> artifacts/monitor_interior_pipeline_${STAMP}.nohup.log 2>&1 &
  echo "started_monitor pid=\$!"
fi
# one monitor tick inline if scripts missing
bash experiments/aerial/scripts/monitor_interior_pipeline_resume.sh &
MPID=\$!
sleep 3
kill \$MPID 2>/dev/null || true
REMOTE
}

echo "[local-monitor] START interval=${INTERVAL_SEC}s host=$SSH_HOST"
while true; do
  # upload on first successful ssh
  tar czf - -C "$ROOT" \
    experiments/aerial/scripts/monitor_interior_pipeline_resume.sh \
    experiments/aerial/scripts/run_interior_pipeline_from_checkpoint.sh \
    2>/dev/null | ssh -o BatchMode=yes -o ConnectTimeout=35 "$SSH_HOST" \
      "cd ~/aerial-wam-v2 && tar xzf - 2>/dev/null" || true
  tick_once
  sleep "$INTERVAL_SEC"
done
