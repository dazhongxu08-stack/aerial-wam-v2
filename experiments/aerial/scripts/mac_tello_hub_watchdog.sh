#!/usr/bin/env bash
# Keep Mac Tello hub alive: Wi-Fi → Tello, Ethernet → Orin, HTTP :8090 for H12 adb reverse.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export TELLO_SKIP_VIDEO=1
export TELLO_FRAME_JPG=/tmp/tello_latest.jpg
PY="${PYTHON:-python3}"
LOG_C=/tmp/mac_tello_ground_console.log
LOG_S=/tmp/tello_video_sidecar.log

start_sidecar() {
  if pgrep -f 'tello_video_sidecar.py' >/dev/null 2>&1; then
    return 0
  fi
  nohup "${PY}" "${ROOT}/experiments/aerial/scripts/tello_video_sidecar.py" >>"${LOG_S}" 2>&1 &
  echo "$(date '+%H:%M:%S') sidecar $!"
}

start_console() {
  if pgrep -f 'tello_ground_console.py --host' >/dev/null 2>&1; then
    return 0
  fi
  nohup "${PY}" "${ROOT}/experiments/aerial/scripts/tello_ground_console.py" \
    --host 0.0.0.0 --port 8090 --local-cmd-port 9000 --local-ip 192.168.10.2 --allow-fly \
    >>"${LOG_C}" 2>&1 &
  echo "$(date '+%H:%M:%S') console $!"
}

echo "watchdog root=${ROOT}"
while true; do
  start_sidecar
  start_console
  sleep 2
done
