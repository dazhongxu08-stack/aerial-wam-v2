#!/usr/bin/env bash
# Mac → Orin USB: start Tello ground console (camera + keyboard), tunnel, open browser.
# Default GROUND LOCK (no takeoff). Pass ALLOW_FLY=1 to unlock.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ORIN_SSH="${ORIN_SSH:-orin-usb}"
ORIN_REPO="${ORIN_REPO:-~/aerial-wam-v2}"
ORIN_PY="${ORIN_PY:-~/sim_verify/.venv/bin/python}"
PORT="${PORT:-8090}"
LOCAL_PORT="${LOCAL_PORT:-$PORT}"
ALLOW_FLY="${ALLOW_FLY:-0}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"

echo "Sync console → ${ORIN_SSH}"
ssh -o ConnectTimeout=8 "${ORIN_SSH}" "mkdir -p ${ORIN_REPO}/experiments/aerial/scripts ${ORIN_REPO}/experiments/aerial/rl/env ${ORIN_REPO}/experiments/aerial/deploy"
rsync -az \
  "${ROOT}/experiments/aerial/scripts/tello_ground_console.py" \
  "${ROOT}/experiments/aerial/scripts/orin_tello_wifi_watchdog.sh" \
  "${ROOT}/experiments/aerial/scripts/mac_tello_ground_console.sh" \
  "${ORIN_SSH}:${ORIN_REPO}/experiments/aerial/scripts/"
rsync -az \
  "${ROOT}/experiments/aerial/rl/env/tello_client.py" \
  "${ROOT}/experiments/aerial/rl/env/tello_env.py" \
  "${ORIN_SSH}:${ORIN_REPO}/experiments/aerial/rl/env/"
rsync -az \
  "${ROOT}/experiments/aerial/deploy/tello_camera.py" \
  "${ORIN_SSH}:${ORIN_REPO}/experiments/aerial/deploy/"
ssh "${ORIN_SSH}" "chmod +x ${ORIN_REPO}/experiments/aerial/scripts/orin_tello_wifi_watchdog.sh"

# Wi-Fi watchdog (auto-reconnect when Tello AP drops)
echo "Restart Tello Wi-Fi watchdog on Orin"
ssh "${ORIN_SSH}" 'pids=$(pgrep -f "[o]rin_tello_wifi_watchdog" || true); if [ -n "$pids" ]; then kill $pids 2>/dev/null || true; sleep 1; fi'
ssh -f "${ORIN_SSH}" \
  "nohup ${ORIN_REPO}/experiments/aerial/scripts/orin_tello_wifi_watchdog.sh \
     > /tmp/orin_tello_wifi_watchdog.log 2>&1 &"
sleep 2
ssh "${ORIN_SSH}" "tail -8 /tmp/orin_tello_wifi_watchdog.log 2>/dev/null || true"

FLY_FLAG=""
if [[ "${ALLOW_FLY}" == "1" ]]; then
  FLY_FLAG="--allow-fly"
  echo "WARNING: ALLOW_FLY=1 — takeoff unlocked"
else
  echo "GROUND LOCK (default) — camera + keys mapped, takeoff blocked"
fi

LOCAL_IP_FLAG=""
ORIN_TELLO_IP="$(ssh "${ORIN_SSH}" "ip -4 -o addr show wlP1p1s0 2>/dev/null | awk '{print \$4}' | cut -d/ -f1" || true)"
if [[ "${ORIN_TELLO_IP}" == 192.168.10.* ]]; then
  LOCAL_IP_FLAG="--local-ip ${ORIN_TELLO_IP}"
  echo "Bind local IP: ${ORIN_TELLO_IP}"
fi

echo "Restart console on Orin :${PORT}"
ssh "${ORIN_SSH}" 'pids=$(pgrep -f "[p]ython.*tello_ground_console.py" || true); if [ -n "$pids" ]; then kill $pids 2>/dev/null || true; sleep 1; fi'
ssh "${ORIN_SSH}" 'pids=$(pgrep -f "[t]ello_camera --worker" || true); if [ -n "$pids" ]; then kill -9 $pids 2>/dev/null || true; sleep 1; fi'
ssh "${ORIN_SSH}" 'fuser -k 11111/udp 2>/dev/null || true; sleep 0.5'
# Drop stale local tunnel so we rebind cleanly
pkill -f "ssh.*-L ${LOCAL_PORT}:127.0.0.1:${PORT}" 2>/dev/null || true
sleep 0.5
ssh -f "${ORIN_SSH}" \
  "cd ${ORIN_REPO} && export PYTHONPATH=${ORIN_REPO}:\$PYTHONPATH && \
   nohup ${ORIN_PY} experiments/aerial/scripts/tello_ground_console.py \
     --host 127.0.0.1 --port ${PORT} --local-cmd-port 9000 ${LOCAL_IP_FLAG} ${FLY_FLAG} \
     > /tmp/tello_ground_console.log 2>&1 &"

sleep 4
ssh "${ORIN_SSH}" "tail -20 /tmp/tello_ground_console.log 2>/dev/null || true"

if lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Local :${LOCAL_PORT} already listening — reusing"
else
  echo "Tunnel localhost:${LOCAL_PORT} → Orin:${PORT}"
  ssh -f -N -L "${LOCAL_PORT}:127.0.0.1:${PORT}" "${ORIN_SSH}"
fi

URL="http://127.0.0.1:${LOCAL_PORT}/"
echo ""
echo "Open: ${URL}"
echo "Stop: ssh ${ORIN_SSH} \"pkill -f tello_ground_console\""
if [[ "${OPEN_BROWSER}" == "1" ]]; then
  open "${URL}" 2>/dev/null || true
fi
