#!/usr/bin/env bash
# Mac: sync stream helper to Orin, start MJPEG server, SSH tunnel, open browser.
#
# Usage:
#   ./experiments/aerial/scripts/mac_orin_camera_view.sh
#   ORIN_SSH=yao@192.168.1.16 ./experiments/aerial/scripts/mac_orin_camera_view.sh
#   ORIN_SSH=wsy@192.168.55.1 CAMERA=0 PORT=8088 ./experiments/aerial/scripts/mac_orin_camera_view.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ORIN_SSH="${ORIN_SSH:-yao@192.168.55.1}"
ORIN_REPO="${ORIN_REPO:-~/aerial-wam-v2}"
ORIN_PY="${ORIN_PY:-~/sim_verify/.venv/bin/python3}"
PORT="${PORT:-8088}"
LOCAL_PORT="${LOCAL_PORT:-$PORT}"
CAMERA="${CAMERA:-0}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"

pick_orin_host() {
  if ssh -o ConnectTimeout=4 -o BatchMode=yes "${ORIN_SSH}" true 2>/dev/null; then
    return 0
  fi
  for candidate in yao@192.168.55.1 yao@192.168.1.16 yao@10.229.66.164; do
    if ssh -o ConnectTimeout=4 -o BatchMode=yes "$candidate" true 2>/dev/null; then
      ORIN_SSH="$candidate"
      return 0
    fi
  done
  echo "ERROR: cannot SSH to Orin. Set ORIN_SSH=user@host" >&2
  exit 1
}

pick_orin_host

echo "Orin target: ${ORIN_SSH}"
echo "Syncing stream module..."
ssh "${ORIN_SSH}" "mkdir -p ${ORIN_REPO}/experiments/aerial/deploy ${ORIN_REPO}/experiments/aerial/scripts"
_sync() {
  local src="$1" dst="$2"
  if scp -q -o ConnectTimeout=10 "$src" "${ORIN_SSH}:${dst}" 2>/dev/null; then
    return 0
  fi
  cat "$src" | ssh "${ORIN_SSH}" "cat > ${dst}"
}
_sync "${ROOT}/experiments/aerial/deploy/orin_camera_stream.py" \
  "${ORIN_REPO}/experiments/aerial/deploy/orin_camera_stream.py"
_sync "${ROOT}/experiments/aerial/scripts/run_orin_camera_stream.sh" \
  "${ORIN_REPO}/experiments/aerial/scripts/run_orin_camera_stream.sh"

echo "Starting camera stream on Orin (port ${PORT})..."
STREAM_EXTRA_ARGS="${STREAM_EXTRA_ARGS:-}"
if [[ "${USE_MOCK:-0}" == "1" ]]; then
  STREAM_EXTRA_ARGS="--mock ${STREAM_EXTRA_ARGS}"
elif ! ssh "${ORIN_SSH}" "test -e /dev/video${CAMERA} || test -e /dev/video/${CAMERA}" 2>/dev/null; then
  echo "WARN: no /dev/video${CAMERA} on Orin — starting --mock (plug USB camera and re-run)"
  STREAM_EXTRA_ARGS="--mock ${STREAM_EXTRA_ARGS}"
fi
ssh "${ORIN_SSH}" "pkill -f 'orin_camera_stream.*--port ${PORT}' 2>/dev/null || true"
ssh -f "${ORIN_SSH}" \
  "chmod +x ${ORIN_REPO}/experiments/aerial/scripts/run_orin_camera_stream.sh && \
   REPO=${ORIN_REPO} PY=${ORIN_PY} HOST=127.0.0.1 PORT=${PORT} CAMERA=${CAMERA} \
   nohup ${ORIN_REPO}/experiments/aerial/scripts/run_orin_camera_stream.sh \
   ${STREAM_EXTRA_ARGS} > /tmp/orin_camera_stream.log 2>&1 &"
sleep 2
ssh "${ORIN_SSH}" "tail -5 /tmp/orin_camera_stream.log 2>/dev/null || true"

if lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Local port ${LOCAL_PORT} already in use — reusing existing tunnel if present."
else
  echo "Opening SSH tunnel localhost:${LOCAL_PORT} -> Orin:${PORT} ..."
  ssh -f -N -L "${LOCAL_PORT}:127.0.0.1:${PORT}" "${ORIN_SSH}"
fi

URL="http://127.0.0.1:${LOCAL_PORT}/"
echo ""
echo "View camera: ${URL}"
echo "Snapshot:    http://127.0.0.1:${LOCAL_PORT}/snapshot.jpg"
echo ""
echo "Stop stream on Orin:"
echo "  ssh ${ORIN_SSH} \"pkill -f orin_camera_stream\""
echo "Stop local tunnel:"
echo "  pkill -f 'ssh -f -N -L ${LOCAL_PORT}:127.0.0.1:${PORT}'"

if [[ "${OPEN_BROWSER}" == "1" ]]; then
  open "${URL}" || true
fi
