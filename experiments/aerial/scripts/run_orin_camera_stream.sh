#!/usr/bin/env bash
# Start MJPEG camera stream on Orin (bind localhost; use mac_orin_camera_view.sh from Mac).
set -euo pipefail

REPO="${REPO:-$HOME/aerial-wam-v2}"
PY="${PY:-$HOME/sim_verify/.venv/bin/python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8088}"
CAMERA="${CAMERA:-0}"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" -m experiments.aerial.deploy.orin_camera_stream \
  --host "$HOST" \
  --port "$PORT" \
  --camera "$CAMERA" \
  --stream-fps "${STREAM_FPS:-15}" \
  "$@"
