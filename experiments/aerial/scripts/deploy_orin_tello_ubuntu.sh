#!/usr/bin/env bash
# Deploy Tello ground console onto this ubuntu Orin (Lumos box).
# USB Wi-Fi stays on the phone hotspot for H12↔dog; Tello uses onboard Wi-Fi
# when a TELLO-* AP is visible. Console binds 0.0.0.0:8090, ground-locked.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ORIN="${ORIN:-ubuntu@172.20.10.4}"
ORIN_REPO="${ORIN_REPO:-/home/ubuntu/aerial-wam-v2}"
PORT="${PORT:-8090}"

echo "Sync Tello stack → ${ORIN}:${ORIN_REPO}"
ssh -o ConnectTimeout=8 "${ORIN}" "mkdir -p \
  ${ORIN_REPO}/experiments/aerial/scripts \
  ${ORIN_REPO}/experiments/aerial/rl/env \
  ${ORIN_REPO}/experiments/aerial/deploy"

rsync -az \
  "${ROOT}/experiments/aerial/__init__.py" \
  "${ORIN}:${ORIN_REPO}/experiments/aerial/"
rsync -az \
  "${ROOT}/experiments/aerial/rl/__init__.py" \
  "${ORIN}:${ORIN_REPO}/experiments/aerial/rl/"
# Avoid fat env/__init__.py (pulls AirSim obs).
printf '' | ssh "${ORIN}" "cat > ${ORIN_REPO}/experiments/aerial/rl/env/__init__.py"
rsync -az \
  "${ROOT}/experiments/aerial/rl/env/tello_client.py" \
  "${ORIN}:${ORIN_REPO}/experiments/aerial/rl/env/"
rsync -az \
  "${ROOT}/experiments/aerial/deploy/__init__.py" \
  "${ROOT}/experiments/aerial/deploy/tello_camera.py" \
  "${ORIN}:${ORIN_REPO}/experiments/aerial/deploy/"
rsync -az \
  "${ROOT}/experiments/aerial/scripts/tello_ground_console.py" \
  "${ROOT}/experiments/aerial/scripts/orin_tello_wifi_watchdog.sh" \
  "${ROOT}/experiments/aerial/scripts/orin_tello_wifi_rejoin.sh" \
  "${ROOT}/experiments/aerial/scripts/run_orin_vgoal_tello.sh" \
  "${ORIN}:${ORIN_REPO}/experiments/aerial/scripts/"
ssh "${ORIN}" "chmod +x ${ORIN_REPO}/experiments/aerial/scripts/*.sh"

echo "Done sync. Start console on Orin with:"
echo "  cd ${ORIN_REPO} && PYTHONPATH=${ORIN_REPO} python3 experiments/aerial/scripts/tello_ground_console.py --host 0.0.0.0 --port ${PORT} --local-cmd-port 9000"
