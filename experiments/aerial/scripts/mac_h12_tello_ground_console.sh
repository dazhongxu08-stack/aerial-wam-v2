#!/usr/bin/env bash
# Mac USB hub: Orin (Tello radio) + H12 (display/control UI).
# Orin runs the ground console; Mac only bridges; H12 browser shows video.
#
# Topology:
#   Tello AP ←USB-WiFi→ Orin ←USB gadget→ Mac ←USB adb→ H12 screen
#   H12 opens http://127.0.0.1:8090 via: adb reverse → Mac SSH -L → Orin
#
# Default GROUND LOCK. ALLOW_FLY=1 to unlock.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ORIN_SSH="${ORIN_SSH:-orin-usb}"
ORIN_REPO="${ORIN_REPO:-~/aerial-wam-v2}"
ORIN_PY="${ORIN_PY:-~/sim_verify/.venv/bin/python}"
PORT="${PORT:-8090}"
LOCAL_PORT="${LOCAL_PORT:-$PORT}"
ALLOW_FLY="${ALLOW_FLY:-0}"
ADB="${ADB:-}"
OPEN_H12="${OPEN_H12:-0}"
OPEN_MAC_BROWSER="${OPEN_MAC_BROWSER:-0}"

find_adb() {
  if [[ -n "${ADB}" && -x "${ADB}" ]]; then
    return 0
  fi
  for p in /tmp/platform-tools/adb \
           "${HOME}/Library/Android/sdk/platform-tools/adb" \
           "$(command -v adb 2>/dev/null || true)"; do
    if [[ -n "${p}" && -x "${p}" ]]; then
      ADB="${p}"
      return 0
    fi
  done
  echo "ERROR: adb not found. Install platform-tools or set ADB=..." >&2
  exit 1
}

require_h12() {
  "${ADB}" start-server >/dev/null 2>&1 || true
  local line
  line="$("${ADB}" devices -l 2>/dev/null | awk '/model:H12|product:H12/ {print; exit}')"
  if [[ -z "${line}" ]]; then
    echo "ERROR: H12 not in adb devices. Check USB + authorize debugging on H12." >&2
    "${ADB}" devices -l >&2 || true
    exit 1
  fi
  echo "H12: ${line}"
}

echo "Sync console → ${ORIN_SSH}"
ssh -o ConnectTimeout=8 "${ORIN_SSH}" "mkdir -p ${ORIN_REPO}/experiments/aerial/scripts ${ORIN_REPO}/experiments/aerial/rl/env ${ORIN_REPO}/experiments/aerial/deploy"
rsync -az \
  "${ROOT}/experiments/aerial/scripts/tello_ground_console.py" \
  "${ROOT}/experiments/aerial/scripts/orin_tello_wifi_rejoin.sh" \
  "${ORIN_SSH}:${ORIN_REPO}/experiments/aerial/scripts/"
rsync -az \
  "${ROOT}/experiments/aerial/rl/env/tello_client.py" \
  "${ROOT}/experiments/aerial/rl/env/tello_env.py" \
  "${ORIN_SSH}:${ORIN_REPO}/experiments/aerial/rl/env/"
rsync -az \
  "${ROOT}/experiments/aerial/deploy/tello_camera.py" \
  "${ORIN_SSH}:${ORIN_REPO}/experiments/aerial/deploy/"

# Prefer USB Wi-Fi iface for Tello; reconnect if L3 dead.
echo "Ensure Orin↔Tello link"
ssh "${ORIN_SSH}" 'bash -s' <<'EOF'
PASS="${ORIN_SUDO_PASS:-cupcake777}"
s(){ printf "%s\n" "$PASS" | sudo -S -p "" "$@"; }
IFACE=""
for d in $(nmcli -t -f DEVICE,TYPE device 2>/dev/null | awk -F: '$2=="wifi"{print $1}'); do
  if ip -4 -o addr show "$d" 2>/dev/null | grep -q '192.168.10.'; then IFACE=$d; break; fi
done
# fall back to USB Realtek name if known
if [[ -z "${IFACE}" ]]; then
  for d in wlx6c1ff783eb62 $(ls /sys/class/net | grep '^wlx' || true); do
    [[ -e "/sys/class/net/$d" ]] || continue
    IFACE=$d; break
  done
fi
echo "tello_iface=${IFACE:-none}"
if [[ -n "${IFACE}" ]] && ! ping -c1 -W1 192.168.10.1 >/dev/null 2>&1; then
  echo "Tello L3 down — reconnect ${IFACE}"
  s nmcli connection down TELLO-USB 2>/dev/null || true
  sleep 1
  s nmcli device wifi rescan ifname "${IFACE}" 2>/dev/null || true
  sleep 2
  s nmcli connection up TELLO-USB ifname "${IFACE}" 2>/dev/null \
    || s nmcli device wifi connect TELLO-E9F873 ifname "${IFACE}" name TELLO-USB 2>/dev/null || true
  sleep 3
  s ip addr replace 192.168.10.2/24 dev "${IFACE}" 2>/dev/null || true
fi
ping -c2 -W1 192.168.10.1 | tail -2 || true
ip -4 -o addr show | grep 192.168.10 || true
EOF

FLY_FLAG=""
if [[ "${ALLOW_FLY}" == "1" ]]; then
  FLY_FLAG="--allow-fly"
  echo "WARNING: ALLOW_FLY=1 — takeoff unlocked"
else
  echo "GROUND LOCK (default)"
fi

LOCAL_IP="$(ssh "${ORIN_SSH}" "ip -4 -o addr show | awk '/192\\.168\\.10\\./{print \$4}' | head -1 | cut -d/ -f1" || true)"
LOCAL_IP_FLAG=""
if [[ "${LOCAL_IP}" == 192.168.10.* ]]; then
  LOCAL_IP_FLAG="--local-ip ${LOCAL_IP}"
  echo "Bind local IP: ${LOCAL_IP}"
fi

echo "Restart console on Orin :${PORT}"
ssh "${ORIN_SSH}" 'pids=$(pgrep -f "[p]ython.*tello_ground_console.py" || true); if [ -n "$pids" ]; then kill $pids 2>/dev/null || true; sleep 1; fi'
ssh "${ORIN_SSH}" 'fuser -k 11111/udp 2>/dev/null || true; fuser -k '"${PORT}"'/tcp 2>/dev/null || true; sleep 0.5'
pkill -f "ssh.*-L ${LOCAL_PORT}:127.0.0.1:${PORT}" 2>/dev/null || true
sleep 0.3

ssh -f "${ORIN_SSH}" \
  "cd ${ORIN_REPO} && export PYTHONPATH=${ORIN_REPO}:\$PYTHONPATH && \
   nohup ${ORIN_PY} experiments/aerial/scripts/tello_ground_console.py \
     --host 0.0.0.0 --port ${PORT} --local-cmd-port 9000 ${LOCAL_IP_FLAG} ${FLY_FLAG} \
     > /tmp/tello_ground_console.log 2>&1 &"

for i in 1 2 3 4 5 6 7 8 9 10; do
  if ssh "${ORIN_SSH}" "curl -fsS -m2 http://127.0.0.1:${PORT}/api/status" >/tmp/tello_h12_status.json 2>/dev/null; then
    echo "Orin console up ($(head -c 120 /tmp/tello_h12_status.json))..."
    break
  fi
  sleep 2
done

echo "Tunnel Mac 0.0.0.0:${LOCAL_PORT} → Orin :${PORT}  (H12 无 adb，走 Mac 局域网 IP)"
pkill -f "ssh.*-L .*${LOCAL_PORT}:127.0.0.1:${PORT}" 2>/dev/null || true
sleep 0.3
if ! lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  ssh -f -N -o ExitOnForwardFailure=yes -L "0.0.0.0:${LOCAL_PORT}:127.0.0.1:${PORT}" "${ORIN_SSH}"
fi

MAC_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
URL_LAN="http://${MAC_IP:-<Mac-WiFi-IP>}:${LOCAL_PORT}/"

# Optional: adb reverse only if H12 is plugged for debug install
if [[ "${OPEN_H12}" == "1" ]]; then
  find_adb
  if "${ADB}" devices 2>/dev/null | grep -qE 'device$'; then
    "${ADB}" reverse --remove-all 2>/dev/null || true
    "${ADB}" reverse "tcp:${LOCAL_PORT}" "tcp:${LOCAL_PORT}"
    echo "adb reverse also set (debug only)"
  fi
fi

if [[ "${OPEN_MAC_BROWSER}" == "1" ]]; then
  open "${URL_LAN}" 2>/dev/null || true
fi

echo ""
echo "=== H12 正常用法（不连 adb）==="
echo "1) H12 连办公室 WiFi（与 Mac 同网，禁止连 TELLO-*）"
echo "2) Tello GS App 填 Orin URL: ${URL_LAN%/}   点「应用」"
echo "3) 断线重连走该 URL → Mac → Orin"
echo "Stop: ssh ${ORIN_SSH} 'pkill -f tello_ground_console'; pkill -f 'ssh.*${LOCAL_PORT}:127.0.0.1:${PORT}'"
