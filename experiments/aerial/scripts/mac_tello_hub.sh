#!/usr/bin/env bash
# Mac is the Tello hub: USB Wi-Fi → Tello AP; en0 stays on phone hotspot for H12/Orin HTTP.
# H12 and Orin open http://<Mac-hotspot-ip>:8090 (do not join TELLO-*).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PORT="${PORT:-8090}"
SSID="${TELLO_SSID:-TELLO-E9F873}"
IFACE="${TELLO_IFACE:-}"
ALLOW_FLY="${ALLOW_FLY:-1}"
PY="${PYTHON:-python3}"

list_wifi() {
  networksetup -listallhardwareports | awk '
    /Hardware Port:/{p=$0}
    /Device:/{d=$2; print p" "d}
  '
}

if [[ -z "${IFACE}" ]]; then
  echo "Mac Wi-Fi devices:"
  list_wifi
  echo ""
  echo "en0 must stay on the phone hotspot. Plug a USB Wi-Fi into the Mac and rerun:"
  echo "  TELLO_IFACE=<device> $0"
  echo "Example: TELLO_IFACE=en8 $0"
  exit 1
fi

echo "Join ${SSID} on ${IFACE}"
networksetup -setairportpower "${IFACE}" on || true
networksetup -setairportnetwork "${IFACE}" "${SSID}" || true
sleep 2
# Tello expects STA 192.168.10.2
sudo ifconfig "${IFACE}" inet 192.168.10.2 netmask 255.255.255.0
sudo route -n add -ifscope "${IFACE}" -net 192.168.10.0/24 -interface "${IFACE}" 2>/dev/null || true
ping -c2 -W2 192.168.10.1 || { echo "Tello 192.168.10.1 not reachable"; exit 1; }

FLY=()
if [[ "${ALLOW_FLY}" == "1" ]]; then
  FLY=(--allow-fly)
  echo "FLY unlocked"
else
  echo "GROUND LOCK"
fi

pkill -f '[p]ython.*tello_ground_console.py' 2>/dev/null || true
sleep 0.5
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}"
nohup "${PY}" experiments/aerial/scripts/tello_ground_console.py \
  --host 0.0.0.0 --port "${PORT}" --local-cmd-port 9000 --local-ip 192.168.10.2 \
  "${FLY[@]}" \
  > /tmp/mac_tello_ground_console.log 2>&1 &
echo "console pid $!"
sleep 3
tail -15 /tmp/mac_tello_ground_console.log
MAC_HOTSPOT="$(ipconfig getifaddr en0 || true)"
echo ""
echo "H12 / Orin URL: http://${MAC_HOTSPOT:-<en0-ip>}:${PORT}/"
echo "H12: do not join TELLO-*. Keep USB adb if hotspot isolation blocks LAN."
echo "Stop: pkill -f tello_ground_console.py"
