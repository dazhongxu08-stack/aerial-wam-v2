#!/usr/bin/env bash
# ESP32 Wi-Fi setup helper (DroneBridge for ESP32).
#
# Usage:
#   ./esp32_wifi_setup.sh detect          # list serial ports, guess ESP vs Pixhawk
#   ./esp32_wifi_setup.sh ap-info         # print default DroneBridge AP credentials
#   ./esp32_wifi_setup.sh add-orin UDP_IP # register Orin as UDP target via DroneBridge API
#
set -euo pipefail

ORIN_IP="${ORIN_IP:-}"
ESP_API="${ESP_API:-http://192.168.2.1}"
UDP_PORT="${UDP_PORT:-14550}"

detect() {
  echo "=== USB serial devices (Mac) ==="
  ls -1 /dev/cu.usb* /dev/cu.wchusb* /dev/cu.SLAB* /dev/cu.usbserial* 2>/dev/null || true
  echo ""
  echo "Pixhawk USB 多为: cu.usbmodem*  (当前飞控 ArduPilot 用此口)"
  echo "ESP32 常见为:     cu.usbserial* / cu.wchusbserial* / cu.SLAB_USBtoUART"
  echo ""
  for p in /dev/cu.usbmodem*; do
    [[ -e "$p" ]] || continue
    if python3 -c "
from pymavlink import mavutil
m=mavutil.mavlink_connection('${p//cu./tty.}', baud=57600)
hb=m.wait_heartbeat(timeout=2)
exit(0 if hb else 1)
" 2>/dev/null; then
      echo "  $p  -> MAVLink heartbeat (likely Pixhawk, not ESP32 flash port)"
    fi
  done
  echo ""
  echo "Next: flash DroneBridge if needed, then: $0 ap-info"
}

ap_info() {
  cat <<'EOF'
=== DroneBridge default Wi-Fi AP ===
  SSID:     DroneBridge for ESP32
  Password: dronebridge
  Web UI:   http://192.168.2.1

Steps:
  1. Power ESP32 (USB). Connect Mac Wi-Fi to the SSID above.
  2. Open http://192.168.2.1 → Settings → WiFi Client.
  3. Enter your router / Orin LAN SSID + password → Save → Reboot.
  4. Note ESP's new IP from router DHCP list.
  5. On Orin: orin_fc_monitor --port udpin:0.0.0.0:14550

Full runbook: docs/handover/RUNBOOK_esp32_wifi_mavlink.md
EOF
}

add_orin() {
  local ip="${1:-$ORIN_IP}"
  if [[ -z "$ip" ]]; then
    echo "Usage: ORIN_IP=10.229.20.127 $0 add-orin"
    echo "   or: $0 add-orin 10.229.20.127"
    exit 2
  fi
  echo "Registering UDP target ${ip}:${UDP_PORT} on ${ESP_API} ..."
  curl -fsS -X POST "${ESP_API}/api/settings/clients/udp" \
    -H 'Content-Type: application/json' \
    -d "{\"ip\":\"${ip}\",\"port\":${UDP_PORT},\"save\":true}"
  echo ""
  echo "OK. Reboot ESP if traffic does not appear."
}

case "${1:-detect}" in
  detect) detect ;;
  ap-info|ap) ap_info ;;
  add-orin) add_orin "${2:-}" ;;
  *)
    echo "Usage: $0 detect|ap-info|add-orin [ORIN_IP]"
    exit 2
    ;;
esac
