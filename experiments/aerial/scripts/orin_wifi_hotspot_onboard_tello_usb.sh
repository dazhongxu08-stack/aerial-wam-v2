#!/usr/bin/env bash
# Orin: onboard Wi-Fi → phone hotspot; USB Wi-Fi → Tello AP.
# USB Tello is never the default route. Requires ORIN_HOTSPOT_SSID + ORIN_HOTSPOT_PSK.
set -euo pipefail

ONBOARD="${ONBOARD_IFACE:-wlP1p1s0}"
USB="${USB_IFACE:-wlx6c1ff783eb62}"
TELLO_SSID="${TELLO_SSID:-TELLO-E9F873}"
HOTSPOT_SSID="${ORIN_HOTSPOT_SSID:?set ORIN_HOTSPOT_SSID}"
HOTSPOT_PSK="${ORIN_HOTSPOT_PSK:?set ORIN_HOTSPOT_PSK}"
SUDO_PASS="${ORIN_SUDO_PASS:-}"

s() {
  if sudo -n true 2>/dev/null; then
    sudo "$@"
    return
  fi
  printf '%s\n' "${SUDO_PASS}" | sudo -S -p '' "$@"
}

echo "USB ${USB} → ${TELLO_SSID}"
s nmcli device set "${USB}" managed yes || true
s nmcli connection delete TELLO-USB 2>/dev/null || true
s nmcli device wifi connect "${TELLO_SSID}" ifname "${USB}" name TELLO-USB
s nmcli connection modify TELLO-USB \
  ipv4.method manual \
  ipv4.addresses 192.168.10.2/24 \
  ipv4.gateway "" \
  ipv4.never-default yes \
  ipv6.method disabled \
  connection.autoconnect yes \
  connection.interface-name "${USB}"
s nmcli connection up TELLO-USB ifname "${USB}" || true
s ip addr replace 192.168.10.2/24 dev "${USB}" || true

echo "onboard ${ONBOARD} → hotspot"
s nmcli device set "${ONBOARD}" managed yes || true
s ip addr del 192.168.10.2/24 dev "${ONBOARD}" 2>/dev/null || true
s nmcli connection delete HOTSPOT-ONBOARD 2>/dev/null || true
s nmcli device wifi connect "${HOTSPOT_SSID}" password "${HOTSPOT_PSK}" ifname "${ONBOARD}" name HOTSPOT-ONBOARD
s nmcli connection modify HOTSPOT-ONBOARD \
  connection.interface-name "${ONBOARD}" \
  connection.autoconnect yes \
  ipv4.never-default no \
  ipv4.route-metric 50 \
  ipv6.method auto
s nmcli connection modify TELLO-USB ipv4.route-metric 600 || true
s nmcli connection up HOTSPOT-ONBOARD ifname "${ONBOARD}" || true

echo "=== addrs ==="
ip -br -4 addr
echo "=== routes ==="
ip route
echo "=== ping tello ==="
ping -c2 -W1 192.168.10.1 || true
echo "=== ping hotspot gw ==="
ping -c2 -W1 172.20.10.1 || true
