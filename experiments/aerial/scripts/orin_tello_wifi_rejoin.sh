#!/usr/bin/env bash
# Recreate open Tello AP profile on Orin (run as wsy with sudo).
set -euo pipefail
PASS="${WSY_PASS:-wsy}"
IFACE="${WIFI_IFACE:-wlP1p1s0}"
SSID="${TELLO_SSID:-TELLO-E9F873}"

s() { printf '%s\n' "$PASS" | sudo -S -p '' "$@"; }

echo "== disconnect =="
s nmcli device disconnect "$IFACE" || true
sleep 1
echo "== delete old profile =="
s nmcli connection delete "$SSID" || true
sleep 1
echo "== rescan =="
s nmcli device wifi rescan ifname "$IFACE" || true
sleep 2
echo "== scan hits =="
s nmcli -t -f SSID,SIGNAL,SECURITY device wifi list ifname "$IFACE" | grep -i TELLO || true
echo "== connect open AP =="
s nmcli device wifi connect "$SSID" ifname "$IFACE" name "$SSID"
sleep 2
echo "== static IP =="
s nmcli connection modify "$SSID" \
  ipv4.method manual \
  ipv4.addresses 192.168.10.2/24 \
  ipv4.never-default yes \
  connection.autoconnect yes \
  connection.autoconnect-priority 100 \
  802-11-wireless.band bg
s nmcli connection up "$SSID"
sleep 3
echo "== status =="
nmcli -t -f GENERAL.CONNECTION,GENERAL.STATE device show "$IFACE" | head -2
ip -4 -o addr show "$IFACE" || true
ping -c 3 -W 2 192.168.10.1 || true
