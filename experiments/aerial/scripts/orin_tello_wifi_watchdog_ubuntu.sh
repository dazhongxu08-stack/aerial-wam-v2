#!/usr/bin/env bash
# Tello Wi-Fi watchdog for ubuntu Orin: onboard wlP1p1s0 only.
# Leaves USB hotspot (wlx*) for H12 ↔ Lumos.
set -euo pipefail

TELLO_SSID_PREFIX="${TELLO_SSID_PREFIX:-TELLO}"
WIFI_IFACE="${WIFI_IFACE:-wlP1p1s0}"
CHECK_EVERY_S="${CHECK_EVERY_S:-5}"
LOG="${LOG:-/tmp/orin_tello_wifi_watchdog.log}"
SUDO_PASS="${ORIN_SUDO_PASS:-${WSY_PASS:-}}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

sudo_nm() {
  if sudo -n true 2>/dev/null; then
    sudo nmcli "$@"
    return
  fi
  printf '%s\n' "${SUDO_PASS}" | sudo -S -p '' nmcli "$@"
}

sudo_ip() {
  if sudo -n true 2>/dev/null; then
    sudo ip "$@"
    return
  fi
  printf '%s\n' "${SUDO_PASS}" | sudo -S -p '' ip "$@"
}

connected_to_tello() {
  local raw ssid
  raw="$(nmcli -t -f GENERAL.CONNECTION device show "$WIFI_IFACE" 2>/dev/null | head -1 || true)"
  ssid="${raw#GENERAL.CONNECTION:}"
  [[ -n "$ssid" && "$ssid" == ${TELLO_SSID_PREFIX}* ]]
}

has_tello_ip() {
  ip -4 -o addr show "${WIFI_IFACE}" 2>/dev/null | grep -q '192.168.10.'
}

link_ok() {
  connected_to_tello && has_tello_ip
}

ensure_tello_ip() {
  if has_tello_ip; then
    return 0
  fi
  sudo_ip addr replace 192.168.10.2/24 dev "${WIFI_IFACE}" || true
  sudo_ip link set "${WIFI_IFACE}" up || true
  has_tello_ip
}

scan_tello_ssid() {
  sudo_nm device wifi rescan ifname "$WIFI_IFACE" >/dev/null 2>&1 || true
  sleep 2
  nmcli -t -f SSID,SIGNAL device wifi list ifname "$WIFI_IFACE" 2>/dev/null \
    | awk -F: -v p="$TELLO_SSID_PREFIX" 'index($1,p)==1 {print; exit}'
}

reconnect() {
  local hit ssid state i
  state="$(nmcli -t -f GENERAL.STATE device show "${WIFI_IFACE}" 2>/dev/null | head -1 || true)"
  if echo "${state}" | grep -qiE 'configuring|connecting|preparing'; then
    log "reconnect: skip — device busy (${state})"
    return 1
  fi
  hit="$(scan_tello_ssid || true)"
  if [[ -z "${hit}" ]]; then
    log "reconnect: Tello SSID not in scan"
    return 1
  fi
  ssid="${hit%%:*}"
  log "reconnect: connecting to ${ssid} (signal ${hit#*:})"
  sudo_nm device wifi connect "${ssid}" ifname "${WIFI_IFACE}" || return 1
  for i in 1 2 3 4 5 6; do
    sleep 1
    ensure_tello_ip || true
    if link_ok; then
      log "reconnect: OK $(ip -br addr show "${WIFI_IFACE}" | awk '{print $3}')"
      return 0
    fi
  done
  log "reconnect: associated but no usable 192.168.10.x yet"
  return 1
}

log "watchdog start iface=${WIFI_IFACE} prefix=${TELLO_SSID_PREFIX}"
fail_streak=0
backoff_s=5
while true; do
  if connected_to_tello; then
    ensure_tello_ip || true
  fi
  if link_ok; then
    fail_streak=0
    backoff_s=5
  else
    fail_streak=$((fail_streak + 1))
    raw="$(nmcli -t -f GENERAL.CONNECTION device show "${WIFI_IFACE}" 2>/dev/null | head -1 || true)"
    ssid_now="${raw#GENERAL.CONNECTION:}"
    log "link bad streak=${fail_streak} conn=${ssid_now:-none} ip=$(has_tello_ip && echo ok || echo miss)"
    reconnect || true
    sleep "${backoff_s}"
    if [[ "${backoff_s}" -lt 30 ]]; then
      backoff_s=$((backoff_s + 5))
    fi
    continue
  fi
  sleep "${CHECK_EVERY_S}"
done
