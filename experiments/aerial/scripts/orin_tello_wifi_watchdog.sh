#!/usr/bin/env bash
# Orin-side Tello Wi-Fi watchdog: keep wlP1p1s0 on TELLO-* and recover when the
# AP vanishes (common after ~7–10 min: ssid-not-found / link timed out).
#
# Run on Orin::
#   ./experiments/aerial/scripts/orin_tello_wifi_watchdog.sh
#
# Env: TELLO_SSID_PREFIX WIFI_IFACE CHECK_EVERY_S PING_IP WSY_PASS
set -euo pipefail

TELLO_SSID_PREFIX="${TELLO_SSID_PREFIX:-TELLO}"
WIFI_IFACE="${WIFI_IFACE:-wlP1p1s0}"
CHECK_EVERY_S="${CHECK_EVERY_S:-5}"
PING_IP="${PING_IP:-192.168.10.1}"
LOG="${LOG:-/tmp/orin_tello_wifi_watchdog.log}"
WSY_PASS="${WSY_PASS:-wsy}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

sudo_nm() {
  if sudo -n true 2>/dev/null; then
    sudo nmcli "$@"
    return
  fi
  local qargs
  qargs="$(printf '%q ' "$@")"
  # shellcheck disable=SC2029
  printf '%s\n' "$WSY_PASS" | su - wsy -c "printf '%s\\n' '${WSY_PASS}' | sudo -S -p '' nmcli ${qargs}"
}

connected_to_tello() {
  local raw ssid
  raw="$(nmcli -t -f GENERAL.CONNECTION device show "$WIFI_IFACE" 2>/dev/null | head -1 || true)"
  # nmcli -t -f FIELD may print "FIELD:value" or just "value".
  ssid="${raw#GENERAL.CONNECTION:}"
  ssid="${ssid#"${ssid%%[![:space:]]*}"}"
  [[ -n "$ssid" && "$ssid" == ${TELLO_SSID_PREFIX}* ]]
}

has_tello_ip() {
  ip -4 -o addr show "${WIFI_IFACE}" 2>/dev/null | grep -q '192.168.10.'
}

link_ok() {
  # Do NOT UDP-probe the Tello command port: the ground console already binds
  # local_cmd_port and owns the reply lock. Extra "command" probes look like
  # dead links and cause reconnect thrash.
  connected_to_tello && has_tello_ip
}

ensure_tello_ip() {
  if has_tello_ip; then
    return 0
  fi
  if sudo -n true 2>/dev/null; then
    sudo ip addr replace 192.168.10.2/24 dev "${WIFI_IFACE}" || true
    sudo ip link set "${WIFI_IFACE}" up || true
  else
    printf '%s\n' "${WSY_PASS}" | su - wsy -c "printf '%s\\n' '${WSY_PASS}' | sudo -S -p '' ip addr replace 192.168.10.2/24 dev ${WIFI_IFACE}" 2>/dev/null || true
    printf '%s\n' "${WSY_PASS}" | su - wsy -c "printf '%s\\n' '${WSY_PASS}' | sudo -S -p '' ip link set ${WIFI_IFACE} up" 2>/dev/null || true
  fi
  has_tello_ip
}

scan_tello_ssid() {
  sudo_nm device wifi rescan ifname "$WIFI_IFACE" >/dev/null 2>&1 || true
  sleep 2
  nmcli -t -f SSID,SIGNAL device wifi list ifname "$WIFI_IFACE" 2>/dev/null \
    | awk -F: -v p="$TELLO_SSID_PREFIX" 'index($1,p)==1 {print; exit}'
}

harden_nm_profiles() {
  sudo_nm connection modify TELLO-E9F873 \
    connection.autoconnect yes \
    connection.autoconnect-priority 100 \
    connection.autoconnect-retries 0 \
    ipv4.method manual \
    ipv4.addresses 192.168.10.2/24 \
    ipv4.never-default yes \
    802-11-wireless.band bg \
    2>/dev/null || true
  for c in autel Guest USER_1802_5G wsy; do
    sudo_nm connection modify "$c" connection.autoconnect no 2>/dev/null || true
  done
  log "NM harden: TELLO priority=100 band=2.4G; office autoconnect=no"
}

reconnect() {
  local hit ssid state i
  # Don't stomp an in-flight association.
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
harden_nm_profiles || true

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
