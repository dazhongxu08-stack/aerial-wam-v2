#!/usr/bin/env bash
# Supervise 125 overnight stack: Cursor/bridge + C-round train + watch + chain.
# NEVER kill Cursor IDE, web-bridge, cloudflared, or train processes unless restarting
# watch/chain only when they are DEAD.
#
#   nohup bash experiments/aerial/scripts/supervise_125_overnight.sh &
set -euo pipefail
ROOT="${ROOT:-/home/yao/aerial-wam-v2}"
C_STAMP="${C_STAMP:-20260925_2100}"
AB_STAMP="${AB_STAMP:-20260926_after_c}"
INTERVAL="${INTERVAL:-180}"
ART="$ROOT/experiments/aerial/rl/artifacts"
LOG="$ART/logs/supervise_125_overnight_${C_STAMP}.log"
STATUS="$ART/logs/supervise_125_STATUS.md"
PIDFILE="$ART/logs/supervise_125_overnight.pid"
mkdir -p "$ART/logs"
echo $$ >"$PIDFILE"
exec >>"$LOG" 2>&1

ts() { date -Is; }
echo "[$(ts)] supervise_125 start interval=${INTERVAL}s"

check_http() {
  curl -sS -m 3 -o /dev/null -w "%{http_code}" "$1" 2>/dev/null || echo 000
}

ensure_watch() {
  if pgrep -f 'watch_c1234_oa_terminal' >/dev/null 2>&1; then
    echo watch_ok
    return
  fi
  echo "[$(ts)] RESTART watch_c1234"
  cd "$ROOT"
  nohup env STAMP="$C_STAMP" ITERS=100 \
    bash experiments/aerial/scripts/watch_c1234_oa_terminal.sh >/dev/null 2>&1 &
}

ensure_chain() {
  if [[ -f "$ART/logs/c1234_oa_terminal_${C_STAMP}_ALL_DONE.txt" ]] \
     && [[ -f "$ART/logs/ab1234_${AB_STAMP}_DEPLOYED.txt" ]]; then
    echo chain_done
    return
  fi
  if pgrep -f 'chain_c1234_to_ab' >/dev/null 2>&1; then
    echo chain_ok
    return
  fi
  # only restart chain if C not fully done yet OR AB not deployed
  echo "[$(ts)] RESTART chain_c1234_to_ab"
  cd "$ROOT"
  nohup env C_STAMP="$C_STAMP" AB_STAMP="$AB_STAMP" ITERS=100 BC_ITERS=40 \
    bash experiments/aerial/scripts/chain_c1234_to_ab.sh >/dev/null 2>&1 &
}

while true; do
  bridge=$(check_http http://127.0.0.1:8787/api/health)
  cursor_n=$(pgrep -c -f '/usr/share/cursor/cursor' 2>/dev/null || echo 0)
  train_n=$(pgrep -c -f 'train_v4_ac.*c1_ol' 2>/dev/null || echo 0)
  airsim_n=$(pgrep -c -f 'AirVLN-Linux-Shipping' 2>/dev/null || echo 0)
  iter_n=$(ls "$ART/v4_ac_ckpt_mainchannel_c1_ol_bc_${C_STAMP}_noshield_h125"/v4_ac_iter_*.pt 2>/dev/null | wc -l | tr -d ' ')
  last=$(tail -1 "$ART/logs/mainchannel_c1_ol_bc_${C_STAMP}_h125.log" 2>/dev/null | tr '|' '/' | head -c 140)

  w=$(ensure_watch)
  c=$(ensure_chain)

  # Alert only — do not kill Cursor/bridge
  alert=""
  [[ "$bridge" != "200" ]] && alert+="BRIDGE_DOWN "
  [[ "$cursor_n" -lt 1 ]] && alert+="CURSOR_GONE "
  [[ "$train_n" -lt 1 ]] && [[ ! -f "$ART/logs/c1234_oa_terminal_${C_STAMP}_ALL_DONE.txt" ]] && alert+="C1_TRAIN_DEAD "
  [[ "$airsim_n" -lt 1 ]] && alert+="AIRSIM_DOWN "

  {
    echo "# supervise_125 STATUS"
    echo
    echo "updated: $(ts)"
    echo
    echo "| check | value |"
    echo "|-------|-------|"
    echo "| bridge /api/health | $bridge |"
    echo "| cursor procs | $cursor_n |"
    echo "| C1 train_v4_ac | $train_n |"
    echo "| AirSim | $airsim_n |"
    echo "| C1 ckpt iters | $iter_n / 100 |"
    echo "| watch | $w |"
    echo "| chain | $c |"
    echo "| alert | ${alert:-none} |"
    echo
    echo "last_log: $last"
  } >"$STATUS"

  echo "[$(ts)] bridge=$bridge cursor=$cursor_n train=$train_n airsim=$airsim_n iters=$iter_n watch=$w chain=$c alert=${alert:-none}"

  # if AirSim down and train still expected, try recover (safe)
  if [[ "$airsim_n" -lt 1 ]] && [[ "$train_n" -ge 1 || -n "$(pgrep -f run_mainchannel_c1 || true)" ]]; then
    R="$HOME/aerial_airsim_persistent/recover_renderer.sh"
    if [[ -x "$R" ]]; then
      echo "[$(ts)] recover_renderer"
      bash "$R" outdoor >/dev/null 2>&1 || true
    fi
  fi

  sleep "$INTERVAL"
done
