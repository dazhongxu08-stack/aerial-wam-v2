#!/usr/bin/env bash
# Overnight watch: keep C1–C4 alive on 125/84/14/11; refresh STATUS.
# Runs on 125 (deploy launches it). Restart any arm that died before DONE.
#
#   STAMP=20260925_2100 bash experiments/aerial/scripts/watch_c1234_oa_terminal.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STAMP="${STAMP:?set STAMP}"
ITERS="${ITERS:-100}"
INTERVAL="${INTERVAL:-120}"
ART="$ROOT/experiments/aerial/rl/artifacts"
STATUS="$ART/logs/c1234_oa_terminal_${STAMP}_STATUS.md"
LOG="$ART/logs/watch_c1234_oa_terminal_${STAMP}.log"
PIDFILE="$ART/logs/watch_c1234_oa_terminal_${STAMP}.pid"
mkdir -p "$ART/logs"
echo $$ >"$PIDFILE"
exec >>"$LOG" 2>&1

KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"
ts() { date -Is; }

echo "[$(ts)] watch start stamp=$STAMP interval=${INTERVAL}s"

is_done() {
  local host=$1 marker=$2
  case "$host" in
    125) grep -q "$marker" "$ART/logs/mainchannel_c1_ol_bc_${STAMP}_h125.log" 2>/dev/null && return 0 ;;
    84) ssh -n -o BatchMode=yes -o ConnectTimeout=12 ubantu@10.229.20.84 \
         "grep -q '$marker' /data/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/mainchannel_c2_wm_escape_${STAMP}_h84.log 2>/dev/null" && return 0 ;;
    14) ssh -n -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" -p 31126 a25689@10.239.121.14 \
         "grep -q '$marker' /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/mainchannel_c3_antistick_${STAMP}_h14.log 2>/dev/null" && return 0 ;;
    11) ssh -n -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" -p 30627 a25689@10.239.121.11 \
         "grep -q '$marker' /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/mainchannel_c4_terminal_${STAMP}_h11.log 2>/dev/null" && return 0 ;;
  esac
  return 1
}

is_alive() {
  local host=$1 pat=$2
  case "$host" in
    125) pgrep -af "$pat" | grep -v pgrep >/dev/null ;;
    84) ssh -n -o BatchMode=yes -o ConnectTimeout=12 ubantu@10.229.20.84 \
         "pgrep -af '$pat' | grep -v pgrep >/dev/null" ;;
    14) ssh -n -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" -p 31126 a25689@10.239.121.14 \
         "pgrep -af '$pat' | grep -v pgrep >/dev/null" ;;
    11) ssh -n -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" -p 30627 a25689@10.239.121.11 \
         "pgrep -af '$pat' | grep -v pgrep >/dev/null" ;;
  esac
}

restart_arm() {
  local host=$1
  echo "[$(ts)] RESTART host=$host"
  case "$host" in
    125)
      cd "$ROOT"; export PYTHONPATH="$ROOT"
      nohup env HOST=125 STAMP="$STAMP" ITERS="$ITERS" SKIP_BASELINE=1 \
        bash experiments/aerial/scripts/run_mainchannel_c1_ol_bc_noshield.sh >/dev/null 2>&1 &
      ;;
    84)
      ssh -o BatchMode=yes ubantu@10.229.20.84 \
        "export STAMP='$STAMP' ITERS='$ITERS'; bash -s" <<'R84'
ROOT=/data/aerial-wam-v2; cd "$ROOT"; export PYTHONPATH="$ROOT"
PYTHON_BIN=/data/venvs/sim_verify/bin/python
nohup env HOST=84 STAMP="$STAMP" ITERS="$ITERS" SKIP_BASELINE=1 \
  PYTHON_BIN="$PYTHON_BIN" OA_PLANNER_MOCK=wm_escape \
  bash experiments/aerial/scripts/run_mainchannel_c2_wm_escape_noshield.sh >/dev/null 2>&1 &
R84
      ;;
    14)
      ssh -o BatchMode=yes -i "$KEY" -p 31126 a25689@10.239.121.14 \
        "export STAMP='$STAMP' ITERS='$ITERS'; bash -s" <<'R14'
ROOT=/home/a25689/aerial-wam-v2; cd "$ROOT"; export PYTHONPATH="$ROOT"
PY=""; [[ -x $ROOT/.venv/bin/python ]] && PY=$ROOT/.venv/bin/python
nohup env HOST=14 STAMP="$STAMP" ITERS="$ITERS" SKIP_BASELINE=1 ${PY:+PYTHON_BIN=$PY} \
  bash experiments/aerial/scripts/run_mainchannel_c3_antistick_noshield.sh >/dev/null 2>&1 &
R14
      ;;
    11)
      ssh -o BatchMode=yes -i "$KEY" -p 30627 a25689@10.239.121.11 \
        "export STAMP='$STAMP' ITERS='$ITERS'; bash -s" <<'R11'
ROOT=/home/a25689/aerial-wam-v2; cd "$ROOT"; export PYTHONPATH="$ROOT"
PY=""; [[ -x $ROOT/.venv/bin/python ]] && PY=$ROOT/.venv/bin/python
nohup env HOST=11 STAMP="$STAMP" ITERS="$ITERS" SKIP_BASELINE=1 NEAR_GOAL_FRAC=0.5 ${PY:+PYTHON_BIN=$PY} \
  bash experiments/aerial/scripts/run_mainchannel_c4_terminal_noshield.sh >/dev/null 2>&1 &
R11
      ;;
  esac
}

tail_line() {
  local host=$1
  case "$host" in
    125) tail -1 "$ART/logs/mainchannel_c1_ol_bc_${STAMP}_h125.log" 2>/dev/null || echo — ;;
    84) ssh -n -o BatchMode=yes -o ConnectTimeout=12 ubantu@10.229.20.84 \
         "tail -1 /data/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/mainchannel_c2_wm_escape_${STAMP}_h84.log 2>/dev/null" || echo — ;;
    14) ssh -n -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" -p 31126 a25689@10.239.121.14 \
         "tail -1 /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/mainchannel_c3_antistick_${STAMP}_h14.log 2>/dev/null" || echo — ;;
    11) ssh -n -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" -p 30627 a25689@10.239.121.11 \
         "tail -1 /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/mainchannel_c4_terminal_${STAMP}_h11.log 2>/dev/null" || echo — ;;
  esac
}

declare -A MARKER PAT
MARKER[125]=C1_OL_BC_NOSHIELD_DONE
MARKER[84]=C2_WM_ESCAPE_NOSHIELD_DONE
MARKER[14]=C3_ANTISTICK_NOSHIELD_DONE
MARKER[11]=C4_TERMINAL_NOSHIELD_DONE
PAT[125]='c1_ol_bc|train_v4_ac.*c1_ol'
PAT[84]='c2_wm_escape|train_v4_ac.*c2_wm'
PAT[14]='c3_antistick|train_v4_ac.*c3_'
PAT[11]='c4_terminal|train_v4_ac.*c4_'

while true; do
  all_done=1
  {
    echo "# C1234 overnight STATUS · $STAMP"
    echo
    echo "updated: $(ts)"
    echo
    echo "| host | arm | state | last_log |"
    echo "|------|-----|-------|----------|"
  } >"$STATUS.tmp"

  for h in 125 84 14 11; do
    arm=C1; [[ $h == 84 ]] && arm=C2; [[ $h == 14 ]] && arm=C3; [[ $h == 11 ]] && arm=C4
    if is_done "$h" "${MARKER[$h]}"; then
      state=DONE
    elif is_alive "$h" "${PAT[$h]}"; then
      state=RUNNING
      all_done=0
    else
      state=DEAD_RESTART
      all_done=0
      restart_arm "$h" || true
    fi
    ll=$(tail_line "$h" | tr '|' '/' | head -c 120)
    echo "| $h | $arm | $state | $ll |" >>"$STATUS.tmp"
  done
  mv "$STATUS.tmp" "$STATUS"
  echo "[$(ts)] tick all_done=$all_done"
  cat "$STATUS"

  if [[ "$all_done" -eq 1 ]]; then
    echo "[$(ts)] ALL FOUR DONE — watch exit"
    echo C1234_ALL_DONE >"$ART/logs/c1234_oa_terminal_${STAMP}_ALL_DONE.txt"
    exit 0
  fi
  sleep "$INTERVAL"
done
