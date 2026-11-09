#!/usr/bin/env bash
# Host-side watchdog for mainchannel fullfix. Survives SSH disconnect.
# If the pipeline dies before MAINCHANNEL_FULLFIX_ALL_DONE, auto-resume
# with APPEND / SKIP_* inferred from DS/OUT progress.
#
#   HOST=84 STAMP=20260923_1218 INTERVAL=90 \
#     nohup bash experiments/aerial/scripts/watch_mainchannel_fullfix.sh &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
HOST="${HOST:?}"
STAMP="${STAMP:?}"
INTERVAL="${INTERVAL:-90}"
ART="$ROOT/experiments/aerial/rl/artifacts"
DS="${DS:-$ART/dataset_urban_inland_r01_depth_scene_${STAMP}}"
# fallback if older stamp used r014 naming
if [[ ! -d "$DS" && -d "$ART/dataset_urban_inland_r014_depth_scene_${STAMP}" ]]; then
  DS="$ART/dataset_urban_inland_r014_depth_scene_${STAMP}"
fi
OUT="${OUT:-$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}_expert_bc_h${HOST}}"
MASTER="$ART/logs/mainchannel_fullfix_${STAMP}_h${HOST}.log"
WLOG="$ART/logs/watch_mainchannel_fullfix_${STAMP}_h${HOST}.log"
PIDFILE="$ART/logs/watch_mainchannel_fullfix_${STAMP}_h${HOST}.pid"
DONE_MARK="MAINCHANNEL_FULLFIX_ALL_DONE"
mkdir -p "$ART/logs"
echo $$ >"$PIDFILE"

echo "[$(date -Is)] watch start HOST=$HOST STAMP=$STAMP DS=$DS INTERVAL=${INTERVAL}s" | tee -a "$WLOG"

alive() {
  # Any phase of this stamp, or the wrapper / resume itself.
  pgrep -af "mainchannel_fullfix_${STAMP}|collect_depth_scene_expert_dataset.*${STAMP}|train_v4_ac.*${STAMP}|hbclear_diag_${STAMP}|run_mainchannel_fullfix" \
    | grep -vE "pgrep|watch_mainchannel_fullfix" >/dev/null 2>&1
}

resume() {
  local skip_c=0 skip_b=0 append=0
  if [[ -f "$DS/AUTO_REVIEW.json" ]]; then
    if python3 -c "import json,sys; s=json.load(open('$DS/AUTO_REVIEW.json')).get('summary') or {}; sys.exit(0 if s.get('gate_passed') else 1)" 2>/dev/null; then
      skip_c=1
    else
      append=1
    fi
  fi
  if [[ -f "$OUT/v4_ac_best.pt" || -f "$OUT/v4_ac_latest.pt" ]]; then
    if grep -q "BC_TRAIN_DONE\|MAINCHANNEL_FULLFIX_ALL_DONE" "$MASTER" 2>/dev/null; then
      skip_b=1
    fi
  fi
  echo "[$(date -Is)] RESUME skip_collect=$skip_c append=$append skip_bc=$skip_b" | tee -a "$WLOG"
  # Recover renderer before relaunch
  RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
  if [[ -x "$RECOVER" ]]; then
    bash "$RECOVER" outdoor >/dev/null 2>&1 || true
    sleep 8
  fi
  # Preserve route/gate knobs from env or state file
  STATE="$ART/logs/mainchannel_fullfix_${STAMP}_h${HOST}.env"
  # shellcheck disable=SC1090
  [[ -f "$STATE" ]] && source "$STATE" || true
  nohup env HOST="$HOST" STAMP="$STAMP" DS="$DS" OUT="$OUT" \
    SKIP_COLLECT="$skip_c" SKIP_BC="${SKIP_BC:-${skip_b}}" APPEND="$append" \
    ROUTES="${ROUTES:-0,1}" MIN_KEPT="${MIN_KEPT:-4}" MIN_KEPT_PER_ROUTE="${MIN_KEPT_PER_ROUTE:-2}" \
    MAX_DYAW="${MAX_DYAW:-0.45}" MAX_DDYAW="${MAX_DDYAW:-0.12}" \
    REPLAN_PERIOD_S="${REPLAN_PERIOD_S:-2.0}" W_JUMP="${W_JUMP:-0.05}" \
    SKIP_EVAL="${SKIP_EVAL:-0}" \
    ${WM:+WM="$WM"} ${PYTHON_BIN:+PYTHON_BIN="$PYTHON_BIN"} \
    bash "$ROOT/experiments/aerial/scripts/run_mainchannel_fullfix.sh" \
    >>"$ART/logs/mainchannel_fullfix_${STAMP}_h${HOST}_resume_nohup.log" 2>&1 &
  echo "[$(date -Is)] resume wrapper pid=$!" | tee -a "$WLOG"
}

while true; do
  if [[ -f "$MASTER" ]] && grep -q "$DONE_MARK" "$MASTER"; then
    echo "[$(date -Is)] DONE — watch exit" | tee -a "$WLOG"
    rm -f "$PIDFILE"
    exit 0
  fi
  if alive; then
    echo "[$(date -Is)] alive ok" >>"$WLOG"
  else
    echo "[$(date -Is)] DEAD — recovering" | tee -a "$WLOG"
    resume
    sleep 45
  fi
  sleep "$INTERVAL"
done
