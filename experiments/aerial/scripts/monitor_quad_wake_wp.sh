#!/usr/bin/env bash
# 125: watch 125/84/14/11 geomfix busy state; write STATUS; on idle/anomaly
# emit WAKE_WORK_PROGRESS.json for Mac → Work progress summary.
#
#   STAMP=20260925_geomfix nohup bash experiments/aerial/scripts/monitor_quad_wake_wp.sh &
set -eo pipefail
STAMP="${STAMP:-20260925_geomfix}"
INTERVAL="${INTERVAL:-90}"
WAKE_COOLDOWN_S="${WAKE_COOLDOWN_S:-600}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ART_MAC="$ROOT/artifacts"
ART="$ROOT/experiments/aerial/rl/artifacts"
OUT="$ART/geomfix_${STAMP}"
LOG="$ART/logs/monitor_quad_wake_wp_${STAMP}.log"
KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"
mkdir -p "$ART_MAC" "$ART/logs" "$OUT"
echo $$ >"$ART/logs/monitor_quad_wake_wp_${STAMP}.pid"
exec >>"$LOG" 2>&1
ts(){ date -Is; }
echo "[$(ts)] monitor_quad_wake_wp start stamp=$STAMP interval=${INTERVAL}s"

busy_pgrep() { pgrep -af "$1" 2>/dev/null | grep -v pgrep | grep -v monitor_quad_wake >/dev/null; }

host_125() {
  if busy_pgrep 'auto_advance_geomfix|collect_obstacle_cost_gt_frames|train_obstacle_cost_labels|wam_obstacle_cost_cone|run_geomfix_r1'; then
    echo busy; return
  fi
  echo idle
}
host_84() {
  if timeout 12 ssh -n -o BatchMode=yes -o ConnectTimeout=8 ubantu@10.229.20.84 \
    "pgrep -af 'collect_obstacle|run_geomfix_r2' | grep -v pgrep >/dev/null" 2>/dev/null; then
    echo busy; return
  fi
  echo idle
}
host_14() {
  if ssh -n -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -p 31126 a25689@10.239.121.14 \
    "pgrep -af '_wm_train_validate|run_geomfix_r3' | grep -v pgrep >/dev/null" 2>/dev/null; then
    echo busy; return
  fi
  echo idle
}
host_11() {
  if ssh -n -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -p 30627 a25689@10.239.121.11 \
    "pgrep -af '_wm_train_validate|run_geomfix_r4' | grep -v pgrep >/dev/null" 2>/dev/null; then
    echo busy; return
  fi
  echo idle
}

ssh_ok_84() {
  timeout 10 ssh -n -o BatchMode=yes -o ConnectTimeout=8 ubantu@10.229.20.84 'true' 2>/dev/null
}
ssh_ok_14() {
  ssh -n -o BatchMode=yes -o ConnectTimeout=12 -i "$KEY" -p 31126 a25689@10.239.121.14 'true' 2>/dev/null
}
ssh_ok_11() {
  ssh -n -o BatchMode=yes -o ConnectTimeout=12 -i "$KEY" -p 30627 a25689@10.239.121.11 'true' 2>/dev/null
}

emit_wake() {
  local reason="$1" detail="$2"
  local lastf="$ART_MAC/WAKE_WORK_PROGRESS_LAST.txt"
  local now epoch_last=0
  now=$(date +%s)
  if [[ -f "$lastf" ]]; then
    epoch_last=$(cat "$lastf" 2>/dev/null || echo 0)
  fi
  if (( now - epoch_last < WAKE_COOLDOWN_S )); then
    echo "[$(ts)] WAKE suppressed cooldown reason=$reason"
    return
  fi
  echo "$now" >"$lastf"
  # Avoid unquoted heredoc + set -u surprises; pass state via env to python.
  ART_MAC_WAKE="$ART_MAC/WAKE_WORK_PROGRESS.json" \
  REASON="$reason" DETAIL="$detail" TS="$(ts)" STAMP="$STAMP" \
  H125="${h125:-unknown}" H84="${h84:-unknown}" H14="${h14:-unknown}" H11="${h11:-unknown}" \
  READY_JSON="$ready_json" AA="${aa:-unknown}" AIRSIM="${airsim:-unknown}" \
  python3 - <<'PY'
import json, os
from pathlib import Path
Path(os.environ["ART_MAC_WAKE"]).write_text(json.dumps({
  "ts": os.environ["TS"],
  "reason": os.environ["REASON"],
  "detail": os.environ["DETAIL"],
  "hosts": {"125": os.environ["H125"], "84": os.environ["H84"],
            "14": os.environ["H14"], "11": os.environ["H11"]},
  "stamp": os.environ["STAMP"],
  "READY_SOFT": os.environ["READY_JSON"] == "True",
  "auto_advance": os.environ["AA"],
  "airsim": os.environ["AIRSIM"],
}, indent=2), encoding="utf-8")
PY
  echo "[$(ts)] WAKE emitted reason=$reason detail=$detail"
}

while true; do
  h125=$(host_125)
  h84=$(host_84)
  h14=$(host_14)
  h11=$(host_11)
  aa=dead
  busy_pgrep 'auto_advance_geomfix' && aa=alive
  airsim=down
  busy_pgrep 'AirVLN-Linux-Shipping' && airsim=up
  ready=false
  ready_json=False
  [[ -f "$OUT/READY_SOFT.txt" ]] && ready=true && ready_json=True

  anomalies=()
  [[ "$aa" == dead && "$ready" == false ]] && anomalies+=("auto_advance_dead")
  ssh_ok_84 || anomalies+=("ssh_84_fail")
  ssh_ok_14 || anomalies+=("ssh_14_fail")
  ssh_ok_11 || anomalies+=("ssh_11_fail")
  if [[ "$airsim" == down && "$h125" == busy ]]; then
    anomalies+=("airsim_down_while_125_busy")
  fi

  idle_hosts=()
  if [[ "$ready" == false ]]; then
    [[ "$h125" == idle ]] && idle_hosts+=("125")
    [[ "$h84" == idle ]] && idle_hosts+=("84")
    [[ "$h14" == idle ]] && idle_hosts+=("14")
    [[ "$h11" == idle ]] && idle_hosts+=("11")
  fi

  {
    echo "# QUAD_WATCH_STATUS"
    echo
    echo "updated: $(ts)"
    echo "stamp: $STAMP"
    echo "READY_SOFT: $ready"
    echo "auto_advance: $aa"
    echo "airsim: $airsim"
    echo
    echo "| host | state |"
    echo "|------|-------|"
    echo "| 125 | $h125 |"
    echo "| 84 | $h84 |"
    echo "| 14 | $h14 |"
    echo "| 11 | $h11 |"
    echo
    echo "idle: ${idle_hosts[*]:-none}"
    echo "anomalies: ${anomalies[*]:-none}"
  } >"$ART_MAC/QUAD_WATCH_STATUS.md"

  IDLE_CSV=$(IFS=,; echo "${idle_hosts[*]-}")
  ANOM_CSV=$(IFS=,; echo "${anomalies[*]-}")
  python3 - <<PY
import json
from pathlib import Path
idle=[x for x in "${IDLE_CSV}".split(",") if x]
anom=[x for x in "${ANOM_CSV}".split(",") if x]
Path("$ART_MAC/QUAD_WATCH_STATUS.json").write_text(json.dumps({
  "updated": "$(ts)",
  "stamp": "$STAMP",
  "READY_SOFT": $ready_json,
  "auto_advance": "$aa",
  "airsim": "$airsim",
  "hosts": {"125":"$h125","84":"$h84","14":"$h14","11":"$h11"},
  "idle": idle,
  "anomalies": anom,
}, indent=2), encoding="utf-8")
PY

  echo "[$(ts)] 125=$h125 84=$h84 14=$h14 11=$h11 aa=$aa airsim=$airsim idle=${idle_hosts[*]:-none} anom=${anomalies[*]:-none}"

  if ((${#anomalies[@]})); then
    emit_wake "anomaly" "${anomalies[*]}"
  elif ((${#idle_hosts[@]})); then
    # 单机短空闲可能是切换间隙；≥2 机空闲或 125 空且 auto_advance 死 → wake
    if ((${#idle_hosts[@]} >= 2)) || [[ "$h125" == idle && "$aa" == dead ]]; then
      emit_wake "idle" "idle_hosts=${idle_hosts[*]}"
    elif ((${#idle_hosts[@]} == 1)); then
      # 单机空闲：记一笔，连续两轮再 wake（用 marker）
      mark="$ART_MAC/QUAD_WATCH_IDLE_ONCE_${idle_hosts[0]}.txt"
      if [[ -f "$mark" ]]; then
        emit_wake "idle" "persistent_idle=${idle_hosts[0]}"
        rm -f "$mark"
      else
        date -Is >"$mark"
      fi
    fi
  else
    rm -f "$ART_MAC"/QUAD_WATCH_IDLE_ONCE_*.txt 2>/dev/null || true
  fi

  sleep "$INTERVAL"
done
