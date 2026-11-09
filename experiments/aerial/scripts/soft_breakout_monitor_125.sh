#!/usr/bin/env bash
# Soft breakout hub monitor — runs ON 125 (LAN to 84/14/11).
# Survives Mac↔125 network flaps. Auto-restarts gate/sync; Soft train
# restart is opt-in (SOFT_MONITOR_RESTART_TRAIN=1).
#
#   nohup STAMP=20260926_soft bash experiments/aerial/scripts/soft_breakout_monitor_125.sh \
#     >experiments/aerial/rl/artifacts/logs/soft_breakout_monitor_125.out 2>&1 &
set +e
STAMP="${STAMP:-20260926_soft}"
INTERVAL="${SOFT_MONITOR_INTERVAL:-180}"
ROOT="${ROOT:-/home/yao/aerial-wam-v2}"
cd "$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
OUT="$ART/breakout_soft_${STAMP}"
LOG="$ART/logs/soft_breakout_monitor_${STAMP}.log"
STATUS="$ART/soft_breakout_STATUS.json"
STATUS_MD="$ART/soft_breakout_STATUS.md"
ALERTS="$ART/logs/soft_breakout_ALERTS_${STAMP}.log"
mkdir -p "$OUT" "$ART/logs"
KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"
[[ -f "$KEY" ]] || KEY="$HOME/.ssh/id_ed25519"

alert() {
  local msg="$*"
  echo "[$(date -Is)] ALERT $msg" | tee -a "$ALERTS" "$LOG"
}

restart_gate() {
  local host="$1" port="$2"
  alert "restart Soft gate on h${host}"
  ssh -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" -p "$port" "a25689@10.239.121.${host}" "bash -s" <<H
set +e
cd /home/a25689/aerial-wam-v2
pkill -f run_soft_positive_gate_watch 2>/dev/null
# only kill this host's eval cfg
pkill -f "_tmp_soft_gate_h${host}" 2>/dev/null
sleep 2
export PYTHONPATH=\$PWD
PY=\$PWD/.venv/bin/python
nohup env HOST=${host} STAMP=${STAMP} SOFT_POS_K=8 INTERVAL=120 PYTHON_BIN=\$PY \
  bash experiments/aerial/scripts/run_soft_positive_gate_watch.sh \
  >experiments/aerial/rl/artifacts/logs/soft_gate${host}_${STAMP}.out 2>&1 &
echo GATE_PID=\$!
H
}

ensure_sync() {
  if ! pgrep -f sync_soft_ckpt_84_to_14 >/dev/null; then
    alert "restart Soft ckpt sync 84→14"
    nohup bash "$ROOT/experiments/aerial/scripts/sync_soft_ckpt_84_to_14.sh" \
      >"$ART/logs/sync_soft_ckpt_${STAMP}.out" 2>&1 &
  fi
}

echo "[$(date -Is)] soft_breakout_monitor start stamp=$STAMP interval=${INTERVAL}s" | tee -a "$LOG"

while true; do
  ts=$(date -Is)
  soft84=0; nckpt84=0; soft_iter=""
  soft125=0; nckpt125=0
  gate14=0; eval14=0; gate11=0; eval11=0
  hard14=0; hard11=0; nckpt_h14=0; nckpt_h11=0
  pos=-1; ready=0
  events=()

  # --- Soft@84 ---
  out84=$(ssh -o BatchMode=yes -o ConnectTimeout=12 ubantu@10.229.20.84 "
    pgrep -f '[t]rain_v4_ac' >/dev/null && echo ALIVE=1 || echo ALIVE=0
    echo NCKPT=\$(ls /data/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84/*.pt 2>/dev/null | wc -l | tr -d ' ')
    grep -E 'iter [0-9]+:' /data/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/mainchannel_soft_${STAMP}_h84.log 2>/dev/null | tail -1
  " 2>/dev/null)
  [[ "$out84" == *ALIVE=1* ]] && soft84=1
  nckpt84=$(echo "$out84" | sed -n 's/^NCKPT=//p' | head -1)
  [[ -z "$nckpt84" ]] && nckpt84=0
  soft_iter=$(echo "$out84" | grep -oE 'iter [0-9]+' | tail -1)

  # --- Soft@125 ---
  pgrep -f 'train_v4_ac.*soft.*h125|ckpt_mainchannel_soft_.*_h125' >/dev/null && soft125=1
  pgrep -f 'train_v4_ac' >/dev/null && soft125=1
  nckpt125=$(ls "$ART/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h125"/*.pt 2>/dev/null | wc -l | tr -d ' ')

  # --- Gate@14 ---
  out14=$(ssh -o BatchMode=yes -o ConnectTimeout=12 -i "$KEY" -p 31126 a25689@10.239.121.14 "
    pgrep -f '[s]oft_positive_gate' >/dev/null && echo GATE=1 || echo GATE=0
    pgrep -f '[w]am_phase2_long_eval' >/dev/null && echo EVAL=1 || echo EVAL=0
    pgrep -f '[r]un_mainchannel_hard_noshield|[t]rain_v4_ac' >/dev/null && echo HARD=1 || echo HARD=0
    echo NCKPT=\$(ls /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_hard_${STAMP}_noshield_h14/*.pt 2>/dev/null | wc -l | tr -d ' ')
    echo POS=\$(cat /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/breakout_soft_${STAMP}/SOFT_POS_COUNT.txt 2>/dev/null || echo -1)
    test -f /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/breakout_soft_${STAMP}/READY_HARD.txt && echo READY=1 || echo READY=0
  " 2>/dev/null)
  [[ "$out14" == *GATE=1* ]] && gate14=1
  [[ "$out14" == *EVAL=1* ]] && eval14=1
  [[ "$out14" == *READY=1* ]] && ready=1
  pos=$(echo "$out14" | sed -n 's/^POS=//p' | head -1)
  [[ -z "$pos" ]] && pos=-1

  # --- Gate@11 ---
  out11=$(ssh -o BatchMode=yes -o ConnectTimeout=12 -i "$KEY" -p 30627 a25689@10.239.121.11 "
    pgrep -f '[s]oft_positive_gate' >/dev/null && echo GATE=1 || echo GATE=0
    pgrep -f '[w]am_phase2_long_eval' >/dev/null && echo EVAL=1 || echo EVAL=0
    pgrep -f '[r]un_mainchannel_hard_noshield|[t]rain_v4_ac' >/dev/null && echo HARD=1 || echo HARD=0
    echo NCKPT=\$(ls /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_hard_${STAMP}_noshield_h11/*.pt 2>/dev/null | wc -l | tr -d ' ')
  " 2>/dev/null)
  [[ "$out11" == *GATE=1* ]] && gate11=1
  [[ "$out11" == *EVAL=1* ]] && eval11=1
  hard14=0; hard11=0; nckpt_h14=0; nckpt_h11=0
  [[ "$out14" == *HARD=1* ]] && hard14=1
  [[ "$out11" == *HARD=1* ]] && hard11=1
  nckpt_h14=$(echo "$out14" | sed -n 's/^NCKPT=//p' | head -1)
  nckpt_h11=$(echo "$out11" | sed -n 's/^NCKPT=//p' | head -1)
  [[ -z "$nckpt_h14" ]] && nckpt_h14=0
  [[ -z "$nckpt_h11" ]] && nckpt_h11=0

  ensure_sync

  # Heal Soft gates only while Hard does not own those machines.
  if [[ -f "$OUT/HARD_PHASE" ]]; then
    bash "$ROOT/experiments/aerial/scripts/ensure_hard_quad.sh" >>"$LOG" 2>&1 || true
  else
    if [[ "$gate14" -eq 0 ]]; then
      events+=("gate14_down")
      restart_gate 14 31126
      gate14=1
    fi
    if [[ "$gate11" -eq 0 ]]; then
      events+=("gate11_down")
      restart_gate 11 30627
      gate11=1
    fi
  fi
  if [[ "$soft84" -eq 0 && "${SOFT_MONITOR_RESTART_TRAIN:-0}" == "1" ]]; then
    events+=("soft84_down")
    alert "Soft@84 train NOT alive — restarting"
    if [[ "${SOFT_MONITOR_RESTART_TRAIN:-0}" == "1" ]]; then
      ssh -o BatchMode=yes -o ConnectTimeout=20 ubantu@10.229.20.84 "bash -s" <<H84
set +e
cd /data/aerial-wam-v2
export PYTHONPATH=\$PWD
PY=/data/venvs/sim_verify/bin/python
nohup env HOST=84 STAMP=$STAMP ITERS=120 SKIP_BASELINE=1 SOFT_EARLY_STOP=1 PYTHON_BIN=\$PY \
  bash experiments/aerial/scripts/run_mainchannel_soft_noshield.sh \
  >experiments/aerial/rl/artifacts/logs/soft84_${STAMP}_relaunch.out 2>&1 &
echo SOFT84_RELAUNCH=\$!
H84
    fi
  fi

  # Delta vs previous status
  prev_pos=-1; prev_ready=0; prev_nckpt=0; prev_soft84=1
  if [[ -f "$STATUS" ]]; then
    prev_pos=$(python3 -c "import json; d=json.load(open('$STATUS')); print(d.get('pos',-1))" 2>/dev/null || echo -1)
    prev_ready=$(python3 -c "import json; d=json.load(open('$STATUS')); print(int(d.get('ready',0)))" 2>/dev/null || echo 0)
    prev_nckpt=$(python3 -c "import json; d=json.load(open('$STATUS')); print(int(d.get('nckpt84',0)))" 2>/dev/null || echo 0)
    prev_soft84=$(python3 -c "import json; d=json.load(open('$STATUS')); print(int(d.get('soft84',1)))" 2>/dev/null || echo 1)
  fi
  [[ "$pos" =~ ^[0-9]+$ && "$prev_pos" =~ ^-?[0-9]+$ && "$pos" -gt "$prev_pos" ]] && events+=("pos_${prev_pos}_to_${pos}")
  [[ "$ready" -eq 1 && "$prev_ready" -eq 0 ]] && events+=("READY_HARD")
  [[ "$nckpt84" =~ ^[0-9]+$ && "$prev_nckpt" =~ ^[0-9]+$ && "$nckpt84" -gt "$prev_nckpt" ]] && events+=("ckpt84_${nckpt84}")
  [[ "$soft84" -eq 1 && "$prev_soft84" -eq 0 ]] && events+=("soft84_recovered")

  ev_str="${events[*]}"
  python3 - <<PY
import json
events = """$ev_str""".split() if """$ev_str""".strip() else []
def to_int(x, default=-1):
    try:
        return int(str(x).strip())
    except Exception:
        return default
cur = {
  "ts": "$ts",
  "stamp": "$STAMP",
  "soft84": $soft84,
  "soft125": $soft125,
  "nckpt84": to_int("$nckpt84", 0),
  "nckpt125": to_int("$nckpt125", 0),
  "soft_iter": "$soft_iter",
  "gate14": $gate14,
  "eval14": $eval14,
  "gate11": $gate11,
  "eval11": $eval11,
  "hard14": $hard14,
  "hard11": $hard11,
  "nckpt_h14": to_int("$nckpt_h14", 0),
  "nckpt_h11": to_int("$nckpt_h11", 0),
  "pos": to_int("$pos", -1),
  "ready": $ready,
  "events": events,
}
open("$STATUS", "w").write(json.dumps(cur, indent=2) + "\n")
md = f"""# Soft breakout status ({cur['ts']})
- Soft@84: alive={cur['soft84']} {cur['soft_iter']} nckpt={cur['nckpt84']}
- Soft@125: alive={cur['soft125']} nckpt={cur['nckpt125']}
- Gate@14: alive={cur['gate14']} eval={cur['eval14']}
- Gate@11: alive={cur['gate11']} eval={cur['eval11']}
- Hard@14: alive={cur['hard14']} nckpt={cur['nckpt_h14']}
- Hard@11: alive={cur['hard11']} nckpt={cur['nckpt_h11']}
- POS={cur['pos']}/8 READY_HARD={cur['ready']}
- events: {', '.join(events) or 'none'}
"""
open("$STATUS_MD", "w").write(md)
print(json.dumps(cur, ensure_ascii=False))
PY

  if [[ ${#events[@]} -gt 0 ]]; then
    echo "[$ts] NEW ${events[*]}" | tee -a "$LOG" "$ALERTS"
  else
    echo "[$ts] OK soft84=$soft84 nckpt=$nckpt84 pos=$pos/8 g14=$gate14 g11=$gate11" >>"$LOG"
  fi

  sleep "$INTERVAL"
done
