#!/usr/bin/env bash
# Local dual-host monitor: keep 84+125 mainchannel fullfix watchers alive,
# refresh artifacts/daytime_mainchannel_STATUS.md, notify on Darwin.
#
#   nohup bash experiments/aerial/scripts/monitor_mainchannel_fullfix_dual.sh &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
INTERVAL="${INTERVAL:-120}"
STAMP84="${STAMP84:-20260923_1347}"
STAMP125="${STAMP125:-20260923_1347}"
SSH84="${SSH84:-cursor-84-public}"
SSH125="${SSH125:-cursor-125-public}"
ROOT84="${ROOT84:-/data/aerial-wam-v2}"
ROOT125="${ROOT125:-/home/yao/aerial-wam-v2}"
STATUS="$ROOT/artifacts/daytime_mainchannel_STATUS.md"
LOG="$ROOT/artifacts/monitor_mainchannel_fullfix_dual.log"
mkdir -p "$ROOT/artifacts"
ts() { date +%Y-%m-%dT%H:%M:%S%z; }

echo "[$(ts)] dual monitor start 84=$STAMP84 125=$STAMP125 every ${INTERVAL}s" >>"$LOG"

remote_ensure_watch() {
  local ssh_host=$1 root=$2 host_id=$3 stamp=$4
  ssh -o BatchMode=yes -o ConnectTimeout=25 "$ssh_host" \
    "ROOT='$root' HOST='$host_id' STAMP='$stamp' bash -s" <<'REMOTE'
set +e
ART="$ROOT/experiments/aerial/rl/artifacts"
PIDFILE="$ART/logs/watch_mainchannel_fullfix_${STAMP}_h${HOST}.pid"
MASTER="$ART/logs/mainchannel_fullfix_${STAMP}_h${HOST}.log"
mkdir -p "$ART/logs"
if [[ -f "$MASTER" ]] && grep -q MAINCHANNEL_FULLFIX_ALL_DONE "$MASTER"; then
  echo DONE; exit 0
fi
if [[ -f "$PIDFILE" ]]; then
  wpid=$(tr -d " \n" <"$PIDFILE")
  if [[ -n "$wpid" ]] && kill -0 "$wpid" 2>/dev/null; then
    echo WATCH_ALIVE; exit 0
  fi
fi
# any live watch script (not this ssh cmdline)
if ps -eo pid=,args= | grep -v grep | grep -F "watch_mainchannel_fullfix.sh" | grep -qv "bash -s\|bash -c"; then
  echo WATCH_ALIVE; exit 0
fi
chmod +x "$ROOT/experiments/aerial/scripts/watch_mainchannel_fullfix.sh" \
  "$ROOT/experiments/aerial/scripts/run_mainchannel_fullfix.sh" 2>/dev/null
nohup env HOST="$HOST" STAMP="$STAMP" INTERVAL=90 \
  ROUTES="${ROUTES:-0,1}" MIN_KEPT="${MIN_KEPT:-4}" MIN_KEPT_PER_ROUTE="${MIN_KEPT_PER_ROUTE:-2}" \
  /bin/bash "$ROOT/experiments/aerial/scripts/watch_mainchannel_fullfix.sh" \
  >>"$ART/logs/watch_mainchannel_fullfix_${STAMP}_h${HOST}_nohup.log" 2>&1 &
echo "WATCH_STARTED pid=$!"
REMOTE
}

remote_snapshot() {
  local ssh_host=$1 root=$2 host_id=$3 stamp=$4
  ssh -o BatchMode=yes -o ConnectTimeout=25 "$ssh_host" bash -s <<REMOTE || echo "HOST=$host_id SSH_FAIL"
ROOT="$root"
HOST="$host_id"
STAMP="$stamp"
ART="\$ROOT/experiments/aerial/rl/artifacts"
DS="\$ART/dataset_urban_inland_r01_depth_scene_\${STAMP}"
if [[ ! -d "\$DS" && -d "\$ART/dataset_urban_inland_r014_depth_scene_\${STAMP}" ]]; then
  DS="\$ART/dataset_urban_inland_r014_depth_scene_\${STAMP}"
fi
OUT="\$ART/v4_ac_ckpt_urban_directional_oa_\${STAMP}_expert_bc_h\${HOST}"
MASTER="\$ART/logs/mainchannel_fullfix_\${STAMP}_h\${HOST}.log"
alive=0
phase=unknown
if pgrep -af "collect_depth_scene_expert_dataset.*\${STAMP}" | grep -v pgrep >/dev/null; then
  alive=1; phase=collect
elif pgrep -af "train_v4_ac.*\${STAMP}" | grep -v pgrep >/dev/null; then
  alive=1; phase=bc
elif pgrep -af "wam_phase2_long_eval|run_hbclear_p0_diag|hbclear_diag_\${STAMP}" | grep -v pgrep >/dev/null; then
  alive=1; phase=eval
elif pgrep -af "run_mainchannel_fullfix|mainchannel_fullfix_\${STAMP}" | grep -v pgrep >/dev/null; then
  alive=1; phase=wrapper
fi
watch=0
pgrep -af "watch_mainchannel_fullfix" | grep -v pgrep >/dev/null && watch=1
done=0
grep -q MAINCHANNEL_FULLFIX_ALL_DONE "\$MASTER" 2>/dev/null && done=1
qk=0; gate=8; kb="{}"; gpass=0; rounds=0
if [[ -f "\$DS/AUTO_REVIEW.json" ]]; then
  meta=\$(python3 - "\$DS/AUTO_REVIEW.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
s=d.get("summary") or {}
print("%d|%d|%s|%d|%d" % (
    int(s.get("quality_kept") or 0),
    int(s.get("min_kept_gate") or 8),
    json.dumps(s.get("keep_by_route") or {}, separators=(",", ":")),
    int(bool(s.get("gate_passed"))),
    int(s.get("rounds") or 0),
))
PY
)
  IFS='|' read -r qk gate kb gpass rounds <<<"\$meta"
fi
actor=0
[[ -f "\$OUT/v4_ac_best.pt" || -f "\$OUT/v4_ac_latest.pt" ]] && actor=1
tail=\$(tail -1 "\$MASTER" 2>/dev/null | tr '\\n' ' ' | cut -c1-100)
echo "HOST=\$HOST done=\$done alive=\$alive phase=\$phase watch=\$watch qk=\${qk}/\${gate} kb=\${kb} gpass=\${gpass} rounds=\${rounds} actor=\$actor tail=\${tail:-na}"
REMOTE
}

write_status() {
  local s84=$1 s125=$2
  cat >"$STATUS" <<EOF
# Daytime main-channel LIVE

**updated:** $(date +%Y-%m-%dT%H:%M:%S%z)
**monitor:** dual watchdog (auto-resume on interrupt)

## 两机
| Host | stamp | 快照 |
|------|-------|------|
| **84** | \`$STAMP84\` | \`$s84\` |
| **125** | \`$STAMP125\` | \`$s125\` |

## 流水线
采齐 R0/1/3/4 → expert BC → P0 wm_bare eval  
中断自动：\`watch_mainchannel_fullfix.sh\`（APPEND / SKIP_*）

## 日志
- 84: \`.../logs/mainchannel_fullfix_${STAMP84}_h84.log\`
- 125: \`.../logs/mainchannel_fullfix_${STAMP125}_h125.log\`
- dual: \`artifacts/monitor_mainchannel_fullfix_dual.log\`
EOF
}

while true; do
  echo "[$(ts)] tick" >>"$LOG"
  ensure84="$(remote_ensure_watch "$SSH84" "$ROOT84" 84 "$STAMP84" 2>>"$LOG" || echo ENSURE_FAIL)"
  ensure125="$(remote_ensure_watch "$SSH125" "$ROOT125" 125 "$STAMP125" 2>>"$LOG" || echo ENSURE_FAIL)"
  echo "ensure84=$ensure84 ensure125=$ensure125" >>"$LOG"

  snap84="$(remote_snapshot "$SSH84" "$ROOT84" 84 "$STAMP84" 2>>"$LOG" || echo "HOST=84 SSH_FAIL")"
  snap125="$(remote_snapshot "$SSH125" "$ROOT125" 125 "$STAMP125" 2>>"$LOG" || echo "HOST=125 SSH_FAIL")"
  echo "$snap84" >>"$LOG"
  echo "$snap125" >>"$LOG"
  write_status "$snap84" "$snap125"

  if [[ "$(uname -s)" == "Darwin" ]]; then
    osascript -e "display notification \"84: ${snap84:0:80}\" with title \"mainchannel 84\"" 2>/dev/null || true
  fi

  # Both done → exit
  if echo "$snap84" | grep -q "done=1" && echo "$snap125" | grep -q "done=1"; then
    echo "[$(ts)] both DONE — dual monitor exit" >>"$LOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
