#!/usr/bin/env bash
# Monitor urban-interior DepthScene expert collect on 125 (until-gate).
set -euo pipefail

SSH_HOST="${SSH_HOST:-cursor-125-public}"
STAMP="${STAMP:-20260917_interior2}"

fetch_status() {
  ssh -o BatchMode=yes -o ConnectTimeout=25 "$SSH_HOST" bash -s "$STAMP" <<'REMOTE'
STAMP="$1"
ROOT=~/aerial-wam-v2
OUT="$ROOT/experiments/aerial/rl/artifacts/dataset_urban_depth_scene_expert_${STAMP}"
LOG="$ROOT/artifacts/collect_urban_depth_scene_expert_${STAMP}.log"
NOHUP="$ROOT/artifacts/collect_urban_depth_scene_expert_${STAMP}_nohup.log"

alive=0
if pgrep -f "experiments.aerial.rl.collect_depth_scene_expert_dataset" >/dev/null 2>&1; then
  alive=1
fi
overnight=0
ophase=na
if pgrep -f "monitor_urban_humanlike_overnight_autopilot" >/dev/null 2>&1; then
  overnight=1
  if [[ -f "$ROOT/artifacts/urban_humanlike_overnight_state.env" ]]; then
    # shellcheck disable=SC1090
    source "$ROOT/artifacts/urban_humanlike_overnight_state.env" 2>/dev/null || true
    ophase="${PHASE:-na}"
  fi
fi

qk=0
gate=12
rounds=0
n_rev=0
n_keep=0
last=""
if [[ -f "$OUT/AUTO_REVIEW.json" ]]; then
  eval "$(python3 - <<PY
import json
d=json.load(open("$OUT/AUTO_REVIEW.json"))
s=d.get("summary") or {}
eps=d.get("episodes") or []
print(f"qk={int(s.get('quality_kept') or sum(1 for e in eps if e.get('keep')))}")
print(f"gate={int(s.get('min_kept_gate') or s.get('gate') or 12)}")
print(f"rounds={int(s.get('rounds') or 0)}")
print(f"n_rev={len(eps)}")
print(f"n_keep={sum(1 for e in eps if e.get('keep'))}")
if eps:
  e=eps[-1]
  print("last=%s" % repr(f"r{e.get('route_idx')} keep={e.get('keep')} arr={e.get('arrived')} infl={e.get('inflate')} drop={e.get('drop_reason') or 'ok'}").replace(" ", "\\ "))
else:
  print("last=none")
PY
)"
fi

gate_ok=0
if [[ -f "$OUT/QUALITY_SUMMARY.json" ]]; then
  gate_ok=$(python3 -c "import json;print(int(bool(json.load(open('$OUT/QUALITY_SUMMARY.json')).get('gate_passed'))))" 2>/dev/null || echo 0)
fi

tail_line=$(tail -1 "$LOG" 2>/dev/null | tr '\n' ' ' | cut -c1-120)
if [[ -z "$tail_line" ]]; then
  tail_line=$(tail -1 "$NOHUP" 2>/dev/null | tr '\n' ' ' | cut -c1-120)
fi

echo "ALIVE=${alive} QK=${qk:-0}/${gate:-12} KEEP=${n_keep:-0} REV=${n_rev:-0} ROUND=${rounds:-0} GATE_OK=${gate_ok} OVERNIGHT=${overnight} OPHASE=${ophase} LAST=${last:-na} TAIL=${tail_line:-na}"
REMOTE
}

notify_tick() {
  local msg="$1"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    osascript -e "display notification \"${msg//\"/\\\"}\" with title \"DepthScene collect 125\"" 2>/dev/null || true
  fi
  echo "AGENT_LOOP_TICK_depth_scene_collect {\"prompt\":\"追进度 depth-scene interior expert collect\",\"status\":\"${msg//\"/\\\"}\"}"
}

line="$(fetch_status 2>/dev/null || echo "SSH_FAIL")"
notify_tick "$line"
