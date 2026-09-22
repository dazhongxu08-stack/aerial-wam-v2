#!/usr/bin/env bash
# Monitor Step1 baseline20 + route1 stage2 chain on 125.
set -euo pipefail

SSH_HOST="${SSH_HOST:-cursor-125-public}"
BASE_STAMP="${BASE_STAMP:-20260917}"

fetch_status() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$SSH_HOST" bash -s "$BASE_STAMP" <<'REMOTE'
BASE="$1"
ROOT=~/aerial-wam-v2
BL="$ROOT/artifacts/urban_complex_baseline20_${BASE}_z38base/run.log"
CH="$ROOT/artifacts/urban_complex_step1_chain_${BASE}.log"
TRAIN_LOG="$ROOT/artifacts/train_urban_complex_p2c_${BASE}_r1s2.log"

phase=unknown
detail=""

if pgrep -af "wam_phase2_long_eval.*baseline20_${BASE}_z38base" >/dev/null 2>&1; then
  phase=baseline20
elif pgrep -af "train_v4_ac.*urban_complex_p2c_${BASE}_r1s2" >/dev/null 2>&1; then
  phase=train_r1
elif pgrep -af "wam_phase2_long_eval.*r1gate" >/dev/null 2>&1; then
  phase=gate_r1
elif grep -q "PIPELINE DONE" "$CH" 2>/dev/null; then
  phase=ALL_DONE
else
  phase=idle_or_done
fi

bl_done=0
bl_route=""
if [[ -f "$BL" ]]; then
  bl_done=$(grep -cE "Route [0-9]+ \([0-9]+/20\)" "$BL" 2>/dev/null || echo 0)
  bl_route=$(grep -E "Route [0-9]+ \([0-9]+/20\)" "$BL" 2>/dev/null | tail -1 | sed -n 's/.*Route \([0-9]*\).*/\1/p')
fi

train_iter=""
if [[ -f "$TRAIN_LOG" ]]; then
  train_iter=$(grep -E "corrector:iter [0-9]+:" "$TRAIN_LOG" 2>/dev/null | tail -1 | sed -n 's/.*iter \([0-9]*\): \([0-9]*\) steps.*/iter=\1 steps=\2/p')
fi

gate_sr=""
if [[ -f "$ROOT/artifacts/urban_complex_toward_g_gate_${BASE}_r1gate/eval_all.json" ]]; then
  gate_sr=$(python3 -c "import json;d=json.load(open('$ROOT/artifacts/urban_complex_toward_g_gate_${BASE}_r1gate/eval_all.json'));e=d.get('episodes',[]);print(f\"SR {sum(1 for x in e if x.get('arrived'))}/{len(e)} prog={round(float(e[0].get('progress_ratio',0))*100,1) if e else 0}%\")" 2>/dev/null || true)
fi

trainable=""
if grep -q TRAINABLE_ROUTES "$BL" 2>/dev/null; then
  trainable=$(grep TRAINABLE_ROUTES "$BL" | tail -1 | sed 's/.*TRAINABLE_ROUTES: //')
fi

echo "PHASE=${phase} BASELINE=${bl_done}/20 last_r=${bl_route:-?} TRAIN=${train_iter:-na} GATE=${gate_sr:-na} TRAINABLE=${trainable:-pending}"
REMOTE
}

notify_tick() {
  local msg="$1"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    osascript -e "display notification \"${msg//\"/\\\"}\" with title \"Urban chain 125\"" 2>/dev/null || true
  fi
  echo "AGENT_LOOP_TICK_urban_chain {\"prompt\":\"追进度 urban step1 chain\",\"status\":\"${msg//\"/\\\"}\"}"
}

line="$(fetch_status 2>/dev/null || echo "SSH_FAIL")"
notify_tick "$line"
