#!/usr/bin/env bash
# Local 3-min poll: 125 aligned autopilot + rsync analysis when ready.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
INTERVAL="${INTERVAL_SEC:-180}"
SSH_HOST="${SSH_HOST:-cursor-125-public}"
TRAIN_STAMP="${TRAIN_STAMP:-20260917_aligned}"
BASE_STAMP="${BASE_STAMP:-20260917}"
PULLED="${ROOT}/artifacts/.urban_aligned_${TRAIN_STAMP}_pulled"

while true; do
  TS="$(date '+%Y-%m-%d %H:%M:%S')"
  OUT="$(ssh -o ConnectTimeout=25 "$SSH_HOST" \
    "source ~/aerial-wam-v2/experiments/aerial/scripts/env_4090.sh 2>/dev/null; \
     cd ~/aerial-wam-v2; \
     ITER=\$(grep -c 'wrote ckpt iter' artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log 2>/dev/null || echo 0); \
     TR=\$(pgrep -c -f 'train_v4_ac.*${TRAIN_STAMP}' 2>/dev/null || echo 0); \
     AP=\$(pgrep -c -f monitor_urban_aligned_autopilot 2>/dev/null || echo 0); \
     EV=\$(pgrep -c -f wam_phase2_long_eval 2>/dev/null || echo 0); \
     ANALYSIS=artifacts/urban_complex_posttrain_analysis_${TRAIN_STAMP}.json; \
     AD=false; [ -f \"\$ANALYSIS\" ] && AD=true; \
     ALIGN=\$(grep deploy-align artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log 2>/dev/null | tail -1 | head -c 100); \
     echo iter=\${ITER}/32 train=\${TR} autopilot=\${AP} eval=\${EV} analysis=\${AD} align=\${ALIGN}" \
    2>&1)" || OUT="ssh_fail"
  echo "AGENT_LOOP_TICK_aligned [$TS] $OUT"

  if [[ ! -f "$PULLED" ]] && echo "$OUT" | grep -q 'analysis=true'; then
    mkdir -p "$ROOT/artifacts/urban_aligned_${TRAIN_STAMP}"
    rsync -az "${SSH_HOST}:~/aerial-wam-v2/artifacts/urban_complex_posttrain_analysis_${TRAIN_STAMP}.json" \
      "${SSH_HOST}:~/aerial-wam-v2/artifacts/urban_complex_toward_g_gate_${TRAIN_STAMP}_gate/eval_all.json" \
      "$ROOT/artifacts/urban_aligned_${TRAIN_STAMP}/" 2>/dev/null || true
    touch "$PULLED"
    echo "AGENT_LOOP_TICK_aligned [$TS] pulled analysis to artifacts/urban_aligned_${TRAIN_STAMP}/"
  fi
  sleep "$INTERVAL"
done
