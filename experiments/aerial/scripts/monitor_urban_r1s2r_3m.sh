#!/usr/bin/env bash
# Poll 125 every 3 minutes: r1s2r train + posttrain pipeline; pull analysis when ready.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
INTERVAL="${INTERVAL_SEC:-180}"
TRAIN_STAMP="${TRAIN_STAMP:-20260917_r1s2r}"
BASE_STAMP="${BASE_STAMP:-20260917}"
SSH_HOST="${SSH_HOST:-cursor-125-public}"
PULLED="${ROOT}/artifacts/.urban_posttrain_analysis_pulled"

while true; do
  TS="$(date '+%Y-%m-%d %H:%M:%S')"
  OUT="$(ssh -o ConnectTimeout=20 "$SSH_HOST" \
    "LOG=~/aerial-wam-v2/artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log; \
     CKPT=~/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${TRAIN_STAMP}; \
     PTLOG=~/aerial-wam-v2/artifacts/urban_complex_posttrain_auto_${BASE_STAMP}.log; \
     ANALYSIS=~/aerial-wam-v2/artifacts/urban_complex_posttrain_analysis_${BASE_STAMP}.json; \
     ITER=\$(grep -c 'wrote ckpt iter' \"\$LOG\" 2>/dev/null || echo 0); \
     RUN=\$(pgrep -c -f 'train_v4_ac.*${TRAIN_STAMP}' 2>/dev/null || echo 0); \
     PT=\$(pgrep -c -f run_urban_complex_posttrain_auto 2>/dev/null || echo 0); \
     NCKPT=\$(ls \"\$CKPT\"/v4_ac_iter_*.pt 2>/dev/null | wc -l); \
     AGE=\$((\$(date +%s)-\$(stat -c %Y \"\$LOG\" 2>/dev/null || echo 0))); \
     REM=\$((64-ITER)); ETA=\$((REM*3)); \
     AD=false; [ -f \"\$ANALYSIS\" ] && AD=true; \
     PTL=\$(tail -1 \"\$PTLOG\" 2>/dev/null | head -c 80); \
     LAST=\$(grep 'iter ' \"\$LOG\" 2>/dev/null | tail -1 | head -c 80); \
     echo iters=\${ITER}/64 ckpts=\${NCKPT} log_age_s=\${AGE} eta_min=\${ETA} train_run=\${RUN} posttrain_run=\${PT} analysis_done=\${AD} pt_last=\${PTL} last=\${LAST}" \
    2>&1)" || OUT="ssh_fail"
  echo "AGENT_LOOP_TICK_r1s2r [$TS] $OUT"
  if [[ ! -f "$PULLED" ]] && echo "$OUT" | grep -q 'analysis_done=true'; then
    rsync -az "${SSH_HOST}:~/aerial-wam-v2/artifacts/urban_complex_posttrain_analysis_${BASE_STAMP}.json" \
      "$ROOT/artifacts/" 2>/dev/null || true
    rsync -az "${SSH_HOST}:~/aerial-wam-v2/artifacts/urban_complex_posttrain_auto_${BASE_STAMP}.log" \
      "$ROOT/artifacts/" 2>/dev/null || true
    touch "$PULLED"
    echo "AGENT_LOOP_TICK_r1s2r [$TS] analysis pulled to artifacts/"
  fi
  sleep "$INTERVAL"
done
