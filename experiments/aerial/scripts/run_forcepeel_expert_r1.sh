#!/usr/bin/env bash
# R1 force open-side peel trial (DepthScene expert).
# Pauses mainchannel collect, runs a short R1 gate with --force-peel-rem,
# then resumes APPEND main collect for stamp 1347.
#
#   HOST=84 bash experiments/aerial/scripts/run_forcepeel_expert_r1.sh
set -euo pipefail
HOST="${HOST:?set HOST=84}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PYTHON_BIN:-}"
for c in \
  "${PYTHON_BIN:-}" \
  /data/venvs/sim_verify/bin/python \
  "${HOME}/sim_verify/.venv/bin/python"; do
  [[ -n "$c" && -x "$c" ]] && PY="$c" && break
done
PY="${PY:-python3}"

ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$ART/forcepeel_expert_r1_${STAMP}}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
FORCE_REM="${FORCE_PEEL_REM:-45,70}"
MIN_YAW="${FORCE_PEEL_MIN_YAW_DEG:-75}"
SIDE="${FORCE_PEEL_SIDE:-0}"
MIN_KEPT="${MIN_KEPT:-3}"
MAX_ROUNDS="${MAX_ROUNDS:-8}"
RESUME_STAMP="${RESUME_STAMP:-20260923_1347}"
RESUME_DS="${RESUME_DS:-$ART/dataset_urban_inland_r01_depth_scene_${RESUME_STAMP}}"
LOG="$ART/logs/forcepeel_expert_r1_${STAMP}_h${HOST}.log"
STATUS="${STATUS_MD:-$ART/../../../../artifacts/forcepeel_expert_STATUS.md}"
# Prefer repo-root artifacts/ for local STATUS mirrors
if [[ -d "$ROOT/artifacts" ]]; then
  STATUS="$ROOT/artifacts/forcepeel_expert_STATUS.md"
fi
mkdir -p "$ART/logs" "$OUT"

exec >>"$LOG" 2>&1
echo "=== forcepeel expert R1 HOST=$HOST stamp=$STAMP rem=$FORCE_REM ==="
hostname; date -Is

write_status() {
  local phase=$1
  cat >"$STATUS" <<EOF
# 84 R1 force open-side peel LIVE

**updated:** $(date -Is)
**stamp:** \`${STAMP}\` · rem=[${FORCE_REM}] · min_yaw=${MIN_YAW}°

| | |
|--|--|
| phase | ${phase} |
| out | \`${OUT}\` |
| log | \`${LOG}\` |
EOF
  if [[ -f "$OUT/AUTO_REVIEW.json" ]]; then
    "$PY" - <<PY >>"$STATUS"
import json
from pathlib import Path
d=json.loads(Path("$OUT/AUTO_REVIEW.json").read_text())
s=d.get("summary") or {}
eps=d.get("episodes") or []
print()
print(f"| n | keep | gate |")
print(f"|---|------|------|")
print(f"| {len(eps)} | {s.get('quality_kept')} | {s.get('gate_passed')} |")
for e in eps[-5:]:
    print(f"- route={e.get('route_idx')} keep={e.get('quality_keep')} arr={e.get('arrived')} "
          f"df={e.get('d_final_m')} reason={e.get('drop_reason') or e.get('keep_reason')}")
PY
  fi
}

# Pause mainchannel (keep AirSim)
pkill -f 'watch_mainchannel_fullfix' 2>/dev/null || true
pkill -f 'run_mainchannel_fullfix' 2>/dev/null || true
pkill -f 'collect_depth_scene_expert_dataset' 2>/dev/null || true
sleep 4
if [[ -x "$RECOVER" ]]; then
  bash "$RECOVER" outdoor >/dev/null 2>&1 || true
  sleep 6
fi
write_status "collecting"

echo "=== COLLECT force peel R1 ==="
"$PY" -m experiments.aerial.rl.collect_depth_scene_expert_dataset \
  --backend airsim --device cuda \
  --annotation "$ANN" \
  --routes 1 \
  --out "$OUT" \
  --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 \
  --max-inflate 1.8 --require-arrived --require-interior \
  --min-spawn-inland-m 100 --min-path-inland-m 80 \
  --require-all-requested-routes \
  --min-kept "$MIN_KEPT" --min-kept-per-route "$MIN_KEPT" \
  --until-gate --max-rounds "$MAX_ROUNDS" \
  --shield-fwd-only --tti-coeff 2.5 \
  --r-m 100 --cruise-speed 10 --max-steps 600 \
  --d-clear 35 --d-danger 4 \
  --max-dyaw 0.45 --max-ddyaw 0.12 \
  --replan-period-s 2.0 --w-jump 0.05 \
  --stuck-escape-after-s 6 --stuck-escape-hold-s 6 \
  --force-peel-rem "$FORCE_REM" \
  --force-peel-min-yaw-deg "$MIN_YAW" \
  --force-peel-side "$SIDE" \
  --episodes 1 \
  || true

write_status "collect_done"
echo "=== FORCEPEEL EXPERT DONE ==="
date -Is
if [[ -f "$OUT/AUTO_REVIEW.json" ]]; then
  "$PY" -c "import json; d=json.load(open('$OUT/AUTO_REVIEW.json')); s=d.get('summary')or{}; print('keep',s.get('quality_kept'),'gate',s.get('gate_passed'),'n',len(d.get('episodes')or[]))"
fi

# Resume main APPEND collect
if [[ -d "$RESUME_DS" ]]; then
  echo "=== RESUME main collect stamp=$RESUME_STAMP ==="
  write_status "resuming_main_${RESUME_STAMP}"
  nohup env HOST="$HOST" STAMP="$RESUME_STAMP" APPEND=1 SKIP_BC=1 SKIP_EVAL=1 \
    ROUTES=0,1 MIN_KEPT=4 MIN_KEPT_PER_ROUTE=2 \
    MAX_DYAW=0.45 MAX_DDYAW=0.12 \
    bash "$ROOT/experiments/aerial/scripts/run_mainchannel_fullfix.sh" \
    >>"$ART/logs/mainchannel_fullfix_${RESUME_STAMP}_h${HOST}.log" 2>&1 < /dev/null &
  echo "resume_pid=$!"
  # optional watcher
  if [[ -x "$ROOT/experiments/aerial/scripts/watch_mainchannel_fullfix.sh" ]]; then
    nohup env HOST="$HOST" STAMP="$RESUME_STAMP" \
      bash "$ROOT/experiments/aerial/scripts/watch_mainchannel_fullfix.sh" \
      >>"$ART/logs/watch_mainchannel_${RESUME_STAMP}_h${HOST}.log" 2>&1 < /dev/null &
  fi
else
  echo "WARN no RESUME_DS=$RESUME_DS — not restarting main collect"
  write_status "done_no_resume"
fi
echo "=== ALL DONE forcepeel expert ==="
date -Is
