#!/usr/bin/env bash
# V12 smoke: V11 + terminal goal pin (default 20m) + near-miss retry (d<=5m).
# Targets routes that failed near-miss in wam_phase2_v11_full16_20260915.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v12_mainline.inc.sh"
PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"

OUT_ROOT="${OUT_ROOT:-artifacts/wam_phase2_v12_nearmiss_20260916}"
ROUTES="${ROUTES:-1,4,9,14}"

mkdir -p "$OUT_ROOT/traj"
LOG="$OUT_ROOT/run.log"
echo "=== V12 near-miss validate routes=[$ROUTES] -> $OUT_ROOT ===" | tee "$LOG"

if [[ -x "$RECOVER" ]]; then
  echo "[$(date '+%H:%M:%S')] recover_renderer" | tee -a "$LOG"
  bash "$RECOVER" >>"$LOG" 2>&1
  sleep 5
fi

set +e
"$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
  "${WAM_PHASE2_CKPTS[@]}" --routes "$ROUTES" \
  --traj-out "$OUT_ROOT/traj" \
  --out "$OUT_ROOT/nearmiss.json" \
  "${WAM_PHASE2_V12_STACK[@]}" 2>&1 | tee -a "$LOG"
set -e

"$PY" << PY | tee -a "$LOG"
import json
from pathlib import Path
d = json.load(open("${OUT_ROOT}/nearmiss.json"))
eps = sorted(d["episodes"], key=lambda x: x["route_idx"])
arr = sum(1 for e in eps if e["arrived"])
print(f"\nV12 near-miss: SR {arr}/{len(eps)}")
for e in eps:
    ri = e["route_idx"] + 1
    att = e.get("near_miss_attempt", 1)
    print(
        f"  R{ri:02d} arr={e['arrived']} d_min={e['d_min_m']:.1f} "
        f"SPL={e['spl']:.3f} attempt={att}"
    )
PY
