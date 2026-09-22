#!/usr/bin/env bash
# R15 terminal closure: pin=30m, creep=20m, recover on near-miss retry, 3 reps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v12_mainline.inc.sh"
PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"

OUT_ROOT="${OUT_ROOT:-artifacts/wam_phase2_v12_route15_validate_20260916}"
ROUTE_IDX=14
mkdir -p "$OUT_ROOT/traj"
LOG="$OUT_ROOT/run.log"

recover_renderer() {
  if [[ -x "$RECOVER" ]]; then
    echo "[$(date '+%H:%M:%S')] recover_renderer" | tee -a "$LOG"
    bash "$RECOVER" >>"$LOG" 2>&1
    sleep 5
  fi
}

run_rep() {
  local tag="$1"
  local out="$OUT_ROOT/${tag}.json"
  echo "=== ${tag} ===" | tee -a "$LOG"
  recover_renderer
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" \
    --routes "$ROUTE_IDX" \
    --traj-out "$OUT_ROOT/traj/${tag}" \
    --out "$out" \
    "${WAM_PHASE2_V12_STACK[@]}" \
    --terminal-pin-rem-m 30 \
    --terminal-creep-rem-m 20 \
    --recover-on-near-miss-retry \
    --recover-script "$RECOVER" \
    2>&1 | tee -a "$LOG"
  set -e
}

echo "=== V12 R15 validate (3 reps) -> $OUT_ROOT ===" | tee "$LOG"
for i in 1 2 3; do
  run_rep "r${i}"
done

"$PY" << PY | tee -a "$LOG"
import json
from pathlib import Path
out = Path("${OUT_ROOT}")
rows = []
for p in sorted(out.glob("r*.json")):
    e = json.load(open(p))["episodes"][0]
    rows.append(e)
    print(p.stem, "arr", e["arrived"], "d_min", e["d_min_m"], "att", e.get("near_miss_attempt", 1))
arr = sum(1 for e in rows if e["arrived"])
print(f"\nR15: {arr}/{len(rows)} arrived")
PY
