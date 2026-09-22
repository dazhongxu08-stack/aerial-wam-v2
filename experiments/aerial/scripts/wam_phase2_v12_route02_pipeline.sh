#!/usr/bin/env bash
# R02 forensics (V12 traj) + terminal homing / shield-relax experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v12_mainline.inc.sh"
PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"

OUT_ROOT="${OUT_ROOT:-artifacts/wam_phase2_v12_route02_pipeline_20260916}"
ROUTE_IDX=1
BASE_TRAJ="${BASE_TRAJ:-artifacts/wam_phase2_v12_nearmiss_20260916/traj/route01.jsonl}"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run.log"

recover_renderer() {
  if [[ -x "$RECOVER" ]]; then
    echo "[$(date '+%H:%M:%S')] recover_renderer" | tee -a "$LOG"
    bash "$RECOVER" >>"$LOG" 2>&1
    sleep 5
  fi
}

echo "=== R02 pipeline -> $OUT_ROOT ===" | tee "$LOG"

if [[ -f "$BASE_TRAJ" ]]; then
  echo "--- forensics on $BASE_TRAJ ---" | tee -a "$LOG"
  "$PY" << PY | tee -a "$LOG"
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("${BASE_TRAJ}").read_text().splitlines() if l.strip()]
d2g = [r["d_to_g"] for r in rows]
imin = min(range(len(d2g)), key=lambda i: d2g[i])
ir = sum(1 for r in rows if r.get("intervened"))
last = rows[-50:]
print(f"steps={len(rows)} d_min={min(d2g):.2f}@step{imin} d_end={d2g[-1]:.2f}")
print(f"IR={ir/len(rows):.1%}  last50 IR={sum(1 for r in last if r.get('intervened'))/len(last):.1%}")
print(f"last50 d2g: min={min(r['d_to_g'] for r in last):.1f} max={max(r['d_to_g'] for r in last):.1f}")
PY
else
  echo "WARN: missing $BASE_TRAJ" | tee -a "$LOG"
fi

run_eval() {
  local tag="$1"
  shift
  local out="$OUT_ROOT/${tag}.json"
  echo "=== eval ${tag} ===" | tee -a "$LOG"
  recover_renderer
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" \
    --routes "$ROUTE_IDX" \
    --traj-out "$OUT_ROOT/traj/${tag}" \
    --out "$out" \
    "${WAM_PHASE2_V12_STACK[@]}" \
    "$@" \
    2>&1 | tee -a "$LOG"
  set -e
}

# Baseline V12 on R02
run_eval "v12_baseline"

# Terminal direct goal homing (R02 hypothesis: shield+polyline can't close)
run_eval "v12_terminal_direct" \
  --terminal-direct-rem-m 25 \
  --terminal-direct-d-m 20

# Terminal shield relax (high IR on R02)
run_eval "v12_terminal_shield" \
  --terminal-shield-tti-rem-m 25 \
  --terminal-shield-tti-cte-m 4 \
  --terminal-shield-tti-scale 1.35 \
  --shield-tti-hysteresis 0.10

# Combined
run_eval "v12_combined" \
  --terminal-direct-rem-m 25 \
  --terminal-direct-d-m 20 \
  --terminal-shield-tti-rem-m 25 \
  --terminal-shield-tti-cte-m 4 \
  --terminal-shield-tti-scale 1.35 \
  --shield-tti-hysteresis 0.10

"$PY" << PY | tee -a "$LOG"
import json
from pathlib import Path
out = Path("${OUT_ROOT}")
print("\n=== R02 experiment summary ===")
for p in sorted(out.glob("v12_*.json")):
    e = json.load(open(p))["episodes"][0]
    print(f"{p.stem:22s} arr={e['arrived']} d_min={e['d_min_m']:5.1f} IR={e['intervention_rate']:.2f} L={e['actual_length_m']:.0f}")
PY
