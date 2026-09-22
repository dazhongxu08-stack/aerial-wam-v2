#!/usr/bin/env bash
# Diagnostic SR eval: terminal-goal stack (toward_g, no polyline / rolling-global).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
ACTOR="${ACTOR:-experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_20260916_night/v4_ac_latest.pt}"
export ACTOR
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v11_mainline.inc.sh"

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
OUT_ROOT="${OUT_ROOT:-artifacts/wam_phase2_v11_interior_terminal_eval_20260916_night}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
ROUTES="${ROUTES:-0,1,3,4,5,10,19}"

WAM_PHASE2_TERMINAL_STACK=(
  --subgoal-source toward_g
  --r-m-intent 100
  --planner --planner-horizon 1
  --tti-coeff 2.5
  --heading-reentry-cos 0.5
  --cte-reentry-m 1.5
  --cruise-speed 10.0
  --max-steps 600
)

mkdir -p "$OUT_ROOT/traj"
LOG="$OUT_ROOT/run.log"
echo "=== terminal-goal SR eval routes=[$ROUTES] actor=$ACTOR -> $OUT_ROOT ===" | tee "$LOG"

if [[ -x "$RECOVER" ]]; then
  bash "$RECOVER" >>"$LOG" 2>&1 || true
  sleep 8
fi

"$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
  "${WAM_PHASE2_CKPTS[@]}" \
  --annotation "$ANNO" \
  --routes "$ROUTES" \
  --traj-out "$OUT_ROOT/traj" \
  --out "$OUT_ROOT/eval_all.json" \
  "${WAM_PHASE2_TERMINAL_STACK[@]}" 2>&1 | tee -a "$LOG"

"$PY" - <<'PY' "$OUT_ROOT" | tee -a "$LOG"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = json.loads((out / "eval_all.json").read_text()).get("episodes", [])
n = len(rows)
arr = sum(1 for r in rows if r.get("arrived"))
print(f"[SR-terminal] routes={n} arrived={arr} SR={arr/n:.1%}" if n else "[SR-terminal] no results")
for r in sorted(rows, key=lambda x: int(x.get("route_idx", -1))):
    print(
        f"  route={r.get('route_idx')} arrived={r.get('arrived')} "
        f"prog={float(r.get('progress_ratio', 0))*100:.1f}%"
    )
PY

echo "[$(date '+%H:%M:%S')] terminal SR eval done" | tee -a "$LOG"
