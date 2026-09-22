#!/usr/bin/env bash
# Step 1: 20-route toward_g baseline table (frozen phase2_toward_g, flyable spawn).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh
# shellcheck disable=SC1091
ACTOR="${ACTOR:-experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt}"
export ACTOR
# shellcheck disable=SC1091
source experiments/aerial/scripts/wam_phase2_v11_mainline.inc.sh

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
STAMP="${STAMP:-$(date +%Y%m%d)_z38base}"
OUT_ROOT="${OUT_ROOT:-artifacts/urban_complex_baseline20_${STAMP}}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
ROUTES="${ROUTES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"
MIN_SPAWN_Z="${MIN_SPAWN_Z:-38.0}"
SPAWN_RETRY_M="${SPAWN_RETRY_M:-14.0}"
SPAWN_Z_MAX_RETRIES="${SPAWN_Z_MAX_RETRIES:-3}"

mkdir -p "$OUT_ROOT/traj"
LOG="$OUT_ROOT/run.log"
echo "=== baseline20 actor=$ACTOR z>=${MIN_SPAWN_Z} routes=[$ROUTES] -> $OUT_ROOT ===" | tee "$LOG"

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
  --subgoal-source toward_g \
  --r-m-intent 100 \
  --planner --planner-horizon 1 \
  --tti-coeff 2.5 \
  --heading-reentry-cos 0.5 \
  --cte-reentry-m 1.5 \
  --cruise-speed 10.0 \
  --max-steps 600 \
  --success-dist 3.0 \
  --terminal-pin-rem-m 20 \
  --min-spawn-z "$MIN_SPAWN_Z" \
  --spawn-z-retry-m "$SPAWN_RETRY_M" \
  --spawn-z-max-retries "$SPAWN_Z_MAX_RETRIES" 2>&1 | tee -a "$LOG"

"$PY" - <<'PY' "$OUT_ROOT" | tee -a "$LOG"
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = json.loads((out / "eval_all.json").read_text()).get("episodes", [])
n = len(rows)
arr = sum(1 for r in rows if r.get("arrived"))
print(f"[baseline20] SR {arr}/{n} = {arr/n:.1%}" if n else "[baseline20] no results")

def cls(r):
    if r.get("spawn_fail"):
        return "SPAWN_DEAD"
    if r.get("arrived"):
        return "ARRIVED"
    prog = float(r.get("progress_ratio", 0)) * 100
    dmin = float(r.get("d_min_m") or 999)
    if prog >= 70 or dmin <= 20:
        return "NEAR_MISS"
    if prog >= 20:
        return "PARTIAL"
    if prog >= 5:
        return "WEAK"
    return "NO_PROGRESS"

print(f"{'route':>5s} {'class':12s} {'arr':5s} {'prog':>6s} {'min_d':>7s} {'d_fin':>7s}")
for r in sorted(rows, key=lambda x: int(x.get("route_idx", -1))):
    print(
        f"{int(r.get('route_idx',-1)):5d} {cls(r):12s} {str(r.get('arrived')):5s} "
        f"{float(r.get('progress_ratio',0))*100:6.1f}% "
        f"{str(r.get('d_min_m','')):>7s} {str(r.get('d_final_m','')):>7s}"
    )
from collections import Counter
print("CLASS:", dict(Counter(cls(r) for r in rows)))
trainable = [int(r["route_idx"]) for r in rows if cls(r) in ("ARRIVED", "NEAR_MISS", "PARTIAL")]
print("TRAINABLE_ROUTES:", ",".join(str(x) for x in trainable))
PY

echo "[$(date '+%H:%M:%S')] baseline20 done" | tee -a "$LOG"
