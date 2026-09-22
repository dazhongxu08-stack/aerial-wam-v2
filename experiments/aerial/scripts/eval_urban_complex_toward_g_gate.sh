#!/usr/bin/env bash
# Post-train gate: toward_g eval on urban-complex focus routes (train-aligned deploy).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh
# shellcheck disable=SC1091
ACTOR="${ACTOR:-${1:-experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt}}"
export ACTOR
# Match urban OA FT / yaml default (depth-aux). Override with WM=…/wm_ckpt_d_full_*
# for Aug-28 A/B. Must be set BEFORE sourcing mainline.inc (reads ${WM:-…}).
export WM="${WM:-experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
# shellcheck disable=SC1091
source experiments/aerial/scripts/wam_phase2_v11_mainline.inc.sh

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
OUT_ROOT="${OUT_ROOT:-artifacts/urban_complex_toward_g_gate_${STAMP}}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
ROUTES="${ROUTES:-0,1,3,4}"
MIN_SPAWN_Z="${MIN_SPAWN_Z:-24.0}"
SPAWN_RETRY_M="${SPAWN_RETRY_M:-5.0}"
SPAWN_Z_MAX_RETRIES="${SPAWN_Z_MAX_RETRIES:-4}"
SHIELD_EXCLUSION_FORWARD_ONLY="${SHIELD_EXCLUSION_FORWARD_ONLY:-1}"
# Imagination steps. Training urban imagination horizon is 15.
PLANNER_HORIZON="${PLANNER_HORIZON:-5}"
# open_loop = frozen hand-rule scoring. closed_loop = actor tail, WM return only.
PLANNER_ROLLOUT="${PLANNER_ROLLOUT:-open_loop}"
# Optional ablation: rules | wm_bare | pass (empty = full scoring for that rollout).
PLANNER_MOCK="${PLANNER_MOCK:-}"

mkdir -p "$OUT_ROOT/traj"
LOG="$OUT_ROOT/run.log"
echo "=== urban complex toward_g gate actor=$ACTOR wm=$WM routes=[$ROUTES] horizon=$PLANNER_HORIZON rollout=$PLANNER_ROLLOUT mock=${PLANNER_MOCK:-none} -> $OUT_ROOT ===" | tee "$LOG"

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
  --planner --planner-horizon "$PLANNER_HORIZON" \
  --planner-rollout "$PLANNER_ROLLOUT" \
  ${PLANNER_MOCK:+--planner-mock "$PLANNER_MOCK"} \
  --tti-coeff 2.5 \
  --heading-reentry-cos 0.5 \
  --cte-reentry-m 1.5 \
  --cruise-speed 10.0 \
  --max-steps 600 \
  --success-dist 3.0 \
  --terminal-pin-rem-m 20 \
  --min-spawn-z "$MIN_SPAWN_Z" \
  --spawn-z-retry-m "$SPAWN_RETRY_M" \
  --spawn-z-max-retries "$SPAWN_Z_MAX_RETRIES" \
  ${SHIELD_EXCLUSION_FORWARD_ONLY:+--shield-exclusion-forward-only} 2>&1 | tee -a "$LOG"

"$PY" - <<'PY' "$OUT_ROOT" | tee -a "$LOG"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = json.loads((out / "eval_all.json").read_text()).get("episodes", [])
n = len(rows)
arr = sum(1 for r in rows if r.get("arrived"))
print(f"[gate] SR {arr}/{n} = {arr/n:.1%}" if n else "[gate] no results")
for r in sorted(rows, key=lambda x: int(x.get("route_idx", -1))):
    print(
        f"  route={r.get('route_idx')} arr={r.get('arrived')} "
        f"prog={float(r.get('progress_ratio',0))*100:.1f}% "
        f"min_d={r.get('d_min_m')} d_fin={r.get('d_final_m')}"
    )
PY

echo "[$(date '+%H:%M:%S')] gate done" | tee -a "$LOG"
