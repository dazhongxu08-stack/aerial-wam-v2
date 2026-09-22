#!/usr/bin/env bash
# Shield ablation on urban focus routes (0,1,3,4): off vs forward-only exclusion.
# Compare intervention_rate / SR / path length vs baseline20 shield-on table.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
ACTOR="${ACTOR:-experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt}"
# Panel route indices 0,1,3,4 into outdoor_complex_only (not focus134 row indices 0..3).
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
ROUTES="${ROUTES:-0,1,3,4}"
MIN_SPAWN_Z="${MIN_SPAWN_Z:-38.0}"
SPAWN_RETRY_M="${SPAWN_RETRY_M:-14.0}"
SPAWN_Z_MAX_RETRIES="${SPAWN_Z_MAX_RETRIES:-3}"
OUT_ROOT="${OUT_ROOT:-artifacts/urban_complex_shield_ablation_${STAMP}}"
BASELINE20="${BASELINE20:-artifacts/urban_complex_baseline20_20260917_z38base/eval_all.json}"

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run.log"
echo "=== shield ablation actor=$ACTOR routes=[$ROUTES] -> $OUT_ROOT ===" | tee "$LOG"

if pgrep -f "train_v4_ac.*urban_complex" >/dev/null 2>&1; then
  echo "ERROR: urban p2c training still running — stop or wait before eval (AirSim single-consumer)" | tee -a "$LOG"
  exit 2
fi

if [[ -x "$RECOVER" ]]; then
  bash "$RECOVER" >>"$LOG" 2>&1 || true
  sleep 8
fi

export ACTOR
# shellcheck disable=SC1091
source experiments/aerial/scripts/wam_phase2_v11_mainline.inc.sh
COMMON=(
  "${WAM_PHASE2_CKPTS[@]}"
  --annotation "$ANNO"
  --routes "$ROUTES"
  --subgoal-source toward_g
  --r-m-intent 100
  --planner --planner-horizon 1
  --tti-coeff 2.5
  --heading-reentry-cos 0.5
  --cte-reentry-m 1.5
  --cruise-speed 10.0
  --max-steps 600
  --success-dist 3.0
  --min-spawn-z "$MIN_SPAWN_Z"
  --spawn-z-retry-m "$SPAWN_RETRY_M"
  --spawn-z-max-retries "$SPAWN_Z_MAX_RETRIES"
)

run_arm() {
  local tag="$1"
  shift
  local out="$OUT_ROOT/eval_${tag}.json"
  echo "[$(date '+%H:%M:%S')] arm=$tag -> $out" | tee -a "$LOG"
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${COMMON[@]}" \
    "$@" \
    --out "$out" \
    --traj-out "$OUT_ROOT/traj_${tag}" 2>&1 | tee -a "$LOG"
}

run_arm shield_off --no-shield
run_arm shield_fwd_only --shield-exclusion-forward-only

"$PY" - <<'PY' "$OUT_ROOT" "$BASELINE20" | tee -a "$LOG"
import json, sys
from pathlib import Path

out_root = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
focus = {0, 1, 3, 4}

def load(p):
    return json.loads(p.read_text()) if p.is_file() else {"episodes": []}

def row(e):
    return {
        "route": int(e.get("route_idx", -1)),
        "arr": bool(e.get("arrived")),
        "prog": round(float(e.get("progress_ratio", 0)) * 100, 1),
        "ir": round(float(e.get("intervention_rate", 0)), 3),
        "len_m": round(float(e.get("actual_length_m", 0)), 1),
        "min_d": e.get("d_min_m"),
        "coll": bool(e.get("collided")),
    }

base_eps = [e for e in load(baseline_path).get("episodes", []) if int(e.get("route_idx", -1)) in focus]
print("\n=== compare focus routes (shield-on baseline20 vs ablation) ===")
print(f"{'route':>5} {'arm':14} {'arr':5} {'prog%':>6} {'IR':>6} {'len_m':>7} {'coll':5}")
for e in sorted(base_eps, key=lambda x: int(x["route_idx"])):
    r = row(e)
    print(f"{r['route']:5d} {'shield_on_ref':14} {str(r['arr']):5} {r['prog']:6.1f} {r['ir']:6.3f} {r['len_m']:7.1f} {str(r['coll']):5}")

for tag in ("shield_off", "shield_fwd_only"):
    eps = [e for e in load(out_root / f"eval_{tag}.json").get("episodes", []) if int(e.get("route_idx", -1)) in focus]
    for e in sorted(eps, key=lambda x: int(x["route_idx"])):
        r = row(e)
        print(f"{r['route']:5d} {tag:14} {str(r['arr']):5} {r['prog']:6.1f} {r['ir']:6.3f} {r['len_m']:7.1f} {str(r['coll']):5}")
PY

echo "[$(date '+%H:%M:%S')] shield ablation done -> $OUT_ROOT" | tee -a "$LOG"
