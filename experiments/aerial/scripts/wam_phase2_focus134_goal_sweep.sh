#!/usr/bin/env bash
# focus134 goal-alignment diagnostic: toward_g on routes 0,1,3,4 + terminal_pin / success_dist sweep.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
ACTOR="${ACTOR:-experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_20260916_night_focus134/v4_ac_latest.pt}"
export ACTOR
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v11_mainline.inc.sh"

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
ROUTES="${ROUTES:-0,1,3,4}"
LOG_ROOT="${LOG_ROOT:-artifacts/wam_phase2_focus134_goal_sweep_20260917}"
mkdir -p "$LOG_ROOT"

TERMINAL_STACK=(
  --subgoal-source toward_g
  --r-m-intent 100
  --planner --planner-horizon 1
  --tti-coeff 2.5
  --heading-reentry-cos 0.5
  --cte-reentry-m 1.5
  --cruise-speed 10.0
  --max-steps 600
)

run_case() {
  local tag="$1"
  local success_dist="$2"
  local pin_rem="$3"
  local direct_rem="$4"
  local out="$LOG_ROOT/${tag}"
  if [[ -f "$out/eval_all.json" ]]; then
    echo "[goal-sweep] SKIP ${tag} (eval_all.json exists)" | tee -a "$LOG_ROOT/resume.log"
    return 0
  fi
  mkdir -p "$out/traj"
  echo "=== CASE ${tag} sd=${success_dist} pin=${pin_rem} direct=${direct_rem} routes=[${ROUTES}] ===" | tee "$out/run.log"
  if [[ -x "$RECOVER" ]]; then
    bash "$RECOVER" >>"$out/run.log" 2>&1 || true
    sleep 8
  fi
  local extra=(--terminal-pin-rem-m "$pin_rem")
  if [[ "$direct_rem" != "0" ]]; then
    extra+=(--terminal-direct-rem-m "$direct_rem" --terminal-direct-d-m 25)
  fi
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" \
    --annotation "$ANNO" \
    --routes "$ROUTES" \
    --success-dist "$success_dist" \
    --traj-out "$out/traj" \
    --out "$out/eval_all.json" \
    "${TERMINAL_STACK[@]}" \
    "${extra[@]}" 2>&1 | tee -a "$out/run.log"
  "$PY" - <<'PY' "$out" "$tag" | tee -a "$out/run.log"
import json, sys
from pathlib import Path
out, tag = Path(sys.argv[1]), sys.argv[2]
rows = json.loads((out / "eval_all.json").read_text()).get("episodes", [])
n = len(rows)
arr = sum(1 for r in rows if r.get("arrived"))
print(f"[{tag}] SR {arr}/{n} = {arr/n:.1%}" if n else f"[{tag}] no results")
for r in sorted(rows, key=lambda x: int(x.get("route_idx", -1))):
    print(
        f"  route={r.get('route_idx')} arr={r.get('arrived')} "
        f"prog={float(r.get('progress_ratio',0))*100:.1f}% "
        f"min_d={r.get('d_min_m')} d_fin={r.get('d_final_m')}"
    )
PY
}

echo "[goal-sweep] START actor=$ACTOR routes=[$ROUTES] -> $LOG_ROOT"
run_case "term_sd3" 3 0 0
run_case "term_sd5" 5 0 0
run_case "term_pin20_sd3" 3 20 25
run_case "term_pin20_sd5" 5 20 25
echo "[goal-sweep] DONE -> $LOG_ROOT/summary.txt"
"$PY" - <<'PY' "$LOG_ROOT"
import json, glob, sys
from pathlib import Path
root = Path(sys.argv[1])
lines = ["focus134 goal-alignment sweep summary", ""]
for p in sorted(root.glob("*/eval_all.json")):
    tag = p.parent.name
    rows = json.load(open(p)).get("episodes", [])
    arr = sum(1 for r in rows if r.get("arrived"))
    best = max((float(r.get("progress_ratio",0)) for r in rows), default=0)
    lines.append(f"{tag}: SR {arr}/{len(rows)} best_prog={best*100:.1f}%")
lines.append("")
Path(root / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
