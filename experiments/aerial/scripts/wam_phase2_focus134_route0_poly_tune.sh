#!/usr/bin/env bash
# focus134 + V11 polyline: terminal-tuning sweep on route 0 (baseline min_d~21m, prog~89%).
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
ROUTES="${ROUTES:-0}"
LOG_ROOT="${LOG_ROOT:-artifacts/wam_phase2_focus134_route0_poly_tune_20260917}"
mkdir -p "$LOG_ROOT"

run_arm() {
  local tag="$1"
  shift
  local out="$LOG_ROOT/${tag}"
  if [[ -f "$out/eval_all.json" ]]; then
    echo "[route0-tune] SKIP $tag" | tee -a "$LOG_ROOT/resume.log"
    return 0
  fi
  mkdir -p "$out/traj"
  echo "=== ARM ${tag} routes=[${ROUTES}] ===" | tee "$out/run.log"
  if [[ -x "$RECOVER" ]]; then
    bash "$RECOVER" >>"$out/run.log" 2>&1 || true
    sleep 8
  fi
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" \
    --annotation "$ANNO" \
    --routes "$ROUTES" \
    --traj-out "$out/traj" \
    --out "$out/eval_all.json" \
    "${WAM_PHASE2_V11_STACK[@]}" \
    "$@" 2>&1 | tee -a "$out/run.log"
  "$PY" - <<'PY' "$out" "$tag" | tee -a "$out/run.log"
import json, sys
from pathlib import Path
out, tag = Path(sys.argv[1]), sys.argv[2]
rows = json.loads((out / "eval_all.json").read_text()).get("episodes", [])
for r in rows:
    print(
        f"[{tag}] route={r.get('route_idx')} arr={r.get('arrived')} "
        f"prog={float(r.get('progress_ratio',0))*100:.1f}% "
        f"min_d={r.get('d_min_m')} d_fin={r.get('d_final_m')} steps={r.get('steps')}"
    )
PY
}

echo "[route0-tune] START actor=$ACTOR -> $LOG_ROOT"
# Reproduce baseline (long_eval default terminal_pin_rem_m=20).
run_arm "v11_base"
# Strong terminal pull: subgoal=goal when rem<=30m and eucl<=25m.
run_arm "direct30" \
  --terminal-direct-rem-m 30 --terminal-direct-d-m 25
# Slower cruise + direct (route-10 sweep winner pattern).
run_arm "cs8_direct30" \
  --cruise-speed 8.0 \
  --terminal-direct-rem-m 30 --terminal-direct-d-m 25
# V12 stack: pin20 + creep15 + near-miss retry.
run_arm "v12_nearmiss" \
  --terminal-pin-rem-m 20 --terminal-creep-rem-m 15 \
  --near-miss-retry-m 5 --near-miss-retries 1
# Wider pin + creep for last 25m.
run_arm "pin30_creep25_direct30" \
  --terminal-pin-rem-m 30 --terminal-creep-rem-m 25 \
  --terminal-direct-rem-m 30 --terminal-direct-d-m 25
# Relaxed success + direct (only helps if d_fin < 5m).
run_arm "sd5_direct30" \
  --success-dist 5 \
  --terminal-direct-rem-m 30 --terminal-direct-d-m 25

echo "[route0-tune] DONE -> $LOG_ROOT/summary.txt"
"$PY" - <<'PY' "$LOG_ROOT"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
lines = ["focus134 route0 polyline terminal tuning", ""]
for p in sorted(root.glob("*/eval_all.json")):
    tag = p.parent.name
    r = json.load(open(p)).get("episodes", [{}])[0]
    lines.append(
        f"{tag}: arr={r.get('arrived')} prog={float(r.get('progress_ratio',0))*100:.1f}% "
        f"min_d={r.get('d_min_m')} d_fin={r.get('d_final_m')} steps={r.get('steps')}"
    )
Path(root / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
