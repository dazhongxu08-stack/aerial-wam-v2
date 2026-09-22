#!/usr/bin/env bash
# SR-focused eval: FT actor on curated train routes only (no dual-view video).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
ACTOR="${ACTOR:-experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_20260916_v2/v4_ac_latest.pt}"
export ACTOR
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v11_mainline.inc.sh"

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
OUT_ROOT="${OUT_ROOT:-artifacts/wam_phase2_v11_interior_ft_sr_eval_20260916_v2}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
# Routes present in curated PathExpert FT set (urban inland, y>=0).
ROUTES="${ROUTES:-0,1,3,4,5,10}"

mkdir -p "$OUT_ROOT/traj"
LOG="$OUT_ROOT/run.log"
echo "=== interior FT SR eval routes=[$ROUTES] actor=$ACTOR -> $OUT_ROOT ===" | tee "$LOG"

# One long_eval process for all routes: WM/depth/actor load once (~60s), then
# sequential env episodes. Per-route bash loops + recover_renderer caused false
# "hangs" (cold WM reload) and AirSim port conflicts (bind: Address already in use).
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
  "${WAM_PHASE2_V11_STACK[@]}" 2>&1 | tee -a "$LOG"

"$PY" - <<'PY' "$OUT_ROOT" | tee -a "$LOG"
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
all_path = out / "eval_all.json"
if all_path.is_file():
    rows.extend(json.loads(all_path.read_text(encoding="utf-8")).get("episodes", []))
for p in sorted(out.glob("eval_interior*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    rows.extend(d.get("episodes", []))
# de-dupe by route_idx (prefer eval_all if re-run partial dirs exist)
best = {}
for r in rows:
    ri = int(r.get("route_idx", -1))
    if ri >= 0:
        best[ri] = r
rows = [best[k] for k in sorted(best)]

n = len(rows)
arr = sum(1 for r in rows if r.get("arrived"))
print(f"[SR] routes={n} arrived={arr} SR={arr/n:.1%}" if n else "[SR] no results")
for r in rows:
    print(
        f"  route={r.get('route_idx')} arrived={r.get('arrived')} "
        f"prog={float(r.get('progress_ratio', 0))*100:.1f}% "
        f"min_d={r.get('d_min_m')} d_final={r.get('d_final_m')}"
    )
PY

echo "[$(date '+%H:%M:%S')] SR eval done" | tee -a "$LOG"
