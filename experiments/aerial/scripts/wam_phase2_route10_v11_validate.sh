#!/usr/bin/env bash
# Route-10 repeat validation for V12 mainline stack.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v12_mainline.inc.sh"
PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"

ROUTE_IDX=9
OUT_ROOT="${OUT_ROOT:-artifacts/wam_phase2_route10_v12_validate_20260916}"

run_rep() {
  local tag="$1"
  local out="${OUT_ROOT}/${tag}.json"
  local traj="${OUT_ROOT}/${tag}_traj"
  mkdir -p "$OUT_ROOT" "$traj"
  echo "=== ${tag} ==="
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" --routes "$ROUTE_IDX" \
    --traj-out "${traj}/route09.jsonl" --out "$out" \
    "${WAM_PHASE2_V12_STACK[@]}" || true
}

summary() {
  "$PY" << PY
import json, statistics as st
from pathlib import Path
out = Path("${OUT_ROOT}")
rows = []
for p in sorted(out.glob("v11_r*.json")):
    e = json.load(open(p))["episodes"][0]
    rows.append(e)
    print(p.stem, "arr", e["arrived"], "SPL", round(e["spl"], 3), "L", round(e["actual_length_m"], 1))
arr = [e for e in rows if e["arrived"]]
ge07 = [e for e in arr if e["spl"] >= 0.70]
print(f"\n{len(arr)}/{len(rows)} arrived, {len(ge07)}/{len(rows)} SPL>=0.7")
if arr:
    print(f"SPL mean={st.fmean(e['spl'] for e in arr):.3f}  L mean={st.fmean(e['actual_length_m'] for e in arr):.1f}")
PY
}

case "${1:-all}" in
  all) run_rep v11_r1; run_rep v11_r2; summary ;;
  summary) summary ;;
  v11_r1) run_rep v11_r1 ;;
  v11_r2) run_rep v11_r2 ;;
  *) echo "Usage: $0 [all|summary|v11_r1|v11_r2]"; exit 1 ;;
esac
