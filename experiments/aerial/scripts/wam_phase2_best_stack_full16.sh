#!/usr/bin/env bash
# 16-route eval — V12 mainline stack (2026-09-16).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v12_mainline.inc.sh"
PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"

OUT_ROOT="${OUT_ROOT:-artifacts/wam_phase2_v12_full16_20260916}"
mkdir -p "$OUT_ROOT"
OUT_JSON="${OUT_ROOT}/full16.json"
LOG="${OUT_ROOT}/run.log"

echo "=== Phase-2 V12 mainline 16-route eval -> ${OUT_JSON} ===" | tee "$LOG"

set +e
"$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
  "${WAM_PHASE2_CKPTS[@]}" \
  "${WAM_PHASE2_V12_STACK[@]}" \
  --traj-out "${OUT_ROOT}/traj" \
  --out "$OUT_JSON" 2>&1 | tee -a "$LOG"
set -e

"$PY" << PY
import json, statistics as st
d = json.load(open("${OUT_JSON}"))
eps = d["episodes"]
arr = [e for e in eps if e["arrived"]]
spls = [e["spl"] for e in arr if e["spl"]]
ge07 = sum(1 for s in spls if s >= 0.70)
print(f"\nSR={len(arr)}/{len(eps)} ({100*len(arr)/len(eps):.1f}%)")
if spls:
    print(f"SPL mean={st.fmean(spls):.3f} med={st.median(spls):.3f}  SPL>=0.7: {ge07}/{len(arr)}")
    for e in sorted(arr, key=lambda x: -x["spl"])[:5]:
        print(f"  R{e['route_idx']+1:02d} SPL={e['spl']:.3f} L={e['actual_length_m']:.0f}m")
PY
