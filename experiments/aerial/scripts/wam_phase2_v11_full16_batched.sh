#!/usr/bin/env bash
# V12 full16 batched with recover_renderer every 4 routes (AirSim drift mitigation).
# Run on 125: bash experiments/aerial/scripts/wam_phase2_v11_full16_batched.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v12_mainline.inc.sh"
PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"

OUT_ROOT="${OUT_ROOT:-artifacts/wam_phase2_v12_full16_batched_20260916}"
CHUNK_DIR="${OUT_ROOT}/chunks"
FINAL_JSON="${OUT_ROOT}/full16.json"
LOG="${OUT_ROOT}/run.log"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"

mkdir -p "$CHUNK_DIR" "${OUT_ROOT}/traj"
echo "=== V12 full16 batched (recover every 4 routes) -> ${FINAL_JSON} ===" | tee "$LOG"

recover_renderer() {
  echo "[$(date '+%H:%M:%S')] recover_renderer" | tee -a "$LOG"
  if [[ -x "$RECOVER" ]]; then
    bash "$RECOVER" >>"$LOG" 2>&1
    sleep 5
  else
    echo "WARN: missing $RECOVER" | tee -a "$LOG"
  fi
}

# Batches: 0-3, 4-7, 8-11, 12-15
BATCHES=("0,1,2,3" "4,5,6,7" "8,9,10,11" "12,13,14,15")
for i in "${!BATCHES[@]}"; do
  routes="${BATCHES[$i]}"
  chunk="${CHUNK_DIR}/chunk_${i}.json"
  echo "[$(date '+%H:%M:%S')] batch $i routes=[$routes]" | tee -a "$LOG"
  recover_renderer
  # Eval exits 1 on FAIL verdict; do not abort remaining batches.
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" \
    "${WAM_PHASE2_V12_STACK[@]}" \
    --routes "$routes" \
    --traj-out "${OUT_ROOT}/traj" \
    --out "$chunk" 2>&1 | tee -a "$LOG"
  eval_rc=${PIPESTATUS[0]}
  set -e
  if [[ "$eval_rc" -ne 0 ]]; then
    echo "WARN: batch $i eval exit=$eval_rc (verdict may be FAIL); continuing" | tee -a "$LOG"
  fi
done

echo "[$(date '+%H:%M:%S')] merge chunks" | tee -a "$LOG"
"$PY" -m experiments.aerial.scripts.merge_phase2_split_eval \
  --out "$FINAL_JSON" \
  "${CHUNK_DIR}"/chunk_0.json \
  "${CHUNK_DIR}"/chunk_1.json \
  "${CHUNK_DIR}"/chunk_2.json \
  "${CHUNK_DIR}"/chunk_3.json \
  2>&1 | tee -a "$LOG"

"$PY" << PY
import json, statistics as st
d = json.load(open("${FINAL_JSON}"))
eps = d["episodes"]
arr = [e for e in eps if e["arrived"]]
spls = [e["spl"] for e in arr if e["spl"]]
ge07 = sum(1 for s in spls if s >= 0.70)
print(f"\nSR={len(arr)}/{len(eps)} ({100*len(arr)/len(eps):.1f}%)")
print(f"verdict={d.get('verdict')}")
if spls:
    print(f"SPL mean={st.fmean(spls):.3f} med={st.median(spls):.3f}  SPL>=0.7: {ge07}/{len(arr)}")
for e in sorted(eps, key=lambda x: x["route_idx"]):
    ri = e["route_idx"]
    print(f"  R{ri+1:02d} arr={e['arrived']} SPL={e['spl']:.3f} L={e['actual_length_m']:.0f}m d_min={e['d_min_m']:.1f}")
PY
