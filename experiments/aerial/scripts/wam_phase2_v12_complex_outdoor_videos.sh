#!/usr/bin/env bash
# V12 stack: fly + record dual-view video for each outdoor-complex route (6 default).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v12_mainline.inc.sh"

PY="${AERIAL_PY:-${PYTHON_BIN:-python3}}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
OUT_ROOT="${OUT_ROOT:-artifacts/wam_phase2_v12_complex_outdoor_videos_20260916}"
ANNO_COMPLEX="${ANNO_COMPLEX:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
ROUTES="${ROUTES:-0,1,2,3,4,5}"

mkdir -p "$OUT_ROOT/traj" "$OUT_ROOT/videos"
LOG="$OUT_ROOT/run.log"

echo "=== V12 complex outdoor videos routes=[$ROUTES] -> $OUT_ROOT ===" | tee "$LOG"

"$PY" -m experiments.aerial.scripts.build_phase3_outdoor_complex_annotation \
  --regenerate --out "$ANNO_COMPLEX" 2>&1 | tee -a "$LOG"

IFS=',' read -r -a ROUTE_ARR <<< "$ROUTES"
for idx in "${ROUTE_ARR[@]}"; do
  idx="${idx// /}"
  [[ -z "$idx" ]] && continue
  label="$("$PY" -c "
import json
from pathlib import Path
ri = int('${idx}')
anno = json.loads(Path('${ANNO_COMPLEX}').read_text())
ep = anno['episodes'][ri]
meta = ep.get('complexity_meta') or {}
print(meta.get('route_label', f'R{ri+1:02d}'))
")"
  tag="V12_complex_${label}"
  traj_dir="$OUT_ROOT/traj/${label}"
  mkdir -p "$traj_dir"

  echo "" | tee -a "$LOG"
  echo "[$(date '+%H:%M:%S')] === ${label} (route_idx=${idx}) ===" | tee -a "$LOG"

  if [[ -x "$RECOVER" ]]; then
    bash "$RECOVER" >>"$LOG" 2>&1 || true
    sleep 5
  fi

  set +e
  ANNO="$ANNO_COMPLEX" "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    "${WAM_PHASE2_CKPTS[@]}" \
    --annotation "$ANNO_COMPLEX" \
    --routes "$idx" \
    --traj-out "$traj_dir" \
    --out "$OUT_ROOT/eval_${label}.json" \
    "${WAM_PHASE2_V12_STACK[@]}" 2>&1 | tee -a "$LOG"
  eval_rc=$?
  set -e

  traj_jsonl="$traj_dir/route$(printf '%02d' "$idx").jsonl"
  if [[ ! -f "$traj_jsonl" ]]; then
    echo "MISSING traj $traj_jsonl (eval_rc=$eval_rc) — skip video" | tee -a "$LOG"
    continue
  fi

  "$PY" -m experiments.aerial.scripts.wam_phase2_dual_view_from_traj \
    --traj-jsonl "$traj_jsonl" \
    --ref-polyline-json "$ANNO_COMPLEX" \
    --route-idx "$idx" \
    --plan-mode polyline \
    --route-label "V12 ${label} outdoor-complex" \
    --out-dir "$OUT_ROOT/videos" \
    --out-prefix "$tag" \
    --fps 5 --source-hz 5 2>&1 | tee -a "$LOG"

  dual="$OUT_ROOT/videos/${tag}_dual_view_dashboard.mp4"
  if [[ -f "$dual" ]]; then
    echo "OK video: $dual" | tee -a "$LOG"
  else
    echo "FAIL video for ${label}" | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "=== Video batch done -> $OUT_ROOT/videos ===" | tee -a "$LOG"
ls -la "$OUT_ROOT/videos/"*_dual_view_dashboard.mp4 2>/dev/null | tee -a "$LOG" || true
