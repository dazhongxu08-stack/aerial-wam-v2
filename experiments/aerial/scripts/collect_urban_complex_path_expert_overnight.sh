#!/usr/bin/env bash
# Overnight PathExpert urban-inland teacher densify (multi-pass append).
# Goal: enough arrived + long-path samples for interior-complex FT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
export PYTHONUNBUFFERED=1

STAMP="${STAMP:-20260916_night}"
OUT="${OUT:-experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}}"
CURATED="${CURATED:-experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}_curated}"
LOG="${LOG:-artifacts/collect_urban_complex_path_expert_${STAMP}_overnight.log}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"

# Urban inland panel (y>=0); excludes south waterfront 12-18.
URBAN_ROUTES="0,1,2,3,4,5,6,7,8,9,10,11,19"
PRIORITY_ROUTES="3,4,5,10,0,1,6,2,7,8,9,11,19"

TARGET_CURATED_USABLE="${TARGET_CURATED_USABLE:-15}"
TARGET_CURATED_ARRIVED="${TARGET_CURATED_ARRIVED:-6}"
MAX_PASSES="${MAX_PASSES:-8}"

mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[overnight-pe] $(date -Is) $*"; }

recover_airsim() {
  if [[ -x "$RECOVER" ]]; then
    say "recover_renderer"
    bash "$RECOVER" || true
    sleep 10
  fi
}

run_pass() {
  local tag="$1"
  local routes="$2"
  local min_z="$3"
  local retry_m="$4"
  local max_retry="$5"
  local append_flag=1
  if [[ "$(find "$ROOT/$OUT" -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l | tr -d ' ')" == "0" ]]; then
    append_flag=0
  fi
  say "=== PASS ${tag} routes=[${routes}] z>=${min_z} retry=${retry_m}x${max_retry} append=${append_flag} ==="
  recover_airsim
  STAMP="$STAMP" OUT="$OUT" LOG="$LOG" \
    ROUTE_INDICES="$routes" \
    MIN_SPAWN_Z="$min_z" SPAWN_RETRY_M="$retry_m" SPAWN_Z_MAX_RETRIES="$max_retry" \
    APPEND="${append_flag}" SKIP_MIN_OK_GATE=1 N=20 \
    bash experiments/aerial/scripts/collect_urban_complex_path_expert.sh
  report_stats "${tag}"
}

report_stats() {
  local tag="${1:-}"
  local npz
  npz="$(find "$ROOT/$OUT" -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
  say "stats after ${tag}: npz_total=${npz}"
  if [[ -f "$ROOT/experiments/aerial/scripts/review_urban_complex_dataset.py" ]]; then
    "$AERIAL_PY" experiments/aerial/scripts/review_urban_complex_dataset.py \
      --src "$OUT" \
      --dst "$CURATED" \
      --write-curated 2>&1 | tee -a "$LOG" || true
    if [[ -f "$ROOT/$CURATED/manifest.json" ]]; then
      "$AERIAL_PY" - <<'PY' "$ROOT/$CURATED/manifest.json"
import json, sys
m = json.load(open(sys.argv[1]))
eps = m.get("episodes", [])
u = sum(1 for e in eps if e.get("usable"))
a = sum(1 for e in eps if e.get("arrived"))
print(f"[overnight-pe] curated usable={u} arrived={a} total={len(eps)}")
PY
    fi
  fi
}

curated_ok() {
  [[ -f "$ROOT/$CURATED/manifest.json" ]] || return 1
  local u a
  u="$("$AERIAL_PY" - <<'PY' "$ROOT/$CURATED/manifest.json"
import json, sys
eps = json.load(open(sys.argv[1])).get("episodes", [])
print(sum(1 for e in eps if e.get("usable")))
PY
)"
  a="$("$AERIAL_PY" - <<'PY' "$ROOT/$CURATED/manifest.json"
import json, sys
eps = json.load(open(sys.argv[1])).get("episodes", [])
print(sum(1 for e in eps if e.get("arrived")))
PY
)"
  [[ "$u" -ge "$TARGET_CURATED_USABLE" && "$a" -ge "$TARGET_CURATED_ARRIVED" ]]
}

say "START out=${OUT} targets usable>=${TARGET_CURATED_USABLE} arrived>=${TARGET_CURATED_ARRIVED}"

# Pass schedule (urban-inland only after baseline).
PASSES=(
  "baseline_all|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19|24|5|5"
  "urban_z28|${URBAN_ROUTES}|28|5|7"
  "priority_z30|${PRIORITY_ROUTES}|30|5|8"
  "priority_z34|${PRIORITY_ROUTES}|34|4|10"
  "urban_z26|${URBAN_ROUTES}|26|5|6"
  "priority_z32|${PRIORITY_ROUTES}|32|4|9"
  "urban_z30|${URBAN_ROUTES}|30|5|8"
  "priority_z36|${PRIORITY_ROUTES}|36|3|12"
)

pass_n=0
for spec in "${PASSES[@]}"; do
  pass_n=$((pass_n + 1))
  if [[ "$pass_n" -gt "$MAX_PASSES" ]]; then
    say "MAX_PASSES=${MAX_PASSES} reached"
    break
  fi
  IFS='|' read -r tag routes min_z retry_m max_retry <<< "$spec"
  run_pass "$tag" "$routes" "$min_z" "$retry_m" "$max_retry"
  if curated_ok; then
    say "TARGETS MET after ${tag}"
    break
  fi
done

report_stats "FINAL"
say "DONE out=${OUT} curated=${CURATED}"
