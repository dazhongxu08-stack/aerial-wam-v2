#!/usr/bin/env bash
# Continue inland PathExpert densify on night dataset (append, multi-pass).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
export PYTHONUNBUFFERED=1

STAMP="${STAMP:-20260916_night}"
OUT="${OUT:-experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}}"
CURATED="${CURATED:-experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}_curated}"
LOG="${LOG:-artifacts/collect_urban_complex_path_expert_${STAMP}_inland_push.log}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"

URBAN_ROUTES="0,1,2,3,4,5,6,7,8,9,10,11,19"
# Missing curated / no-arrival / short-trajectory inland routes.
WEAK_ROUTES="2,6,7,8,9,11"
NO_ARRIVAL_ROUTES="0,1,2,5,6,7,8,9,10,11"
PRIORITY_ROUTES="3,4,5,10,0,1,6,2,7,8,9,11,19"

TARGET_CURATED_USABLE="${TARGET_CURATED_USABLE:-20}"
TARGET_CURATED_ARRIVED="${TARGET_CURATED_ARRIVED:-8}"
REVIEW_MAX_PER_ROUTE="${REVIEW_MAX_PER_ROUTE:-3}"
MAX_PASSES="${MAX_PASSES:-10}"

mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[inland-push] $(date -Is) $*"; }

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
  say "=== PASS ${tag} routes=[${routes}] z>=${min_z} retry=${retry_m}x${max_retry} append=1 ==="
  recover_airsim
  STAMP="$STAMP" OUT="$OUT" LOG="$LOG" \
    ROUTE_INDICES="$routes" \
    MIN_SPAWN_Z="$min_z" SPAWN_RETRY_M="$retry_m" SPAWN_Z_MAX_RETRIES="$max_retry" \
    APPEND=1 SKIP_MIN_OK_GATE=1 N=20 \
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
      --max-per-route "$REVIEW_MAX_PER_ROUTE" \
      --write-curated 2>&1 | tee -a "$LOG" || true
    if [[ -f "$ROOT/$CURATED/manifest.json" ]]; then
      "$AERIAL_PY" - <<'PY' "$ROOT/$CURATED/manifest.json"
import json, sys
m = json.load(open(sys.argv[1]))
eps = m.get("episodes", [])
u = sum(1 for e in eps if e.get("usable"))
a = sum(1 for e in eps if e.get("arrived"))
routes = sorted({e.get("route_idx") for e in eps})
arr = sorted({e.get("route_idx") for e in eps if e.get("arrived")})
print(f"[inland-push] curated usable={u} arrived={a} routes={routes} arr_routes={arr}")
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

PASSES=(
  "weak_z30|${WEAK_ROUTES}|30|5|10"
  "noarr_z32|${NO_ARRIVAL_ROUTES}|32|4|12"
  "urban_z34|${URBAN_ROUTES}|34|4|12"
  "weak_z36|${WEAK_ROUTES}|36|3|14"
  "priority_z38|${PRIORITY_ROUTES}|38|3|14"
  "route6_z40|6|40|3|16"
  "urban_z36|${URBAN_ROUTES}|36|4|12"
  "noarr_z40|${NO_ARRIVAL_ROUTES}|40|3|16"
  "priority_z42|${PRIORITY_ROUTES}|42|2|18"
  "urban_z38|${URBAN_ROUTES}|38|3|14"
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
