#!/usr/bin/env bash
# Loop inland PathExpert collect until curated targets met (default 20 usable / 8 arrived).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
export PYTHONUNBUFFERED=1

STAMP="${STAMP:-20260916_night}"
OUT="${OUT:-experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}}"
CURATED="${CURATED:-experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}_curated}"
LOG="${LOG:-artifacts/collect_urban_complex_path_expert_${STAMP}_until_targets.log}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"

URBAN_ROUTES="0,1,2,3,4,5,6,7,8,9,10,11,19"
WEAK_ROUTES="2,6,7,8,9,11"
NO_ARRIVAL_ROUTES="0,1,2,6,7,8,9,10,11"
ARRIVAL_PUSH="0,1,3,4,5,10,19"
PRIORITY_ROUTES="3,4,5,10,0,1,6,2,7,8,9,11,19"

TARGET_CURATED_USABLE="${TARGET_CURATED_USABLE:-20}"
TARGET_CURATED_ARRIVED="${TARGET_CURATED_ARRIVED:-8}"
REVIEW_MAX_PER_ROUTE="${REVIEW_MAX_PER_ROUTE:-3}"
MAX_CYCLES="${MAX_CYCLES:-50}"

mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[until-targets] $(date -Is) $*"; }

recover_airsim() {
  if [[ -x "$RECOVER" ]]; then
    say "recover_renderer"
    bash "$RECOVER" || true
    sleep 10
  fi
}

report_stats() {
  local tag="${1:-}"
  local npz
  npz="$(find "$ROOT/$OUT" -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
  say "stats after ${tag}: npz_total=${npz}"
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
print(f"[until-targets] curated usable={u} arrived={a} routes={routes} arr_routes={arr}")
PY
  fi
}

curated_counts() {
  [[ -f "$ROOT/$CURATED/manifest.json" ]] || return 1
  "$AERIAL_PY" - <<'PY' "$ROOT/$CURATED/manifest.json"
import json, sys
eps = json.load(open(sys.argv[1])).get("episodes", [])
print(sum(1 for e in eps if e.get("usable")), sum(1 for e in eps if e.get("arrived")))
PY
}

curated_ok() {
  local u a
  read -r u a < <(curated_counts || echo "0 0")
  [[ "$u" -ge "$TARGET_CURATED_USABLE" && "$a" -ge "$TARGET_CURATED_ARRIVED" ]]
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

say "START out=${OUT} targets usable>=${TARGET_CURATED_USABLE} arrived>=${TARGET_CURATED_ARRIVED} max_per_route=${REVIEW_MAX_PER_ROUTE}"

report_stats "BOOT"

cycle=0
while ! curated_ok; do
  cycle=$((cycle + 1))
  if [[ "$cycle" -gt "$MAX_CYCLES" ]]; then
    say "MAX_CYCLES=${MAX_CYCLES} reached without targets"
    break
  fi
  say "=== CYCLE ${cycle}/${MAX_CYCLES} ==="
  z_base=$((30 + (cycle % 6) * 2))
  z_weak=$((z_base + 6))
  z_arr=$((z_base + 4))
  z_route6=$((z_base + 10))

  run_pass "c${cycle}_weak_z${z_weak}" "$WEAK_ROUTES" "$z_weak" 4 14
  curated_ok && break

  run_pass "c${cycle}_noarr_z${z_arr}" "$NO_ARRIVAL_ROUTES" "$z_arr" 3 16
  curated_ok && break

  run_pass "c${cycle}_arrival_z${z_arr}" "$ARRIVAL_PUSH" "$z_arr" 3 16
  curated_ok && break

  run_pass "c${cycle}_urban_z${z_base}" "$URBAN_ROUTES" "$z_base" 4 12
  curated_ok && break

  run_pass "c${cycle}_priority_z$((z_base + 8))" "$PRIORITY_ROUTES" "$((z_base + 8))" 2 18
  curated_ok && break

  run_pass "c${cycle}_route6_z${z_route6}" "6" "$z_route6" 2 20
  curated_ok && break
done

if curated_ok; then
  say "TARGETS MET usable>=${TARGET_CURATED_USABLE} arrived>=${TARGET_CURATED_ARRIVED}"
else
  read -r u a < <(curated_counts || echo "0 0")
  say "STOPPED at usable=${u} arrived=${a} (targets ${TARGET_CURATED_USABLE}/${TARGET_CURATED_ARRIVED})"
fi

report_stats "FINAL"
say "DONE out=${OUT} curated=${CURATED}"
