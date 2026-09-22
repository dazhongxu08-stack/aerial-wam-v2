#!/usr/bin/env bash
# Monitor interior pipeline on 125; auto-resume on idle interrupt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"

STAMP="${STAMP:-20260916_night}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"
IDLE_THRESHOLD_SEC="${IDLE_THRESHOLD_SEC:-420}"
LOG="${LOG:-artifacts/monitor_interior_pipeline_${STAMP}.log}"
PIPE_LOG="artifacts/run_interior_full_pipeline_${STAMP}.log"
STATE="artifacts/pipeline_checkpoint_${STAMP}.json"
LOCK="/tmp/interior_pipeline_${STAMP}.lock"

mkdir -p "$(dirname "$LOG")"

say() { echo "[monitor-pipeline] $(date -Is) $*"; }

active_jobs() {
  pgrep -af "run_interior_pipeline|collect_path_expert_dataset|train_v4_ac|wam_phase2_long_eval" \
    | grep -v "monitor_interior_pipeline" | grep -v "grep" || true
}

pipeline_done() {
  "$AERIAL_PY" - <<PY "$ROOT/$STATE"
import json, sys
path = sys.argv[1]
try:
    done = set(json.load(open(path)).get("done", []))
except FileNotFoundError:
    done = set()
need = {"step0_review","step1a_terminal","step1b_poly_backfill","step2_poly_ft",
        "step2b_focus_ft","step3_weak_collect","step4_evals","step5_demo"}
raise SystemExit(0 if need <= done else 1)
PY
}

status_line() {
  local npz cur_u cur_a jobs
  npz="$(find "$ROOT/experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}" \
    -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ -f "$ROOT/experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}_curated/manifest.json" ]]; then
    read -r cur_u cur_a < <("$AERIAL_PY" - <<PY
import json
e=json.load(open("$ROOT/experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}_curated/manifest.json"))["episodes"]
print(sum(1 for x in e if x.get("usable")), sum(1 for x in e if x.get("arrived")))
PY
)
  else
    cur_u=0 cur_a=0
  fi
  jobs="$(active_jobs | head -1 | sed 's/.*collect_path/collect/' | cut -c1-90)"
  echo "NPZ=${npz} CUR=${cur_u}u/${cur_a}a JOB=${jobs:-IDLE}"
}

resume_if_needed() {
  local jobs idle_marker now last
  jobs="$(active_jobs)"
  if [[ -n "$jobs" ]]; then
    return 0
  fi
  if pipeline_done; then
    say "pipeline complete — monitor exit"
    exit 0
  fi
  now="$(date +%s)"
  idle_marker="/tmp/interior_pipeline_idle_${STAMP}.ts"
  if [[ ! -f "$idle_marker" ]]; then
    echo "$now" >"$idle_marker"
    say "idle first seen — waiting ${IDLE_THRESHOLD_SEC}s before resume"
    return 0
  fi
  last="$(cat "$idle_marker")"
  if (( now - last < IDLE_THRESHOLD_SEC )); then
    return 0
  fi
  say "idle>${IDLE_THRESHOLD_SEC}s and pipeline incomplete — RESUME"
  rm -f "$idle_marker"
  nohup bash "$ROOT/experiments/aerial/scripts/run_interior_pipeline_from_checkpoint.sh" \
    >>"$PIPE_LOG" 2>&1 &
  say "resumed pid=$!"
}

# macOS notify when run locally via SSH wrapper
notify_local() {
  local msg="$1"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    osascript -e "display notification \"${msg//\"/\\\"}\" with title \"Interior Pipeline 125\"" 2>/dev/null || true
  fi
}

say "START interval=${INTERVAL_SEC}s idle_threshold=${IDLE_THRESHOLD_SEC}s stamp=$STAMP"
while true; do
  line="$(status_line)"
  say "$line"
  notify_local "$line"
  if [[ -z "$(active_jobs)" ]]; then
    : # idle marker handled in resume_if_needed
  else
    rm -f "/tmp/interior_pipeline_idle_${STAMP}.ts"
  fi
  resume_if_needed || true
  sleep "$INTERVAL_SEC"
done
