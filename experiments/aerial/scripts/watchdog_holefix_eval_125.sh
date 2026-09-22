#!/usr/bin/env bash
# Watchdog: finish holefix dual-eval on 125; if a run dies/hangs, resume.
# Usage (on 125):
#   nohup bash experiments/aerial/scripts/watchdog_holefix_eval_125.sh >>artifacts/logs/watchdog_holefix_eval.out 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${PYTHON_BIN:-$HOME/sim_verify/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3
ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-20260922_141414}"
WM="${WM:-$ART/wm_ckpt_obstacle_cost_20260922_022358/wm_obs.pt}"
ACTOR="${ACTOR:-$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}_hard014/v4_ac_latest.pt}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
DEPTH="${DEPTH:-$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU:-$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
CFG="${CFG:-configs/_tmp_directional_oa_eval.yaml}"
LOG="$ART/logs/watchdog_holefix_eval_${STAMP}.log"
# Per-eval wall budget (4 routes @ ~3min ≈ 12min; allow slack for spawn retries).
EVAL_TIMEOUT_S="${EVAL_TIMEOUT_S:-1200}"
POLL_S="${POLL_S:-20}"
mkdir -p "$ART/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

need_done() {
  local missing=0
  for TAG in cl_noshield_b0 cl_noshield_b1 cl_noshield_b2 actor_noshield_a0 actor_noshield_a1 actor_noshield_a2; do
    if [[ ! -f "$ART/eval_directional_oa_${STAMP}_${TAG}/eval_all.json" ]]; then
      missing=1
      break
    fi
  done
  return $missing
}

run_one() {
  local TAG=$1; shift
  local OUT="$ART/eval_directional_oa_${STAMP}_${TAG}"
  local ELOG="$ART/logs/directional_oa_holefix_eval_${TAG}_${STAMP}.log"
  if [[ -f "$OUT/eval_all.json" ]]; then
    log "SKIP $TAG (eval_all exists)"
    return 0
  fi
  mkdir -p "$OUT/traj"
  # Kill any stale eval holding AirSim.
  pkill -f 'wam_phase2_long_eval.py' 2>/dev/null || true
  sleep 2
  log "START $TAG timeout=${EVAL_TIMEOUT_S}s"
  # Run eval in background; watchdog polls for json (eval sometimes hangs after write).
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$ACTOR" \
    --depth-ckpt "$DEPTH" --tau-ckpt "$TAU" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$OUT/traj" --out "$OUT/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10.0 --max-steps 600 --success-dist 3.0 --terminal-pin-rem-m 20 \
    --min-spawn-z 24.0 --spawn-z-retry-m 5.0 --spawn-z-max-retries 4 \
    "$@" >"$ELOG" 2>&1 &
  local epid=$!
  local t0=$SECONDS
  while true; do
    if [[ -f "$OUT/eval_all.json" ]]; then
      # Give a moment then kill hung post-complete process.
      sleep 5
      if kill -0 "$epid" 2>/dev/null; then
        log "DONE $TAG (json ready; killing hung pid=$epid)"
        kill "$epid" 2>/dev/null || true
        sleep 2
        kill -9 "$epid" 2>/dev/null || true
      else
        log "DONE $TAG (pid exited)"
      fi
      wait "$epid" 2>/dev/null || true
      "$PY" -c "import json;from pathlib import Path;p=Path('$OUT/eval_all.json');r=json.loads(p.read_text())['episodes'];print(f'[$TAG] SR {sum(1 for x in r if x.get(\"arrived\"))}/{len(r)}')" | tee -a "$LOG"
      return 0
    fi
    if ! kill -0 "$epid" 2>/dev/null; then
      wait "$epid" 2>/dev/null || true
      if [[ -f "$OUT/eval_all.json" ]]; then
        log "DONE $TAG after exit"
        return 0
      fi
      log "FAIL $TAG exited without eval_all.json — will retry"
      return 1
    fi
    if (( SECONDS - t0 > EVAL_TIMEOUT_S )); then
      log "TIMEOUT $TAG killing pid=$epid"
      kill "$epid" 2>/dev/null || true
      sleep 2
      kill -9 "$epid" 2>/dev/null || true
      wait "$epid" 2>/dev/null || true
      return 1
    fi
    sleep "$POLL_S"
  done
}

summarize() {
  "$PY" - <<PY | tee -a "$LOG"
import json, statistics as st
from pathlib import Path
art=Path("$ART"); stamp="$STAMP"
print("=== HOLEFIX SUMMARY ===")
for prefix, seeds in [("cl_noshield",["b0","b1","b2"]),("actor_noshield",["a0","a1","a2"])]:
    srs=[]
    for s in seeds:
        p=art/f"eval_directional_oa_{stamp}_{prefix}_{s}/eval_all.json"
        if not p.exists():
            print(f"  {prefix}_{s}: MISSING"); continue
        rows=json.loads(p.read_text())["episodes"]
        arr=sum(1 for r in rows if r.get("arrived")); srs.append(arr)
        progs=[round(100*float(r.get("progress_ratio",0)),1) for r in sorted(rows, key=lambda x:int(x["route_idx"]))]
        print(f"  {prefix}_{s}: SR {arr}/4 prog={progs}")
    if srs:
        print(f"  {prefix} median={st.median(srs)}/4 range={min(srs)}-{max(srs)}")
print("=== WATCHDOG COMPLETE ===")
PY
}

log "watchdog start root=$ROOT stamp=$STAMP actor=$ACTOR"
# Single-instance lock
LOCK="$ART/logs/watchdog_holefix_eval_${STAMP}.lock"
if [[ -f "$LOCK" ]]; then
  old=$(cat "$LOCK" 2>/dev/null || true)
  if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
    log "another watchdog pid=$old alive — exit"
    exit 0
  fi
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

round=0
while ! need_done; do
  round=$((round + 1))
  log "round=$round"
  run_one cl_noshield_b0 --planner --planner-horizon 15 --planner-rollout closed_loop --no-shield || true
  run_one cl_noshield_b1 --planner --planner-horizon 15 --planner-rollout closed_loop --no-shield || true
  run_one cl_noshield_b2 --planner --planner-horizon 15 --planner-rollout closed_loop --no-shield || true
  run_one actor_noshield_a0 --no-shield || true
  run_one actor_noshield_a1 --no-shield || true
  run_one actor_noshield_a2 --no-shield || true
  if ! need_done; then
    log "still missing — sleep 15s then retry"
    sleep 15
  fi
  if (( round > 30 )); then
    log "ABORT after 30 rounds"
    break
  fi
done

if need_done; then
  log "all eval_all.json present"
  summarize
else
  log "incomplete after retries"
  summarize
  exit 2
fi
