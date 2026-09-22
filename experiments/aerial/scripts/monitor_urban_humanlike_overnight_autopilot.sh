#!/usr/bin/env bash
# Overnight autopilot: DepthScene interior collect (until gate) → humanlike FT → gate eval.
# Never idle if work remains; restart hung collect/train.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source experiments/aerial/scripts/env_4090.sh

INTERVAL="${INTERVAL_SEC:-180}"
COLLECT_STAMP="${COLLECT_STAMP:-20260917_interior2}"
TRAIN_STAMP="${TRAIN_STAMP:-20260918_humanlike}"
MIN_KEPT="${MIN_KEPT:-12}"
MAX_ROUNDS="${MAX_ROUNDS:-80}"
ITERS_TARGET="${ITERS_TARGET:-32}"
HANG_SEC="${HANG_SEC:-900}"
RESUME_CKPT="${RESUME_CKPT:-experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_20260917_aligned/v4_ac_latest.pt}"
EXPERT_OUT="experiments/aerial/rl/artifacts/dataset_urban_depth_scene_expert_${COLLECT_STAMP}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
FOCUS_ROUTES="${FOCUS_ROUTES:-1,2,3,5,6,7,8,9,10,12,13,17,18,19}"
STATE="${STATE:-artifacts/urban_humanlike_overnight_state.env}"
LOG="${LOG:-artifacts/urban_humanlike_overnight_autopilot.log}"
SCENE_SH="${SCENE_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"

mkdir -p artifacts
[[ -f "$STATE" ]] || cat >"$STATE" <<EOF
PHASE=collect
COLLECT_STAMP=$COLLECT_STAMP
TRAIN_STAMP=$TRAIN_STAMP
ITERS_TARGET=$ITERS_TARGET
MIN_KEPT=$MIN_KEPT
TRAIN_STARTED=0
GATE_STARTED=0
EOF
# shellcheck disable=SC1090
source "$STATE"

log() { echo "[overnight $(date '+%F %T')] $*" | tee -a "$LOG"; }

persist() {
  cat >"$STATE" <<EOF
PHASE=$PHASE
COLLECT_STAMP=$COLLECT_STAMP
TRAIN_STAMP=$TRAIN_STAMP
ITERS_TARGET=$ITERS_TARGET
MIN_KEPT=$MIN_KEPT
TRAIN_STARTED=${TRAIN_STARTED:-0}
GATE_STARTED=${GATE_STARTED:-0}
EOF
}

collect_running() {
  pgrep -f "experiments.aerial.rl.collect_depth_scene_expert_dataset" >/dev/null 2>&1
}

train_running() {
  pgrep -f "train_v4_ac.*${TRAIN_STAMP}" >/dev/null 2>&1
}

eval_running() {
  pgrep -f "wam_phase2_long_eval" >/dev/null 2>&1
}

quality_kept() {
  local f="$EXPERT_OUT/AUTO_REVIEW.json"
  [[ -f "$f" ]] || { echo 0; return; }
  python3 -c "import json;d=json.load(open('$f'));s=d.get('summary') or {};print(int(s.get('quality_kept') or sum(1 for e in (d.get('episodes') or []) if e.get('keep'))))" 2>/dev/null || echo 0
}

gate_passed() {
  local f="$EXPERT_OUT/QUALITY_SUMMARY.json"
  [[ -f "$f" ]] || return 1
  python3 -c "import json,sys;sys.exit(0 if json.load(open('$f')).get('gate_passed') else 1)" 2>/dev/null
}

file_age_sec() {
  local f="$1"
  [[ -f "$f" ]] || { echo 999999; return; }
  echo $(( $(date +%s) - $(stat -c %Y "$f" 2>/dev/null || echo 0) ))
}

count_train_iters() {
  grep -c "wrote ckpt iter" "artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log" 2>/dev/null || echo 0
}

kill_collect() {
  # Avoid matching this script's cmdline: kill by pid from pgrep -f module path only.
  local p
  for p in $(pgrep -f "experiments.aerial.rl.collect_depth_scene_expert_dataset" || true); do
    kill "$p" 2>/dev/null || true
  done
  for p in $(pgrep -f "collect_urban_depth_scene_expert.sh" || true); do
    # skip self
    [[ "$p" == "$$" ]] && continue
    kill "$p" 2>/dev/null || true
  done
  sleep 3
}

start_collect() {
  local append_flag="${1:-}"
  log "START collect stamp=$COLLECT_STAMP out=$EXPERT_OUT append=${append_flag:-0}"
  mkdir -p artifacts "$(dirname "$EXPERT_OUT")"
  # Prefer existing AirSim; recover only if port dead.
  if ! timeout 8 "$PYTHON_BIN" -c 'import airsim;c=airsim.MultirotorClient("127.0.0.1",41451);c.confirmConnection()' >/dev/null 2>&1; then
    if [[ -x "$SCENE_SH" ]]; then
      bash "$SCENE_SH" outdoor >>"$LOG" 2>&1 || true
      sleep 25
    fi
  fi
  if [[ "$append_flag" == "1" && -d "$EXPERT_OUT" ]]; then
    nohup env PYTHONUNBUFFERED=1 RECOVER_SCRIPT=/bin/true STAMP="$COLLECT_STAMP" \
      MIN_KEPT="$MIN_KEPT" MAX_ROUNDS="$MAX_ROUNDS" OUT="$EXPERT_OUT" ANNO="$ANNO" \
      bash -c '
        source experiments/aerial/scripts/env_4090.sh
        "$PYTHON_BIN" -m experiments.aerial.rl.collect_depth_scene_expert_dataset \
          --backend airsim --device cuda --annotation "'"$ANNO"'" --out "'"$EXPERT_OUT"'" \
          --min-spawn-z 38 --spawn-z-retry-m 14 --spawn-z-max-retries 3 \
          --max-inflate 1.8 --require-arrived --require-interior \
          --min-spawn-inland-m 100 --min-path-inland-m 80 \
          --min-kept "'"$MIN_KEPT"'" --until-gate --max-rounds "'"$MAX_ROUNDS"'" \
          --shield-fwd-only --tti-coeff 2.5 --r-m 100 --cruise-speed 10 \
          --max-steps 600 --episodes 20 --append \
          2>&1 | tee -a "artifacts/collect_urban_depth_scene_expert_'"$COLLECT_STAMP"'.log"
      ' >>"artifacts/collect_urban_depth_scene_expert_${COLLECT_STAMP}_nohup.log" 2>&1 &
  else
    nohup env PYTHONUNBUFFERED=1 RECOVER_SCRIPT=/bin/true \
      STAMP="$COLLECT_STAMP" MIN_KEPT="$MIN_KEPT" MAX_ROUNDS="$MAX_ROUNDS" \
      OUT="$EXPERT_OUT" ANNO="$ANNO" \
      bash experiments/aerial/scripts/collect_urban_depth_scene_expert.sh \
      >>"artifacts/collect_urban_depth_scene_expert_${COLLECT_STAMP}_nohup.log" 2>&1 &
  fi
  echo "COLLECT_PID=$!" | tee -a "$LOG"
}

ensure_collect() {
  local qk age
  qk="$(quality_kept)"
  if gate_passed || [[ "$qk" -ge "$MIN_KEPT" ]]; then
    log "collect GATE PASS qk=$qk/$MIN_KEPT → phase=train"
    PHASE=train
    persist
    kill_collect
    return 0
  fi
  if collect_running; then
    age="$(file_age_sec "artifacts/collect_urban_depth_scene_expert_${COLLECT_STAMP}.log")"
    # also consider AUTO_REVIEW freshness
    local age2
    age2="$(file_age_sec "$EXPERT_OUT/AUTO_REVIEW.json")"
    if [[ "$age" -gt "$HANG_SEC" && "$age2" -gt "$HANG_SEC" ]]; then
      log "WARN collect hung log_age=${age}s review_age=${age2}s — restart append"
      kill_collect
      if [[ -x "$SCENE_SH" ]]; then
        bash "$SCENE_SH" outdoor >>"$LOG" 2>&1 || true
        sleep 25
      fi
      start_collect 1
    else
      log "collect running qk=$qk/$MIN_KEPT log_age=${age}s"
    fi
    return 0
  fi
  # process dead, gate not met
  log "collect dead qk=$qk/$MIN_KEPT — restart append"
  start_collect 1
}

start_train() {
  local start_iter="${1:-0}"
  log "START humanlike train stamp=$TRAIN_STAMP start_iter=$start_iter/$ITERS_TARGET expert=$EXPERT_OUT"
  kill_collect
  pkill -f "wam_phase2_long_eval" 2>/dev/null || true
  # careful: don't pkill our overnight script
  local p
  for p in $(pgrep -f "train_v4_ac" || true); do
    kill "$p" 2>/dev/null || true
  done
  sleep 3
  if [[ -x "$SCENE_SH" ]]; then
    bash "$SCENE_SH" outdoor >>"artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log" 2>&1 || true
    sleep 20
  fi
  mkdir -p "experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${TRAIN_STAMP}" artifacts
  # Build focus anno from interior-friendly indices into a dedicated file.
  local anno_train="experiments/aerial/phase3_unified/annotations/outdoor_complex_humanlike_overnight.json"
  "$PYTHON_BIN" -m experiments.aerial.scripts.build_outdoor_complex_focus_annotation \
    --routes "$FOCUS_ROUTES" \
    --src "$ANNO" \
    --out "$anno_train" >>"$LOG" 2>&1 || {
      # fallback: use full inland annotation
      anno_train="$ANNO"
    }
  local init_ckpt="$RESUME_CKPT"
  if [[ "$start_iter" -gt 0 && -f "experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${TRAIN_STAMP}/v4_ac_latest.pt" ]]; then
    init_ckpt="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${TRAIN_STAMP}/v4_ac_latest.pt"
  fi
  nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m experiments.aerial.rl.train_v4_ac \
    --config configs/aerial_rl_urban_complex_p2c.yaml \
    --backend airsim --device cuda --dynamics torch --phase2 --r-m 100 \
    --wm-ckpt experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt \
    --annotation "$anno_train" \
    --dataset "$EXPERT_OUT" \
    --init-actor-ckpt "$init_ckpt" \
    --iters "$ITERS_TARGET" --start-iter "$start_iter" --save-every-iter \
    --episodes-per-iter 1 --imagine-batch 16 --imagine-horizon 15 \
    --near-goal-frac 0.5 --near-goal-dist-min 5 --near-goal-dist-max 25 \
    --min-spawn-z 38 --spawn-z-retry-m 14 --spawn-z-max-retries 3 \
    --renderer-restart-every 8 \
    --renderer-restart-script "$SCENE_SH" \
    --renderer-restart-scene outdoor \
    --w-collision 1.0 \
    --w-eff-strafe 0.08 --w-eff-heading 0.08 --w-eff-idle 0.03 \
    --tti-coeff 2.5 --planner --planner-horizon 1 --shield-exclusion-forward-only \
    --ckpt-dir "experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${TRAIN_STAMP}" \
    >>"artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log" 2>&1 &
  TRAIN_STARTED=1
  persist
  log "TRAIN_PID=$!"
}

ensure_train() {
  local iter
  iter="$(count_train_iters)"
  if [[ "$iter" -ge "$ITERS_TARGET" ]]; then
    log "train DONE iter=$iter/$ITERS_TARGET → phase=gate"
    PHASE=gate
    persist
    return 0
  fi
  if train_running; then
    log "train running iter=$iter/$ITERS_TARGET"
    return 0
  fi
  local age
  age="$(file_age_sec "artifacts/train_urban_complex_p2c_${TRAIN_STAMP}.log")"
  if [[ "$age" -lt 180 ]]; then
    log "train absent but log fresh (${age}s) — wait"
    return 0
  fi
  log "train dead/missing iter=$iter — resume"
  start_train "$iter"
}

start_gate() {
  log "START gate eval stamp=$TRAIN_STAMP"
  pkill -f "wam_phase2_long_eval" 2>/dev/null || true
  sleep 2
  nohup env \
    ACTOR="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${TRAIN_STAMP}/v4_ac_latest.pt" \
    MIN_SPAWN_Z=38 \
    SHIELD_EXCLUSION_FORWARD_ONLY=1 \
    STAMP="${TRAIN_STAMP}_gate" \
    bash experiments/aerial/scripts/eval_urban_complex_toward_g_gate.sh \
    >>"artifacts/urban_complex_toward_g_gate_${TRAIN_STAMP}_gate.log" 2>&1 &
  GATE_STARTED=1
  persist
  log "GATE_PID=$!"
}

ensure_gate() {
  local gate_json="artifacts/urban_complex_toward_g_gate_${TRAIN_STAMP}_gate/eval_all.json"
  if [[ -f "$gate_json" ]]; then
    local sr
    sr="$(python3 -c "import json;d=json.load(open('$gate_json'));e=d.get('episodes') or [];print(f\"{sum(1 for x in e if x.get('arrived'))}/{len(e)}\")" 2>/dev/null || echo "?")"
    log "gate DONE SR=$sr → phase=done"
    PHASE=done
    persist
    return 0
  fi
  if eval_running; then
    log "gate eval running"
    return 0
  fi
  if [[ "${GATE_STARTED:-0}" == "1" ]]; then
    local age
    age="$(file_age_sec "artifacts/urban_complex_toward_g_gate_${TRAIN_STAMP}_gate.log")"
    if [[ "$age" -lt 300 ]]; then
      log "gate log fresh — wait"
      return 0
    fi
  fi
  start_gate
}

# If collect finished with max_rounds fail, keep appending forever overnight.
maybe_extend_collect_rounds() {
  local meta="$EXPERT_OUT/path_expert_meta.json"
  [[ -f "$meta" ]] || return 0
  python3 -c "import json,sys;d=json.load(open('$meta'));sys.exit(0 if (not d.get('gate_passed') and int(d.get('rounds') or 0)>=int(d.get('max_rounds') or 40)) else 1)" 2>/dev/null || return 0
  log "max_rounds hit without gate — bump MAX_ROUNDS and append"
  MAX_ROUNDS=$((MAX_ROUNDS + 40))
  start_collect 1
}

tick() {
  log "TICK phase=$PHASE qk=$(quality_kept)/$MIN_KEPT collect=$(collect_running && echo 1 || echo 0) train=$(train_running && echo 1 || echo 0) eval=$(eval_running && echo 1 || echo 0)"
  case "$PHASE" in
    collect)
      ensure_collect
      maybe_extend_collect_rounds
      ;;
    train)
      ensure_train
      ;;
    gate)
      ensure_gate
      ;;
    done)
      log "ALL DONE overnight pipeline — idle heartbeat"
      ;;
    *)
      log "unknown PHASE=$PHASE — reset to collect"
      PHASE=collect
      persist
      ;;
  esac
}

log "overnight autopilot START collect=$COLLECT_STAMP train=$TRAIN_STAMP interval=${INTERVAL}s hang=${HANG_SEC}s"
while true; do
  tick
  sleep "$INTERVAL"
done
