#!/usr/bin/env bash
# Short-path visual demo in AirSim: detect target (e.g. car) → autonomous approach.
#
# Run on 125 (4090 + AirSim). No Orin / Pixhawk required.
#
# Usage:
#   source experiments/aerial/scripts/env_4090.sh   # or let this script source it
#   export TARGET_CLASS=car
#   ./experiments/aerial/scripts/run_sim_vgoal_demo_short.sh mock    # stack smoke (no AirSim)
#   ./experiments/aerial/scripts/run_sim_vgoal_demo_short.sh gt      # GT nearest object smoke
#   ./experiments/aerial/scripts/run_sim_vgoal_demo_short.sh eval      # A: pure visual closed-loop
#   ./experiments/aerial/scripts/run_sim_vgoal_demo_short.sh waypoint  # B: fly route + YOLO overlay
#
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/aerial-wam-v2")}"
cd "$REPO"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  # env_4090 sets PYTHON_BIN when sourced
  source "$REPO/experiments/aerial/scripts/env_4090.sh"
else
  export REPO_ROOT="${REPO_ROOT:-$REPO}"
  export PYTHONPATH="$REPO:${PYTHONPATH:-}"
fi

STAGE="${1:-eval}"
TARGET_CLASS="${TARGET_CLASS:-car}"
VGOAL_REPO="${VGOAL_REPO:-${HOME}/aerial-vgoal-wam}"
OUT_DIR="${OUT_DIR:-$REPO/artifacts/sim_vgoal_demo_short}"
ANNO="${ANNO:-$REPO/experiments/aerial/annotations/sim_vgoal_demo_short_route.json}"
# YOLO needs native 1080p; WAM/depth stay 224 via fan-out.
CAPTURE_W="${CAPTURE_W:-1920}"
CAPTURE_H="${CAPTURE_H:-1080}"
WAM_SIZE="${WAM_SIZE:-224}"
# Crowded urban interior: keep cruise low so shield/subgoal can thread blocks.
CRUISE_SPEED="${CRUISE_SPEED:-3}"
AERIAL_PERSIST_ROOT="${AERIAL_PERSIST_ROOT:-$HOME/aerial_airsim_persistent}"

mkdir -p "$OUT_DIR"

_vision_flags=(
  --fanout-rgb
  --capture-w "$CAPTURE_W"
  --capture-h "$CAPTURE_H"
  --wam-encode-size "$WAM_SIZE"
)

_common_eval_flags=(
  --annotation "$ANNO"
  --routes 0
  --episodes 1
  --target-class "$TARGET_CLASS"
  --vgoal-repo "$VGOAL_REPO"
  "${_vision_flags[@]}"
  --max-steps 400
  --cruise-speed "$CRUISE_SPEED"
  --success-dist 4
  --search-det-steer
  --reject-far-lock-m 40
  --spawn-yaw-acquire-steps 60
  --spawn-yaw-acquire-deg 120
  --search-area-half-m 25
  --search-yaw-hold-deg 45
  --planner
  --out "$OUT_DIR/result.json"
  --traj-out "$OUT_DIR/traj"
  --perception-log "$OUT_DIR/perception"
  --video-out "$OUT_DIR/demo_ego.mp4"
)

ensure_1080_renderer() {
  local settings=(
    "$AERIAL_PERSIST_ROOT/AirSim/settings.json"
    "$HOME/Documents/AirSim/settings.json"
    "$AERIAL_PERSIST_ROOT/scene/env_airsim_16/LinuxNoEditor/AirVLN/Binaries/Linux/settings.json"
  )
  echo "=== Patch AirSim Scene capture → ${CAPTURE_W}x${CAPTURE_H} ==="
  "$PYTHON_BIN" experiments/aerial/sim_verify/probes/patch_capture_res.py \
    --w "$CAPTURE_W" --h "$CAPTURE_H" --settings "${settings[@]}"
  if ! nc -z 127.0.0.1 "${AIRSIM_PORT:-41451}" 2>/dev/null; then
    echo "=== Start AirSim renderer ==="
    bash "$AERIAL_PERSIST_ROOT/recover_renderer.sh" >/tmp/airsim_start.log 2>&1 &
  else
    echo "=== Restart AirSim renderer (capture res changed) ==="
    bash "$AERIAL_PERSIST_ROOT/recover_renderer.sh" >/tmp/airsim_start.log 2>&1 &
  fi
  local port="${AIRSIM_PORT:-41451}"
  for _ in $(seq 1 90); do
    if nc -z 127.0.0.1 "$port" 2>/dev/null; then
      if "$PYTHON_BIN" -c "
import airsim
c = airsim.MultirotorClient(ip='127.0.0.1', port=int('${port}'))
c.confirmConnection()
" 2>/dev/null; then
        echo "AirSim ready on :${port} (ping OK)"
        sleep 5
        return 0
      fi
    fi
    sleep 2
  done
  echo "AirSim failed to start; see /tmp/airsim_start.log" >&2
  tail -30 /tmp/airsim_start.log >&2 || true
  return 1
}

mock() {
  echo "=== Mock smoke (no AirSim, detector=mock) ==="
  "$PYTHON_BIN" -m experiments.aerial.scripts.wam_vgoal_eval \
    "${_common_eval_flags[@]}" \
    --mock --device cpu --detector mock \
    --max-steps 30 \
    --out "$OUT_DIR/result_mock.json"
}

gt_smoke() {
  echo "=== GT smoke (AirSim + nearest scene object, no YOLO) ==="
  "$PYTHON_BIN" -m experiments.aerial.scripts.wam_vgoal_eval \
    "${_common_eval_flags[@]}" \
    --detector gt --gt-nearest-scene-object \
    --gt-scene-pattern "PhysXCar.*|Car.*|Vehicle.*" \
    --max-steps 250 \
    --out "$OUT_DIR/result_gt.json"
}

eval_run() {
  ensure_1080_renderer
  echo "=== A: visual closed-loop YOLO @ ${CAPTURE_W}x${CAPTURE_H} ==="
  echo "TARGET_CLASS=$TARGET_CLASS | OUT_DIR=$OUT_DIR | WAM=${WAM_SIZE}"
  "$PYTHON_BIN" -m experiments.aerial.scripts.wam_vgoal_eval \
    "${_common_eval_flags[@]}" \
    --detector yolo --device cuda \
    --out "$OUT_DIR/result.json"
}

waypoint_run() {
  ensure_1080_renderer
  echo "=== B: waypoint route + YOLO overlay @ ${CAPTURE_W}x${CAPTURE_H} cs=${CRUISE_SPEED}m/s ==="
  "$PYTHON_BIN" -m experiments.aerial.scripts.wam_vgoal_record_route_demo \
    --annotation "$ANNO" \
    --route-idx 0 \
    --target-class "$TARGET_CLASS" \
    --vgoal-repo "$VGOAL_REPO" \
    --capture-w "$CAPTURE_W" \
    --capture-h "$CAPTURE_H" \
    --wam-encode-size "$WAM_SIZE" \
    --cruise-speed "$CRUISE_SPEED" \
    --max-steps 300 \
    --success-dist 4 \
    --out-dir "$OUT_DIR" \
    --out-mp4 "$OUT_DIR/demo_waypoint_yolo.mp4"
}

case "$STAGE" in
  mock) mock ;;
  setup) ensure_1080_renderer ;;
  gt)
    ensure_1080_renderer
    gt_smoke
    ;;
  eval) eval_run ;;
  waypoint) waypoint_run ;;
  all)
    mock
    echo ""
    echo "Mock OK. Then:"
    echo "  $0 setup     # patch 1080p + restart renderer"
    echo "  $0 waypoint  # B: fly route + YOLO video (recommended demo)"
    echo "  $0 eval      # A: pure visual closed-loop"
    ;;
  *)
    echo "Unknown stage: $STAGE (mock|setup|gt|eval|waypoint|all)"
    exit 2
    ;;
esac
