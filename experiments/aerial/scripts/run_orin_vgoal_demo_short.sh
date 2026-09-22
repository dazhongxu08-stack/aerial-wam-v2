#!/usr/bin/env bash
# Short-path visual demo on Orin: detect object → autonomous approach.
#
# Usage (on Orin, props OFF unless stage=fly):
#   export TARGET_CLASS=car          # COCO class or open-vocab prompt
#   ./experiments/aerial/scripts/run_orin_vgoal_demo_short.sh preflight
#   ./experiments/aerial/scripts/run_orin_vgoal_demo_short.sh detect
#   ./experiments/aerial/scripts/run_orin_vgoal_demo_short.sh fly
#   ./experiments/aerial/scripts/run_orin_vgoal_demo_short.sh all
#
set -euo pipefail

REPO="${REPO:-$HOME/aerial-wam-v2}"
PY="${PY:-$HOME/sim_verify/.venv/bin/python3}"
STAGE="${1:-all}"
TARGET_CLASS="${TARGET_CLASS:-car}"
VGOAL_REPO="${VGOAL_REPO:-$HOME/Projects/aerial-vgoal-wam}"
CAMERA="${CAMERA:-0}"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

_common_deploy_flags=(
  --mavlink-port /dev/ttyACM0
  --camera "$CAMERA"
  --vgoal-repo "$VGOAL_REPO"
  --target-class "$TARGET_CLASS"
  --demo-short
  --device cuda
)

preflight() {
  echo "=== Preflight: MAVLink + battery ==="
  "$PY" -m experiments.aerial.scripts.orin_reset_h12_control
  echo "=== Preflight: YOLO one-shot ==="
  "$PY" -m experiments.aerial.scripts.camera_yolo_probe \
    --camera "$CAMERA" --target-class "$TARGET_CLASS" --device cuda \
    --out "$HOME/camera_yolo_probe.jpg"
}

detect() {
  echo "=== Detect bench (40 frames, props OFF) ==="
  echo "Place $TARGET_CLASS in camera FOV (~10-30m ahead)"
  "$PY" -m experiments.aerial.scripts.wam_vgoal_deploy \
    "${_common_deploy_flags[@]}" --detect-bench
}

load_stack() {
  echo "=== Load full stack (bench, no flight) ==="
  "$PY" -m experiments.aerial.scripts.wam_vgoal_deploy \
    "${_common_deploy_flags[@]}"
}

fly() {
  echo "=== Closed-loop visual approach (PROPS ON) ==="
  echo "H12: STABILIZE, throttle low, ARM, then script takes GUIDED control."
  echo "ch8=start record, ch9=stop. RC ready to disarm."
  read -r -p "Props on and area clear? Type YES: " ok
  if [[ "$ok" != "YES" ]]; then
    echo "Aborted."
    exit 1
  fi
  "$PY" -m experiments.aerial.scripts.wam_vgoal_deploy \
    "${_common_deploy_flags[@]}" \
    --offboard --arm --run \
    --i-know-props-are-on
}

case "$STAGE" in
  preflight) preflight ;;
  detect) detect ;;
  load) load_stack ;;
  fly) fly ;;
  all)
    preflight
    detect
    load_stack
    echo ""
    echo "Preflight + detect OK. When ready for flight:"
    echo "  $0 fly"
    ;;
  *)
    echo "Unknown stage: $STAGE (preflight|detect|load|fly|all)"
    exit 2
    ;;
esac
