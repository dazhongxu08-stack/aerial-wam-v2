#!/usr/bin/env bash
# Orin → Tello: hard-route V5 baseline (hbclear + open_loop planner) via vgoal.
# Prereq: Orin Wi-Fi joined to TELLO-XXXX; Mac can SSH via USB (orin-usb).
set -euo pipefail

ROOT="${ROOT:-$HOME/aerial-wam-v2}"
VGOAL_REPO="${VGOAL_REPO:-$HOME/Projects/aerial-vgoal-wam}"
PY="${PY:-$HOME/sim_verify/.venv/bin/python}"
TARGET_CLASS="${TARGET_CLASS:-person}"
STAGE="${1:-bench}"

cd "$ROOT"
export PYTHONPATH="$ROOT:${VGOAL_REPO}:${PYTHONPATH:-}"

common=(
  -m experiments.aerial.scripts.wam_vgoal_deploy
  --backend tello
  --preset v5
  --tello-ip 192.168.10.1
  --camera tello
  --demo-short
  --vgoal-repo "$VGOAL_REPO"
  --target-class "$TARGET_CLASS"
  --device cuda
  --record-auto
)

case "$STAGE" in
  bench)
    # SDK + V5 models + one frame; no takeoff
    exec "$PY" "${common[@]}" --offboard
    ;;
  fly)
    echo "WARNING: Tello will TAKEOFF with V5 planner+shield. Clear space. Type YES:"
    read -r ans
    [[ "$ans" == "YES" ]] || { echo "aborted"; exit 1; }
    exec "$PY" "${common[@]}" --offboard --arm --run --i-know-props-are-on
    ;;
  *)
    echo "usage: $0 [bench|fly]"
    exit 2
    ;;
esac
