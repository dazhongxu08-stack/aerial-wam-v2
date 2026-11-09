#!/usr/bin/env bash
# Decrypt weights into tmpfs, then run the chassis wire. Does not arm the base.
set -eo pipefail
set +u
source /opt/ros/jazzy/setup.bash
source /home/user/work/robot-sensors/install/setup.bash
set -u
export ROS_LOCALHOST_ONLY=1
export AERIAL_ORIN_HOME=/home/user/rt
export AERIAL_DA3_FP16=1
export AERIAL_ARM="${AERIAL_ARM:-0}"
export PYTHONPATH=/home/user/rt/app/pydeps:/home/user/rt/app${PYTHONPATH:+:$PYTHONPATH}
PY=/home/user/lln_ws/VlmDetector/.pixi/envs/default/bin/python
"$PY" /home/user/rt/bin/rt_crypt.pyc decrypt >/dev/null
trap '"$PY" /home/user/rt/bin/rt_crypt.pyc wipe >/dev/null 2>&1 || true' EXIT
exec "$PY" /home/user/rt/app/experiments/aerial/deploy/yhs_wire.py
