#!/usr/bin/env bash
# Independent office detour stack. Does not decrypt, wipe, or start the others.
# Publishing stays off unless DETOUR_ARM=1, and only after they have stopped.
set -eo pipefail
ROOT=/home/user/rt/office_detour
PY=/home/user/lln_ws/VlmDetector/.pixi/envs/default/bin/python
export PYTHONPATH=/home/user/rt/app/pydeps:/home/user/rt/app${PYTHONPATH:+:$PYTHONPATH}
if [[ "${1:-}" == "--check" ]]; then
  exec "$PY" "$ROOT/yhs_detour.py" --check
fi
if pgrep -f 'yhs_wire.py' >/dev/null \
  || pgrep -f 'yhs_wire.pyc' >/dev/null \
  || pgrep -f 'bin/yhs_wire.sh' >/dev/null \
  || pgrep -f 'yhs_align.py' >/dev/null \
  || pgrep -f 'yhs_align.sh' >/dev/null; then
  echo "another office stack is running; detour stack not started" >&2
  exit 3
fi
set +u
source /opt/ros/jazzy/setup.bash
source /home/user/work/robot-sensors/install/setup.bash
set -u
export ROS_LOCALHOST_ONLY=1
export AERIAL_ORIN_HOME=/home/user/rt
export AERIAL_DA3_FP16=0
exec "$PY" "$ROOT/yhs_detour.py"
