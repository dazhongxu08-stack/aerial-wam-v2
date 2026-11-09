#!/usr/bin/env bash
# Independent office align stack. Does not decrypt, wipe, or start yhs_wire.
# Publishing stays off unless ALIGN_ARM=1, and only after yhs_wire has stopped.
set -eo pipefail
ROOT=/home/user/rt/office_align
PY=/home/user/lln_ws/VlmDetector/.pixi/envs/default/bin/python
export PYTHONPATH=/home/user/rt/app/pydeps:/home/user/rt/app${PYTHONPATH:+:$PYTHONPATH}
if [[ "${1:-}" == "--check" ]]; then
  exec "$PY" "$ROOT/yhs_align.py" --check
fi
if pgrep -f 'yhs_wire.py' >/dev/null \
  || pgrep -f 'yhs_wire.pyc' >/dev/null \
  || pgrep -f 'bin/yhs_wire.sh' >/dev/null \
  || pgrep -f 'yhs_detour.py' >/dev/null \
  || pgrep -f 'yhs_detour.sh' >/dev/null; then
  echo "another office stack is running; align stack not started" >&2
  exit 3
fi
set +u
source /opt/ros/jazzy/setup.bash
source /home/user/work/robot-sensors/install/setup.bash
set -u
export ROS_LOCALHOST_ONLY=1
export AERIAL_ORIN_HOME=/home/user/rt
export AERIAL_DA3_FP16=0
exec "$PY" "$ROOT/yhs_align.py"
