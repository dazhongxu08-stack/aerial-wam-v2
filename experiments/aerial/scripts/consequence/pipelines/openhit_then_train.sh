#!/bin/bash
# Wait until the open-side collector exits, check the pairs, train only if they pass.
cd /home/yao/aerial-wam-v2 || exit 1
export PYTHONPATH=$PWD
PY=$HOME/sim_verify/.venv/bin/python
LOG=/tmp/openc13_then_train.log
exec >>"$LOG" 2>&1
echo "WAIT START $(date -Is)"
while pgrep -f "python -u -m experiments.aerial.rl.collect_onpolicy_branches --branch-mode side_open" >/dev/null; do
  sleep 60
done
sleep 5
if pgrep -f "python -u -m experiments.aerial.rl.collect_onpolicy_branches --branch-mode side_open" >/dev/null; then
  echo "collector still running, not training $(date -Is)"
  exit 1
fi
echo "COLLECTOR STOPPED $(date -Is)"
"$PY" -u experiments/aerial/scripts/consequence/check_openhit_quality.py
rc=$?
echo "QUALITY exit $rc $(date -Is)"
if [ "$rc" -ne 0 ]; then
  echo "quality gate not met, not training"
  exit "$rc"
fi
echo "QUALITY PASS, training $(date -Is)"
bash experiments/aerial/scripts/consequence/pipelines/side_corner_krs6_openhit.sh
echo "TRAIN WRAPPER EXIT $? $(date -Is)"
