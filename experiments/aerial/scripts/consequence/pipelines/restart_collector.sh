#!/bin/bash
set -uo pipefail
cd /home/yao/aerial-wam-v2
export PYTHONPATH=$PWD
PY=$HOME/sim_verify/.venv/bin/python
R=experiments/aerial/rl/artifacts/consequence_data_20261004/AUTO_REVIEW.json
m0=$(stat -c %Y "$R")
# Wait for the next review write (end of an episode), then stop right away.
for i in $(seq 1 300); do
  sleep 0.5
  [ "$(stat -c %Y "$R")" != "$m0" ] && break
done
pkill -INT -f "collect_consequence_dataset"; sleep 3
pkill -9 -f "collect_consequence_dataset"; sleep 2
echo "stopped; episodes=$($PY -c "import json;print(len(json.load(open('$R'))['episodes']))")"
timeout 20 $PY -c "
import airsim
c=airsim.MultirotorClient(); c.confirmConnection(); print('ping', c.ping())
r=c.simGetImages([airsim.ImageRequest('front_custom', airsim.ImageType.Scene, False, True)])[0]
print('image bytes', len(r.image_data_uint8))
" 2>&1 | grep -E "ping|image|Error|Timeout" || echo "AIRSIM_CHECK_FAILED"
