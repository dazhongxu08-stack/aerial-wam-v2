#!/bin/bash
cd /home/yao/aerial-wam-v2
export PYTHONPATH=$PWD
PY=$HOME/sim_verify/.venv/bin/python
L=experiments/aerial/rl/artifacts/logs/consequence_schedule_v1_20261004.log
for i in $(seq 1 120); do
  grep -q "branch exit" $L && break
  sleep 60
done
cat $L
grep -v -E "spawn|Warn|Connected|Client Ver|^$" experiments/aerial/rl/artifacts/logs/consequence_branch_20261004.log | grep -E "probe [0-9]+|failed|Error|Traceback" | tail -30 | cut -c1-170
$PY -c "
import json
r=json.load(open('experiments/aerial/rl/artifacts/consequence_branch_train_20261004/AUTO_REVIEW.json'))
p=r['probes']; m=[x for x in p if 0<sum(x['collided'])<len(x['collided'])]
print('branch probes',len(p),'attempts',r['attempts'],'mixed',len(m),'all_safe',sum(sum(x['collided'])==0 for x in p),'dropped',len(r.get('dropped',[])))
"
$PY -c "
import json
r=json.load(open('experiments/aerial/rl/artifacts/consequence_data_20261004/AUTO_REVIEW.json'))
print('random gate', r['gate'])
"
timeout 900 $PY -u -m experiments.aerial.rl.train_consequence --data-dir experiments/aerial/rl/artifacts/consequence_data_20261004 \
  --branch-dirs experiments/aerial/rl/artifacts/consequence_branch_train_20261004 --out-dir /tmp/cq_branch_dry \
  --val-routes od_train_03,od_train_17,od_train_25,od_train_33,od_train_07,od_train_21 --steps 300 --eval-every 100 2>&1 \
  | grep -E "windows|branch states|step\": 300|Error|Traceback" | cut -c1-400
rm -rf /tmp/cq_branch_dry
