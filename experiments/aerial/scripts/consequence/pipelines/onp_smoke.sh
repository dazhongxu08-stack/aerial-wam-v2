#!/bin/bash
cd /home/yao/aerial-wam-v2; export PYTHONPATH=$PWD; PY=$HOME/sim_verify/.venv/bin/python
V3=experiments/aerial/rl/artifacts/consequence_v3_20261005
CK=$V3/m_full/consequence_best.pt,$V3/m_full_s1/consequence_best.pt,$V3/m_full_s2/consequence_best.pt
rm -rf /tmp/onp_smoke
$PY -u -m experiments.aerial.rl.collect_onpolicy_branches --ckpts $CK --out-dir /tmp/onp_smoke --n-states 2 > /tmp/onp_smoke.log 2>&1
echo "exit $?" >> /tmp/onp_smoke.log
$PY - >> /tmp/onp_smoke.log 2>&1 <<PY
import json, numpy as np
from experiments.aerial.rl.consequence_model import branch_windows
r=json.load(open("/tmp/onp_smoke/AUTO_REVIEW.json"))
print("flights", r["flights"]); print("probes", [(p["why"], p["collided"]) for p in r["probes"]])
z=np.load("/tmp/onp_smoke/probes/"+r["probes"][0]["file"])
w=branch_windows(z, z["gt_depth56"], 10); print({k:v.shape for k,v in w.items()}); print("hist", z["hist_action"].shape, z["hist_jpeg_offsets"].shape)
PY
echo SMOKE_DONE >> /tmp/onp_smoke.log
