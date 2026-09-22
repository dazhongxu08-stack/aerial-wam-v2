#!/usr/bin/env bash
# Resume directional OA after a failed gate (e.g. missing left_near).
# Reclass existing frames → pack → train → gate → policy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ART="$ROOT/experiments/aerial/rl/artifacts"
GT="$ART/obstacle_cost_gt_depth"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WM_CKPT="${WM_CKPT:-$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
ACTOR_INIT="${ACTOR_INIT:-$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear/v4_ac_latest.pt}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
LOG="$ART/logs/directional_oa_resume_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== resume stamp=$STAMP ==="

# If left_near still missing, recollect with updated script.
NEED_RECOLLECT=0
if [[ -f "$GT/frames.npz" ]]; then
  "$PYTHON_BIN" - <<'PY' || NEED_RECOLLECT=1
import numpy as np
from experiments.aerial.rl.depth_geometry import directional_clearance_m
from pathlib import Path
d=np.load("experiments/aerial/rl/artifacts/obstacle_cost_gt_depth/frames.npz", allow_pickle=True)
depth=d["depth"]; rgb=d["rgb"]; prop=d["proprio"]
PROBES=[("fwd",[1.,0,0]),("left",[0,1.,0])]
d_near,d_far=5.0,18.0
counts={"fwd_empty":0,"fwd_near":0,"left_near":0}
rgbs,depths,acts,proprios,groups=[],[],[],[],[]
for i in range(len(depth)):
    clear={n:directional_clearance_m(depth[i], np.array(a)) for n,a in PROBES}
    for name,a in PROBES:
        grp=None
        if name=="fwd":
            if clear["fwd"]<=d_near: grp="fwd_near"
            elif clear["fwd"]>=d_far: grp="fwd_empty"
        elif name=="left":
            if clear["left"]<=d_near and clear["fwd"]>=max(8.0, clear["left"]+3.0):
                grp="left_near"
        if grp is None: continue
        rgbs.append(rgb[i]); depths.append(depth[i]); acts.append(np.array(a,np.float32))
        proprios.append(prop[i]); groups.append(grp); counts[grp]+=1
print("reclass", counts)
out=Path("experiments/aerial/rl/artifacts/obstacle_cost_gt_depth/frames_reclass.npz")
if counts["left_near"]<5:
    raise SystemExit(2)
np.savez_compressed(out, rgb=np.stack(rgbs), depth=np.stack(depths),
    action_xyz=np.stack(acts), proprio=np.stack(proprios), group=np.array(groups,dtype=str))
print("wrote", out, "n", len(groups))
PY
else
  NEED_RECOLLECT=1
fi

if [[ "${NEED_RECOLLECT}" == "1" ]]; then
  echo "=== recollect GT (need left_near) ==="
  "$PYTHON_BIN" -m experiments.aerial.rl.collect_obstacle_cost_gt_frames \
    --config configs/aerial_rl_urban_complex_p2c.yaml \
    --annotation "$ANN" \
    --out "$GT/frames.npz" \
    --host 127.0.0.1 --port 41451 \
    --episodes "${EPISODES:-40}" \
    --steps-per-ep "${STEPS_PER_EP:-16}" \
    --d-near 5.0 --d-far 18.0 \
    --min-per-group 20 --max-frames 900
  FRAMES="$GT/frames.npz"
else
  FRAMES="$GT/frames_reclass.npz"
fi

LABELS="$GT/labels_resume.npz"
OBS_CKPT="$ART/wm_ckpt_obstacle_cost_${STAMP}/wm_obs.pt"
GATE_JSON="$GT/gate_${STAMP}.json"
POLICY_DIR="$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}"

"$PYTHON_BIN" -m experiments.aerial.rl.train_obstacle_cost_labels pack \
  --wm-ckpt "$WM_CKPT" --frames "$FRAMES" --out "$LABELS" --device cuda

mkdir -p "$(dirname "$OBS_CKPT")"
"$PYTHON_BIN" -m experiments.aerial.rl.train_obstacle_cost_labels train \
  --wm-ckpt "$WM_CKPT" --labels "$LABELS" --out-ckpt "$OBS_CKPT" \
  --steps "${HEAD_STEPS:-1000}" --batch 64 --device cuda

set +e
"$PYTHON_BIN" -m experiments.aerial.rl.train_obstacle_cost_labels gate \
  --wm-ckpt "$OBS_CKPT" --labels "$LABELS" --report "$GATE_JSON" --device cuda
GATE_RC=$?
set -e
[[ "$GATE_RC" -eq 0 ]] || { echo "GATE FAILED"; exit 2; }

mkdir -p "$POLICY_DIR"
"$PYTHON_BIN" -m experiments.aerial.rl.train_v4_ac \
  --config configs/aerial_rl_urban_complex_p2c.yaml \
  --config-overlay configs/aerial_rl_urban_complex_directional_oa.yaml \
  --backend airsim --dynamics torch \
  --wm-ckpt "$OBS_CKPT" \
  --init-actor-ckpt "$ACTOR_INIT" \
  --no-planner --no-shield \
  --iters "${ITERS:-40}" --episodes-per-iter 1 \
  --imagine-batch 16 --imagine-horizon 15 \
  --device cuda --annotation "$ANN" \
  --ckpt-dir "$POLICY_DIR" --save-every-iter

echo "=== DONE policy=$POLICY_DIR ==="
