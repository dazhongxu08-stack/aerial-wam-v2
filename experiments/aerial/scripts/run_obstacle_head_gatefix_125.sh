#!/usr/bin/env bash
# Retrain obstacle_cost_head with action-probe expand + ranking; refuse policy train until gate passes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
GT="$ART/obstacle_cost_gt_depth"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PY="${PYTHON_BIN:-${HOME}/sim_verify/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3
WM_BASE="${WM_BASE:-$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
FRAMES="${FRAMES:-$GT/frames_multi2.npz}"
OUT_CKPT="${OUT_CKPT:-$ART/wm_ckpt_obstacle_cost_gatefix_${STAMP}/wm_obs.pt}"
LOG="$ART/logs/obstacle_cost_gatefix_${STAMP}.log"
mkdir -p "$(dirname "$OUT_CKPT")" "$ART/logs"
exec > >(tee -a "$LOG") 2>&1

echo "=== gatefix stamp=$STAMP ==="
test -f "$FRAMES" || { echo "FATAL missing $FRAMES"; exit 1; }
test -f "$WM_BASE" || { echo "FATAL missing $WM_BASE"; exit 1; }

LABELS="$GT/labels_gatefix_${STAMP}.npz"
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels pack \
  --wm-ckpt "$WM_BASE" --frames "$FRAMES" --out "$LABELS" --device cuda

"$PY" - <<PY
import numpy as np
from collections import Counter
d=np.load("$LABELS", allow_pickle=True)
g=d["group"].astype(str)
print("groups", dict(Counter(g.tolist())))
print("n", len(g), "label_mean", float(d["obstacle_label"].mean()))
# On fwd_near scenes, fwd probe labels should exceed side/climb
a=d["action"]; y=d["obstacle_label"]
m=g=="fwd_near"
fwd=(np.abs(a[:,0]-1)<0.05)&(np.abs(a[:,1])<0.05)&(np.abs(a[:,2])<0.05)
side=(np.abs(a[:,0])<0.05)&(np.abs(a[:,1])>0.5)
print("fwd_near fwd_y", float(y[m&fwd].mean()) if np.any(m&fwd) else None,
      "side_y", float(y[m&side].mean()) if np.any(m&side) else None)
PY

"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels train \
  --wm-ckpt "$WM_BASE" --labels "$LABELS" --out-ckpt "$OUT_CKPT" \
  --steps "${HEAD_STEPS:-2000}" --batch 128 --device cuda

GATE_JSON="$GT/gate_gatefix_${STAMP}.json"
set +e
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels gate \
  --wm-ckpt "$OUT_CKPT" --labels "$LABELS" --report "$GATE_JSON" --device cuda
RC=$?
set -e
echo "gate_rc=$RC report=$GATE_JSON"
cat "$GATE_JSON"
[[ "$RC" -eq 0 ]] || { echo "GATE STILL FAILED — no policy train"; exit 2; }
echo "$OUT_CKPT" > "$GT/LATEST_REFLOW_CKPT.txt"
echo "GATEFIX_OK $OUT_CKPT"
