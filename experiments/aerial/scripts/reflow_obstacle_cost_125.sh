#!/usr/bin/env bash
# Reflow obstacle_cost_head from GT frames (prefer multi2 with left_near), then gate.
# Usage (repo root on 125):
#   bash experiments/aerial/scripts/reflow_obstacle_cost_125.sh
#
# Prefer frames_multi2.npz (has left_near). Raw frames.npz alone fails task-5
# group 3 (left_near=0) — that was the 20260922_141414 holefix fallback bug.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
GT="$ART/obstacle_cost_gt_depth"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PY="${PYTHON_BIN:-python3}"
WM_BASE="${WM_BASE:-$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
OUT_CKPT="${OUT_CKPT:-$ART/wm_ckpt_obstacle_cost_reflow_${STAMP}/wm_obs.pt}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
LOG="$ART/logs/obstacle_cost_reflow_${STAMP}.log"
MIN_LEFT_NEAR="${MIN_LEFT_NEAR:-20}"
mkdir -p "$(dirname "$OUT_CKPT")" "$ART/logs" "$GT"
exec > >(tee -a "$LOG") 2>&1

echo "=== reflow stamp=$STAMP ==="

# Frame source priority: explicit FRAMES → multi2 → multi → raw (last resort).
if [[ -n "${FRAMES:-}" ]]; then
  :
elif [[ -f "$GT/frames_multi2.npz" ]]; then
  FRAMES="$GT/frames_multi2.npz"
elif [[ -f "$GT/frames_multi.npz" ]]; then
  FRAMES="$GT/frames_multi.npz"
else
  FRAMES="$GT/frames.npz"
fi

if [[ "${FORCE_RECOLLECT:-0}" == "1" || ! -f "$FRAMES" ]]; then
  echo "=== collect GT frames (force or missing) ==="
  FRAMES="$GT/frames_reflow_${STAMP}.npz"
  "$PY" -m experiments.aerial.rl.collect_obstacle_cost_gt_frames \
    --config configs/aerial_rl_urban_complex_p2c.yaml \
    --annotation "$ANN" \
    --out "$FRAMES" \
    --host 127.0.0.1 --port 41451 \
    --episodes "${EPISODES:-40}" \
    --steps-per-ep "${STEPS_PER_EP:-16}" \
    --min-per-group "${MIN_PER_GROUP:-20}" \
    --max-frames "${MAX_FRAMES:-900}"
fi
echo "FRAMES=$FRAMES"

LABELS="$GT/labels_reflow_${STAMP}.npz"
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels pack \
  --wm-ckpt "$WM_BASE" --frames "$FRAMES" --out "$LABELS" --device cuda

# Fail fast if left_near missing (task-5 group 3).
"$PY" - <<PY
import numpy as np, sys
from collections import Counter
d = np.load("$LABELS", allow_pickle=True)
g = d["group"].astype(str)
c = Counter(g.tolist())
print("label_groups", dict(c))
n_left = int(c.get("left_near", 0))
need = int("$MIN_LEFT_NEAR")
if n_left < need:
    print(f"FATAL left_near={n_left} < {need}; refuse reflow (use frames_multi2 or FORCE_RECOLLECT)")
    sys.exit(3)
PY

"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels train \
  --wm-ckpt "$WM_BASE" --labels "$LABELS" --out-ckpt "$OUT_CKPT" \
  --steps "${HEAD_STEPS:-1000}" --batch 64 --device cuda

GATE_JSON="$GT/gate_reflow_${STAMP}.json"
set +e
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels gate \
  --wm-ckpt "$OUT_CKPT" --labels "$LABELS" --report "$GATE_JSON" --device cuda
RC=$?
set -e
echo "gate_rc=$RC report=$GATE_JSON out=$OUT_CKPT"
[[ "$RC" -eq 0 ]] || { echo "GATE FAILED"; exit 2; }
echo "REFLOW_OK $OUT_CKPT"
echo "$OUT_CKPT" > "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt"
