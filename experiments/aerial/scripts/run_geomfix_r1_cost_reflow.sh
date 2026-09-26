#!/usr/bin/env bash
# R1@125 — left_near-required obstacle_cost reflow + cone rank (mainchannel, no shield).
# Uses frames_multi2 (has left_near) then optional AirSim top-up collect.
set -euo pipefail
HOST="${HOST:-125}"
STAMP="${STAMP:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
GT="$ART/obstacle_cost_gt_depth"
OUT="$ART/geomfix_${STAMP}"
LOG="$ART/logs/geomfix_R1_${STAMP}_h${HOST}.log"
mkdir -p "$OUT" "$ART/logs" "$GT"
exec > >(tee -a "$LOG") 2>&1

if [[ -x "$HOME/sim_verify/.venv/bin/python" ]]; then
  PY="$HOME/sim_verify/.venv/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON_BIN:-python3}"
fi
[[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]] && PY="$PYTHON_BIN"

WM_BASE="${WM_BASE:-$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
# Prefer backbone that already has depth_aux; fall back to gatefix file if needed.
[[ -f "$WM_BASE" ]] || WM_BASE="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
FRAMES="${FRAMES:-$GT/frames_multi2.npz}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
HEAD_STEPS="${HEAD_STEPS:-2500}"
MIN_LEFT="${MIN_LEFT_NEAR:-20}"
DS_NEAR="$ART/dataset_v0_three_zone_near_20260823fg"

echo "=== R1 cost reflow stamp=$STAMP $(date -Is) ==="
echo "PY=$PY WM_BASE=$WM_BASE FRAMES=$FRAMES"
test -f "$WM_BASE"
test -f "$FRAMES"

# Prefer existing packed labels that already keep left_near (avoid reclass wipe).
if [[ -f "$GT/labels_multi2_gate.npz" ]]; then
  echo "=== use labels_multi2_gate.npz (has left_near) ==="
  LABELS="$GT/labels_multi2_gate.npz"
else
  LABELS="$GT/labels_geomfix_${STAMP}.npz"
  "$PY" -m experiments.aerial.rl.train_obstacle_cost_labels pack \
    --wm-ckpt "$WM_BASE" --frames "$FRAMES" --out "$LABELS" --device cuda \
    --d-near 5.0 --d-far 18.0
fi

"$PY" - <<PY
import numpy as np, sys
from collections import Counter
d=np.load("$LABELS", allow_pickle=True)
g=d["group"].astype(str)
c=dict(Counter(g.tolist()))
print("packed_groups", c)
n=int(c.get("left_near", 0))
need=int("$MIN_LEFT")
# labels_multi2 has 63 left_near; allow min(need, 20) floor for first pass
need=min(need, max(20, n if n>0 else need))
if n < 20:
    print(f"FATAL left_near={n} < 20")
    sys.exit(3)
print("left_near_ok", n)
PY

OBS_CKPT="$ART/wm_ckpt_obstacle_cost_geomfix_${STAMP}/wm_obs.pt"
mkdir -p "$(dirname "$OBS_CKPT")"
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels train \
  --wm-ckpt "$WM_BASE" --labels "$LABELS" --out-ckpt "$OBS_CKPT" \
  --steps "$HEAD_STEPS" --batch 128 --device cuda

GATE_JSON="$OUT/R1_gate.json"
set +e
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels gate \
  --wm-ckpt "$OBS_CKPT" --labels "$LABELS" --report "$GATE_JSON" --device cuda \
  --require-left-near --min-left-near "$MIN_LEFT"
GATE_RC=$?
set -e
echo "gate_rc=$GATE_RC"
[[ "$GATE_RC" -eq 0 ]] || { echo R1_GATE_FAIL; echo FAIL >"$OUT/R1_FAIL.txt"; exit 2; }

# cone rank on hard near episodes
CONE_JSON="$OUT/R1_cone_rank.json"
set +e
"$PY" experiments/aerial/scripts/wam_obstacle_cost_cone_rank.py \
  --dataset "$DS_NEAR" --wm-ckpt "$OBS_CKPT" --device cuda \
  --max-samples 400 --stride 2 --window 8 \
  --out "$CONE_JSON"
CONE_RC=$?
set -e
echo "cone_rc=$CONE_RC"
echo "$OBS_CKPT" > "$GT/LATEST_REFLOW_CKPT.txt"
echo "$OBS_CKPT" > "$OUT/R1_WM_OBS.txt"
cp -f "$GATE_JSON" "$OUT/" 2>/dev/null || true

# AirSim top-up collect to grow left_near (keep machine busy); then optional retrain loop
if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
  echo "=== R1 top-up collect left_near ==="
  TOP="$GT/frames_geomfix_topup_${STAMP}.npz"
  set +e
  "$PY" -m experiments.aerial.rl.collect_obstacle_cost_gt_frames \
    --config configs/aerial_rl_urban_complex_p2c.yaml \
    --annotation "$ANN" \
    --out "$TOP" \
    --host 127.0.0.1 --port 41451 \
    --episodes "${EPISODES:-60}" \
    --steps-per-ep "${STEPS_PER_EP:-20}" \
    --d-near 5.0 --d-far 18.0 \
    --min-per-group 30 --max-frames 1200
  COL_RC=$?
  set -e
  if [[ $COL_RC -eq 0 && -f "$TOP" ]]; then
    echo "=== merge multi2 + topup → retrain ==="
    MERGED="$GT/frames_geomfix_merged_${STAMP}.npz"
    "$PY" - <<PY
import numpy as np
from pathlib import Path
a=np.load("$FRAMES", allow_pickle=True)
b=np.load("$TOP", allow_pickle=True)
keys=["rgb","depth","action_xyz","proprio","group"]
out={}
for k in keys:
  out[k]=np.concatenate([a[k], b[k]], axis=0)
if "obstacle_label" in a.files and "obstacle_label" in b.files:
  out["obstacle_label"]=np.concatenate([a["obstacle_label"], b["obstacle_label"]], axis=0)
Path("$MERGED").parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed("$MERGED", **out)
print("merged_n", len(out["group"]))
from collections import Counter
print(dict(Counter(out["group"].astype(str).tolist())))
PY
    LABELS2="$GT/labels_geomfix_merged_${STAMP}.npz"
    "$PY" -m experiments.aerial.rl.train_obstacle_cost_labels pack \
      --wm-ckpt "$WM_BASE" --frames "$MERGED" --out "$LABELS2" --device cuda \
      --d-near 5.0 --d-far 18.0
    OBS2="$ART/wm_ckpt_obstacle_cost_geomfix_${STAMP}_v2/wm_obs.pt"
    mkdir -p "$(dirname "$OBS2")"
    "$PY" -m experiments.aerial.rl.train_obstacle_cost_labels train \
      --wm-ckpt "$WM_BASE" --labels "$LABELS2" --out-ckpt "$OBS2" \
      --steps "$HEAD_STEPS" --batch 128 --device cuda
    "$PY" -m experiments.aerial.rl.train_obstacle_cost_labels gate \
      --wm-ckpt "$OBS2" --labels "$LABELS2" --report "$OUT/R1_gate_v2.json" --device cuda \
      --require-left-near --min-left-near "$MIN_LEFT"
    echo "$OBS2" > "$GT/LATEST_REFLOW_CKPT.txt"
    echo "$OBS2" > "$OUT/R1_WM_OBS.txt"
    "$PY" experiments/aerial/scripts/wam_obstacle_cost_cone_rank.py \
      --dataset "$DS_NEAR" --wm-ckpt "$OBS2" --device cuda \
      --max-samples 400 --out "$OUT/R1_cone_rank_v2.json" || true
  fi
fi

echo "GEOMFIX_R1_DONE $(date -Is)" | tee "$OUT/R1_DONE.txt"
rm -f "$OUT/R1_FAIL.txt"
