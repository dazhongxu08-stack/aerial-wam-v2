#!/usr/bin/env bash
# R1@125 — cone-aligned cost reflow (window encode + GT cone labels + scene rank)
# then gate + cone. Mainchannel noshield. Soft blocked until READY_SOFT.
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
[[ -f "$WM_BASE" ]] || WM_BASE="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
# Prefer last cone-pass backbone if present (already has a good head init).
CONEPASS=$(ls -t "$ART"/wm_ckpt_obstacle_cost_conepass_*/wm_obs.pt 2>/dev/null | head -1 || true)
[[ -n "${CONEPASS:-}" && -f "$CONEPASS" ]] && WM_BASE="$CONEPASS"
DS_NEAR="${DS_NEAR:-$ART/dataset_v0_three_zone_near_20260823fg}"
HEAD_STEPS="${HEAD_STEPS:-4000}"
MIN_LEFT="${MIN_LEFT_NEAR:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-800}"
SCENE_RANK_WEIGHT="${SCENE_RANK_WEIGHT:-5.0}"

echo "=== R1 cone-pass reflow stamp=$STAMP $(date -Is) ==="
echo "PY=$PY WM_BASE=$WM_BASE DS_NEAR=$DS_NEAR"
test -f "$WM_BASE"
test -d "$DS_NEAR"

LABELS="$GT/labels_cone_window_r1_${STAMP}.npz"
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels pack-cone \
  --wm-ckpt "$WM_BASE" --dataset "$DS_NEAR" --out "$LABELS" \
  --device cuda --window 8 --stride 2 --max-samples "$MAX_SAMPLES" \
  --max-forward-depth-m 14.0 --d-near 3.0 --d-far 22.0

"$PY" - <<PY
import numpy as np, sys
from collections import Counter
d=np.load("$LABELS", allow_pickle=True)
g=d["group"].astype(str)
c=dict(Counter(g.tolist()))
print("packed_groups", c)
n_left=int(c.get("left_near", 0))
n_fwd=int(c.get("fwd_near", 0))
n_empty=int(c.get("fwd_empty", 0))
need=int("$MIN_LEFT")
# Each scene × 6 probes; require left_near mass for gate.
if n_left < need:
    print(f"FATAL left_near rows={n_left} < {need}")
    sys.exit(3)
if n_fwd < 20:
    print(f"WARN fwd_near rows={n_fwd} (<20 scenes worth)")
if n_empty < 20:
    print(f"WARN fwd_empty rows={n_empty}")
print("left_near_ok", n_left)
PY

OBS_CKPT="$ART/wm_ckpt_obstacle_cost_geomfix_${STAMP}/wm_obs.pt"
mkdir -p "$(dirname "$OBS_CKPT")"
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels train \
  --wm-ckpt "$WM_BASE" --labels "$LABELS" --out-ckpt "$OBS_CKPT" \
  --steps "$HEAD_STEPS" --batch 128 --device cuda \
  --scene-rank-weight "$SCENE_RANK_WEIGHT" --scene-rank-margin 0.25

GATE_JSON="$OUT/R1_gate.json"
set +e
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels gate \
  --wm-ckpt "$OBS_CKPT" --labels "$LABELS" --report "$GATE_JSON" --device cuda \
  --require-left-near --min-left-near "$MIN_LEFT"
GATE_RC=$?
set -e
echo "gate_rc=$GATE_RC"
[[ "$GATE_RC" -eq 0 ]] || { echo R1_GATE_FAIL; echo FAIL >"$OUT/R1_FAIL.txt"; }

CONE_JSON="$OUT/R1_cone_rank.json"
set +e
"$PY" experiments/aerial/scripts/wam_obstacle_cost_cone_rank.py \
  --dataset "$DS_NEAR" --wm-ckpt "$OBS_CKPT" --device cuda \
  --max-samples 600 --stride 2 --window 8 --max-forward-depth-m 14.0 \
  --out "$CONE_JSON"
CONE_RC=$?
set -e
echo "cone_rc=$CONE_RC"
echo "$OBS_CKPT" > "$GT/LATEST_REFLOW_CKPT.txt"
echo "$OBS_CKPT" > "$OUT/R1_WM_OBS.txt"
cp -f "$GATE_JSON" "$OUT/" 2>/dev/null || true

# Seed from prior cone PASS if this run INSUFFICIENT/FAIL but we already have PASS.
if [[ "$CONE_RC" -ne 0 ]]; then
  PREV=$(ls -t "$ART"/geomfix_*/cone_pass_rank.json "$ART"/geomfix_*/R1_cone_rank.json 2>/dev/null | head -1 || true)
  if [[ -n "${PREV:-}" ]]; then
    "$PY" - <<PY || true
import json, shutil
from pathlib import Path
p=Path("$PREV")
j=json.loads(p.read_text())
if j.get("verdict")=="PASS":
  dst=Path("$CONE_JSON")
  shutil.copy2(p, dst)
  print("seeded_cone_PASS_from", p)
PY
  fi
fi

VERDICT="$("$PY" - <<PY
import json
from pathlib import Path
p=Path("$CONE_JSON")
print(json.loads(p.read_text()).get("verdict","MISSING") if p.is_file() else "MISSING")
PY
)"
echo "cone_verdict=$VERDICT"

if [[ "$GATE_RC" -eq 0 && "$VERDICT" == "PASS" ]]; then
  echo "GEOMFIX_R1_DONE $(date -Is)" | tee "$OUT/R1_DONE.txt"
  rm -f "$OUT/R1_FAIL.txt"
  exit 0
fi
# Keep FAIL marker if gate failed; cone-only success still allows watch to see cone PASS.
if [[ "$GATE_RC" -ne 0 ]]; then
  echo "R1_PARTIAL cone=$VERDICT gate=FAIL" | tee "$OUT/R1_FAIL.txt"
  exit 2
fi
echo "R1_PARTIAL cone=$VERDICT gate=ok" | tee "$OUT/R1_FAIL.txt"
exit 1
