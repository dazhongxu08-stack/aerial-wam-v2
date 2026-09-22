#!/usr/bin/env bash
# Post-holefix repairs (2026-09-22 review leftovers):
#   1) reflow obstacle head from frames_multi2 (left_near) + gate
#   2) short closed_loop finetune from hard014 best with new WM
#   3) dual eval n=3 (cl + actor), prefer v4_ac_best.pt
#
# Does NOT stop a running holefix train — run after hard014 finishes, or
# set WAIT_HARD=1 to poll. Safe to run on 125 while AirSim free.
#
#   WAIT_HARD=1 bash experiments/aerial/scripts/run_directional_oa_fix2_125.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PY="${PYTHON_BIN:-${HOME}/sim_verify/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3
HOLE_STAMP="${HOLE_STAMP:-20260922_141414}"
HARD_DIR="$ART/v4_ac_ckpt_urban_directional_oa_${HOLE_STAMP}_hard014"
EASY_DIR="$ART/v4_ac_ckpt_urban_directional_oa_${HOLE_STAMP}_easy"
ANN_HARD="${ANN_HARD:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ANN_HARD_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
DEPTH="${DEPTH:-experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU:-experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
MASTER="$ART/logs/directional_oa_fix2_${STAMP}.log"
mkdir -p "$ART/logs"
exec > >(tee -a "$MASTER") 2>&1

echo "=== fix2 stamp=$STAMP hole=$HOLE_STAMP ==="

if [[ "${WAIT_HARD:-0}" == "1" ]]; then
  echo "waiting for hard014 train_v4_ac to finish..."
  while pgrep -af 'train_v4_ac.*141414_hard014|train_v4_ac.*hard014' | grep -v pgrep >/dev/null; do
    sleep 30
  done
  echo "hard014 idle"
fi

# --- 1) reflow with multi2 ---
export FRAMES="$ART/obstacle_cost_gt_depth/frames_multi2.npz"
export STAMP
export PYTHON_BIN="$PY"
bash experiments/aerial/scripts/reflow_obstacle_cost_125.sh
WM="$(cat "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt")"
test -f "$WM"

# Pick actor init: hard best > hard latest > easy best > easy latest
pick_actor() {
  local d=$1
  if [[ -f "$d/v4_ac_best.pt" ]]; then echo "$d/v4_ac_best.pt"
  elif [[ -f "$d/v4_ac_latest.pt" ]]; then echo "$d/v4_ac_latest.pt"
  else return 1
  fi
}
INIT="$(pick_actor "$HARD_DIR" || pick_actor "$EASY_DIR")"
echo "INIT=$INIT WM=$WM"

# Ensure hard focus ann
"$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
  --src "$ANN_HARD" --out "$ANN_HARD_FOCUS" --routes 0,1,3,4

# --- 2) short finetune ---
OUT="$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}_fix2"
mkdir -p "$OUT"
FT_ITERS="${FT_ITERS:-20}"
"$PY" -m experiments.aerial.rl.train_v4_ac \
  --config configs/aerial_rl_urban_complex_p2c.yaml \
  --config-overlay configs/aerial_rl_urban_complex_directional_oa.yaml \
  --backend airsim --dynamics torch \
  --wm-ckpt "$WM" --init-actor-ckpt "$INIT" \
  --planner --planner-horizon 15 --planner-rollout closed_loop \
  --no-shield \
  --iters "$FT_ITERS" --episodes-per-iter 1 \
  --imagine-batch 16 --imagine-horizon 15 \
  --device cuda --annotation "$ANN_HARD_FOCUS" \
  --ckpt-dir "$OUT" --save-every-iter \
  --near-goal-frac 0.25 \
  --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 6 \
  2>&1 | tee -a "$ART/logs/directional_oa_fix2_train_${STAMP}.log"

if [[ -f "$OUT/v4_ac_best.pt" ]]; then ACTOR="$OUT/v4_ac_best.pt"
else ACTOR="$OUT/v4_ac_latest.pt"
fi
echo "ACTOR=$ACTOR"

# Eval yaml
CFG="$ROOT/configs/_tmp_directional_oa_eval.yaml"
"$PY" - <<PY
import yaml
from pathlib import Path
def deep_merge(a,b):
    out=dict(a or {})
    for k,v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k]=deep_merge(out[k], v)
        else:
            out[k]=v
    return out
base=yaml.safe_load(Path("configs/aerial_rl_urban_complex_p2c.yaml").read_text())
ov=yaml.safe_load(Path("configs/aerial_rl_urban_complex_directional_oa.yaml").read_text())
cfg=deep_merge(base, ov)
cfg.setdefault("planner", {})["enable"]=True
cfg["planner"]["rollout_mode"]="closed_loop"
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

run_eval() {
  local TAG=$1; shift
  local EOUT="$ART/eval_directional_oa_${STAMP}_${TAG}"
  mkdir -p "$EOUT/traj"
  echo "=== EVAL $TAG ==="
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN_HARD" --wm-ckpt "$WM" --actor-ckpt "$ACTOR" \
    --depth-ckpt "$DEPTH" --tau-ckpt "$TAU" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$EOUT/traj" --out "$EOUT/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10.0 --max-steps 600 --success-dist 3.0 --terminal-pin-rem-m 20 \
    --min-spawn-z 24.0 --spawn-z-retry-m 5.0 --spawn-z-max-retries 4 \
    "$@" \
    > "$ART/logs/directional_oa_fix2_eval_${TAG}_${STAMP}.log" 2>&1 || true
  "$PY" - <<PY
import json
from pathlib import Path
p=Path("$EOUT/eval_all.json")
if not p.exists():
    print("[$TAG] MISSING")
else:
    rows=json.loads(p.read_text())["episodes"]
    arr=sum(1 for r in rows if r.get("arrived"))
    print(f"[$TAG] SR {arr}/{len(rows)}")
PY
}

for SEED in b0 b1 b2; do
  run_eval "cl_noshield_${SEED}" --planner --planner-horizon 15 --planner-rollout closed_loop --no-shield
done
for SEED in a0 a1 a2; do
  run_eval "actor_noshield_${SEED}" --no-shield
done
echo "=== FIX2 DONE stamp=$STAMP actor=$ACTOR wm=$WM ==="
