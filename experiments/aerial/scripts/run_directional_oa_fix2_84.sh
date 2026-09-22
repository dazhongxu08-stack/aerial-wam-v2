#!/usr/bin/env bash
# Directional-OA fix2 on **84 only** (independent of 125).
#   1) reflow obstacle head from frames_multi2 (left_near) + gate
#   2) short closed_loop finetune from hard014/easy best
#   3) dual eval n=3 (cl + actor)
#
#   bash experiments/aerial/scripts/run_directional_oa_fix2_84.sh
set -euo pipefail
ROOT="${AERIAL_ROOT:-/data/aerial-wam-v2}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PY="${PYTHON_BIN:-/data/venvs/sim_verify/bin/python}"
[[ -x "$PY" ]] || PY=python3
HOLE_STAMP="${HOLE_STAMP:-20260922_141414}"
HARD_DIR="$ART/v4_ac_ckpt_urban_directional_oa_${HOLE_STAMP}_hard014"
EASY_DIR="$ART/v4_ac_ckpt_urban_directional_oa_${HOLE_STAMP}_easy"
ANN_HARD="${ANN_HARD:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ANN_HARD_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
DEPTH="${DEPTH:-$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU:-$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
MASTER="$ART/logs/directional_oa_fix2_84_${STAMP}.log"
mkdir -p "$ART/logs" "$ART"
exec > >(tee -a "$MASTER") 2>&1

echo "=== fix2-84 stamp=$STAMP hole=$HOLE_STAMP root=$ROOT ==="
hostname
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
ss -ltnp | grep 41451 || { echo "FATAL AirSim :41451 down"; exit 1; }

# --- 1) reflow with multi2 (84-local assets) ---
export FRAMES="${FRAMES:-$ART/obstacle_cost_gt_depth/frames_multi2.npz}"
export WM_BASE="${WM_BASE:-$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
# Fallback: retrain head on existing wm_obs if raw depth-aux base missing
if [[ ! -f "$WM_BASE" && -f "$ART/wm_ckpt_obstacle_cost_20260922_022358/wm_obs.pt" ]]; then
  WM_BASE="$ART/wm_ckpt_obstacle_cost_20260922_022358/wm_obs.pt"
  echo "WARN: using wm_obs as WM_BASE=$WM_BASE"
fi
test -f "$FRAMES" || { echo "FATAL missing FRAMES=$FRAMES"; exit 1; }
test -f "$WM_BASE" || { echo "FATAL missing WM_BASE=$WM_BASE"; exit 1; }
export STAMP PYTHON_BIN="$PY" OUT_CKPT="$ART/wm_ckpt_obstacle_cost_reflow84_${STAMP}/wm_obs.pt"
bash experiments/aerial/scripts/reflow_obstacle_cost_125.sh
WM="$(cat "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt")"
test -f "$WM"
echo "WM=$WM"

pick_actor() {
  local d=$1
  if [[ -f "$d/v4_ac_best.pt" ]]; then echo "$d/v4_ac_best.pt"
  elif [[ -f "$d/v4_ac_latest.pt" ]]; then echo "$d/v4_ac_latest.pt"
  else return 1
  fi
}
INIT="$(pick_actor "$HARD_DIR" || pick_actor "$EASY_DIR" || pick_actor "$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear")"
echo "INIT=$INIT"

"$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
  --src "$ANN_HARD" --out "$ANN_HARD_FOCUS" --routes 0,1,3,4

# --- 2) short finetune ---
OUT="$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}_fix2_84"
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
  2>&1 | tee -a "$ART/logs/directional_oa_fix2_84_train_${STAMP}.log"

if [[ -f "$OUT/v4_ac_best.pt" ]]; then ACTOR="$OUT/v4_ac_best.pt"
else ACTOR="$OUT/v4_ac_latest.pt"
fi
echo "ACTOR=$ACTOR"

CFG="$ROOT/configs/_tmp_directional_oa_eval_84.yaml"
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
print("wrote", "$CFG")
PY

run_eval() {
  local TAG=$1; shift
  local EOUT="$ART/eval_directional_oa_${STAMP}_${TAG}"
  mkdir -p "$EOUT/traj"
  echo "=== EVAL $TAG ==="
  local depth_args=()
  [[ -f "$DEPTH" ]] && depth_args+=(--depth-ckpt "$DEPTH")
  [[ -f "$TAU" ]] && depth_args+=(--tau-ckpt "$TAU")
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN_HARD" --wm-ckpt "$WM" --actor-ckpt "$ACTOR" \
    "${depth_args[@]}" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$EOUT/traj" --out "$EOUT/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10.0 --max-steps 600 --success-dist 3.0 --terminal-pin-rem-m 20 \
    --min-spawn-z 24.0 --spawn-z-retry-m 5.0 --spawn-z-max-retries 4 \
    "$@" \
    > "$ART/logs/directional_oa_fix2_84_eval_${TAG}_${STAMP}.log" 2>&1 || true
  "$PY" - <<PY
import json
from pathlib import Path
p=Path("$EOUT/eval_all.json")
if not p.exists():
    print("[$TAG] MISSING")
else:
    rows=json.loads(p.read_text())["episodes"]
    arr=sum(1 for r in rows if r.get("arrived"))
    progs=[round(100*float(r.get("progress_ratio",0)),1) for r in sorted(rows, key=lambda x:int(x["route_idx"]))]
    print(f"[$TAG] SR {arr}/{len(rows)} prog={progs}")
PY
}

for SEED in b0 b1 b2; do
  run_eval "cl_noshield_${SEED}" --planner --planner-horizon 15 --planner-rollout closed_loop --no-shield
done
for SEED in a0 a1 a2; do
  run_eval "actor_noshield_${SEED}" --no-shield
done

"$PY" - <<PY
import json, statistics as st
from pathlib import Path
art=Path("$ART"); stamp="$STAMP"
print("=== FIX2-84 SUMMARY ===")
for prefix in ["cl_noshield", "actor_noshield"]:
    srs=[]
    seeds=["b0","b1","b2"] if prefix.startswith("cl") else ["a0","a1","a2"]
    for s in seeds:
        p=art/f"eval_directional_oa_{stamp}_{prefix}_{s}/eval_all.json"
        if not p.exists():
            print(f"  {prefix}_{s}: MISSING"); continue
        rows=json.loads(p.read_text())["episodes"]
        arr=sum(1 for r in rows if r.get("arrived"))
        srs.append(arr)
        print(f"  {prefix}_{s}: SR {arr}/4")
    if srs:
        print(f"  {prefix} median_SR={st.median(srs)}/4 range={min(srs)}-{max(srs)}")
print("actor=$ACTOR wm=$WM")
print("=== FIX2-84 DONE ===")
PY
