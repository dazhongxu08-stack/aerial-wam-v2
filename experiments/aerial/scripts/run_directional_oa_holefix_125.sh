#!/usr/bin/env bash
# Hole-fix pipeline for directional OA (review 2026-09-22):
#   1) reflow obstacle_cost_head + gate
#   2) curriculum stage-A easy inland (closed_loop collect, no-shield)
#   3) stage-B hard 0/1/3/4 (closed_loop collect, no-shield)
#   4) dual eval: closed_loop + actor-only, n=3 seeds, reverse stats
#
# Usage on 125 (repo root):
#   bash experiments/aerial/scripts/run_directional_oa_holefix_125.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PY="${PYTHON_BIN:-python3}"
ANN_EASY="${ANN_EASY:-experiments/aerial/phase3_unified/annotations/outdoor_complex_easy_inland.json}"
ANN_HARD="${ANN_HARD:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
DEPTH="${DEPTH:-experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU:-experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
ACTOR0="${ACTOR0:-$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear/v4_ac_latest.pt}"
MASTER="$ART/logs/directional_oa_holefix_${STAMP}.log"
mkdir -p "$ART/logs" "$ART"
exec > >(tee -a "$MASTER") 2>&1

echo "=== holefix stamp=$STAMP ==="

# Ensure easy annotation exists
if [[ ! -f "$ROOT/$ANN_EASY" ]]; then
  "$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
    --src "$ANN_HARD" \
    --out "$ANN_EASY" \
    --routes "${EASY_ROUTES:-11,14,15,16}"
fi

# --- 1) reflow obstacle head ---
if [[ "${SKIP_REFLOW:-0}" == "1" ]]; then
  WM="${WM:-$ART/wm_ckpt_obstacle_cost_20260922_022358/wm_obs.pt}"
  echo "SKIP_REFLOW=1 using WM=$WM"
else
  set +e
  STAMP="$STAMP" bash experiments/aerial/scripts/reflow_obstacle_cost_125.sh
  RC=$?
  set -e
  if [[ "$RC" -eq 0 && -f "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt" ]]; then
    WM="$(cat "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt")"
  else
    WM="${WM_FALLBACK:-$ART/wm_ckpt_obstacle_cost_20260922_022358/wm_obs.pt}"
    echo "WARN: reflow gate failed (rc=$RC) — fallback WM=$WM"
  fi
fi
test -f "$WM" || { echo "FATAL missing WM $WM"; exit 1; }
test -f "$ACTOR0" || { echo "FATAL missing actor $ACTOR0"; exit 1; }

# Merge eval yaml (planner on for closed_loop eval)
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
print("wrote", "$CFG")
PY

train_stage() {
  local TAG=$1 ANN=$2 INIT=$3 ITERS=$4
  local OUT="$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}_${TAG}"
  local PTR="$ART/logs/holefix_actor_${TAG}_${STAMP}.txt"
  mkdir -p "$OUT"
  echo "=== TRAIN $TAG iters=$ITERS ann=$ANN ==="
  "$PY" -m experiments.aerial.rl.train_v4_ac \
    --config configs/aerial_rl_urban_complex_p2c.yaml \
    --config-overlay configs/aerial_rl_urban_complex_directional_oa.yaml \
    --backend airsim --dynamics torch \
    --wm-ckpt "$WM" \
    --init-actor-ckpt "$INIT" \
    --planner --planner-horizon 15 --planner-rollout closed_loop \
    --no-shield \
    --iters "$ITERS" --episodes-per-iter 1 \
    --imagine-batch 16 --imagine-horizon 15 \
    --device cuda --annotation "$ANN" \
    --ckpt-dir "$OUT" --save-every-iter \
    --near-goal-frac "${NEAR_GOAL_FRAC:-0.25}" \
    --min-spawn-z "${MIN_SPAWN_Z:-24}" \
    --spawn-z-retry-m "${SPAWN_Z_RETRY_M:-5}" \
    --spawn-z-max-retries "${SPAWN_Z_MAX_RETRIES:-6}" \
    2>&1 | tee -a "$ART/logs/directional_oa_holefix_train_${TAG}_${STAMP}.log"
  # Prefer collect-return best over latest (guards +160-style overtrain collapse).
  if [[ -f "$OUT/v4_ac_best.pt" ]]; then
    echo "$OUT/v4_ac_best.pt" > "$PTR"
    echo "TRAIN_DONE $TAG -> $OUT/v4_ac_best.pt (best)"
  else
    echo "$OUT/v4_ac_latest.pt" > "$PTR"
    echo "TRAIN_DONE $TAG -> $OUT/v4_ac_latest.pt"
  fi
}

# --- 2) curriculum easy ---
STAGE_A_ITERS="${STAGE_A_ITERS:-40}"
train_stage easy "$ANN_EASY" "$ACTOR0" "$STAGE_A_ITERS"
ACTOR_A="$(cat "$ART/logs/holefix_actor_easy_${STAMP}.txt")"

# --- 3) hard 014 ---
STAGE_B_ITERS="${STAGE_B_ITERS:-40}"
# Restrict hard stage to routes 0,1,3,4 via focus annotation
ANN_HARD_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
"$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
  --src "$ANN_HARD" --out "$ANN_HARD_FOCUS" --routes 0,1,3,4
train_stage hard014 "$ANN_HARD_FOCUS" "$ACTOR_A" "$STAGE_B_ITERS"
ACTOR_B="$(cat "$ART/logs/holefix_actor_hard014_${STAMP}.txt")"

# --- 4) dual eval ---
run_eval() {
  local TAG=$1; shift
  local OUT="$ART/eval_directional_oa_${STAMP}_${TAG}"
  mkdir -p "$OUT/traj"
  echo "=== EVAL $TAG ==="
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN_HARD" --wm-ckpt "$WM" --actor-ckpt "$ACTOR_B" \
    --depth-ckpt "$DEPTH" --tau-ckpt "$TAU" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$OUT/traj" --out "$OUT/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10.0 --max-steps 600 --success-dist 3.0 --terminal-pin-rem-m 20 \
    --min-spawn-z 24.0 --spawn-z-retry-m 5.0 --spawn-z-max-retries 4 \
    "$@" \
    > "$ART/logs/directional_oa_holefix_eval_${TAG}_${STAMP}.log" 2>&1
  "$PY" - <<PY
import json
from pathlib import Path
import numpy as np
p=Path("$OUT/eval_all.json")
rows=json.loads(p.read_text())["episodes"]
arr=sum(1 for r in rows if r.get("arrived"))
progs=[round(100*float(r.get("progress_ratio",0)),1) for r in sorted(rows, key=lambda x:int(x["route_idx"]))]
print(f"[$TAG] SR {arr}/{len(rows)} prog={progs}")
anno=json.loads(Path("$ANN_HARD").read_text())
routes=anno.get("routes") or anno.get("episodes")
for r in sorted(rows, key=lambda x:int(x["route_idx"])):
    ri=int(r["route_idx"])
    print(f"  r={ri} arr={r.get('arrived')} col={r.get('collided')} prog={float(r.get('progress_ratio',0))*100:.1f}% steps={r.get('steps')}")
    if not r.get("arrived"):
        continue
    traj=Path("$OUT")/f"traj/route{ri:02d}.jsonl"
    if not traj.exists():
        continue
    rows_t=[json.loads(l) for l in traj.read_text().strip().splitlines()]
    pos=np.array([t["pos"] for t in rows_t], float)
    yaw=np.deg2rad(np.array([t["yaw_deg"] for t in rows_t], float))
    dp=np.diff(pos, axis=0)
    c,s=np.cos(yaw[:-1]), np.sin(yaw[:-1])
    body_dx=dp[:,0]*c+dp[:,1]*s
    g=np.asarray(routes[ri]["pos"][-1], float)
    rel=g[:2]-pos[:-1,:2]
    goal_fwd=rel[:,0]*c+rel[:,1]*s
    goal_lat=-rel[:,0]*s+rel[:,1]*c
    bearing_err=np.rad2deg(np.arctan2(goal_lat, goal_fwd))
    print(f"    reverse: frac_body_dx<0={float(np.mean(body_dx<-0.05)):.0%} frac_nose_away={float(np.mean(np.abs(bearing_err)>90)):.0%} mean_dx={body_dx.mean():+.3f}")
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
print("=== HOLEFIX SUMMARY ===")
for prefix in ["cl_noshield", "actor_noshield"]:
    srs=[]
    for s in (["b0","b1","b2"] if prefix.startswith("cl") else ["a0","a1","a2"]):
        p=art/f"eval_directional_oa_{stamp}_{prefix}_{s}/eval_all.json"
        if not p.exists():
            print(prefix, s, "MISSING"); continue
        rows=json.loads(p.read_text())["episodes"]
        arr=sum(1 for r in rows if r.get("arrived"))
        srs.append(arr)
        print(f"{prefix}_{s}: SR {arr}/4")
    if srs:
        print(f"{prefix} median={st.median(srs)}/4 mean={sum(srs)/len(srs):.2f}/4")
print("HOLEFIX_DONE actor=$ACTOR_B wm=$WM stamp=$STAMP")
PY
