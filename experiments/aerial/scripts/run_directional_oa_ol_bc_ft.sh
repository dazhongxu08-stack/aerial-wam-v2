#!/usr/bin/env bash
# Open-loop planner teacher + online BC distill (peel behavior into actor).
# Hot-start from hbclear (not failed closed_loop CL). Then P0 stick eval.
#
#   HOST=125|84 bash experiments/aerial/scripts/run_directional_oa_ol_bc_ft.sh
set -euo pipefail
HOST="${HOST:?set HOST=125 or HOST=84}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PYTHON_BIN:-}"
if [[ -z "$PY" ]]; then
  for c in "${HOME}/sim_verify/.venv/bin/python" /data/venvs/sim_verify/bin/python; do
    [[ -x "$c" ]] && PY="$c" && break
  done
fi
PY="${PY:-python3}"

ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ANN_HARD="${ANN_HARD:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ANN_HARD_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
DEPTH="${DEPTH:-experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU:-experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
MASTER="$ART/logs/directional_oa_ol_bc_ft_${STAMP}_h${HOST}.log"
OUT="$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}_ol_bc_ft_h${HOST}"
CFG="$ROOT/configs/_tmp_ol_bc_eval_h${HOST}.yaml"
FT_ITERS="${FT_ITERS:-40}"
EP_PER_ITER="${EP_PER_ITER:-2}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
mkdir -p "$ART/logs" "$OUT"
exec > >(tee -a "$MASTER") 2>&1

echo "=== ol-bc-ft HOST=$HOST stamp=$STAMP iters=$FT_ITERS ==="
hostname; date -Is

pick_actor() {
  local d=$1
  if [[ -f "$d/v4_ac_best.pt" ]]; then echo "$d/v4_ac_best.pt"
  elif [[ -f "$d/v4_ac_latest.pt" ]]; then echo "$d/v4_ac_latest.pt"
  else return 1
  fi
}
INIT="${INIT:-}"
if [[ -z "$INIT" ]]; then
  INIT="$(pick_actor "$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear" \
    || pick_actor "$ART/v4_ac_ckpt_urban_directional_oa_20260922_162118_actor_ft" \
    || pick_actor "$ART/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2")"
fi
echo "INIT=$INIT"
[[ -n "${INIT:-}" && -f "$INIT" ]] || { echo "FATAL no INIT"; exit 1; }

if [[ -n "${WM:-}" && -f "$WM" ]]; then
  :
elif [[ -f "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt" ]]; then
  WM="$(cat "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt")"
else
  echo "FATAL need WM"; exit 1
fi
echo "WM=$WM"

"$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
  --src "$ANN_HARD" --out "$ANN_HARD_FOCUS" --routes 0,1,3,4

if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ -x "$RECOVER" ]]; then bash "$RECOVER" || true; sleep 6; fi
  "$PY" -m experiments.aerial.rl.train_v4_ac \
    --config configs/aerial_rl_urban_complex_p2c.yaml \
    --config-overlay configs/aerial_rl_urban_complex_directional_oa.yaml \
    --backend airsim --dynamics torch \
    --wm-ckpt "$WM" --init-actor-ckpt "$INIT" \
    --planner --planner-horizon 15 --planner-rollout open_loop \
    --no-shield \
    --enable-bc --bc-batch 64 --bc-updates-per-iter 8 --bc-loss-scale 1.0 \
    --iters "$FT_ITERS" --episodes-per-iter "$EP_PER_ITER" \
    --imagine-batch 16 --imagine-horizon 15 \
    --device cuda --annotation "$ANN_HARD_FOCUS" \
    --ckpt-dir "$OUT" --save-every-iter \
    --min-steps-for-best 20 \
    --near-goal-frac 0.25 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 6 \
    --min-spawn-clear-m 5.0 \
    2>&1 | tee -a "$ART/logs/directional_oa_ol_bc_ft_train_${STAMP}_h${HOST}.log"
fi

if [[ -f "$OUT/v4_ac_best.pt" ]]; then ACTOR_BEST="$OUT/v4_ac_best.pt"
elif [[ -f "$OUT/v4_ac_latest.pt" ]]; then ACTOR_BEST="$OUT/v4_ac_latest.pt"
else echo "FATAL no actor after train"; exit 1
fi
ACTOR_LATEST="$OUT/v4_ac_latest.pt"
[[ -f "$ACTOR_LATEST" ]] || ACTOR_LATEST="$ACTOR_BEST"
echo "ACTOR_BEST=$ACTOR_BEST"

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
cfg["planner"]["rollout_mode"]="open_loop"
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

run_eval() {
  local TAG=$1; shift
  local EOUT="$ART/eval_ol_bc_${STAMP}_h${HOST}_${TAG}"
  [[ -f "$EOUT/eval_all.json" ]] && { echo "SKIP eval $TAG"; return 0; }
  mkdir -p "$EOUT/traj"
  if [[ -x "$RECOVER" ]]; then bash "$RECOVER" || true; sleep 6; fi
  echo "=== EVAL $TAG ==="
  local depth_args=()
  [[ -f "$DEPTH" ]] && depth_args+=(--depth-ckpt "$DEPTH")
  [[ -f "$TAU" ]] && depth_args+=(--tau-ckpt "$TAU")
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN_HARD" --wm-ckpt "$WM" --actor-ckpt "$ACTOR_BEST" \
    "${depth_args[@]}" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$EOUT/traj" --out "$EOUT/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10.0 --max-steps 600 --success-dist 3.0 --terminal-pin-rem-m 20 \
    --min-spawn-z 24.0 --spawn-z-retry-m 5.0 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5.0 --shield-exclusion-forward-only \
    "$@" \
    > "$ART/logs/ol_bc_eval_${TAG}_${STAMP}_h${HOST}.log" 2>&1 || true
  "$PY" - <<PY
import json
from pathlib import Path
p=Path("$EOUT/eval_all.json"); tag="$TAG"
if not p.exists():
    print(f"[{tag}] MISSING"); raise SystemExit(0)
rows=json.loads(p.read_text())["episodes"]
by={int(r["route_idx"]): r for r in rows}
arr=sum(1 for r in rows if r.get("arrived"))
sev=sum(1 for r in rows if r.get("severe_collision"))
stick=sum(1 for ri in (0,1) if by.get(ri,{}).get("severe_collision"))
print(f"[{tag}] SR {arr}/{len(rows)} severe={sev}/{len(rows)} P0_stick={stick}/2")
for ri in (0,1,3,4):
    r=by.get(ri)
    if not r: continue
    print(f"  R{ri} arr={r.get('arrived')} sev={r.get('severe_collision')} d_f={round(float(r.get('d_final_m') or 0),1)}")
PY
}

# P0 matrix: open peel ± shield (same as antistick H1 follow-up)
run_eval ol_open_noshield --planner --planner-horizon 15 --planner-rollout open_loop --no-shield
run_eval ol_open_shield --planner --planner-horizon 15 --planner-rollout open_loop
run_eval ol_closed_noshield --planner --planner-horizon 15 --planner-rollout closed_loop --no-shield

echo "=== OL-BC-FT DONE host=$HOST stamp=$STAMP best=$ACTOR_BEST ==="
date -Is
