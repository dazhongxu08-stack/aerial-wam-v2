#!/usr/bin/env bash
# Anti-stick (wall-first) eval matrix — P0 = R0/R1 severe, not SR/climb.
#
# Plan: docs/superpowers/plans/2026-09-22-antistick-wall-first.md
#
# Usage (on the box, or via ssh):
#   HOST=125 bash experiments/aerial/scripts/run_antistick_wall_first.sh
#   HOST=84  bash experiments/aerial/scripts/run_antistick_wall_first.sh
#
# 125 arms (serial): V5 ceiling → CL open noshield → CL open shield
# 84  arms (serial): CL open noshield → CL open shield → CL closed shield
#
# Skips closed+noshield baselines (already measured on both hosts).
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
OUT_ROOT="${OUT_ROOT:-$ART/antistick_${STAMP}_h${HOST}}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
DEPTH="${DEPTH:-experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU:-experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
WM_OA="${WM_OA:-}"
if [[ -z "$WM_OA" && -f "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt" ]]; then
  WM_OA="$(cat "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt")"
fi
WM_V5="${WM_V5:-experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
ROUTES="${ROUTES:-0,1,3,4}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
CFG="$ROOT/configs/_tmp_antistick_eval_h${HOST}.yaml"
MASTER="$ART/logs/antistick_${STAMP}_h${HOST}.log"
mkdir -p "$ART/logs" "$OUT_ROOT"

exec > >(tee -a "$MASTER") 2>&1
echo "=== antistick HOST=$HOST stamp=$STAMP root=$ROOT out=$OUT_ROOT ==="
hostname; date -Is
echo "PY=$PY"
echo "WM_OA=$WM_OA"
echo "WM_V5=$WM_V5"

case "$HOST" in
  125)
    ACTOR_CL="${ACTOR_CL:-$ART/v4_ac_ckpt_urban_directional_oa_20260922_172751_cl_ft/v4_ac_best.pt}"
    ACTOR_V5="${ACTOR_V5:-$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear/v4_ac_latest.pt}"
    ARMS=(V5 CL_B_open_noshield CL_C_open_shield)
    ;;
  84)
    ACTOR_CL="${ACTOR_CL:-$ART/v4_ac_ckpt_urban_directional_oa_20260922_175214_cl_ft/v4_ac_best.pt}"
    ACTOR_V5=""
    ARMS=(CL_B_open_noshield CL_C_open_shield CL_D_closed_shield)
    ;;
  *)
    echo "FATAL HOST must be 125 or 84"; exit 2
    ;;
esac

[[ -f "$ACTOR_CL" ]] || { echo "FATAL missing ACTOR_CL=$ACTOR_CL"; exit 1; }
[[ -n "$WM_OA" && -f "$WM_OA" ]] || { echo "FATAL missing WM_OA"; exit 1; }
if [[ "$HOST" == "125" ]]; then
  [[ -f "$ACTOR_V5" ]] || { echo "FATAL missing ACTOR_V5=$ACTOR_V5"; exit 1; }
  [[ -f "$WM_V5" ]] || { echo "FATAL missing WM_V5=$WM_V5"; exit 1; }
fi

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
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", "$CFG")
PY

summarize() {
  local TAG=$1 EOUT=$2
  "$PY" - <<PY
import json
from pathlib import Path
p=Path("$EOUT/eval_all.json")
tag="$TAG"
if not p.exists():
    print(f"[{tag}] MISSING")
    raise SystemExit(0)
rows=json.loads(p.read_text())["episodes"]
by={int(r["route_idx"]): r for r in rows}
arr=sum(1 for r in rows if r.get("arrived"))
sev=sum(1 for r in rows if r.get("severe_collision"))
def cell(ri, key, fmt=lambda x: x):
    r=by.get(ri)
    if not r: return "-"
    return fmt(r.get(key))
print(
    f"[{tag}] SR {arr}/{len(rows)} severe={sev}/{len(rows)} | "
    f"R0 sev={cell(0,'severe_collision')} d_f={cell(0,'d_final_m', lambda x: round(float(x),1) if x is not None else None)} | "
    f"R1 sev={cell(1,'severe_collision')} d_f={cell(1,'d_final_m', lambda x: round(float(x),1) if x is not None else None)} | "
    f"R3 sev={cell(3,'severe_collision')} d_f={cell(3,'d_final_m', lambda x: round(float(x),1) if x is not None else None)} | "
    f"R4 sev={cell(4,'severe_collision')} d_f={cell(4,'d_final_m', lambda x: round(float(x),1) if x is not None else None)}"
)
# stick gate: R0+R1 severe count
stick = sum(1 for ri in (0,1) if by.get(ri,{}).get("severe_collision"))
print(f"[{tag}] P0_stick_severe_R0R1={stick}/2")
PY
}

run_arm() {
  local TAG=$1
  local EOUT="$OUT_ROOT/$TAG"
  if [[ -f "$EOUT/eval_all.json" ]]; then
    echo "SKIP completed $TAG"
    summarize "$TAG" "$EOUT"
    return 0
  fi
  mkdir -p "$EOUT/traj"

  local ACTOR="$ACTOR_CL"
  local WM="$WM_OA"
  local H=15
  local ROLLOUT=closed_loop
  local SHIELD_ARGS=(--no-shield)

  case "$TAG" in
    V5)
      ACTOR="$ACTOR_V5"
      WM="$WM_V5"
      H=5
      ROLLOUT=open_loop
      SHIELD_ARGS=()
      ;;
    CL_B_open_noshield)
      ROLLOUT=open_loop
      SHIELD_ARGS=(--no-shield)
      ;;
    CL_C_open_shield)
      ROLLOUT=open_loop
      SHIELD_ARGS=()
      ;;
    CL_D_closed_shield)
      ROLLOUT=closed_loop
      SHIELD_ARGS=()
      ;;
    *)
      echo "FATAL unknown TAG=$TAG"; return 2
      ;;
  esac

  if [[ -x "$RECOVER" ]]; then
    echo "=== recover renderer before $TAG ==="
    bash "$RECOVER" || true
    sleep 6
  fi

  echo "=== EVAL $TAG actor=$(basename "$ACTOR") wm=$(basename "$WM") H=$H rollout=$ROLLOUT shield=${SHIELD_ARGS[*]:-ON} ==="
  local depth_args=()
  [[ -f "$DEPTH" ]] && depth_args+=(--depth-ckpt "$DEPTH")
  [[ -f "$TAU" ]] && depth_args+=(--tau-ckpt "$TAU")

  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$ACTOR" \
    "${depth_args[@]}" --goal-feat-mode meter \
    --routes "$ROUTES" --traj-out "$EOUT/traj" --out "$EOUT/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --planner --planner-horizon "$H" --planner-rollout "$ROLLOUT" \
    --cruise-speed 10.0 --max-steps 600 --success-dist 3.0 --terminal-pin-rem-m 20 \
    --min-spawn-z 24.0 --spawn-z-retry-m 5.0 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5.0 \
    --shield-exclusion-forward-only \
    "${SHIELD_ARGS[@]}" \
    > "$ART/logs/antistick_${STAMP}_h${HOST}_${TAG}.log" 2>&1 || true

  summarize "$TAG" "$EOUT"
}

for TAG in "${ARMS[@]}"; do
  run_arm "$TAG"
done

{
  echo "=== ANTISTICK DONE host=$HOST stamp=$STAMP out=$OUT_ROOT ==="
  echo "Sealed V5 reference (125, prior): artifacts/urban_complex_toward_g_gate_20260921_hbclear_facegoal_v5  SR 3/4 severe R1 only"
  echo "P0 gate: look at P0_stick_severe_R0R1 per arm (want decline vs closed+noshield baseline)."
} | tee -a "$MASTER"
echo "DONE $(date -Is)"
