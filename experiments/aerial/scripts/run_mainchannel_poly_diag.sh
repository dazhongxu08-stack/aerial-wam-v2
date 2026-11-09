#!/usr/bin/env bash
# Main-channel diagnostic: polyline + rolling_global on hard134 inland routes.
# Compare to antistick toward_g numbers (same actors).
#
#   HOST=125 bash experiments/aerial/scripts/run_mainchannel_poly_diag.sh
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
OUT_ROOT="${OUT_ROOT:-$ART/mainchannel_poly_${STAMP}_h${HOST}}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
DEPTH="${DEPTH:-experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU:-experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
WM_OA="${WM_OA:-}"
[[ -z "$WM_OA" && -f "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt" ]] && WM_OA="$(cat "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt")"
WM_V5="${WM_V5:-experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
CFG="$ROOT/configs/_tmp_mainchannel_poly_h${HOST}.yaml"
MASTER="$ART/logs/mainchannel_poly_${STAMP}_h${HOST}.log"
mkdir -p "$ART/logs" "$OUT_ROOT"
exec > >(tee -a "$MASTER") 2>&1

echo "=== mainchannel poly diag HOST=$HOST stamp=$STAMP ==="
hostname; date -Is

case "$HOST" in
  125)
    ACTOR_CL="${ACTOR_CL:-$ART/v4_ac_ckpt_urban_directional_oa_20260922_172751_cl_ft/v4_ac_best.pt}"
    ACTOR_V5="${ACTOR_V5:-$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear/v4_ac_latest.pt}"
    ARMS=(POLY_V5_shield POLY_CL_noshield POLY_CL_shield)
    ;;
  84)
    ACTOR_CL="${ACTOR_CL:-$ART/v4_ac_ckpt_urban_directional_oa_20260922_175214_cl_ft/v4_ac_best.pt}"
    ACTOR_V5="${ACTOR_V5:-$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear/v4_ac_latest.pt}"
    ARMS=(POLY_CL_noshield POLY_CL_shield)
    ;;
  *) echo "FATAL HOST"; exit 2 ;;
esac

[[ -f "$ACTOR_CL" && -n "$WM_OA" && -f "$WM_OA" ]] || { echo "FATAL ckpt/WM"; exit 1; }

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
p=Path("$EOUT/eval_all.json"); tag="$TAG"
if not p.exists():
    print(f"[{tag}] MISSING"); raise SystemExit(0)
rows=json.loads(p.read_text())["episodes"]
by={int(r["route_idx"]): r for r in rows}
arr=sum(1 for r in rows if r.get("arrived"))
sev=sum(1 for r in rows if r.get("severe_collision"))
stick=sum(1 for ri in (0,1) if by.get(ri,{}).get("severe_collision"))
def df(ri):
    r=by.get(ri); return None if not r else round(float(r.get("d_final_m") or 0),1)
print(f"[{tag}] SR {arr}/{len(rows)} severe={sev}/{len(rows)} P0={stick}/2 R0df={df(0)} R1df={df(1)} R3df={df(3)} R4df={df(4)}")
PY
}

run_arm() {
  local TAG=$1
  local EOUT="$OUT_ROOT/$TAG"
  [[ -f "$EOUT/eval_all.json" ]] && { echo "SKIP $TAG"; summarize "$TAG" "$EOUT"; return 0; }
  mkdir -p "$EOUT/traj"
  local ACTOR="$ACTOR_CL" WM="$WM_OA" H=5 ROLLOUT=open_loop
  local SHIELD_ARGS=()
  case "$TAG" in
    POLY_V5_shield)
      ACTOR="$ACTOR_V5"; WM="$WM_V5"; H=5; ROLLOUT=open_loop; SHIELD_ARGS=()
      [[ -f "$ACTOR" && -f "$WM" ]] || { echo "SKIP no V5"; return 0; }
      ;;
    POLY_CL_noshield) SHIELD_ARGS=(--no-shield); H=15; ROLLOUT=open_loop ;;
    POLY_CL_shield) SHIELD_ARGS=(); H=15; ROLLOUT=open_loop ;;
  esac
  if [[ -x "$RECOVER" ]]; then bash "$RECOVER" || true; sleep 6; fi
  echo "=== EVAL $TAG actor=$(basename "$ACTOR") poly+RG H=$H ==="
  local depth_args=()
  [[ -f "$DEPTH" ]] && depth_args+=(--depth-ckpt "$DEPTH")
  [[ -f "$TAU" ]] && depth_args+=(--tau-ckpt "$TAU")
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$ACTOR" \
    "${depth_args[@]}" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$EOUT/traj" --out "$EOUT/eval_all.json" \
    --subgoal-source polyline --rolling-global --heading-assist \
    --global-horizon-m 60 --global-replan-period-s 1.0 \
    --r-m-intent 100 \
    --planner --planner-horizon "$H" --planner-rollout "$ROLLOUT" \
    --cruise-speed 10.0 --max-steps 600 --success-dist 3.0 --terminal-pin-rem-m 20 \
    --min-spawn-z 24.0 --spawn-z-retry-m 5.0 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5.0 --shield-exclusion-forward-only \
    "${SHIELD_ARGS[@]}" \
    > "$ART/logs/mainchannel_poly_${STAMP}_h${HOST}_${TAG}.log" 2>&1 || true
  summarize "$TAG" "$EOUT"
}

for TAG in "${ARMS[@]}"; do run_arm "$TAG"; done
echo "=== MAINCHANNEL POLY DONE host=$HOST stamp=$STAMP ==="
date -Is
