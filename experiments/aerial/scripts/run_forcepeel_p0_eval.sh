#!/usr/bin/env bash
# Force-peel P0 eval: ol_bc actor + updated planner (tight-mid leave-wall exec).
#   HOST=125|84 bash experiments/aerial/scripts/run_forcepeel_p0_eval.sh
set -euo pipefail
HOST="${HOST:?}"
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
STAMP="${STAMP:-20260923_0219_forcepeel}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
DEPTH="${DEPTH:-experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU:-experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
WM="$(cat "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt")"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
CFG="$ROOT/configs/_tmp_forcepeel_eval_h${HOST}.yaml"
MASTER="$ART/logs/forcepeel_${STAMP}_h${HOST}.log"
mkdir -p "$ART/logs"
# Prefer append to master (nohup/setsid friendly); avoid process-sub tee.
exec >>"$MASTER" 2>&1
echo "=== forcepeel HOST=$HOST stamp=$STAMP $(date -Is) ==="

case "$HOST" in
  125) ACTOR="$ART/v4_ac_ckpt_urban_directional_oa_20260923_001432_ol_bc_ft_h125/v4_ac_best.pt" ;;
  84)  ACTOR="$ART/v4_ac_ckpt_urban_directional_oa_20260923_001432_ol_bc_ft_h84/v4_ac_best.pt" ;;
  *) echo "FATAL HOST"; exit 2 ;;
esac
[[ -f "$ACTOR" ]] || { echo "FATAL no actor $ACTOR"; exit 1; }
echo "actor=$ACTOR wm=$WM"

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

# Wait up to ~20min if prior ol_bc eval still holds the sim
for _ in $(seq 1 40); do
  if pgrep -f "wam_phase2_long_eval" >/dev/null 2>&1; then
    echo "wait: long_eval busy"; sleep 30
  else
    break
  fi
done

run_one() {
  local TAG=$1; shift
  local EOUT="$ART/eval_${STAMP}_h${HOST}_${TAG}"
  if [[ -f "$EOUT/eval_all.json" ]]; then
    echo "SKIP $TAG"; return 0
  fi
  mkdir -p "$EOUT/traj"
  if [[ -x "$RECOVER" ]]; then bash "$RECOVER" || true; sleep 6; fi
  echo "=== EVAL $TAG ==="
  local depth_args=()
  [[ -f "$DEPTH" ]] && depth_args+=(--depth-ckpt "$DEPTH")
  [[ -f "$TAU" ]] && depth_args+=(--tau-ckpt "$TAU")
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$ACTOR" \
    "${depth_args[@]}" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$EOUT/traj" --out "$EOUT/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --planner --planner-horizon 15 --planner-rollout open_loop \
    --cruise-speed 10.0 --max-steps 600 --success-dist 3.0 --terminal-pin-rem-m 20 \
    --min-spawn-z 24.0 --spawn-z-retry-m 5.0 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5.0 --shield-exclusion-forward-only \
    "$@" \
    > "$ART/logs/forcepeel_${STAMP}_h${HOST}_${TAG}.log" 2>&1 || true
  "$PY" - <<PY
import json
from pathlib import Path
p=Path("$EOUT/eval_all.json"); tag="$TAG"
if not p.exists():
    print(f"[{tag}] MISSING"); raise SystemExit(0)
rows=json.loads(p.read_text())["episodes"]
by={int(r["route_idx"]): r for r in rows}
stick=sum(1 for ri in (0,1) if by.get(ri, {}).get("severe_collision"))
arr=sum(1 for r in rows if r.get("arrived"))
print(f"[{tag}] SR {arr}/{len(rows)} P0={stick}/2 R0df={round(float(by.get(0,{}).get('d_final_m') or 0),1)} R1df={round(float(by.get(1,{}).get('d_final_m') or 0),1)}")
PY
}

run_one forcepeel_open_noshield --no-shield
run_one forcepeel_open_shield
echo "=== FORCEPEEL DONE host=$HOST ==="
date -Is
