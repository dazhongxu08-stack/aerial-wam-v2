#!/usr/bin/env bash
# Quick P0 diag: hbclear V5 actor + open_loop, optional --planner-mock.
#   HOST=84 PLANNER_MOCK=rules bash experiments/aerial/scripts/run_hbclear_p0_diag.sh
set -euo pipefail
HOST="${HOST:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${PYTHON_BIN:-}"
for c in "${HOME}/sim_verify/.venv/bin/python" /data/venvs/sim_verify/bin/python; do
  [[ -x "$c" ]] && PY="$c" && break
done
PY="${PY:-python3}"
ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TAG="${TAG:-hbclear_open_shield}"
MOCK="${PLANNER_MOCK:-}"
ACTOR="${ACTOR:-$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear/v4_ac_latest.pt}"
WM="${WM:-$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
CFG="$ROOT/configs/_tmp_hbclear_diag_h${HOST}.yaml"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
EOUT="$ART/eval_hbclear_diag_${STAMP}_h${HOST}_${TAG}${MOCK:+_$MOCK}"
MASTER="$ART/logs/hbclear_diag_${STAMP}_h${HOST}.log"
mkdir -p "$ART/logs" "$EOUT/traj"
exec >>"$MASTER" 2>&1
echo "=== hbclear diag HOST=$HOST mock=${MOCK:-none} ==="
[[ -f "$ACTOR" && -f "$WM" ]] || { echo FATAL ckpt; exit 1; }
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
PY
[[ -x "$RECOVER" ]] && { bash "$RECOVER" || true; sleep 6; }
mock_args=()
[[ -n "$MOCK" ]] && mock_args+=(--planner-mock "$MOCK")
"$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
  --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$ACTOR" \
  --depth-ckpt "$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt" \
  --tau-ckpt "$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt" --goal-feat-mode meter \
  --routes 0,1,3,4 --traj-out "$EOUT/traj" --out "$EOUT/eval_all.json" \
  --subgoal-source toward_g --r-m-intent 100 \
  --planner --planner-horizon 5 --planner-rollout open_loop \
  "${mock_args[@]}" \
  --cruise-speed 10 --max-steps 600 --success-dist 3 --terminal-pin-rem-m 20 \
  --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 \
  --min-spawn-clear-m 5 --shield-exclusion-forward-only \
  > "$ART/logs/hbclear_diag_${STAMP}_h${HOST}_${TAG}.log" 2>&1 || true
"$PY" - <<PY
import json
from pathlib import Path
p=Path("$EOUT/eval_all.json")
if not p.exists():
    print("MISSING"); raise SystemExit(0)
rows=json.loads(p.read_text())["episodes"]
by={int(r["route_idx"]):r for r in rows}
stick=sum(1 for ri in (0,1) if by.get(ri,{}).get("severe_collision"))
arr=sum(1 for r in rows if r.get("arrived"))
print(f"[hbclear mock={'$MOCK' or 'none'}] SR {arr}/{len(rows)} P0={stick}/2 R0df={round(float(by.get(0,{}).get('d_final_m') or 0),1)} R1df={round(float(by.get(1,{}).get('d_final_m') or 0),1)}")
PY
echo DONE
