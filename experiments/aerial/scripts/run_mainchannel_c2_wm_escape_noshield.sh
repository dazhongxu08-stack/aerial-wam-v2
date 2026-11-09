#!/usr/bin/env bash
# C2 · wm_escape open-side teacher + BC (floor INIT). Gate = noshield capability.
#
#   HOST=84 STAMP=20260925_2100 bash experiments/aerial/scripts/run_mainchannel_c2_wm_escape_noshield.sh
set -euo pipefail
HOST="${HOST:?set HOST}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export OA_PLANNER_MOCK="${OA_PLANNER_MOCK:-wm_escape}"

PY="${PYTHON_BIN:-}"
for c in "${PYTHON_BIN:-}" "$ROOT/.venv/bin/python" "${HOME}/sim_verify/.venv/bin/python" /data/venvs/sim_verify/bin/python; do
  [[ -n "$c" && -x "$c" ]] && PY="$c" && break
done
PY="${PY:-python3}"

ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ITERS="${ITERS:-100}"
EPI_PER_ITER="${EPI_PER_ITER:-2}"
NEAR_GOAL_FRAC="${NEAR_GOAL_FRAC:-0.25}"
SKIP_BASELINE="${SKIP_BASELINE:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ANN_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
OUT="${OUT:-$ART/v4_ac_ckpt_mainchannel_c2_wm_escape_${STAMP}_noshield_h${HOST}}"
MASTER="$ART/logs/mainchannel_c2_wm_escape_${STAMP}_h${HOST}.log"
CFG="$ROOT/configs/_tmp_mainchannel_c2_wm_escape_h${HOST}.yaml"
mkdir -p "$ART/logs" "$OUT"
exec >>"$MASTER" 2>&1

echo "=== C2 wm_escape noshield HOST=$HOST stamp=$STAMP OA_PLANNER_MOCK=$OA_PLANNER_MOCK ==="
hostname; date -Is

pick_actor() {
  local d=$1
  [[ -f "$d/v4_ac_best.pt" ]] && { echo "$d/v4_ac_best.pt"; return 0; }
  [[ -f "$d/v4_ac_latest.pt" ]] && { echo "$d/v4_ac_latest.pt"; return 0; }
  return 1
}

INIT="${INIT:-}"
if [[ -z "$INIT" ]]; then
  INIT="$(pick_actor "$ART/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2" || true)"
fi
[[ -f "${INIT:-}" ]] || { echo FATAL no INIT floor; exit 1; }

if [[ -n "${WM:-}" && -f "$WM" ]]; then :
elif [[ -f "$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt" ]]; then
  WM="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
elif [[ -f "$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt" ]]; then
  WM="$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt"
else
  echo FATAL need WM; exit 1
fi
DEPTH="${DEPTH_CKPT:-$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU_CKPT:-$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
echo "INIT=$INIT WM=$WM"

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
oa=yaml.safe_load(Path("configs/aerial_rl_urban_complex_directional_oa.yaml").read_text())
esc=yaml.safe_load(Path("configs/aerial_rl_urban_complex_c2_wm_escape.yaml").read_text())
cfg=deep_merge(deep_merge(base, oa), esc)
assert cfg["planner"].get("mock_mode") == "wm_escape", cfg["planner"]
assert cfg["planner"].get("enable") is True
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("cfg C2 wm_escape OK", cfg["planner"])
PY

summarize() {
  local tag=$1 json=$2
  TAG="$tag" JSON="$json" "$PY" - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ["JSON"]); tag=os.environ["TAG"]
if not p.exists():
    print(f"[{tag}] MISSING"); raise SystemExit(0)
rows=json.loads(p.read_text())["episodes"]
by={int(r["route_idx"]):r for r in rows}
stick=sum(1 for ri in (0,1) if by.get(ri,{}).get("severe_collision"))
arr=sum(1 for r in rows if r.get("arrived"))
print(f"[{tag}] SR={arr}/{len(rows)} P0_stick={stick}/2 "
      f"R0df={round(float(by.get(0,{}).get('d_final_m') or 0),1)} "
      f"R1df={round(float(by.get(1,{}).get('d_final_m') or 0),1)}")
Path(p).parent.joinpath("CAPABILITY_SUMMARY.json").write_text(json.dumps({
    "tag": tag, "sr": f"{arr}/{len(rows)}", "p0_stick": stick,
    "capability_pass": bool(arr >= 1 and stick <= 1),
    "note": "no-shield C2 wm_escape; pass needs arrived>=1 and P0_stick<=1",
}, indent=2)+"\n")
PY
}

run_eval() {
  local tag=$1 actor=$2
  local eout="$ART/eval_mainchannel_c2_wm_escape_${STAMP}_h${HOST}_${tag}"
  mkdir -p "$eout/traj"
  echo "=== EVAL $tag ==="
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$actor" \
    --depth-ckpt "$DEPTH" --tau-ckpt "$TAU" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$eout/traj" --out "$eout/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10 --max-steps 600 --success-dist 3 --terminal-pin-rem-m 20 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5 \
    --no-shield \
    >"$ART/logs/mainchannel_c2_wm_escape_${STAMP}_h${HOST}_${tag}.log" 2>&1
  set -e
  summarize "$tag" "$eout/eval_all.json"
}

[[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 8; }

if [[ "$SKIP_BASELINE" != "1" ]]; then
  run_eval "baseline_noshield" "$INIT"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "=== TRAIN C2 wm_escape iters=$ITERS ==="
  "$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
    --src "$ANN" --out "$ANN_FOCUS" --routes 0,1,3,4 || true
  [[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 6; }
  "$PY" -m experiments.aerial.rl.train_v4_ac \
    --config "$CFG" \
    --backend airsim --dynamics torch \
    --wm-ckpt "$WM" --init-actor-ckpt "$INIT" \
    --planner --planner-horizon 15 --planner-rollout open_loop \
    --no-shield \
    --enable-bc --bc-batch 64 --bc-updates-per-iter 8 --bc-loss-scale 1.0 \
    --iters "$ITERS" --episodes-per-iter "$EPI_PER_ITER" \
    --imagine-batch 16 --imagine-horizon 15 \
    --device cuda --annotation "$ANN_FOCUS" \
    --ckpt-dir "$OUT" --save-every-iter \
    --min-steps-for-best 20 --near-goal-frac "$NEAR_GOAL_FRAC" \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 6 \
    --min-spawn-clear-m 5.0
  echo TRAIN_DONE
fi

ACTOR="$(pick_actor "$OUT" || echo "$INIT")"
echo "ACTOR=$ACTOR"

if [[ "$SKIP_EVAL" != "1" ]]; then
  run_eval "gate_noshield" "$ACTOR"
fi

echo "C2_WM_ESCAPE_NOSHIELD_DONE stamp=$STAMP out=$OUT"
date -Is
