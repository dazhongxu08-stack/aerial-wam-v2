#!/usr/bin/env bash
# 主航道能力线（正确）：模型自己学飞，用【关罩】证明能力。
#
# 铁律：
#   - 训练：--no-shield（逼 actor 自己绕障）
#   - 能力门闩：关罩 P0（R0+R1 severe）+ d_final；罩开只作部署对照，不过关
#   - 禁止：开罩教师穷采 / 开罩-only BC / 用罩开 SR 宣称「会了」
#
#   HOST=84 bash experiments/aerial/scripts/run_mainchannel_capability.sh
#   HOST=125 SKIP_BASELINE=0 ITERS=40 bash ...
set -euo pipefail
HOST="${HOST:?set HOST}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PYTHON_BIN:-}"
for c in \
  "${PYTHON_BIN:-}" \
  "$ROOT/.venv/bin/python" \
  "${HOME}/sim_verify/.venv/bin/python" \
  /data/venvs/sim_verify/bin/python; do
  [[ -n "$c" && -x "$c" ]] && PY="$c" && break
done
PY="${PY:-python3}"

ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ANN_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
ITERS="${ITERS:-40}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
# Capability = no-shield. Deploy对照 optional.
REPORT_SHIELD_ON="${REPORT_SHIELD_ON:-1}"
OUT="${OUT:-$ART/v4_ac_ckpt_mainchannel_capability_${STAMP}_noshield_h${HOST}}"
MASTER="$ART/logs/mainchannel_capability_${STAMP}_h${HOST}.log"
CFG="$ROOT/configs/_tmp_mainchannel_capability_h${HOST}.yaml"
mkdir -p "$ART/logs" "$OUT"
exec >>"$MASTER" 2>&1

echo "=== MAINCHANNEL CAPABILITY (no-shield) HOST=$HOST stamp=$STAMP ==="
hostname; date -Is
echo "POLICY: train+gate = NO SHIELD; shield-on is deploy对照 only"

pick_actor() {
  local d=$1
  if [[ -f "$d/v4_ac_best.pt" ]]; then echo "$d/v4_ac_best.pt"
  elif [[ -f "$d/v4_ac_latest.pt" ]]; then echo "$d/v4_ac_latest.pt"
  else return 1
  fi
}

INIT="${INIT:-}"
if [[ -z "$INIT" ]]; then
  INIT="$(pick_actor "$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear" || true)"
fi
[[ -f "${INIT:-}" ]] || { echo FATAL no INIT hbclear; exit 1; }

if [[ -n "${WM:-}" && -f "$WM" ]]; then
  :
elif [[ -f "$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt" ]]; then
  WM="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
elif [[ -f "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt" ]]; then
  WM="$(cat "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt")"
elif [[ -f "$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt" ]]; then
  WM="$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt"
else
  echo FATAL need WM; exit 1
fi
DEPTH="${DEPTH_CKPT:-$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU_CKPT:-$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
echo "INIT=$INIT"
echo "WM=$WM"

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
# Capability train: no planner (same as directional OA actor FT) — actor must fly.
cfg.setdefault("planner", {})["enable"]=False
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", "$CFG")
PY

summarize_eval() {
  local tag=$1 json=$2
  TAG="$tag" JSON="$json" "$PY" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["JSON"])
tag = os.environ["TAG"]
if not p.exists():
    print(f"[{tag}] MISSING"); raise SystemExit(0)
rows = json.loads(p.read_text())["episodes"]
by = {int(r["route_idx"]): r for r in rows}
stick = sum(1 for ri in (0, 1) if by.get(ri, {}).get("severe_collision"))
arr = sum(1 for r in rows if r.get("arrived"))
ir = [round(float(r.get("intervention_rate") or 0), 2) for r in rows]
print(
    f"[{tag}] SR={arr}/{len(rows)} P0_stick={stick}/2 "
    f"R0df={round(float(by.get(0,{}).get('d_final_m') or 0),1)} "
    f"R1df={round(float(by.get(1,{}).get('d_final_m') or 0),1)} "
    f"IR={ir}"
)
# gate file
gate = {
    "tag": tag,
    "sr": f"{arr}/{len(rows)}",
    "p0_stick": stick,
    "capability_pass": stick <= 1 and arr >= 1,  # soft: not worse than overnight shield P0=1/2 aspiration
    "note": "CAPABILITY = no-shield; do not pass on shield-on alone",
    "episodes": [
        {
            "r": int(r["route_idx"]),
            "arrived": bool(r.get("arrived")),
            "severe": bool(r.get("severe_collision")),
            "d_final": r.get("d_final_m"),
            "ir": r.get("intervention_rate"),
        }
        for r in sorted(rows, key=lambda x: int(x["route_idx"]))
    ],
}
Path(p).parent.joinpath("CAPABILITY_SUMMARY.json").write_text(
    json.dumps(gate, indent=2) + "\n"
)
print("capability_pass", gate["capability_pass"])
PY
}

run_eval() {
  local tag=$1
  local actor=$2
  shift 2
  local extra=("$@")
  local eout="$ART/eval_mainchannel_capability_${STAMP}_h${HOST}_${tag}"
  mkdir -p "$eout/traj"
  echo "=== EVAL $tag actor=$(basename "$actor") ==="
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$actor" \
    --depth-ckpt "$DEPTH" --tau-ckpt "$TAU" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$eout/traj" --out "$eout/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10 --max-steps 600 --success-dist 3 --terminal-pin-rem-m 20 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5 \
    "${extra[@]}" \
    >"$ART/logs/mainchannel_capability_${STAMP}_h${HOST}_${tag}.log" 2>&1
  set -e
  summarize_eval "$tag" "$eout/eval_all.json"
}

[[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 8; }

# --- A) capability baseline: hbclear, NO SHIELD ---
if [[ "$SKIP_BASELINE" != "1" ]]; then
  echo "=== A) BASELINE no-shield (capability truth) ==="
  run_eval "baseline_noshield" "$INIT" --no-shield
else
  echo "SKIP_BASELINE"
fi

# --- B) train: no-planner + no-shield (model must fly) ---
if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "=== B) TRAIN no-shield FT from hbclear iters=$ITERS ==="
  "$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
    --src "$ANN" --out "$ANN_FOCUS" --routes 0,1,3,4 || true
  [[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 6; }
  "$PY" -m experiments.aerial.rl.train_v4_ac \
    --config configs/aerial_rl_urban_complex_p2c.yaml \
    --config-overlay configs/aerial_rl_urban_complex_directional_oa.yaml \
    --backend airsim --dynamics torch \
    --wm-ckpt "$WM" --init-actor-ckpt "$INIT" \
    --no-planner \
    --no-shield \
    --iters "$ITERS" --episodes-per-iter 2 \
    --imagine-batch 16 --imagine-horizon 15 \
    --device cuda --annotation "$ANN_FOCUS" \
    --ckpt-dir "$OUT" --save-every-iter \
    --min-steps-for-best 20 --near-goal-frac 0.25 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 6 \
    --min-spawn-clear-m 5.0
  echo "TRAIN_DONE"
else
  echo "SKIP_TRAIN"
fi

ACTOR="$(pick_actor "$OUT" || true)"
if [[ -z "${ACTOR:-}" || ! -f "$ACTOR" ]]; then
  echo "WARN no trained actor — eval INIT only"
  ACTOR="$INIT"
fi
echo "ACTOR=$ACTOR"

# --- C) capability gate: NO SHIELD ---
if [[ "$SKIP_EVAL" != "1" ]]; then
  echo "=== C) GATE no-shield (CAPABILITY) ==="
  run_eval "gate_noshield" "$ACTOR" --no-shield
  if [[ "$REPORT_SHIELD_ON" == "1" ]]; then
    echo "=== D) deploy对照 shield-on (NOT a pass criterion) ==="
    run_eval "deploy_shield_on" "$ACTOR" --shield-exclusion-forward-only \
      --planner --planner-horizon 5 --planner-rollout open_loop --planner-mock wm_bare
  fi
fi

echo "MAINCHANNEL_CAPABILITY_DONE stamp=$STAMP out=$OUT"
echo "REMINDER: only gate_noshield counts as capability; deploy_shield_on is对照"
