#!/usr/bin/env bash
# 主航道：关罩 + clearance-centering 奖励学贴墙解法。
# 不重跑已证伪的「关罩 FT 无新奖励」；在 directional OA 权重上叠 antistick 奖励。
#
#   HOST=84 bash experiments/aerial/scripts/run_mainchannel_antistick_reward.sh
set -euo pipefail
HOST="${HOST:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PYTHON_BIN:-}"
for c in "${PYTHON_BIN:-}" "$ROOT/.venv/bin/python" "${HOME}/sim_verify/.venv/bin/python" /data/venvs/sim_verify/bin/python; do
  [[ -n "$c" && -x "$c" ]] && PY="$c" && break
done
PY="${PY:-python3}"

ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ITERS="${ITERS:-40}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ANN_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
OUT="${OUT:-$ART/v4_ac_ckpt_mainchannel_antistick_${STAMP}_noshield_h${HOST}}"
MASTER="$ART/logs/mainchannel_antistick_${STAMP}_h${HOST}.log"
CFG="$ROOT/configs/_tmp_mainchannel_antistick_h${HOST}.yaml"
mkdir -p "$ART/logs" "$OUT"
exec >>"$MASTER" 2>&1

echo "=== ANTISTICK REWARD no-shield HOST=$HOST stamp=$STAMP ==="
hostname; date -Is

pick_actor() {
  local d=$1
  [[ -f "$d/v4_ac_best.pt" ]] && { echo "$d/v4_ac_best.pt"; return 0; }
  [[ -f "$d/v4_ac_latest.pt" ]] && { echo "$d/v4_ac_latest.pt"; return 0; }
  return 1
}

# Prefer prior directional OA (already no-shield trained), else hbclear.
INIT="${INIT:-}"
if [[ -z "$INIT" ]]; then
  INIT="$(pick_actor "$ART/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2" \
    || pick_actor "$ART/v4_ac_ckpt_urban_directional_oa_20260922_162118_actor_ft" \
    || pick_actor "$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear" || true)"
fi
[[ -f "${INIT:-}" ]] || { echo FATAL no INIT; exit 1; }

if [[ -n "${WM:-}" && -f "$WM" ]]; then :
elif [[ -f "$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt" ]]; then
  WM="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
elif [[ -f "$ART/wm_ckpt_obstacle_cost_20260922_022358/wm_obs.pt" ]]; then
  WM="$ART/wm_ckpt_obstacle_cost_20260922_022358/wm_obs.pt"
elif [[ -f "$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt" ]]; then
  WM="$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt"
else
  echo FATAL need WM with obstacle_cost if possible; exit 1
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
anti=yaml.safe_load(Path("configs/aerial_rl_urban_complex_antistick_reward.yaml").read_text())
cfg=deep_merge(deep_merge(base, oa), anti)
cfg.setdefault("planner", {})["enable"]=False
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", "$CFG", "w_center", cfg["reward"].get("w_clearance_center"),
      "w_press", cfg["reward"].get("w_press_wall"))
PY

summarize() {
  local tag=$1 json=$2
  TAG="$tag" JSON="$json" "$PY" - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ["JSON"]); tag=os.environ["TAG"]
if not p.exists():
    print(f"[{tag}] MISSING"); raise SystemExit
rows=json.loads(p.read_text())["episodes"]
by={int(r["route_idx"]):r for r in rows}
stick=sum(1 for ri in (0,1) if by.get(ri,{}).get("severe_collision"))
arr=sum(1 for r in rows if r.get("arrived"))
print(f"[{tag}] SR={arr}/{len(rows)} P0_stick={stick}/2 "
      f"R0df={round(float(by.get(0,{}).get('d_final_m') or 0),1)} "
      f"R1df={round(float(by.get(1,{}).get('d_final_m') or 0),1)} "
      f"IR={[round(float(r.get('intervention_rate') or 0),2) for r in rows]}")
Path(p.parent/"CAPABILITY_SUMMARY.json").write_text(json.dumps({
    "tag": tag, "sr": f"{arr}/{len(rows)}", "p0_stick": stick,
    "capability_pass": stick <= 1,
    "note": "no-shield gate only",
}, indent=2)+"\n")
PY
}

run_eval() {
  local tag=$1 actor=$2
  local eout="$ART/eval_mainchannel_antistick_${STAMP}_h${HOST}_${tag}"
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
    >"$ART/logs/mainchannel_antistick_${STAMP}_h${HOST}_${tag}.log" 2>&1
  set -e
  summarize "$tag" "$eout/eval_all.json"
}

[[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 8; }

if [[ "${SKIP_BASELINE:-0}" != "1" ]]; then
  echo "=== A) baseline no-shield (pre antistick FT) ==="
  run_eval "baseline_noshield" "$INIT"
fi

echo "=== B) TRAIN no-shield + antistick reward iters=$ITERS ==="
"$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
  --src "$ANN" --out "$ANN_FOCUS" --routes 0,1,3,4 || true
[[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 6; }

"$PY" -m experiments.aerial.rl.train_v4_ac \
  --config "$CFG" \
  --backend airsim --dynamics torch \
  --wm-ckpt "$WM" --init-actor-ckpt "$INIT" \
  --no-planner --no-shield \
  --iters "$ITERS" --episodes-per-iter 2 \
  --imagine-batch 16 --imagine-horizon 15 \
  --device cuda --annotation "$ANN_FOCUS" \
  --ckpt-dir "$OUT" --save-every-iter \
  --min-steps-for-best 20 --near-goal-frac 0.25 \
  --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 6 \
  --min-spawn-clear-m 5.0
echo TRAIN_DONE

ACTOR="$(pick_actor "$OUT" || true)"
[[ -n "${ACTOR:-}" && -f "$ACTOR" ]] || ACTOR="$INIT"
echo "ACTOR=$ACTOR"

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  echo "=== C) GATE no-shield (capability) ==="
  run_eval "gate_noshield" "$ACTOR"
fi

echo "MAINCHANNEL_ANTISTICK_DONE stamp=$STAMP out=$OUT"
echo "REMINDER: only gate_noshield counts; do not claim success from shield-on SR"
