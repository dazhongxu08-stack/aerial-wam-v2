#!/usr/bin/env bash
# 主航道 V2：floor 零 FT 基线评测 + floor INIT curiosity FT。
#
#   HOST=84 STAMP=20260925_1007 bash experiments/aerial/scripts/run_mainchannel_v2_floor_then_b.sh
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
: "${REWARD_OVERLAY:=configs/aerial_rl_urban_complex_b_curiosity.yaml}"
: "${ITERS:=80}"
: "${IMAGINE_HORIZON:=40}"
VARIANT=v2_floor_then_curiosity
EPI_PER_ITER="${EPI_PER_ITER:-4}"
NEAR_GOAL_FRAC="${NEAR_GOAL_FRAC:-0.4}"
IMAGINE_BATCH="${IMAGINE_BATCH:-16}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ANN_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
OUT="${OUT:-$ART/v4_ac_ckpt_mainchannel_v2_floor_b_${STAMP}_noshield_h${HOST}}"
MASTER="$ART/logs/mainchannel_v2_floor_b_${STAMP}_h${HOST}.log"
CFG="$ROOT/configs/_tmp_mainchannel_v2_floor_b_h${HOST}.yaml"
mkdir -p "$ART/logs" "$OUT"
exec >>"$MASTER" 2>&1

echo "=== V2 floor_then_B noshield HOST=$HOST stamp=$STAMP variant=$VARIANT ==="
hostname; date -Is
echo "POLICY: V2 eval floor baseline then curiosity FT from floor; NO SHIELD"
echo "REWARD_OVERLAY=$REWARD_OVERLAY ITERS=$ITERS EPI=$EPI_PER_ITER H=$IMAGINE_HORIZON B=$IMAGINE_BATCH"

pick_actor() {
  local d=$1
  if [[ -f "$d/v4_ac_best.pt" ]]; then echo "$d/v4_ac_best.pt"
  elif [[ -f "$d/v4_ac_latest.pt" ]]; then echo "$d/v4_ac_latest.pt"
  else return 1
  fi
}

INIT="${INIT:-}"
if [[ -z "$INIT" ]]; then
  INIT="$(pick_actor "$ART/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2" || true)"
fi
[[ -f "${INIT:-}" ]] || { echo FATAL no INIT floor best_sr2; exit 1; }

if [[ -n "${WM:-}" && -f "$WM" ]]; then
  :
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
ov=yaml.safe_load(Path("$REWARD_OVERLAY").read_text())
cfg=deep_merge(deep_merge(base, oa), ov)
cfg.setdefault("planner", {})["enable"]=False
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
r=cfg.get("reward") or {}
assert float(r.get("w_press_wall") or 0) == 0.0, r
assert float(r.get("w_clearance_center") or 0) == 0.0, r
assert float(r.get("w_curiosity") or 0) > 0.0, r
assert float(r.get("w_progress") or 0) > 0, r
assert float(r.get("success_bonus") or 0) > 0, r
assert bool(r.get("forbid_backward_motion")), r
assert bool(r.get("gate_progress_by_heading")), r
assert float(r.get("w_backward") or 0) > 0, r
keys=("w_progress","w_collision","success_bonus","w_curiosity",
      "w_clearance_center","w_press_wall","forbid_backward_motion")
print("reward V2 floor_then_B OK", {k:r.get(k) for k in keys})
print("variant=$VARIANT host=$HOST H=$IMAGINE_HORIZON iters=$ITERS")
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
      f"R1df={round(float(by.get(1,{}).get('d_final_m') or 0),1)} "
      f"IR={[round(float(r.get('intervention_rate') or 0),2) for r in rows]}")
Path(p).parent.joinpath("CAPABILITY_SUMMARY.json").write_text(json.dumps({
    "tag": tag, "sr": f"{arr}/{len(rows)}", "p0_stick": stick,
    "capability_pass": bool(arr >= 1 and stick <= 1),
    "note": "no-shield V2 floor_then_B; pass needs arrived>=1 and P0_stick<=1",
}, indent=2)+"\n")
PY
}

run_eval() {
  local tag=$1 actor=$2
  local eout="$ART/eval_mainchannel_v2_floor_b_${STAMP}_h${HOST}_${tag}"
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
    >"$ART/logs/mainchannel_v2_floor_b_${STAMP}_h${HOST}_${tag}.log" 2>&1
  set -e
  summarize "$tag" "$eout/eval_all.json"
}

[[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 8; }

if [[ "$SKIP_BASELINE" != "1" ]]; then
  run_eval "baseline_noshield" "$INIT"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "=== TRAIN V2 floor_then_B noshield iters=$ITERS epi=$EPI_PER_ITER ==="
  "$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
    --src "$ANN" --out "$ANN_FOCUS" --routes 0,1,3,4 || true
  "$PY" -m experiments.aerial.rl.train_v4_ac \
    --config "$CFG" \
    --backend airsim --dynamics torch \
    --wm-ckpt "$WM" --init-actor-ckpt "$INIT" \
    --no-planner \
    --no-shield \
    --iters "$ITERS" --episodes-per-iter "$EPI_PER_ITER" \
    --imagine-batch "$IMAGINE_BATCH" --imagine-horizon "$IMAGINE_HORIZON" \
    --device cuda --annotation "$ANN_FOCUS" \
    --ckpt-dir "$OUT" --save-every-iter \
    --min-steps-for-best 20 --near-goal-frac "$NEAR_GOAL_FRAC" \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 6 \
    --min-spawn-clear-m 5.0
  echo "TRAIN_DONE"
fi

ACTOR="$(pick_actor "$OUT" || echo "$INIT")"
echo "ACTOR=$ACTOR"

if [[ "$SKIP_EVAL" != "1" ]]; then
  echo "=== GATE noshield (capability) ==="
  run_eval "gate_noshield" "$ACTOR"
fi

echo "V2_FLOOR_THEN_B_NOSHIELD_DONE stamp=$STAMP out=$OUT"
