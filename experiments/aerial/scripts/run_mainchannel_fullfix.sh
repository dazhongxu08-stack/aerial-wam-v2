#!/usr/bin/env bash
# Main-channel fullfix (2026-09-23):
#   1) DepthScene expert collect on ALL requested routes (R0/R1/R3/R4)
#      with require-all-requested-routes + per-route keep gate
#   2) Offline expert BC (+ short online open_loop BC) from that dataset
#   3) P0 hbclear diag with wm_bare (yaw commit lives in ImaginationPlanner)
#
#   HOST=84 bash experiments/aerial/scripts/run_mainchannel_fullfix.sh
# Resume knobs:
#   APPEND=1          — continue collect into existing DS
#   SKIP_COLLECT=1    — gate already passed
#   SKIP_BC=1         — actor already trained
#   SKIP_EVAL=0       — set 1 to stop after BC
set -euo pipefail
HOST="${HOST:?set HOST=84 or 125}"
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
ROUTES="${ROUTES:-0,1,3,4}"
MIN_KEPT="${MIN_KEPT:-8}"
MIN_KEPT_PER_ROUTE="${MIN_KEPT_PER_ROUTE:-2}"
# Dataset tag from routes (e.g. 0,1 → r01)
_ROUTES_TAG="r$(echo "$ROUTES" | tr -d ',')"
DS="${DS:-$ART/dataset_urban_inland_${_ROUTES_TAG}_depth_scene_${STAMP}}"
OUT="${OUT:-$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}_expert_bc_h${HOST}}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ANN_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
MASTER="$ART/logs/mainchannel_fullfix_${STAMP}_h${HOST}.log"
BC_ITERS="${BC_ITERS:-20}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_BC="${SKIP_BC:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
APPEND="${APPEND:-0}"
MAX_DYAW="${MAX_DYAW:-0.35}"
MAX_DDYAW="${MAX_DDYAW:-0.12}"
# Match morning 0915 R1-success collect defaults (not human_like_commit).
REPLAN_PERIOD_S="${REPLAN_PERIOD_S:-2.0}"
W_JUMP="${W_JUMP:-0.05}"
EPISODES="${EPISODES:-$(echo "$ROUTES" | awk -F, '{print NF}')}"
mkdir -p "$ART/logs" "$OUT"
# nohup/setsid friendly: plain append (no process-sub tee)
exec >>"$MASTER" 2>&1

echo "=== mainchannel fullfix HOST=$HOST stamp=$STAMP ==="
hostname; date -Is
echo "DS=$DS OUT=$OUT ROUTES=$ROUTES MIN_KEPT=$MIN_KEPT/$MIN_KEPT_PER_ROUTE"
echo "APPEND=$APPEND SKIP_COLLECT=$SKIP_COLLECT SKIP_BC=$SKIP_BC"

pick_actor() {
  local d=$1
  if [[ -f "$d/v4_ac_best.pt" ]]; then echo "$d/v4_ac_best.pt"
  elif [[ -f "$d/v4_ac_latest.pt" ]]; then echo "$d/v4_ac_latest.pt"
  else return 1
  fi
}
INIT="${INIT:-}"
if [[ -z "$INIT" ]]; then
  INIT="$(pick_actor "$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear")"
fi
echo "INIT=$INIT"
[[ -f "$INIT" ]] || { echo FATAL no INIT; exit 1; }

if [[ -n "${WM:-}" && -f "$WM" ]]; then
  :
elif [[ -f "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt" ]]; then
  WM="$(cat "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt")"
elif [[ -f "$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt" ]]; then
  WM="$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt"
else
  echo FATAL need WM; exit 1
fi
echo "WM=$WM"

# Auto-skip phases from prior progress
if [[ "$SKIP_COLLECT" != "1" && -f "$DS/AUTO_REVIEW.json" ]]; then
  if "$PY" -c "import json,sys; s=json.load(open('$DS/AUTO_REVIEW.json')).get('summary') or {}; sys.exit(0 if s.get('gate_passed') else 1)"; then
    echo "AUTO SKIP_COLLECT: gate_passed in $DS"
    SKIP_COLLECT=1
  else
    APPEND=1
    echo "AUTO APPEND collect: gate not passed yet"
  fi
fi
if [[ "$SKIP_BC" != "1" ]] && pick_actor "$OUT" >/dev/null 2>&1; then
  if grep -q "BC_TRAIN_DONE\|MAINCHANNEL_FULLFIX_ALL_DONE" "$MASTER" 2>/dev/null; then
    echo "AUTO SKIP_BC: marker or prior DONE"
    SKIP_BC=1
  fi
fi

if [[ "$SKIP_COLLECT" != "1" ]]; then
  [[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor || true; sleep 10; }
  echo "=== PHASE1 collect DepthScene expert ==="
  append_args=()
  if [[ "$APPEND" == "1" ]]; then
    append_args+=(--append)
  fi
  "$PY" -m experiments.aerial.rl.collect_depth_scene_expert_dataset \
    --backend airsim --device cuda \
    --annotation "$ANN" \
    --routes "$ROUTES" \
    --out "$DS" \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 \
    --max-inflate 1.8 --require-arrived \
    --require-interior --min-spawn-inland-m 100 --min-path-inland-m 80 \
    --require-all-requested-routes \
    --min-kept "$MIN_KEPT" --min-kept-per-route "$MIN_KEPT_PER_ROUTE" --until-gate --max-rounds 80 \
    --shield-fwd-only --tti-coeff 2.5 \
    --r-m 100 --cruise-speed 10 --max-steps 600 \
    --d-clear 35 --d-danger 4 --max-dyaw "$MAX_DYAW" --max-ddyaw "$MAX_DDYAW" \
    --replan-period-s "$REPLAN_PERIOD_S" --w-jump "$W_JUMP" \
    --stuck-escape-after-s 6 --stuck-escape-hold-s 6 \
    --episodes "$EPISODES" \
    "${append_args[@]}"
  echo "COLLECT_DONE"
  "$PY" - <<PY
import json, sys
from pathlib import Path
ar = json.loads(Path("$DS/AUTO_REVIEW.json").read_text())
s = ar["summary"]
print("gate", s.get("gate_passed"), "keep_by_route", s.get("keep_by_route"), "required", s.get("required_routes"))
if not s.get("gate_passed"):
    sys.exit(3)
need = s.get("required_routes") or [int(x) for x in "$ROUTES".split(",") if x.strip()!=""]
kb = {int(k): int(v) for k, v in (s.get("keep_by_route") or {}).items()}
missing = [r for r in need if kb.get(int(r), 0) < int("$MIN_KEPT_PER_ROUTE")]
if missing:
    print("FATAL incomplete coverage", missing)
    sys.exit(4)
print("COLLECT_GATE_OK")
PY
else
  echo "SKIP_COLLECT ds=$DS"
fi

if [[ "$SKIP_BC" != "1" ]]; then
  [[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor || true; sleep 8; }
  "$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
    --src "$ANN" --out "$ANN_FOCUS" --routes "$ROUTES"
  echo "=== PHASE2 expert BC + short online open_loop ==="
  "$PY" -m experiments.aerial.rl.train_v4_ac \
    --config configs/aerial_rl_urban_complex_p2c.yaml \
    --config-overlay configs/aerial_rl_urban_complex_directional_oa.yaml \
    --backend airsim --dynamics torch \
    --wm-ckpt "$WM" --init-actor-ckpt "$INIT" \
    --dataset "$DS" \
    --planner --planner-horizon 15 --planner-rollout open_loop \
    --no-shield \
    --enable-bc --bc-batch 64 --bc-updates-per-iter 12 --bc-loss-scale 1.0 \
    --iters "$BC_ITERS" --episodes-per-iter 2 \
    --imagine-batch 16 --imagine-horizon 15 \
    --device cuda --annotation "$ANN_FOCUS" \
    --ckpt-dir "$OUT" --save-every-iter \
    --min-steps-for-best 20 --near-goal-frac 0.25 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 6 \
    --min-spawn-clear-m 5.0
  echo "BC_TRAIN_DONE"
fi

ACTOR="$(pick_actor "$OUT" || true)"
[[ -n "${ACTOR:-}" && -f "$ACTOR" ]] || { echo FATAL no actor after BC; exit 1; }
echo "ACTOR=$ACTOR"

if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "SKIP_EVAL"
  echo "MAINCHANNEL_FULLFIX_ALL_DONE"
  date -Is
  exit 0
fi

EVAL_JSON="$ART/eval_hbclear_diag_${STAMP}_eval_h${HOST}_expert_bc_wmbare_wm_bare/eval_all.json"
if [[ -f "$EVAL_JSON" ]] && grep -q "MAINCHANNEL_FULLFIX_ALL_DONE" "$MASTER" 2>/dev/null; then
  echo "SKIP_EVAL: already have $EVAL_JSON"
else
  echo "=== PHASE3 P0 eval wm_bare ==="
  HOST="$HOST" STAMP="${STAMP}_eval" TAG=expert_bc_wmbare PLANNER_MOCK=wm_bare \
    ACTOR="$ACTOR" WM="$WM" ANN="$ANN" \
    bash experiments/aerial/scripts/run_hbclear_p0_diag.sh
fi

echo "MAINCHANNEL_FULLFIX_ALL_DONE"
date -Is
