#!/usr/bin/env bash
# Cost-reflow prep @11 while Soft trains: pack_cone nearthick for obstacle_cost;
# optional short freeze-backbone train. Do NOT swap Soft INIT until Soft gate
# + cost gate both green.
#
#   HOST=11 STAMP=20260926_soft bash experiments/aerial/scripts/run_soft_cost_reflow_prep.sh
set -euo pipefail
HOST="${HOST:-11}"
STAMP="${STAMP:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
OUT="$ART/breakout_soft_${STAMP}"
LOG="$ART/logs/soft_cost_reflow_prep_${STAMP}_h${HOST}.log"
mkdir -p "$OUT" "$ART/logs"
exec >>"$LOG" 2>&1

PY="${PYTHON_BIN:-}"
[[ -z "$PY" && -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
[[ -z "$PY" ]] && PY=python3

echo "=== cost reflow prep HOST=$HOST stamp=$STAMP $(date -Is) ==="
DS="${DS:-$ART/dataset_geomfix_nearthick_20260926_hardnear}"
[[ -d "$DS" ]] || DS="$ART/dataset_v0_three_zone_near_20260823fg"
WM="${WM:-$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt}"
[[ -f "$WM" ]] || WM="$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt"
LABELS="$OUT/obstacle_cost_cone_prep.npz"
echo "DS=$DS WM=$WM → $LABELS"

"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels pack-cone \
  --dataset "$DS" --wm-ckpt "$WM" --out "$LABELS" \
  --device cuda --max-samples 400 --stride 2 --window 8 \
  2>&1 | tee "$OUT/cost_pack.log"

if [[ ! -f "$LABELS" ]]; then
  echo "FATAL pack-cone produced no labels at $LABELS"
  exit 2
fi
CKPT_OUT="$OUT/wm_obs_cost_reflow_prep.pt"
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels train \
  --labels "$LABELS" --wm-ckpt "$WM" --out-ckpt "$CKPT_OUT" \
  --device cuda --steps 800 \
  2>&1 | tee "$OUT/cost_train.log"
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels gate \
  --wm-ckpt "$CKPT_OUT" --labels "$LABELS" --device cuda \
  --report "$OUT/cost_gate.json" \
  2>&1 | tee "$OUT/cost_gate.log"
echo "COST_REFLOW_PREP_DONE $(date -Is)" | tee "$OUT/COST_PREP_DONE.txt"
echo "NOTE: Soft INIT unchanged until Soft READY_HARD + cost gate PASS"
