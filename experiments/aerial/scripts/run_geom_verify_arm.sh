#!/usr/bin/env bash
# Per-host geometry falsification runner (offline GPU; no AirSim).
# Env: HOST STAMP ARM ROOT WM DS MAX_SAMPLES (optional extras)
set -euo pipefail
HOST="${HOST:?}"
STAMP="${STAMP:?}"
ARM="${ARM:?}"
ROOT="${ROOT:?}"
cd "$ROOT"
export PYTHONPATH="$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif [[ -x /data/venvs/sim_verify/bin/python ]]; then
  PY=/data/venvs/sim_verify/bin/python
elif [[ -x "$HOME/sim_verify/.venv/bin/python" ]]; then
  PY="$HOME/sim_verify/.venv/bin/python"
else
  PY="${PYTHON_BIN:-python3}"
fi
[[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]] && PY="$PYTHON_BIN"

ART="$ROOT/experiments/aerial/rl/artifacts"
OUT_DIR="$ART/geom_verify_${STAMP}"
LOG_DIR="$ART/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"
LOG="$LOG_DIR/geom_${ARM}_${STAMP}_h${HOST}.log"
DONE="$OUT_DIR/${ARM}_DONE.txt"
FAIL="$OUT_DIR/${ARM}_FAIL.txt"
WM="${WM:-$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt}"
MAX_SAMPLES="${MAX_SAMPLES:-400}"
CFG="${CFG:-configs/aerial_rl.yaml}"

exec > >(tee -a "$LOG") 2>&1
echo "=== geom $ARM host=$HOST stamp=$STAMP $(date -Is) ==="
echo "PY=$PY WM=$WM"
test -f "$WM" || { echo "FATAL missing WM $WM"; echo FAIL >"$FAIL"; exit 2; }

run_probe() {
  local ds=$1 out=$2 n=$3
  test -d "$ds" || { echo "FATAL missing dataset $ds"; return 2; }
  "$PY" experiments/aerial/scripts/wam_latent_depth_probe.py \
    --dataset "$ds" --wm-ckpt "$WM" --config "$CFG" --device cuda \
    --max-samples "$n" --stride 2 --window 8 \
    --max-center-depth-m 12.0 \
    --out "$out"
}

run_imagine() {
  local ds=$1 out=$2 n=$3
  test -d "$ds" || { echo "FATAL missing dataset $ds"; return 2; }
  "$PY" experiments/aerial/scripts/wam_imagine_coll_rank.py \
    --dataset "$ds" --wm-ckpt "$WM" --config "$CFG" --device cuda \
    --max-samples "$n" --stride 2 --horizon 15 \
    --encode-mode window --window 8 --max-center-depth-m 12.0 \
    --out "$out"
}

run_cone() {
  local ds=$1 out=$2 n=$3
  test -d "$ds" || { echo "FATAL missing dataset $ds"; return 2; }
  "$PY" experiments/aerial/scripts/wam_obstacle_cost_cone_rank.py \
    --dataset "$ds" --wm-ckpt "$WM" --config "$CFG" --device cuda \
    --max-samples "$n" --stride 2 --window 8 \
    --max-forward-depth-m 12.0 \
    --out "$out" || true
}

run_gate() {
  local labels=$1 report=$2
  test -f "$labels" || { echo "FATAL missing labels $labels"; return 2; }
  "$PY" -m experiments.aerial.rl.train_obstacle_cost_labels gate \
    --wm-ckpt "$WM" --labels "$labels" --report "$report" --device cuda || true
}

rc=0
case "$ARM" in
  G1)
    # 125: latent probe on merged (broad) + near_enrich
    DS1="${DS:-$ART/dataset_v0_p45_merged_20260821}"
    DS2="${DS2:-$ART/dataset_v0_p45_near_enrich_20260820}"
    run_probe "$DS1" "$OUT_DIR/G1_latent_probe_merged_n${MAX_SAMPLES}.json" "$MAX_SAMPLES" || rc=$?
    if [[ -d "$DS2" ]]; then
      run_probe "$DS2" "$OUT_DIR/G1_latent_probe_near_enrich_n${MAX_SAMPLES}.json" "$MAX_SAMPLES" || rc=$?
    fi
    ;;
  G2)
    # 84: obstacle-cost label gate (no episode DS required) + cone rank if DS synced
    LABELS="${LABELS:-$ART/obstacle_cost_gt_depth/labels_gatefix_20260922_161641.npz}"
    run_gate "$LABELS" "$OUT_DIR/G2_obstacle_cost_gate.json" || rc=$?
    # Also re-gate multi2 pack if present
    LABELS2="$ART/obstacle_cost_gt_depth/labels_multi2_gate.npz"
    if [[ -f "$LABELS2" ]]; then
      run_gate "$LABELS2" "$OUT_DIR/G2_obstacle_cost_gate_multi2.json" || true
    fi
    DS="${DS:-$ART/dataset_v0_three_zone_near_20260823fg}"
    if [[ -d "$DS" ]]; then
      run_cone "$DS" "$OUT_DIR/G2_cone_rank_n${MAX_SAMPLES}.json" "$MAX_SAMPLES" || true
    else
      echo "WARN no episode DS on 84 — cone rank skipped (gate-only)"
    fi
    ;;
  G3)
    # 14: latent probe on three_zone_near (hard near-field)
    DS="${DS:-$ART/dataset_v0_three_zone_near_20260823fg}"
    run_probe "$DS" "$OUT_DIR/G3_latent_probe_three_zone_near_n${MAX_SAMPLES}.json" "$MAX_SAMPLES" || rc=$?
    run_probe "$ART/dataset_v0_p45_merged_20260821" \
      "$OUT_DIR/G3_latent_probe_merged_n${MAX_SAMPLES}.json" "$MAX_SAMPLES" || rc=$?
    ;;
  G4)
    # 11: imagine coll rank + cone directional rank
    DS="${DS:-$ART/dataset_v0_three_zone_near_20260823fg}"
    run_imagine "$DS" "$OUT_DIR/G4_imagine_coll_rank_n${MAX_SAMPLES}.json" "$MAX_SAMPLES" || rc=$?
    run_cone "$DS" "$OUT_DIR/G4_cone_rank_n${MAX_SAMPLES}.json" "$MAX_SAMPLES" || true
    if [[ -d "$ART/dataset_v0_p45_merged_20260821" ]]; then
      run_imagine "$ART/dataset_v0_p45_merged_20260821" \
        "$OUT_DIR/G4_imagine_coll_rank_merged_n${MAX_SAMPLES}.json" "$MAX_SAMPLES" || true
    fi
    ;;
  *)
    echo "FATAL unknown ARM=$ARM"; echo FAIL >"$FAIL"; exit 2
    ;;
esac

if [[ $rc -eq 0 ]]; then
  echo "GEOM_${ARM}_DONE $(date -Is)" | tee "$DONE"
else
  echo "GEOM_${ARM}_FAIL rc=$rc $(date -Is)" | tee "$FAIL"
fi
echo "log=$LOG out=$OUT_DIR"
exit "$rc"
