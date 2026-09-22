#!/usr/bin/env bash
# Full interior pipeline: review → diagnostic eval → polyline FT → focus FT → weak collect → final eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
export PYTHONUNBUFFERED=1

STAMP="${STAMP:-20260916_night}"
LOG="${LOG:-artifacts/run_interior_full_pipeline_${STAMP}.log}"
RAW="experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}"
CURATED="experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}_curated"
POLY_CURATED="experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}_poly_curated"
FOCUS_CURATED="experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}_focus134_curated"
ANNO="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json"
ANNO_PATCH="experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json"
ACTOR_BASE="experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_${STAMP}/v4_ac_latest.pt"
ACTOR_POLY="experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_${STAMP}_poly/v4_ac_latest.pt"
ACTOR_FOCUS="experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_${STAMP}_focus134/v4_ac_latest.pt"
ITERS="${ITERS:-120}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[pipeline] $(date -Is) $*"; }

say "=== STEP 0: review all routes ==="
"$AERIAL_PY" experiments/aerial/scripts/review_all_interior_routes.py \
  --raw "$RAW" --curated "$CURATED"

say "=== STEP 1A: terminal-goal diagnostic eval (existing night FT) ==="
ACTOR="$ACTOR_BASE" OUT_ROOT="artifacts/wam_phase2_v11_interior_terminal_eval_${STAMP}" \
  bash experiments/aerial/scripts/wam_phase2_v11_interior_terminal_eval.sh || true

say "=== STEP 1B: polyline subgoal backfill + curated rebuild ==="
"$AERIAL_PY" -m experiments.aerial.rl.backfill_episode_goals \
  --dataset "$CURATED" --annotation "$ANNO" --max-match-m 5.0 || true
cp -a "$CURATED" "$POLY_CURATED"
"$AERIAL_PY" -m experiments.aerial.rl.backfill_polyline_subgoals \
  --dataset "$POLY_CURATED" --annotation "$ANNO"

say "=== STEP 2: polyline-aligned micro-FT ==="
nohup env STAMP="${STAMP}_poly" SKIP_COLLECT=1 \
  DATASET_REL="$POLY_CURATED" \
  CKPT_REL="experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_${STAMP}_poly" \
  LOG_REL="artifacts/train_interior_complex_${STAMP}_poly.log" \
  bash -c '
    source experiments/aerial/scripts/env_4090.sh
    "$AERIAL_PY" -m experiments.aerial.rl.train_v4_ac \
      --config configs/aerial_rl_interior_complex.yaml \
      --iters '"$ITERS"' --episodes-per-iter 0 --skip-collect \
      --imagine-batch 8 --imagine-horizon 8 \
      --device cuda --dynamics torch --backend mock \
      --wm-ckpt experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt \
      --dataset '"$POLY_CURATED"' \
      --annotation '"$ANNO"' \
      --init-actor-ckpt experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt \
      --w-collision 1.0 \
      --ckpt-dir experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_'"${STAMP}"'_poly \
      > artifacts/train_interior_complex_'"${STAMP}"'_poly.log 2>&1
  ' &
POLY_TRAIN_PID=$!
say "poly FT pid=$POLY_TRAIN_PID"

say "=== STEP 2b: focus curated (routes 1,3,4) micro-FT ==="
"$AERIAL_PY" experiments/aerial/scripts/build_focus_curated.py \
  --src "$CURATED" --dst "$FOCUS_CURATED" --routes "1,3,4"
nohup env STAMP="${STAMP}_focus134" SKIP_COLLECT=1 \
  bash -c '
    source experiments/aerial/scripts/env_4090.sh
    "$AERIAL_PY" -m experiments.aerial.rl.train_v4_ac \
      --config configs/aerial_rl_interior_complex.yaml \
      --iters '"$ITERS"' --episodes-per-iter 0 --skip-collect \
      --imagine-batch 8 --imagine-horizon 8 \
      --device cuda --dynamics torch --backend mock \
      --wm-ckpt experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt \
      --dataset '"$FOCUS_CURATED"' \
      --annotation '"$ANNO"' \
      --init-actor-ckpt experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt \
      --w-collision 1.0 \
      --ckpt-dir experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_'"${STAMP}"'_focus134 \
      > artifacts/train_interior_complex_'"${STAMP}"'_focus134.log 2>&1
  ' &
FOCUS_TRAIN_PID=$!
say "focus FT pid=$FOCUS_TRAIN_PID"

wait "$POLY_TRAIN_PID" "$FOCUS_TRAIN_PID" || true

say "=== STEP 3: patch weak spawns + collect ==="
"$AERIAL_PY" experiments/aerial/scripts/patch_weak_route_spawns.py
STAMP="${STAMP}_patch" OUT="${RAW}" APPEND=1 SKIP_MIN_OK_GATE=1 N=15 \
  MIN_SPAWN_Z=36 SPAWN_RETRY_M=4 SPAWN_Z_MAX_RETRIES=14 \
  ROUTE_INDICES="2,6,7,8,9,11" ANNO="$ANNO_PATCH" \
  bash experiments/aerial/scripts/collect_urban_complex_path_expert.sh || true
"$AERIAL_PY" experiments/aerial/scripts/review_urban_complex_dataset.py \
  --src "$RAW" --dst "$CURATED" --max-per-route 3 --write-curated

say "=== STEP 4: final SR eval (polyline + terminal) for all 3 actors ==="
for spec in \
  "night:$ACTOR_BASE" \
  "poly:$ACTOR_POLY" \
  "focus134:$ACTOR_FOCUS"; do
  name="${spec%%:*}"
  ck="${spec#*:}"
  if [[ ! -f "$ck" ]]; then
    say "skip eval $name — missing $ck"
    continue
  fi
  ACTOR="$ck" OUT_ROOT="artifacts/wam_phase2_v11_interior_ft_sr_eval_${STAMP}_${name}" \
    ROUTES="0,1,3,4,5,10,19" \
    bash experiments/aerial/scripts/wam_phase2_v11_interior_ft_sr_eval.sh || true
  ACTOR="$ck" OUT_ROOT="artifacts/wam_phase2_v11_interior_terminal_eval_${STAMP}_${name}" \
    ROUTES="0,1,3,4,5,10,19" \
    bash experiments/aerial/scripts/wam_phase2_v11_interior_terminal_eval.sh || true
done

say "=== STEP 5: route 4 failure demo video ==="
if [[ -f "$ACTOR_POLY" ]]; then
  "$AERIAL_PY" experiments/aerial/scripts/wam_vgoal_record_route_demo.py \
    --annotation "$ANNO" --route-idx 4 \
    --actor-ckpt "$ACTOR_POLY" \
    --out "artifacts/wam_interior_route04_demo_${STAMP}.mp4" \
    || say "demo video skipped (airsim busy)"
fi

say "=== PIPELINE DONE ==="
