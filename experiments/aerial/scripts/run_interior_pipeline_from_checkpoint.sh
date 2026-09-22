#!/usr/bin/env bash
# Idempotent interior pipeline — skips completed steps, safe to re-run after interrupt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
export PYTHONUNBUFFERED=1

STAMP="${STAMP:-20260916_night}"
LOG="${LOG:-artifacts/run_interior_full_pipeline_${STAMP}.log}"
STATE="${STATE:-artifacts/pipeline_checkpoint_${STAMP}.json}"
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
LOCK="${LOCK:-/tmp/interior_pipeline_${STAMP}.lock}"

mkdir -p "$(dirname "$LOG")" "$(dirname "$STATE")"
exec >>"$LOG" 2>&1

say() { echo "[pipeline-ckpt] $(date -Is) $*"; }

acquire_lock() {
  if [[ -f "$LOCK" ]]; then
    local old
    old="$(cat "$LOCK" 2>/dev/null || true)"
    if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
      say "another pipeline holder pid=$old — exit"
      exit 0
    fi
  fi
  echo "$$" >"$LOCK"
}

release_lock() { rm -f "$LOCK"; }
trap release_lock EXIT
acquire_lock

mark() {
  local step="$1"
  "$AERIAL_PY" - <<'PY' "$STATE" "$step"
import json, sys, datetime
path, step = sys.argv[1], sys.argv[2]
try:
    st = json.load(open(path))
except FileNotFoundError:
    st = {"done": []}
done = set(st.get("done", []))
done.add(step)
st["done"] = sorted(done)
st["updated"] = datetime.datetime.now().astimezone().isoformat()
json.dump(st, open(path, "w"), indent=2)
print(f"[pipeline-ckpt] marked {step}")
PY
}

is_done() {
  local step="$1"
  "$AERIAL_PY" - <<'PY' "$STATE" "$step"
import json, sys
path, step = sys.argv[1], sys.argv[2]
try:
    done = set(json.load(open(path)).get("done", []))
except FileNotFoundError:
    done = set()
raise SystemExit(0 if step in done else 1)
PY
}

poly_backfill_ok() {
  [[ -d "$POLY_CURATED" ]] && [[ "$(find "$POLY_CURATED" -maxdepth 1 -name 'episode_*.npz' | wc -l)" -ge 7 ]] && \
    "$AERIAL_PY" - <<'PY' "$ROOT/$POLY_CURATED"
import glob, numpy as np, sys
from pathlib import Path
p = Path(sys.argv[1])
files = sorted(p.glob("episode_*.npz"))
if not files:
    raise SystemExit(1)
d = np.load(files[0])
raise SystemExit(0 if "goals" in d.files else 1)
PY
}

say "START checkpoint pipeline stamp=$STAMP"

if ! is_done "step0_review"; then
  say "=== STEP 0: review ==="
  "$AERIAL_PY" experiments/aerial/scripts/review_all_interior_routes.py --raw "$RAW" --curated "$CURATED"
  mark step0_review
fi

if ! is_done "step1a_terminal"; then
  say "=== STEP 1A: terminal eval ==="
  ACTOR="$ACTOR_BASE" OUT_ROOT="artifacts/wam_phase2_v11_interior_terminal_eval_${STAMP}" \
    bash experiments/aerial/scripts/wam_phase2_v11_interior_terminal_eval.sh || true
  mark step1a_terminal
fi

if ! is_done "step1b_poly_backfill"; then
  say "=== STEP 1B: polyline backfill ==="
  if [[ ! -d "$POLY_CURATED" ]]; then
    cp -a "$CURATED" "$POLY_CURATED"
  fi
  if ! poly_backfill_ok; then
    "$AERIAL_PY" -m experiments.aerial.rl.backfill_polyline_subgoals \
      --dataset "$POLY_CURATED" --annotation "$ANNO"
  fi
  mark step1b_poly_backfill
fi

if ! is_done "step2_poly_ft"; then
  say "=== STEP 2: poly FT ==="
  if [[ ! -f "$ACTOR_POLY" ]]; then
    "$AERIAL_PY" -m experiments.aerial.rl.train_v4_ac \
      --config configs/aerial_rl_interior_complex.yaml \
      --iters "$ITERS" --episodes-per-iter 0 --skip-collect \
      --imagine-batch 8 --imagine-horizon 8 \
      --device cuda --dynamics torch --backend mock \
      --wm-ckpt experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt \
      --dataset "$POLY_CURATED" --annotation "$ANNO" \
      --init-actor-ckpt experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt \
      --w-collision 1.0 \
      --ckpt-dir "experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_${STAMP}_poly" \
      >>"artifacts/train_interior_complex_${STAMP}_poly.log" 2>&1
  fi
  mark step2_poly_ft
fi

if ! is_done "step2b_focus_ft"; then
  say "=== STEP 2b: focus FT ==="
  "$AERIAL_PY" experiments/aerial/scripts/build_focus_curated.py \
    --src "$CURATED" --dst "$FOCUS_CURATED" --routes "1,3,4"
  if [[ ! -f "$ACTOR_FOCUS" ]]; then
    "$AERIAL_PY" -m experiments.aerial.rl.train_v4_ac \
      --config configs/aerial_rl_interior_complex.yaml \
      --iters "$ITERS" --episodes-per-iter 0 --skip-collect \
      --imagine-batch 8 --imagine-horizon 8 \
      --device cuda --dynamics torch --backend mock \
      --wm-ckpt experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt \
      --dataset "$FOCUS_CURATED" --annotation "$ANNO" \
      --init-actor-ckpt experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt \
      --w-collision 1.0 \
      --ckpt-dir "experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_${STAMP}_focus134" \
      >>"artifacts/train_interior_complex_${STAMP}_focus134.log" 2>&1
  fi
  mark step2b_focus_ft
fi

if ! is_done "step3_weak_collect"; then
  say "=== STEP 3: weak spawn patch + collect ==="
  "$AERIAL_PY" experiments/aerial/scripts/patch_weak_route_spawns.py || true
  OUT="$RAW" APPEND=1 SKIP_MIN_OK_GATE=1 N=15 \
    MIN_SPAWN_Z=36 SPAWN_RETRY_M=4 SPAWN_Z_MAX_RETRIES=14 \
    ROUTE_INDICES="2,6,7,8,9,11" ANNO="$ANNO_PATCH" \
    bash experiments/aerial/scripts/collect_urban_complex_path_expert.sh || true
  "$AERIAL_PY" experiments/aerial/scripts/review_urban_complex_dataset.py \
    --src "$RAW" --dst "$CURATED" --max-per-route 3 --write-curated || true
  mark step3_weak_collect
fi

if ! is_done "step4_evals"; then
  say "=== STEP 4: final evals ==="
  for spec in "night:$ACTOR_BASE" "poly:$ACTOR_POLY" "focus134:$ACTOR_FOCUS"; do
    name="${spec%%:*}"
    ck="${spec#*:}"
    [[ -f "$ck" ]] || { say "skip eval $name missing ckpt"; continue; }
    poly_out="artifacts/wam_phase2_v11_interior_ft_sr_eval_${STAMP}_${name}/eval_all.json"
    term_out="artifacts/wam_phase2_v11_interior_terminal_eval_${STAMP}_${name}/eval_all.json"
    if [[ ! -f "$poly_out" ]]; then
      ACTOR="$ck" OUT_ROOT="artifacts/wam_phase2_v11_interior_ft_sr_eval_${STAMP}_${name}" \
        ROUTES="0,1,3,4,5,10,19" \
        bash experiments/aerial/scripts/wam_phase2_v11_interior_ft_sr_eval.sh || true
    fi
    if [[ ! -f "$term_out" ]]; then
      ACTOR="$ck" OUT_ROOT="artifacts/wam_phase2_v11_interior_terminal_eval_${STAMP}_${name}" \
        ROUTES="0,1,3,4,5,10,19" \
        bash experiments/aerial/scripts/wam_phase2_v11_interior_terminal_eval.sh || true
    fi
  done
  mark step4_evals
fi

if ! is_done "step5_demo"; then
  say "=== STEP 5: route 4 demo ==="
  if [[ -f "$ACTOR_POLY" ]]; then
    "$AERIAL_PY" experiments/aerial/scripts/wam_vgoal_record_route_demo.py \
      --annotation "$ANNO" --route-idx 4 --actor-ckpt "$ACTOR_FOCUS" \
      --out-mp4 "artifacts/wam_interior_route04_demo_${STAMP}.mp4" || true
  fi
  mark step5_demo
fi

say "=== PIPELINE CHECKPOINT DONE ==="
