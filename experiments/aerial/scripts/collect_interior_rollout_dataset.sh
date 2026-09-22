#!/usr/bin/env bash
# Collect V11 polyline rollouts on interior20 for micro-FT (RGB + executed actions).
# Keeps partial episodes — scene/polyline validated even when policy does not arrive.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/wam_phase2_v11_mainline.inc.sh"

STAMP="${STAMP:-20260916}"
ANNO="${ANNO:-experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json}"
OUT="${OUT:-experiments/aerial/rl/artifacts/dataset_interior_rollout_v11_${STAMP}}"
LOG="${LOG:-artifacts/collect_interior_rollout_v11_${STAMP}.log}"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
ACTOR="${ACTOR:-experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt}"

mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "[interior-rollout] $(date -Is) V11 save-dataset -> ${OUT}"

if [[ -x "$RECOVER" ]]; then
  bash "$RECOVER" || true
  sleep 8
fi

set +e
"$AERIAL_PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
  "${WAM_PHASE2_CKPTS[@]}" \
  --annotation "$ANNO" \
  --actor-ckpt "$ACTOR" \
  --episodes 20 \
  --save-dataset "$OUT" \
  --traj-out "${OUT}/traj" \
  --out "${OUT}/eval_all.json" \
  "${WAM_PHASE2_V11_STACK[@]}"
rc=$?
set -e

n_npz=$(find "$OUT" -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "[interior-rollout] done exit=$rc npz=$n_npz out=$OUT"
test "$n_npz" -ge 10
