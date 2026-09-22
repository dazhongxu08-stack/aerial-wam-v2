#!/usr/bin/env bash
# Interior urban-complex: PathExpert collect (125) → micro-FT (H100) → interior eval hint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_4090.sh"
# shellcheck disable=SC1091
source "$ROOT/experiments/aerial/scripts/env_h100_from_125.sh" 2>/dev/null || {
  H100_USER="${H100_USER:-a25689}"
  H100_HOST="${H100_HOST:-10.239.121.23}"
  H100_PORT="${H100_PORT:-31126}"
  H100_SSH_KEY="${H100_SSH_KEY:-$HOME/.ssh/id_ed25519_h100}"
  H100_REPO="${H100_REPO:-/home/a25689/aerial-wam-v2}"
}

STAMP="${STAMP:-20260916_v2}"
CONFIG_REL="configs/aerial_rl_interior_complex.yaml"
DATASET_REL="${DATASET_REL:-experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}_curated}"
WM_REL="experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828"
INIT_REL="${INIT_REL:-experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt}"
CKPT_REL="experiments/aerial/rl/artifacts/v4_ac_ckpt_interior_complex_${STAMP}"
LOG_REL="artifacts/train_interior_complex_${STAMP}.log"
ITERS="${ITERS:-120}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"

ssh_h100() {
  ssh -i "$H100_SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new -p "$H100_PORT" \
    "${H100_USER}@${H100_HOST}" "$@"
}

tar_push_h100() {
  local rel="$1"
  tar czf - -C "$ROOT" "$rel" | ssh_h100 "mkdir -p ${H100_REPO}/$(dirname "$rel") && tar xzf - -C ${H100_REPO}"
}

say() { echo "[interior-complex] $*"; }

if [[ ! -f "$INIT_REL" ]]; then
  say "ERROR missing warm-start ckpt: $INIT_REL"
  exit 1
fi

if [[ ! -d "$ROOT/$DATASET_REL" ]] || [[ $(find "$ROOT/$DATASET_REL" -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l | tr -d ' ') -lt 4 ]]; then
  say "=== curate PathExpert dataset (urban-inland gate) ==="
  "$AERIAL_PY" experiments/aerial/scripts/review_urban_complex_dataset.py \
    --src "experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}" \
    --dst "$DATASET_REL" \
    --max-per-route "${REVIEW_MAX_PER_ROUTE:-3}" \
    --write-curated
fi

if [[ "$SKIP_COLLECT" != "1" ]]; then
  say "=== step 1: PathExpert teacher collect on urban-complex20 (125) ==="
  STAMP="$STAMP" OUT="$DATASET_REL" bash experiments/aerial/scripts/collect_urban_complex_path_expert.sh
fi

N_EP=$(find "$ROOT/$DATASET_REL" -maxdepth 1 -name 'episode_*.npz' 2>/dev/null | wc -l | tr -d ' ')
say "dataset episodes=$N_EP"
test "$N_EP" -ge 5

say "=== sync dataset -> H100 ==="
tar_push_h100 "$DATASET_REL"
tar_push_h100 "$(dirname "$INIT_REL")"
tar_push_h100 "$CONFIG_REL"
tar_push_h100 experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json
if ! ssh_h100 "test -f ${H100_REPO}/${WM_REL}/wm_step_3500.pt"; then
  tar_push_h100 "$WM_REL"
fi
for f in experiments/aerial/rl/train_v4_ac.py experiments/aerial/rl/train_rl.py \
  experiments/aerial/rl/corrector.py experiments/aerial/rl/collector.py \
  experiments/aerial/rl/dynamics_torch.py experiments/aerial/rl/actor_critic.py \
  experiments/aerial/rl/scene_profile.py experiments/aerial/eval/run_closed_loop.py; do
  tar_push_h100 "$f"
done

say "=== H100 micro-FT (warm-start Phase-2, iters=$ITERS) ==="
ssh_h100 "cd ${H100_REPO} && source experiments/aerial/scripts/env_h100.sh && \
  mkdir -p $(dirname ${CKPT_REL}) artifacts && \
  nohup \$AERIAL_PY -m experiments.aerial.rl.train_v4_ac \
    --config ${CONFIG_REL} \
    --iters ${ITERS} --episodes-per-iter 0 --skip-collect \
    --imagine-batch 8 --imagine-horizon 8 \
    --device cuda --dynamics torch --backend mock \
    --wm-ckpt ${WM_REL}/wm_step_3500.pt \
    --dataset ${DATASET_REL} \
    --annotation experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json \
    --init-actor-ckpt ${INIT_REL} \
    --w-collision 1.0 \
    --ckpt-dir ${CKPT_REL} \
    > ${LOG_REL} 2>&1 & echo TRAIN_PID=\$!"

say "log: ${H100_REPO}/${LOG_REL}"
say "ckpt: ${H100_REPO}/${CKPT_REL}/v4_ac_latest.pt"
say "post-train eval (125): ACTOR=${CKPT_REL}/v4_ac_latest.pt bash experiments/aerial/scripts/wam_phase2_v11_interior20_smoke.sh"
