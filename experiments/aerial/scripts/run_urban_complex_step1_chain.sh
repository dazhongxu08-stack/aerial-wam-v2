#!/usr/bin/env bash
# Step1 baseline20 → auto Step2 route1 stage2 train (single AirSim queue, no idle gap).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

BASE_STAMP="${STAMP:-$(date +%Y%m%d)}"
LOG="artifacts/urban_complex_step1_chain_${BASE_STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "[chain] $(date -Is) === Step1: baseline20 z38 frozen toward_g ==="
STAMP="${BASE_STAMP}_z38base" bash experiments/aerial/scripts/eval_urban_complex_baseline20.sh

TRAIN_STAMP="${BASE_STAMP}_r1s2"
echo "[chain] $(date -Is) === Step2: route1 stage2 train (64 iter, near_frac=0.7, z38) ==="
STAMP="$TRAIN_STAMP" \
  FOCUS_ROUTES=1 \
  ITERS=64 \
  NEAR_FRAC=0.7 \
  NEAR_MIN=5.0 \
  NEAR_MAX=15.0 \
  MIN_SPAWN_Z=38.0 \
  SPAWN_RETRY_M=14.0 \
  SPAWN_Z_MAX_RETRIES=3 \
  RESUME_CKPT=experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_20260917_z24/v4_ac_latest.pt \
  bash experiments/aerial/scripts/train_urban_complex_p2c.sh

echo "[chain] $(date -Is) === Step2b: route1 gate eval ==="
STAMP="${BASE_STAMP}_r1gate" \
  ACTOR="experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_${TRAIN_STAMP}/v4_ac_latest.pt" \
  ROUTES=1 \
  MIN_SPAWN_Z=38.0 \
  SPAWN_RETRY_M=14.0 \
  SPAWN_Z_MAX_RETRIES=3 \
  bash experiments/aerial/scripts/eval_urban_complex_toward_g_gate.sh

echo "[chain] $(date -Is) PIPELINE DONE"
