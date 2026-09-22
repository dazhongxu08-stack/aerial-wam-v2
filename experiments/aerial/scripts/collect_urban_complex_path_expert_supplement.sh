#!/usr/bin/env bash
# Append PathExpert teacher samples — same stack as main v2 collect (z=24, inland gate).
# Only R09 urban-grid routes (y>=11 corridor); excludes south waterfront variants (y<0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-20260916_v2}"
OUT="${OUT:-experiments/aerial/rl/artifacts/dataset_urban_complex_path_expert_${STAMP}}"
LOG="${LOG:-artifacts/collect_urban_complex_path_expert_${STAMP}_supplement.log}"

# route 6 skipped in v2; others short-collision on urban grid (no y<0 waterfront).
ROUTE_INDICES="${ROUTE_INDICES:-6,1,2,7,8,10,11}"

export STAMP OUT LOG ROUTE_INDICES
export APPEND=1 SKIP_MIN_OK_GATE=1 N=20
# Inherit main collect defaults: MIN_SPAWN_Z=24, SPAWN_RETRY_M=5, SPAWN_Z_MAX_RETRIES=4

exec bash experiments/aerial/scripts/collect_urban_complex_path_expert.sh
