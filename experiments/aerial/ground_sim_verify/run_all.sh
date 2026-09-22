#!/usr/bin/env bash
# Ground sim GO/NO-GO — run on 84 against Avant-AirSim CARLA.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
[[ -f config.env ]] && source config.env || source config.env.example
mkdir -p artifacts
echo "== ground_sim_verify host=$CARLA_HOST:$CARLA_PORT =="
python probes/t0_connectivity.py
python probes/t1_carla_ground.py
python verdict.py
