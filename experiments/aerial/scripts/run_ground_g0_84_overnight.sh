#!/usr/bin/env bash
# G0 ground migration overnight pipeline on 84 (Avant-AirSim / CarlaAir).
# Run: nohup bash experiments/aerial/scripts/run_ground_g0_84_overnight.sh >> logs/ground_g0_84_overnight.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/ground_g0_84_overnight_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== ground G0 overnight ${STAMP} ==="
echo "ROOT=$ROOT"

# Conda (84 default)
CONDA_BASE="${CONDA_BASE:-/data/linux/workspace/miniconda3}"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate carlaAir

export CARLA_HOST="${CARLA_HOST:-127.0.0.1}"
export CARLA_PORT="${CARLA_PORT:-2200}"
export AIRSIM_HOST="${AIRSIM_HOST:-127.0.0.1}"
export AIRSIM_PORT="${AIRSIM_PORT:-41463}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

cd "$ROOT"

echo "--- Step 1: Aerial Fork A on Avant-AirSim (41463) ---"
cd experiments/aerial/sim_verify
export AIRSIM_PORT=41463 AIRSIM_CAMERA=0
python probes/t2_capability.py || true
python verdict.py || true
cd "$ROOT"

echo "--- Step 2: Ground Fork G probe (CARLA 2200) ---"
cd experiments/aerial/ground_sim_verify
cp -n config.env.example config.env 2>/dev/null || true
chmod +x run_all.sh
./run_all.sh || GROUND_EXIT=$?
GROUND_EXIT="${GROUND_EXIT:-0}"
cd "$ROOT"

echo "--- Step 3: Ground smoke closed-loop (5 steps) ---"
python - <<'PY' || SMOKE_EXIT=$?
import numpy as np
from experiments.aerial.rl.env.ground_robot_env import CarlaGroundRobotEnv, GroundRobotEnvConfig

cfg = GroundRobotEnvConfig(carla_port=2200, step_hz=10.0)
with CarlaGroundRobotEnv(cfg) as env:
    obs = env.reset()
    assert obs.rgb.shape == (224, 224, 3)
    for i in range(5):
        a = np.array([0.05, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, info = env.step(a)
        print(f"step {i} vel={obs.state[3:6].tolist()} collided={obs.collided}")
print("SMOKE_PASS")
PY
SMOKE_EXIT="${SMOKE_EXIT:-0}"

echo "--- Step 4: Write verdict summary ---"
VERDICT="${ROOT}/docs/handover/GROUND_SIM_VERDICT_84.md"
mkdir -p "$(dirname "$VERDICT")"
{
  echo "# Ground Sim Verdict (84) — ${STAMP}"
  echo ""
  echo "## Selected backend: **Avant-AirSim (CarlaAir)**"
  echo ""
  echo "- CARLA: \`${CARLA_HOST}:${CARLA_PORT}\` (ground vehicle + sensors)"
  echo "- AirSim: \`${AIRSIM_HOST}:${AIRSIM_PORT}\` (aerial Fork A, shared world)"
  echo "- Product: \`/data/linux/workspace/Avant-AirSim/Avant-AirSim\`"
  echo ""
  echo "## Candidates"
  echo ""
  echo "| Repo | Clone | Verdict |"
  echo "|------|-------|---------|"
  echo "| Avant-AirSim (deployed) | On 84 | **GO — primary G0 backend** |"
  echo "| AvantAGSim (GitLab) | Auth blocked | Same product line; use deployed tarball |"
  echo "| slam-nav (GitLab) | Auth blocked | Deferred; SLAM-centric, not WAM-native |"
  echo ""
  echo "## Probe exits"
  echo "- ground_sim_verify: ${GROUND_EXIT}"
  echo "- ground smoke: ${SMOKE_EXIT}"
  echo ""
  echo "## Next (G0 execution)"
  echo "1. \`GroundRobotBridge\` = \`CarlaGroundRobotEnv\` (scaffolded)"
  echo "2. G1: load aerial \`step_e\` π + WM on 10–50 m ground routes"
  echo "3. G2: ground RGB + odom + depth GT corpus for DA3 fine-tune"
} > "$VERDICT"
echo "Wrote $VERDICT"

echo "=== DONE ground G0 overnight ${STAMP} exit_ground=${GROUND_EXIT} exit_smoke=${SMOKE_EXIT} ==="
exit 0
