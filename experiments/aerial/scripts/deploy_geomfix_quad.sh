#!/usr/bin/env bash
# Deploy geomfix Phase-0 overnight (cost reflow + near WM) on 125/84/14/11.
# Mainchannel: noshield. Soft→Hard only after READY_SOFT.
#
#   STAMP=20260925_geomfix SSH125=cursor-125-public \
#     bash experiments/aerial/scripts/deploy_geomfix_quad.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SSH125="${SSH125:-cursor-125-public}"
cd "$ROOT"

# stop C chain if still around
bash experiments/aerial/scripts/stop_c1234_quad.sh || true

PAYLOAD=(
  experiments/aerial/scripts/run_geomfix_r1_cost_reflow.sh
  experiments/aerial/scripts/run_geomfix_r2_collect_84.sh
  experiments/aerial/scripts/run_geomfix_r3_wm_near.sh
  experiments/aerial/scripts/run_geomfix_r4_wm_probe.sh
  experiments/aerial/scripts/geomfix_launch_on_125.sh
  experiments/aerial/scripts/watch_geomfix_quad.sh
  experiments/aerial/scripts/wam_latent_depth_probe.py
  experiments/aerial/scripts/wam_imagine_coll_rank.py
  experiments/aerial/scripts/wam_obstacle_cost_cone_rank.py
  experiments/aerial/scripts/stop_c1234_quad.sh
  experiments/aerial/rl/train_obstacle_cost_labels.py
  configs/aerial_rl_geomfix_near_wm.yaml
  docs/superpowers/plans/2026-09-25-breakout-soft-hard-quad.md
)

echo "=== deploy geomfix → $SSH125 stamp=$STAMP ==="
ssh -o ConnectTimeout=45 -o BatchMode=yes "$SSH125" 'mkdir -p /tmp/geomfix_deploy'
rsync -avz "${PAYLOAD[@]}" "$SSH125:/tmp/geomfix_deploy/"
ssh -n -o ConnectTimeout=45 -o BatchMode=yes "$SSH125" \
  "chmod +x /tmp/geomfix_deploy/geomfix_launch_on_125.sh /tmp/geomfix_deploy/run_geomfix_r*.sh /tmp/geomfix_deploy/watch_geomfix_quad.sh && \
   export STAMP='$STAMP' && bash /tmp/geomfix_deploy/geomfix_launch_on_125.sh"

mkdir -p "$ROOT/artifacts"
echo "$STAMP" > "$ROOT/artifacts/geomfix_STAMP.txt"
cat > "$ROOT/artifacts/geomfix_STATUS.md" <<EOF
# Geomfix Phase-0 · stamp \`$STAMP\`

| 臂 | 机 | 任务 | 罩 |
|----|----|------|----|
| R1 | 125 | left_near pack+reflow+cone (+AirSim top-up) | 关 |
| R2 | 84 | 并行 GT left_near 采集 | 关（采集） |
| R3 | 14 | 近场 WM FT + B′-1 | n/a |
| R4 | 11 | 更强 hinge WM + imagine/cone | n/a |

出门：\`experiments/aerial/rl/artifacts/geomfix_${STAMP}/PHASE0_VERDICT.md\`  
\`READY_SOFT.txt\` 出现后才允许 Soft→Hard（仍关罩）。
EOF
echo "MAC_DEPLOY_DONE stamp=$STAMP"
