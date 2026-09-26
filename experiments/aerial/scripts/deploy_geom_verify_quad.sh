#!/usr/bin/env bash
# Deploy geometry falsification to 125 and launch G1–G4 on 125/84/14/11.
#   STAMP=20260925_2300 SSH125=cursor-125-public \
#     bash experiments/aerial/scripts/deploy_geom_verify_quad.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SSH125="${SSH125:-cursor-125-public}"
MAX_SAMPLES="${MAX_SAMPLES:-400}"
cd "$ROOT"

PAYLOAD=(
  experiments/aerial/scripts/wam_latent_depth_probe.py
  experiments/aerial/scripts/wam_imagine_coll_rank.py
  experiments/aerial/scripts/wam_obstacle_cost_cone_rank.py
  experiments/aerial/scripts/run_geom_verify_arm.sh
  experiments/aerial/scripts/geom_verify_launch_on_125.sh
  experiments/aerial/scripts/summarize_geom_verify.py
  experiments/aerial/scripts/stop_c1234_quad.sh
  experiments/aerial/rl/train_obstacle_cost_labels.py
  docs/superpowers/plans/2026-09-25-breakout-soft-hard-quad.md
)

echo "=== stop C1234 first via $SSH125 ==="
bash experiments/aerial/scripts/stop_c1234_quad.sh || true

echo "=== rsync geom payloads → $SSH125 stamp=$STAMP ==="
ssh -o ConnectTimeout=40 -o BatchMode=yes "$SSH125" 'mkdir -p /tmp/geom_deploy'
rsync -avz "${PAYLOAD[@]}" "$SSH125:/tmp/geom_deploy/"
ssh -n -o ConnectTimeout=40 -o BatchMode=yes "$SSH125" \
  "chmod +x /tmp/geom_deploy/geom_verify_launch_on_125.sh /tmp/geom_deploy/run_geom_verify_arm.sh && \
   export STAMP='$STAMP' MAX_SAMPLES='$MAX_SAMPLES' && \
   bash /tmp/geom_deploy/geom_verify_launch_on_125.sh"

mkdir -p "$ROOT/artifacts"
echo "$STAMP" > "$ROOT/artifacts/geom_verify_STAMP.txt"
cat > "$ROOT/artifacts/geom_verify_STATUS.md" <<EOF
# Geometry falsification · stamp \`$STAMP\`

| 臂 | 机 | 任务 |
|----|----|------|
| G1 | 125 | latent depth probe (merged + near_enrich) n=$MAX_SAMPLES |
| G2 | 84 | obstacle_cost three-sort gate + cone rank |
| G3 | 14 | latent probe (three_zone_near + merged) |
| G4 | 11 | imagine coll rank + cone directional rank |

WM: \`wm_ckpt_obstacle_cost_gatefix_20260922_161641\`
Out: \`experiments/aerial/rl/artifacts/geom_verify_${STAMP}/SUMMARY.md\`
EOF
echo "MAC_DEPLOY_DONE stamp=$STAMP"
