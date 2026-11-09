#!/usr/bin/env bash
# Deploy Soft→Hard breakout on 4×4090 after READY_SOFT.
# Roles (contract): Soft@84 · Soft-gate watch@14 (NO Hard yet) ·
# ceiling distill short@125 · cost-reflow prep@11.
#
#   STAMP=20260926_soft SSH125=cursor-125-public \
#     bash experiments/aerial/scripts/deploy_soft_breakout_quad.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SSH125="${SSH125:-cursor-125-public}"
ITERS_SOFT="${ITERS_SOFT:-120}"
ITERS_CEIL="${ITERS_CEIL:-40}"
cd "$ROOT"

# Require READY_SOFT locally or on 125
if [[ ! -f "$ROOT/artifacts/geomfix_20260926_hardnear/READY_SOFT.txt" ]]; then
  echo "WARN local READY_SOFT missing — will check on 125"
fi

PAYLOAD=(
  configs/aerial_rl_urban_complex_soft_noshield.yaml
  configs/aerial_rl_urban_complex_directional_oa.yaml
  configs/aerial_rl_urban_complex_p2c.yaml
  configs/aerial_rl_urban_complex_antistick_reward.yaml
  configs/aerial_rl_urban_complex_c2_wm_escape.yaml
  experiments/aerial/rl/reward.py
  experiments/aerial/rl/train_rl.py
  experiments/aerial/scripts/run_mainchannel_soft_noshield.sh
  experiments/aerial/scripts/run_soft_positive_gate_watch.sh
  experiments/aerial/scripts/run_soft_cost_reflow_prep.sh
  experiments/aerial/scripts/run_mainchannel_c1_ol_bc_noshield.sh
  experiments/aerial/scripts/soft_breakout_launch_on_125.sh
)

echo "=== pack Soft breakout → $SSH125 stamp=$STAMP ==="
PACK=/tmp/soft_breakout_${STAMP}.tgz
tar -czf "$PACK" "${PAYLOAD[@]}"
scp -o ConnectTimeout=30 -o BatchMode=yes "$PACK" "$SSH125:/tmp/soft_breakout.tgz"
scp -o ConnectTimeout=30 -o BatchMode=yes \
  experiments/aerial/scripts/soft_breakout_launch_on_125.sh \
  "$SSH125:/tmp/soft_breakout_launch_on_125.sh"

ssh -o ConnectTimeout=40 -o BatchMode=yes "$SSH125" \
  "chmod +x /tmp/soft_breakout_launch_on_125.sh;
   export STAMP='$STAMP' ITERS_SOFT='$ITERS_SOFT' ITERS_CEIL='$ITERS_CEIL';
   bash /tmp/soft_breakout_launch_on_125.sh"

mkdir -p "$ROOT/artifacts"
echo "$STAMP" > "$ROOT/artifacts/soft_breakout_STAMP.txt"
cat > "$ROOT/artifacts/soft_breakout_STATUS.md" <<EOF
# Soft→Hard breakout · stamp \`$STAMP\`

| 机 | 角色 | 状态 |
|----|------|------|
| **84** | Soft L-main (\`--no-shield\`, no terminate-on-coll) | launched |
| **14** | Soft positive gate watch (K=8) — **Hard 未开** | launched |
| **125** | 天花板 OL-BC 短训 (iters≤$ITERS_CEIL) | launched |
| **11** | 代价回流 prep (nearthick pack_cone) | launched |

合同：Hard@14 仅在 \`breakout_soft_\${STAMP}/READY_HARD.txt\` 出现后。
INIT floor=022358_best_sr2 · WM=obstacle_cost_gatefix · Soft 不吃 C1–C4 中途 ckpt。
EOF
echo "MAC_DEPLOY_SOFT_DONE stamp=$STAMP"
