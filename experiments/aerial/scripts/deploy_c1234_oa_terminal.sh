#!/usr/bin/env bash
# Deploy+launch C1–C4 OA+terminal overnight on 125/84/14/11.
#   STAMP=20260925_2100 SSH125=cursor-125-public bash experiments/aerial/scripts/deploy_c1234_oa_terminal.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SSH125="${SSH125:-cursor-125-public}"
ITERS="${ITERS:-100}"
cd "$ROOT"

PAYLOAD=(
  configs/aerial_rl_urban_complex_c2_wm_escape.yaml
  configs/aerial_rl_urban_complex_c4_terminal.yaml
  configs/aerial_rl_urban_complex_antistick_reward.yaml
  configs/aerial_rl_urban_complex_directional_oa.yaml
  experiments/aerial/scripts/run_mainchannel_c1_ol_bc_noshield.sh
  experiments/aerial/scripts/run_mainchannel_c2_wm_escape_noshield.sh
  experiments/aerial/scripts/run_mainchannel_c3_antistick_noshield.sh
  experiments/aerial/scripts/run_mainchannel_c4_terminal_noshield.sh
  experiments/aerial/scripts/watch_c1234_oa_terminal.sh
  experiments/aerial/scripts/c1234_launch_on_125.sh
  docs/superpowers/plans/2026-09-25-overnight-oa-terminal.md
)

echo "=== rsync payloads → $SSH125 stamp=$STAMP iters=$ITERS ==="
ssh -o ConnectTimeout=25 -o BatchMode=yes "$SSH125" 'mkdir -p /tmp/c1234_deploy'
rsync -avz "${PAYLOAD[@]}" "$SSH125:/tmp/c1234_deploy/"
ssh -n -o ConnectTimeout=25 -o BatchMode=yes "$SSH125" \
  "chmod +x /tmp/c1234_deploy/c1234_launch_on_125.sh && export STAMP='$STAMP' ITERS='$ITERS' && bash /tmp/c1234_deploy/c1234_launch_on_125.sh"

echo "MAC_DEPLOY_DONE stamp=$STAMP"
mkdir -p "$ROOT/artifacts"
echo "$STAMP" > "$ROOT/artifacts/c1234_oa_terminal_STAMP.txt"
cat > "$ROOT/artifacts/c1234_oa_terminal_STATUS.md" <<EOF
# C1–C4 overnight · stamp \`$STAMP\`

| 臂 | 机 | 方案 | 状态 |
|----|----|------|------|
| C1 | 125 | ol_bc | launched |
| C2 | 84 | wm_escape | launched |
| C3 | 14 | antistick | launched |
| C4 | 11 | terminal | launched |

Gate: routes 0,1,3,4 toward_g **noshield** · pass arrived≥1 ∧ P0≤1 · INIT floor 022358.
ITERS=$ITERS. Watch on 125 auto-restarts dead arms.
EOF
