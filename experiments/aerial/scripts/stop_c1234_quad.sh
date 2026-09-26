#!/usr/bin/env bash
# Stop C1–C4 + watch + A/B chain on 125/84/14/11. Run from Mac or 125.
#   SSH125=cursor-125-public bash experiments/aerial/scripts/stop_c1234_quad.sh
set -euo pipefail
SSH125="${SSH125:-cursor-125-public}"
ssh -o ConnectTimeout=30 -o BatchMode=yes "$SSH125" 'bash -s' <<'EOF'
set +e
KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"
ts(){ date -Is; }
echo "[$(ts)] stop_c1234_quad"

pkill -9 -f 'watch_c1234_oa_terminal' 2>/dev/null || true
pkill -9 -f 'chain_c1234_to_ab' 2>/dev/null || true
pkill -9 -f 'arm_chain_c_to_ab' 2>/dev/null || true
pkill -9 -f 'run_mainchannel_c1_ol_bc' 2>/dev/null || true
pkill -9 -f 'train_v4_ac.*c1_ol' 2>/dev/null || true
pkill -9 -f 'train_v4_ac' 2>/dev/null || true

ssh -n -o BatchMode=yes -o ConnectTimeout=20 ubantu@10.229.20.84 \
  'pkill -9 -f run_mainchannel_c2; pkill -9 -f train_v4_ac; pgrep -af train_v4_ac|grep -v pgrep||echo 84_CLEAR' || echo 84_SSH_FAIL

ssh -n -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" -p 31126 a25689@10.239.121.14 \
  'pkill -9 -f run_mainchannel_c3; pkill -9 -f train_v4_ac; pgrep -af train_v4_ac|grep -v pgrep||echo 14_CLEAR' || echo 14_SSH_FAIL

ssh -n -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" -p 30627 a25689@10.239.121.11 \
  'pkill -9 -f run_mainchannel_c4; pkill -9 -f train_v4_ac; pgrep -af train_v4_ac|grep -v pgrep||echo 11_CLEAR' || echo 11_SSH_FAIL

sleep 1
pgrep -af 'watch_c1234|chain_c1234|train_v4_ac|run_mainchannel_c' | grep -v pgrep || echo 125_CLEAR
ART=/home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts
mkdir -p "$ART/logs"
echo "C1234_KILLED_FOR_GEOM_VERIFY $(date -Is)" > "$ART/logs/c1234_KILLED_FOR_GEOM.txt"
echo STOP_C1234_QUAD_DONE
EOF
