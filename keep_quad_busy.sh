#!/usr/bin/env bash
# Stop idle Hard gates and put 84 + 125 onto Hard training.
# 14 and 11 keep the trains they already have. Then arm the monitor to
# restart any of the four if a train dies.
set +e
R=/home/yao/aerial-wam-v2
cd "$R"
STAMP=20260926_soft
KEY=${H100_KEY:-$HOME/.ssh/id_ed25519_h100}
[[ -f "$KEY" ]] || KEY=$HOME/.ssh/id_ed25519
chmod +x experiments/aerial/scripts/ensure_hard_quad.sh \
  experiments/aerial/scripts/run_mainchannel_hard_noshield.sh \
  experiments/aerial/scripts/soft_breakout_monitor_125.sh

echo '=== stop idle gates ==='
pkill -f '[r]un_hard_noshield_gate' || true
pkill -f '[w]am_phase2_long_eval' || true
ssh -n -o BatchMode=yes -o ConnectTimeout=15 ubantu@10.229.20.84 \
  "pkill -f '[r]un_hard_noshield_gate' || true; pkill -f '[w]am_phase2_long_eval' || true; echo GATE84_STOPPED"

echo '=== install hard trainer on 84 ==='
scp -o BatchMode=yes -o ConnectTimeout=20 \
  configs/aerial_rl_urban_complex_hard_noshield.yaml \
  configs/aerial_rl_urban_complex_p2c.yaml \
  configs/aerial_rl_urban_complex_directional_oa.yaml \
  experiments/aerial/rl/reward.py \
  experiments/aerial/rl/train_rl.py \
  experiments/aerial/scripts/run_mainchannel_hard_noshield.sh \
  ubantu@10.229.20.84:/tmp/
ssh -n -o BatchMode=yes -o ConnectTimeout=20 ubantu@10.229.20.84 'bash -lc "
cd /data/aerial-wam-v2
cp -f /tmp/aerial_rl_urban_complex_hard_noshield.yaml configs/
cp -f /tmp/aerial_rl_urban_complex_p2c.yaml configs/
cp -f /tmp/aerial_rl_urban_complex_directional_oa.yaml configs/
cp -f /tmp/reward.py experiments/aerial/rl/
cp -f /tmp/train_rl.py experiments/aerial/rl/
cp -f /tmp/run_mainchannel_hard_noshield.sh experiments/aerial/scripts/
chmod +x experiments/aerial/scripts/run_mainchannel_hard_noshield.sh
echo INSTALLED84
"'

echo '=== ensure four hard trains ==='
bash experiments/aerial/scripts/ensure_hard_quad.sh

echo '=== restart monitor ==='
pkill -f '[s]oft_breakout_monitor_125.sh' || true
sleep 1
nohup env STAMP=$STAMP bash experiments/aerial/scripts/soft_breakout_monitor_125.sh \
  >experiments/aerial/rl/artifacts/logs/soft_breakout_monitor_${STAMP}.out 2>&1 &
echo MONITOR_PID=$!
sleep 2
echo '=== procs ==='
pgrep -af '[r]un_mainchannel_hard|[t]rain_v4_ac|[s]oft_breakout_monitor' | grep -v pgrep | head -10
echo BUSY_KICK_DONE
