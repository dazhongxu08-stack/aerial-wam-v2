#!/usr/bin/env bash
# Runs ON 125 after deploy_soft_breakout_quad.sh.
set +e
STAMP="${STAMP:?}"
ITERS_SOFT="${ITERS_SOFT:-120}"
ITERS_CEIL="${ITERS_CEIL:-40}"
echo "STAMP=$STAMP Soft=$ITERS_SOFT Ceil=$ITERS_CEIL on $(hostname) $(date -Is)"

R125=/home/yao/aerial-wam-v2
cd "$R125"
tar -xzf /tmp/soft_breakout.tgz
chmod +x experiments/aerial/scripts/run_mainchannel_soft_noshield.sh \
         experiments/aerial/scripts/run_soft_positive_gate_watch.sh \
         experiments/aerial/scripts/run_soft_cost_reflow_prep.sh \
         experiments/aerial/scripts/run_mainchannel_c1_ol_bc_noshield.sh

# Prove READY_SOFT
READY="$R125/experiments/aerial/rl/artifacts/geomfix_20260926_hardnear/READY_SOFT.txt"
if [[ ! -f "$READY" ]]; then
  echo "FATAL no READY_SOFT at $READY — refuse Soft"
  exit 2
fi
echo "READY_SOFT ok: $(cat "$READY")"

KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"
[[ -f "$KEY" ]] || KEY="$HOME/.ssh/id_ed25519"
PY125="${PYTHON_BIN:-}"
[[ -z "$PY125" && -x "$HOME/sim_verify/.venv/bin/python" ]] && PY125="$HOME/sim_verify/.venv/bin/python"
[[ -z "$PY125" && -x "$R125/.venv/bin/python" ]] && PY125="$R125/.venv/bin/python"
[[ -z "$PY125" ]] && PY125=python3
export PYTHONPATH="$R125"

# Stop geomfix near thrash / auto_advance so Soft owns AirSim
pkill -f 'auto_advance_geomfix' 2>/dev/null
pkill -f 'run_geomfix_near_episode' 2>/dev/null
pkill -f 'run_mainchannel_c[1-4]' 2>/dev/null

# ---------- 125: short ceiling OL-BC (iters≤40) ----------
pkill -f 'run_mainchannel_c1_ol_bc|train_v4_ac.*c1_ol_bc' 2>/dev/null
nohup env HOST=125 STAMP="${STAMP}_ceil" ITERS="$ITERS_CEIL" SKIP_BASELINE=1 \
  PYTHON_BIN="$PY125" \
  bash experiments/aerial/scripts/run_mainchannel_c1_ol_bc_noshield.sh \
  >experiments/aerial/rl/artifacts/logs/soft_ceil125_${STAMP}.out 2>&1 &
echo CEIL_125_PID=$!

# ---------- 84: Soft L-main ----------
scp -o BatchMode=yes -o ConnectTimeout=20 \
  configs/aerial_rl_urban_complex_soft_noshield.yaml \
  configs/aerial_rl_urban_complex_directional_oa.yaml \
  configs/aerial_rl_urban_complex_p2c.yaml \
  experiments/aerial/rl/reward.py \
  experiments/aerial/rl/train_rl.py \
  experiments/aerial/scripts/run_mainchannel_soft_noshield.sh \
  ubantu@10.229.20.84:/tmp/ 2>&1 | tail -3

ssh -o BatchMode=yes -o ConnectTimeout=25 ubantu@10.229.20.84 "bash -s" <<H84
set +e
ROOT=/data/aerial-wam-v2
cd \$ROOT
cp -f /tmp/aerial_rl_urban_complex_soft_noshield.yaml \$ROOT/configs/
cp -f /tmp/aerial_rl_urban_complex_directional_oa.yaml \$ROOT/configs/ 2>/dev/null
cp -f /tmp/aerial_rl_urban_complex_p2c.yaml \$ROOT/configs/ 2>/dev/null
cp -f /tmp/reward.py \$ROOT/experiments/aerial/rl/
cp -f /tmp/train_rl.py \$ROOT/experiments/aerial/rl/
cp -f /tmp/run_mainchannel_soft_noshield.sh \$ROOT/experiments/aerial/scripts/
chmod +x \$ROOT/experiments/aerial/scripts/run_mainchannel_soft_noshield.sh
# prove floor
test -f \$ROOT/experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2/v4_ac_best.pt
pkill -f 'run_mainchannel_soft|train_v4_ac.*soft' 2>/dev/null
pkill -f 'run_geomfix_near|auto_advance_geomfix' 2>/dev/null
export PYTHONPATH=\$ROOT
PY=/data/venvs/sim_verify/bin/python
[[ -x \$PY ]] || PY=\$ROOT/.venv/bin/python
[[ -x \$PY ]] || PY=python3
nohup env HOST=84 STAMP=$STAMP ITERS=$ITERS_SOFT SKIP_BASELINE=1 PYTHON_BIN=\$PY \
  bash experiments/aerial/scripts/run_mainchannel_soft_noshield.sh \
  >experiments/aerial/rl/artifacts/logs/soft84_${STAMP}.out 2>&1 &
echo SOFT_84_PID=\$!
sleep 3
ps -ef | grep -E 'run_mainchannel_soft|train_v4_ac' | grep -v grep | head -5
H84

# ---------- 14: Soft positive gate watch (NO Hard) ----------
scp -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
  -i "$KEY" -P 31126 \
  configs/aerial_rl_urban_complex_soft_noshield.yaml \
  configs/aerial_rl_urban_complex_directional_oa.yaml \
  configs/aerial_rl_urban_complex_p2c.yaml \
  experiments/aerial/scripts/run_soft_positive_gate_watch.sh \
  a25689@10.239.121.14:/tmp/ 2>&1 | tail -3

ssh -o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new \
  -i "$KEY" -p 31126 a25689@10.239.121.14 "bash -s" <<H14
set +e
cd /home/a25689/aerial-wam-v2
cp -f /tmp/aerial_rl_urban_complex_soft_noshield.yaml configs/
cp -f /tmp/aerial_rl_urban_complex_directional_oa.yaml configs/ 2>/dev/null
cp -f /tmp/aerial_rl_urban_complex_p2c.yaml configs/ 2>/dev/null
cp -f /tmp/run_soft_positive_gate_watch.sh experiments/aerial/scripts/
chmod +x experiments/aerial/scripts/run_soft_positive_gate_watch.sh
pkill -f 'run_soft_positive_gate|run_geomfix_r3|_wm_train_validate' 2>/dev/null
export PYTHONPATH=\$PWD
PY=\$PWD/.venv/bin/python
nohup env HOST=14 STAMP=$STAMP SOFT_POS_K=8 INTERVAL=600 PYTHON_BIN=\$PY \
  bash experiments/aerial/scripts/run_soft_positive_gate_watch.sh \
  >experiments/aerial/rl/artifacts/logs/soft_gate14_${STAMP}.out 2>&1 &
echo GATE_14_PID=\$!
H14

# ---------- 11: cost reflow prep ----------
scp -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
  -i "$KEY" -P 30627 \
  experiments/aerial/scripts/run_soft_cost_reflow_prep.sh \
  a25689@10.239.121.11:/tmp/ 2>&1 | tail -3

ssh -o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new \
  -i "$KEY" -p 30627 a25689@10.239.121.11 "bash -s" <<H11
set +e
cd /home/a25689/aerial-wam-v2
cp -f /tmp/run_soft_cost_reflow_prep.sh experiments/aerial/scripts/
chmod +x experiments/aerial/scripts/run_soft_cost_reflow_prep.sh
# keep antisat code; stop idle R4 if done
pkill -f 'run_geomfix_r4|_wm_train_validate' 2>/dev/null
export PYTHONPATH=\$PWD
PY=\$PWD/.venv/bin/python
nohup env HOST=11 STAMP=$STAMP PYTHON_BIN=\$PY \
  bash experiments/aerial/scripts/run_soft_cost_reflow_prep.sh \
  >experiments/aerial/rl/artifacts/logs/soft_cost11_${STAMP}.out 2>&1 &
echo COST_11_PID=\$!
H11

echo "SOFT_BREAKOUT_LAUNCH_DONE stamp=$STAMP $(date -Is)"
echo "Hard blocked until breakout_soft_${STAMP}/READY_HARD.txt"
