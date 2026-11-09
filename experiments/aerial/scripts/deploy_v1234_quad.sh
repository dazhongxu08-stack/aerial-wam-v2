#!/usr/bin/env bash
# Deploy+launch V1–V4 on 125/84/14/11.
#   STAMP=20260925_1007 SSH125=cursor-125-public bash experiments/aerial/scripts/deploy_v1234_quad.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STAMP="${STAMP:-20260925_1007}"
SSH125="${SSH125:-cursor-125-public}"
cd "$ROOT"

echo "=== rsync payloads to $SSH125 ==="
ssh -o ConnectTimeout=25 -o BatchMode=yes "$SSH125" 'mkdir -p /tmp/v1234_deploy'
rsync -avz \
  configs/aerial_rl_urban_complex_b_curiosity.yaml \
  configs/aerial_rl_urban_complex_b_openside.yaml \
  configs/aerial_rl_urban_complex_v4_neargoal.yaml \
  experiments/aerial/scripts/run_mainchannel_b_curiosity_noshield.sh \
  experiments/aerial/scripts/run_mainchannel_b_openside_noshield.sh \
  experiments/aerial/scripts/run_mainchannel_v2_floor_then_b.sh \
  experiments/aerial/scripts/run_mainchannel_v4_neargoal_noshield.sh \
  "$SSH125:/tmp/v1234_deploy/"

# Export stamp into remote via env on the outer ssh command
ssh -o ConnectTimeout=25 -o BatchMode=yes "$SSH125" \
  "export STAMP='$STAMP'; bash -s" <<'REMOTE'
set -euo pipefail
echo "STAMP=$STAMP on $(hostname)"

# ---------- 125 V1 curiosity ----------
R125=/home/yao/aerial-wam-v2
cp -f /tmp/v1234_deploy/aerial_rl_urban_complex_b_curiosity.yaml "$R125/configs/"
cp -f /tmp/v1234_deploy/run_mainchannel_b_curiosity_noshield.sh "$R125/experiments/aerial/scripts/"
chmod +x "$R125/experiments/aerial/scripts/run_mainchannel_b_curiosity_noshield.sh"
pkill -f 'run_mainchannel_b_curiosity|train_v4_ac.*b_curiosity' 2>/dev/null || true
LOGA="$R125/experiments/aerial/rl/artifacts/logs/mainchannel_outcome_a_20260924_1605_h125.log"
mkdir -p "$(dirname "$LOGA")"
grep -q OUTCOME_A_NOSHIELD_DONE "$LOGA" 2>/dev/null || echo OUTCOME_A_NOSHIELD_DONE >>"$LOGA"
cd "$R125"; export PYTHONPATH="$R125"
nohup env HOST=125 STAMP="$STAMP" SKIP_BASELINE=1 \
  bash experiments/aerial/scripts/run_mainchannel_b_curiosity_noshield.sh \
  >/dev/null 2>&1 &
echo V1_125_LAUNCH=$!

# ---------- 84 V2 floor then B ----------
scp -o BatchMode=yes \
  /tmp/v1234_deploy/aerial_rl_urban_complex_b_curiosity.yaml \
  /tmp/v1234_deploy/run_mainchannel_v2_floor_then_b.sh \
  ubantu@10.229.20.84:/tmp/
ssh -o BatchMode=yes ubantu@10.229.20.84 "export STAMP='$STAMP'; bash -s" <<'H84'
set -euo pipefail
ROOT=/data/aerial-wam-v2
cp -f /tmp/aerial_rl_urban_complex_b_curiosity.yaml "$ROOT/configs/"
cp -f /tmp/run_mainchannel_v2_floor_then_b.sh "$ROOT/experiments/aerial/scripts/"
chmod +x "$ROOT/experiments/aerial/scripts/run_mainchannel_v2_floor_then_b.sh"
test -f "$ROOT/experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2/v4_ac_best.pt"
pkill -f 'run_mainchannel_v2_floor|train_v4_ac.*v2_floor' 2>/dev/null || true
pkill -f 'run_mainchannel_outcome_a4|train_v4_ac.*outcome_a4' 2>/dev/null || true
LOGA="$ROOT/experiments/aerial/rl/artifacts/logs/mainchannel_outcome_a_20260924_1605_h84.log"
mkdir -p "$(dirname "$LOGA")"
grep -q OUTCOME_A_NOSHIELD_DONE "$LOGA" 2>/dev/null || echo OUTCOME_A_NOSHIELD_DONE >>"$LOGA"
cd "$ROOT"; export PYTHONPATH="$ROOT" PYTHON_BIN=/data/venvs/sim_verify/bin/python
nohup env HOST=84 STAMP="$STAMP" PYTHON_BIN="$PYTHON_BIN" \
  bash experiments/aerial/scripts/run_mainchannel_v2_floor_then_b.sh \
  >/dev/null 2>&1 &
echo V2_84_LAUNCH=$!
H84

# ---------- 14 V3 openside (center) ----------
scp -o BatchMode=yes -i ~/.ssh/id_ed25519_h100 -P 31126 \
  /tmp/v1234_deploy/aerial_rl_urban_complex_b_openside.yaml \
  /tmp/v1234_deploy/run_mainchannel_b_openside_noshield.sh \
  a25689@10.239.121.14:/tmp/
ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_h100 -p 31126 a25689@10.239.121.14 \
  "export STAMP='$STAMP'; bash -s" <<'H14'
set -euo pipefail
ROOT=/home/a25689/aerial-wam-v2
cp -f /tmp/aerial_rl_urban_complex_b_openside.yaml "$ROOT/configs/"
cp -f /tmp/run_mainchannel_b_openside_noshield.sh "$ROOT/experiments/aerial/scripts/"
chmod +x "$ROOT/experiments/aerial/scripts/run_mainchannel_b_openside_noshield.sh"
pkill -f 'run_mainchannel_b_openside|train_v4_ac.*b_openside|train_v4_ac.*outcome_a_20260924_1605' 2>/dev/null || true
LOGA="$ROOT/experiments/aerial/rl/artifacts/logs/mainchannel_outcome_a_20260924_1605_h14.log"
mkdir -p "$(dirname "$LOGA")"
grep -q OUTCOME_A_NOSHIELD_DONE "$LOGA" 2>/dev/null || echo OUTCOME_A_NOSHIELD_DONE >>"$LOGA"
sleep 2
cd "$ROOT"; export PYTHONPATH="$ROOT"
nohup env HOST=14 STAMP="$STAMP" SKIP_BASELINE=1 \
  bash experiments/aerial/scripts/run_mainchannel_b_openside_noshield.sh \
  >/dev/null 2>&1 &
echo V3_14_LAUNCH=$!
H14

# ---------- 11 V4 neargoal ----------
scp -o BatchMode=yes -i ~/.ssh/id_ed25519_h100 -P 30627 \
  /tmp/v1234_deploy/aerial_rl_urban_complex_v4_neargoal.yaml \
  /tmp/v1234_deploy/run_mainchannel_v4_neargoal_noshield.sh \
  a25689@10.239.121.11:/tmp/
if ! ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_h100 -p 30627 a25689@10.239.121.11 \
  'test -f /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2/v4_ac_best.pt'; then
  rsync -a /home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2/ \
    -e 'ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_h100 -p 30627' \
    a25689@10.239.121.11:/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2/
fi
ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_h100 -p 30627 a25689@10.239.121.11 \
  "export STAMP='$STAMP'; bash -s" <<'H11'
set -euo pipefail
ROOT=/home/a25689/aerial-wam-v2
cp -f /tmp/aerial_rl_urban_complex_v4_neargoal.yaml "$ROOT/configs/"
cp -f /tmp/run_mainchannel_v4_neargoal_noshield.sh "$ROOT/experiments/aerial/scripts/"
chmod +x "$ROOT/experiments/aerial/scripts/run_mainchannel_v4_neargoal_noshield.sh"
pkill -f 'run_mainchannel_v4_neargoal|train_v4_ac.*v4_neargoal|train_v4_ac.*outcome_a_20260924_1605' 2>/dev/null || true
LOGA="$ROOT/experiments/aerial/rl/artifacts/logs/mainchannel_outcome_a_20260924_1605_h11.log"
mkdir -p "$(dirname "$LOGA")"
grep -q OUTCOME_A_NOSHIELD_DONE "$LOGA" 2>/dev/null || echo OUTCOME_A_NOSHIELD_DONE >>"$LOGA"
sleep 2
cd "$ROOT"; export PYTHONPATH="$ROOT"
nohup env HOST=11 STAMP="$STAMP" SKIP_BASELINE=1 \
  bash experiments/aerial/scripts/run_mainchannel_v4_neargoal_noshield.sh \
  >/dev/null 2>&1 &
echo V4_11_LAUNCH=$!
H11

sleep 12
echo "=== verify ==="
echo -n "125: "; pgrep -af 'b_curiosity|train_v4_ac.*curiosity' | grep -v pgrep | head -1 || echo NONE
echo -n "84: "; ssh -o BatchMode=yes ubantu@10.229.20.84 'pgrep -af "v2_floor|train_v4_ac" | grep -v pgrep | head -1' || echo NONE
echo -n "14: "; ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_h100 -p 31126 a25689@10.239.121.14 'pgrep -af "b_openside|train_v4_ac" | grep -v pgrep | head -1' || echo NONE
echo -n "11: "; ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_h100 -p 30627 a25689@10.239.121.11 'pgrep -af "v4_neargoal|train_v4_ac" | grep -v pgrep | head -1' || echo NONE
tail -12 /home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/mainchannel_b_curiosity_${STAMP}_h125.log 2>/dev/null || true
REMOTE

echo "DEPLOY_DONE stamp=$STAMP"
