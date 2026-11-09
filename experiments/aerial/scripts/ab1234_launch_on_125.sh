#!/usr/bin/env bash
# Deploy A1/A2/B1/B2 after C-round. Runs ON 125 (payload already in /tmp/ab1234_deploy or rsynced).
set -euo pipefail
STAMP="${STAMP:?}"
ITERS="${ITERS:-100}"
BC_ITERS="${BC_ITERS:-40}"
echo "AB deploy stamp=$STAMP on $(hostname) $(date -Is)"
D=/tmp/ab1234_deploy
KEY=~/.ssh/id_ed25519_h100
R125=/home/yao/aerial-wam-v2

install() {
  local ROOT=$1
  mkdir -p "$ROOT/configs" "$ROOT/experiments/aerial/scripts" "$ROOT/docs/superpowers/plans"
  cp -f "$D"/*.yaml "$ROOT/configs/" 2>/dev/null || true
  cp -f "$D"/run_mainchannel_a*.sh "$D"/run_mainchannel_b*.sh "$ROOT/experiments/aerial/scripts/" 2>/dev/null || true
  cp -f "$D"/2026-09-25-ab-after-c-round.md "$ROOT/docs/superpowers/plans/" 2>/dev/null || true
  chmod +x "$ROOT/experiments/aerial/scripts"/run_mainchannel_a*.sh \
           "$ROOT/experiments/aerial/scripts"/run_mainchannel_b*.sh 2>/dev/null || true
}

install "$R125"
# stop C arms only (they should be DONE); stop prior AB if any
pkill -f 'run_mainchannel_c[1-4]|train_v4_ac.*c[1-4]_' 2>/dev/null || true
pkill -f 'run_mainchannel_a1_|run_mainchannel_a2_|run_mainchannel_b1_|run_mainchannel_b2_' 2>/dev/null || true
sleep 2

cd "$R125"; export PYTHONPATH="$R125"
nohup env HOST=125 STAMP="$STAMP" ITERS="$ITERS" SKIP_BASELINE=1 \
  bash experiments/aerial/scripts/run_mainchannel_a1_antistick_noshield.sh >/dev/null 2>&1 &
echo A1_125=$!

scp -o BatchMode=yes \
  "$D"/aerial_rl_urban_complex_a2_antistick_strong.yaml \
  "$D"/aerial_rl_urban_complex_antistick_reward.yaml \
  "$D"/aerial_rl_urban_complex_directional_oa.yaml \
  "$D"/run_mainchannel_a2_antistick_strong_noshield.sh \
  ubantu@10.229.20.84:/tmp/
ssh -o BatchMode=yes ubantu@10.229.20.84 "export STAMP='$STAMP' ITERS='$ITERS'; bash -s" <<'H84'
ROOT=/data/aerial-wam-v2
cp -f /tmp/aerial_rl_urban_complex_a2_antistick_strong.yaml /tmp/aerial_rl_urban_complex_antistick_reward.yaml \
     /tmp/aerial_rl_urban_complex_directional_oa.yaml "$ROOT/configs/"
cp -f /tmp/run_mainchannel_a2_antistick_strong_noshield.sh "$ROOT/experiments/aerial/scripts/"
chmod +x "$ROOT/experiments/aerial/scripts/run_mainchannel_a2_antistick_strong_noshield.sh"
pkill -f 'run_mainchannel_a2_|train_v4_ac.*a2_' 2>/dev/null || true
pkill -f 'run_mainchannel_c2_|train_v4_ac.*c2_' 2>/dev/null || true
cd "$ROOT"; export PYTHONPATH="$ROOT" PYTHON_BIN=/data/venvs/sim_verify/bin/python
nohup env HOST=84 STAMP="$STAMP" ITERS="$ITERS" SKIP_BASELINE=1 PYTHON_BIN="$PYTHON_BIN" \
  bash experiments/aerial/scripts/run_mainchannel_a2_antistick_strong_noshield.sh >/dev/null 2>&1 &
echo A2_84=$!
H84

scp -o BatchMode=yes -i "$KEY" -P 31126 \
  "$D"/run_mainchannel_b1_pathexpert_bc_noshield.sh \
  "$D"/aerial_rl_urban_complex_directional_oa.yaml \
  a25689@10.239.121.14:/tmp/
ssh -o BatchMode=yes -i "$KEY" -p 31126 a25689@10.239.121.14 \
  "export STAMP='$STAMP' BC_ITERS='$BC_ITERS'; bash -s" <<'H14'
ROOT=/home/a25689/aerial-wam-v2
cp -f /tmp/run_mainchannel_b1_pathexpert_bc_noshield.sh "$ROOT/experiments/aerial/scripts/"
cp -f /tmp/aerial_rl_urban_complex_directional_oa.yaml "$ROOT/configs/"
chmod +x "$ROOT/experiments/aerial/scripts/run_mainchannel_b1_pathexpert_bc_noshield.sh"
pkill -f 'run_mainchannel_b1_|train_v4_ac.*b1_|collect_path_expert' 2>/dev/null || true
pkill -f 'run_mainchannel_c3_|train_v4_ac.*c3_' 2>/dev/null || true
cd "$ROOT"; export PYTHONPATH="$ROOT"
PY=""; [[ -x $ROOT/.venv/bin/python ]] && PY=$ROOT/.venv/bin/python
nohup env HOST=14 STAMP="$STAMP" BC_ITERS="$BC_ITERS" SKIP_BASELINE=1 ${PY:+PYTHON_BIN=$PY} \
  bash experiments/aerial/scripts/run_mainchannel_b1_pathexpert_bc_noshield.sh >/dev/null 2>&1 &
echo B1_14=$!
H14

scp -o BatchMode=yes -i "$KEY" -P 30627 \
  "$D"/run_mainchannel_b2_pathexpert_deprivilege_noshield.sh \
  "$D"/aerial_rl_urban_complex_directional_oa.yaml \
  a25689@10.239.121.11:/tmp/
ssh -o BatchMode=yes -i "$KEY" -p 30627 a25689@10.239.121.11 \
  "export STAMP='$STAMP' BC_ITERS='$BC_ITERS'; bash -s" <<'H11'
ROOT=/home/a25689/aerial-wam-v2
cp -f /tmp/run_mainchannel_b2_pathexpert_deprivilege_noshield.sh "$ROOT/experiments/aerial/scripts/"
cp -f /tmp/aerial_rl_urban_complex_directional_oa.yaml "$ROOT/configs/"
chmod +x "$ROOT/experiments/aerial/scripts/run_mainchannel_b2_pathexpert_deprivilege_noshield.sh"
pkill -f 'run_mainchannel_b2_|train_v4_ac.*b2_|collect_path_expert' 2>/dev/null || true
pkill -f 'run_mainchannel_c4_|train_v4_ac.*c4_' 2>/dev/null || true
cd "$ROOT"; export PYTHONPATH="$ROOT"
PY=""; [[ -x $ROOT/.venv/bin/python ]] && PY=$ROOT/.venv/bin/python
nohup env HOST=11 STAMP="$STAMP" BC_ITERS="$BC_ITERS" SKIP_BASELINE=1 ${PY:+PYTHON_BIN=$PY} \
  bash experiments/aerial/scripts/run_mainchannel_b2_pathexpert_deprivilege_noshield.sh >/dev/null 2>&1 &
echo B2_11=$!
H11

sleep 12
echo "=== AB verify ==="
echo -n "A1: "; pgrep -af 'a1_antistick|train_v4_ac.*a1_' | grep -v pgrep | head -1 || echo NONE
echo -n "A2: "; ssh -n -o BatchMode=yes ubantu@10.229.20.84 'pgrep -af "a2_antistick|train_v4_ac.*a2_" | grep -v pgrep | head -1' || echo NONE
echo -n "B1: "; ssh -n -o BatchMode=yes -i "$KEY" -p 31126 a25689@10.239.121.14 'pgrep -af "b1_pathexpert|collect_path_expert|train_v4_ac.*b1_" | grep -v pgrep | head -1' || echo NONE
echo -n "B2: "; ssh -n -o BatchMode=yes -i "$KEY" -p 30627 a25689@10.239.121.11 'pgrep -af "b2_pathexpert|collect_path_expert|train_v4_ac.*b2_" | grep -v pgrep | head -1' || echo NONE
echo "AB_DEPLOY_DONE stamp=$STAMP" | tee "$R125/experiments/aerial/rl/artifacts/logs/ab1234_${STAMP}_DEPLOYED.txt"
