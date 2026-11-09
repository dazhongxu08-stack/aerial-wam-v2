#!/usr/bin/env bash
# Runs ON 125. Launched by deploy_c1234_oa_terminal.sh after rsync to /tmp/c1234_deploy.
set -euo pipefail
STAMP="${STAMP:?}"
ITERS="${ITERS:-100}"
echo "STAMP=$STAMP ITERS=$ITERS on $(hostname) $(date -Is)"
D=/tmp/c1234_deploy
FLOOR_REL=experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2
WM_REL=experiments/aerial/rl/artifacts/wm_ckpt_obstacle_cost_gatefix_20260922_161641
DEPTH_REL=experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825
TAU_REL=experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815

install_scripts() {
  local ROOT=$1
  mkdir -p "$ROOT/configs" "$ROOT/experiments/aerial/scripts" "$ROOT/docs/superpowers/plans"
  cp -f "$D"/*.yaml "$ROOT/configs/" 2>/dev/null || true
  cp -f "$D"/run_mainchannel_c*.sh "$ROOT/experiments/aerial/scripts/"
  cp -f "$D"/watch_c1234_oa_terminal.sh "$ROOT/experiments/aerial/scripts/"
  cp -f "$D"/2026-09-25-overnight-oa-terminal.md "$ROOT/docs/superpowers/plans/" 2>/dev/null || true
  chmod +x "$ROOT/experiments/aerial/scripts"/run_mainchannel_c*.sh \
           "$ROOT/experiments/aerial/scripts"/watch_c1234_oa_terminal.sh
}

R125=/home/yao/aerial-wam-v2
install_scripts "$R125"
test -f "$R125/$FLOOR_REL/v4_ac_best.pt"

# ---------- 125 C1 ol_bc ----------
pkill -f 'run_mainchannel_c1_ol_bc|train_v4_ac.*c1_ol_bc' 2>/dev/null || true
pkill -f 'run_mainchannel_b_curiosity|train_v4_ac.*curiosity' 2>/dev/null || true
cd "$R125"; export PYTHONPATH="$R125"
nohup env HOST=125 STAMP="$STAMP" ITERS="$ITERS" SKIP_BASELINE=1 \
  bash experiments/aerial/scripts/run_mainchannel_c1_ol_bc_noshield.sh \
  >/dev/null 2>&1 &
echo C1_125_LAUNCH=$!
sleep 2

# ---------- 84 C2 wm_escape ----------
scp -o BatchMode=yes -o ConnectTimeout=20 \
  "$D"/aerial_rl_urban_complex_c2_wm_escape.yaml \
  "$D"/aerial_rl_urban_complex_directional_oa.yaml \
  "$D"/run_mainchannel_c2_wm_escape_noshield.sh \
  ubantu@10.229.20.84:/tmp/
ssh -o BatchMode=yes -o ConnectTimeout=20 ubantu@10.229.20.84 \
  "export STAMP='$STAMP' ITERS='$ITERS'; bash -s" <<'H84'
set -euo pipefail
ROOT=/data/aerial-wam-v2
cp -f /tmp/aerial_rl_urban_complex_c2_wm_escape.yaml "$ROOT/configs/"
cp -f /tmp/aerial_rl_urban_complex_directional_oa.yaml "$ROOT/configs/"
cp -f /tmp/run_mainchannel_c2_wm_escape_noshield.sh "$ROOT/experiments/aerial/scripts/"
chmod +x "$ROOT/experiments/aerial/scripts/run_mainchannel_c2_wm_escape_noshield.sh"
test -f "$ROOT/experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2/v4_ac_best.pt"
pkill -f 'run_mainchannel_c2_wm_escape|train_v4_ac.*c2_wm_escape' 2>/dev/null || true
pkill -f 'run_mainchannel_v2_floor|train_v4_ac.*v2_floor' 2>/dev/null || true
cd "$ROOT"; export PYTHONPATH="$ROOT" PYTHON_BIN=/data/venvs/sim_verify/bin/python
nohup env HOST=84 STAMP="$STAMP" ITERS="$ITERS" SKIP_BASELINE=1 \
  PYTHON_BIN="$PYTHON_BIN" OA_PLANNER_MOCK=wm_escape \
  bash experiments/aerial/scripts/run_mainchannel_c2_wm_escape_noshield.sh \
  >/dev/null 2>&1 &
echo C2_84_LAUNCH=$!
H84

sync_floor_bundle() {
  local HOSTPORT=$1 KEY=$2 USERHOST=$3 RDEST=$4
  if ! ssh -n -o BatchMode=yes -i "$KEY" -p "$HOSTPORT" "$USERHOST" \
      "test -f $RDEST/$FLOOR_REL/v4_ac_best.pt"; then
    echo "sync floor → $USERHOST:$HOSTPORT"
    ssh -n -o BatchMode=yes -i "$KEY" -p "$HOSTPORT" "$USERHOST" "mkdir -p $RDEST/$FLOOR_REL"
    rsync -a "$R125/$FLOOR_REL/" \
      -e "ssh -o BatchMode=yes -i $KEY -p $HOSTPORT" \
      "$USERHOST:$RDEST/$FLOOR_REL/"
  fi
  if ! ssh -n -o BatchMode=yes -i "$KEY" -p "$HOSTPORT" "$USERHOST" \
      "test -f $RDEST/$WM_REL/wm_obs.pt"; then
    echo "sync wm → $USERHOST:$HOSTPORT"
    if [[ -f "$R125/$WM_REL/wm_obs.pt" ]]; then
      ssh -n -o BatchMode=yes -i "$KEY" -p "$HOSTPORT" "$USERHOST" "mkdir -p $RDEST/$WM_REL"
      rsync -a "$R125/$WM_REL/" \
        -e "ssh -o BatchMode=yes -i $KEY -p $HOSTPORT" \
        "$USERHOST:$RDEST/$WM_REL/"
    elif [[ -f "$R125/experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt" ]]; then
      ssh -n -o BatchMode=yes -i "$KEY" -p "$HOSTPORT" "$USERHOST" \
        "mkdir -p $RDEST/experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920"
      rsync -a "$R125/experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/" \
        -e "ssh -o BatchMode=yes -i $KEY -p $HOSTPORT" \
        "$USERHOST:$RDEST/experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/"
    fi
  fi
  for REL in "$DEPTH_REL" "$TAU_REL"; do
    if ! ssh -n -o BatchMode=yes -i "$KEY" -p "$HOSTPORT" "$USERHOST" \
        "test -d $RDEST/$REL && ls $RDEST/$REL | grep -q ."; then
      echo "sync $REL → $USERHOST:$HOSTPORT"
      ssh -n -o BatchMode=yes -i "$KEY" -p "$HOSTPORT" "$USERHOST" "mkdir -p $RDEST/$REL"
      rsync -a "$R125/$REL/" \
        -e "ssh -o BatchMode=yes -i $KEY -p $HOSTPORT" \
        "$USERHOST:$RDEST/$REL/" || true
    fi
  done
}

KEY=~/.ssh/id_ed25519_h100
R_H100=/home/a25689/aerial-wam-v2

# ---------- 14 C3 ----------
scp -o BatchMode=yes -i "$KEY" -P 31126 \
  "$D"/aerial_rl_urban_complex_antistick_reward.yaml \
  "$D"/aerial_rl_urban_complex_directional_oa.yaml \
  "$D"/run_mainchannel_c3_antistick_noshield.sh \
  a25689@10.239.121.14:/tmp/
sync_floor_bundle 31126 "$KEY" a25689@10.239.121.14 "$R_H100"
ssh -o BatchMode=yes -i "$KEY" -p 31126 a25689@10.239.121.14 \
  "export STAMP='$STAMP' ITERS='$ITERS'; bash -s" <<'H14'
set -euo pipefail
ROOT=/home/a25689/aerial-wam-v2
cp -f /tmp/aerial_rl_urban_complex_antistick_reward.yaml "$ROOT/configs/"
cp -f /tmp/aerial_rl_urban_complex_directional_oa.yaml "$ROOT/configs/"
cp -f /tmp/run_mainchannel_c3_antistick_noshield.sh "$ROOT/experiments/aerial/scripts/"
chmod +x "$ROOT/experiments/aerial/scripts/run_mainchannel_c3_antistick_noshield.sh"
test -f "$ROOT/experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2/v4_ac_best.pt"
pkill -f 'run_mainchannel_c3_antistick|train_v4_ac.*c3_antistick' 2>/dev/null || true
pkill -f 'run_mainchannel_b_openside|train_v4_ac.*b_openside' 2>/dev/null || true
cd "$ROOT"; export PYTHONPATH="$ROOT"
PY=""
[[ -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
nohup env HOST=14 STAMP="$STAMP" ITERS="$ITERS" SKIP_BASELINE=1 ${PY:+PYTHON_BIN=$PY} \
  bash experiments/aerial/scripts/run_mainchannel_c3_antistick_noshield.sh \
  >/dev/null 2>&1 &
echo C3_14_LAUNCH=$!
H14

# ---------- 11 C4 ----------
scp -o BatchMode=yes -i "$KEY" -P 30627 \
  "$D"/aerial_rl_urban_complex_c4_terminal.yaml \
  "$D"/aerial_rl_urban_complex_directional_oa.yaml \
  "$D"/run_mainchannel_c4_terminal_noshield.sh \
  a25689@10.239.121.11:/tmp/
sync_floor_bundle 30627 "$KEY" a25689@10.239.121.11 "$R_H100"
ssh -o BatchMode=yes -i "$KEY" -p 30627 a25689@10.239.121.11 \
  "export STAMP='$STAMP' ITERS='$ITERS'; bash -s" <<'H11'
set -euo pipefail
ROOT=/home/a25689/aerial-wam-v2
cp -f /tmp/aerial_rl_urban_complex_c4_terminal.yaml "$ROOT/configs/"
cp -f /tmp/aerial_rl_urban_complex_directional_oa.yaml "$ROOT/configs/"
cp -f /tmp/run_mainchannel_c4_terminal_noshield.sh "$ROOT/experiments/aerial/scripts/"
chmod +x "$ROOT/experiments/aerial/scripts/run_mainchannel_c4_terminal_noshield.sh"
test -f "$ROOT/experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2/v4_ac_best.pt"
pkill -f 'run_mainchannel_c4_terminal|train_v4_ac.*c4_terminal' 2>/dev/null || true
pkill -f 'run_mainchannel_v4_neargoal|train_v4_ac.*v4_neargoal' 2>/dev/null || true
cd "$ROOT"; export PYTHONPATH="$ROOT"
PY=""
[[ -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
nohup env HOST=11 STAMP="$STAMP" ITERS="$ITERS" SKIP_BASELINE=1 NEAR_GOAL_FRAC=0.5 ${PY:+PYTHON_BIN=$PY} \
  bash experiments/aerial/scripts/run_mainchannel_c4_terminal_noshield.sh \
  >/dev/null 2>&1 &
echo C4_11_LAUNCH=$!
H11

# ---------- watch on 125 ----------
pkill -f 'watch_c1234_oa_terminal' 2>/dev/null || true
nohup env STAMP="$STAMP" ITERS="$ITERS" \
  bash "$R125/experiments/aerial/scripts/watch_c1234_oa_terminal.sh" \
  >/dev/null 2>&1 &
echo WATCH_125=$!

sleep 18
echo "=== verify ==="
echo -n "125 C1: "; pgrep -af 'c1_ol_bc|train_v4_ac.*c1_ol' | grep -v pgrep | head -1 || echo NONE
echo -n "84 C2: "; ssh -n -o BatchMode=yes ubantu@10.229.20.84 'pgrep -af "c2_wm_escape|train_v4_ac.*c2_wm" | grep -v pgrep | head -1' || echo NONE
echo -n "14 C3: "; ssh -n -o BatchMode=yes -i "$KEY" -p 31126 a25689@10.239.121.14 'pgrep -af "c3_antistick|train_v4_ac.*c3_" | grep -v pgrep | head -1' || echo NONE
echo -n "11 C4: "; ssh -n -o BatchMode=yes -i "$KEY" -p 30627 a25689@10.239.121.11 'pgrep -af "c4_terminal|train_v4_ac.*c4_" | grep -v pgrep | head -1' || echo NONE
echo -n "watch: "; pgrep -af watch_c1234_oa_terminal | grep -v pgrep | head -1 || echo NONE
echo "--- c1 log ---"
tail -15 "$R125/experiments/aerial/rl/artifacts/logs/mainchannel_c1_ol_bc_${STAMP}_h125.log" 2>/dev/null || echo "(no log yet)"
echo "DEPLOY_C1234_DONE stamp=$STAMP"
