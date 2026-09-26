#!/usr/bin/env bash
# Runs ON 125. Launch R1–R4 geomfix overnight (noshield mainchannel).
set -euo pipefail
STAMP="${STAMP:?}"
echo "STAMP=$STAMP geomfix launch $(hostname) $(date -Is)"
D=/tmp/geomfix_deploy
KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"
R125=/home/yao/aerial-wam-v2
WM_REL=experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920
DS_NEAR=experiments/aerial/rl/artifacts/dataset_v0_three_zone_near_20260823fg
DS_MERGED=experiments/aerial/rl/artifacts/dataset_v0_p45_merged_20260821
DS_ENRICH=experiments/aerial/rl/artifacts/dataset_v0_p45_near_enrich_20260820
OUT="$R125/experiments/aerial/rl/artifacts/geomfix_${STAMP}"
mkdir -p "$OUT"

install_into() {
  local ROOT=$1
  mkdir -p "$ROOT/experiments/aerial/scripts" "$ROOT/configs" \
           "$ROOT/experiments/aerial/rl" "$ROOT/docs/superpowers/plans"
  cp -f "$D"/run_geomfix_r*.sh "$ROOT/experiments/aerial/scripts/"
  cp -f "$D"/wam_latent_depth_probe.py "$D"/wam_imagine_coll_rank.py \
        "$D"/wam_obstacle_cost_cone_rank.py "$ROOT/experiments/aerial/scripts/" 2>/dev/null || true
  cp -f "$D"/watch_geomfix_quad.sh "$ROOT/experiments/aerial/scripts/" 2>/dev/null || true
  cp -f "$D"/aerial_rl_geomfix_near_wm.yaml "$ROOT/configs/" 2>/dev/null || true
  cp -f "$D"/train_obstacle_cost_labels.py "$ROOT/experiments/aerial/rl/" 2>/dev/null || true
  chmod +x "$ROOT/experiments/aerial/scripts"/run_geomfix_r*.sh \
           "$ROOT/experiments/aerial/scripts"/watch_geomfix_quad.sh 2>/dev/null || true
}

# kill leftover C / geom verify noise
pkill -9 -f 'watch_c1234|chain_c1234|train_v4_ac|run_mainchannel_c' 2>/dev/null || true
pkill -9 -f 'run_geom_verify_arm|watch_geom_verify' 2>/dev/null || true

install_into "$R125"

# ---------- 125 R1 ----------
cd "$R125"; export PYTHONPATH="$R125"
nohup env HOST=125 STAMP="$STAMP" \
  bash experiments/aerial/scripts/run_geomfix_r1_cost_reflow.sh >/dev/null 2>&1 &
echo R1_125=$!

# ---------- 84 R2 ----------
R84=/data/aerial-wam-v2
if timeout 15 ssh -n -o BatchMode=yes -o ConnectTimeout=10 ubantu@10.229.20.84 'echo 84_ok'; then
  scp -o BatchMode=yes -o ConnectTimeout=20 \
    "$D"/run_geomfix_r2_collect_84.sh \
    "$D"/aerial_rl_geomfix_near_wm.yaml \
    ubantu@10.229.20.84:/tmp/
  ssh -o BatchMode=yes ubantu@10.229.20.84 \
    "export STAMP='$STAMP'; bash -s" <<'H84'
set -euo pipefail
ROOT=/data/aerial-wam-v2
cp -f /tmp/run_geomfix_r2_collect_84.sh "$ROOT/experiments/aerial/scripts/"
chmod +x "$ROOT/experiments/aerial/scripts/run_geomfix_r2_collect_84.sh"
pkill -9 -f 'train_v4_ac|run_mainchannel_c2|run_geomfix_r2' 2>/dev/null || true
cd "$ROOT"; export PYTHONPATH="$ROOT" PYTHON_BIN=/data/venvs/sim_verify/bin/python
nohup env HOST=84 STAMP="$STAMP" PYTHON_BIN="$PYTHON_BIN" ROUNDS=4 \
  bash experiments/aerial/scripts/run_geomfix_r2_collect_84.sh >/dev/null 2>&1 &
echo R2_84=$!
H84
else
  echo "WARN 84 down — R2 skipped"
  echo SKIP >"$OUT/R2_SKIP.txt"
fi

launch_h100() {
  local PORT=$1 ARM=$2 HOSTN=$3 SCRIPT=$4
  local RDEST=/home/a25689/aerial-wam-v2
  local SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new -i $KEY -p $PORT"
  local SCP_OPTS="-o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new -i $KEY -P $PORT"
  echo "=== $ARM @$HOSTN ==="
  ssh -n $SSH_OPTS "a25689@10.239.121.$HOSTN" \
    "pkill -9 -f train_v4_ac; pkill -9 -f run_geomfix_r; true" || true
  scp $SCP_OPTS \
    "$D"/$SCRIPT \
    "$D"/aerial_rl_geomfix_near_wm.yaml \
    "$D"/wam_latent_depth_probe.py \
    "$D"/wam_imagine_coll_rank.py \
    "$D"/wam_obstacle_cost_cone_rank.py \
    "$D"/train_obstacle_cost_labels.py \
    "a25689@10.239.121.$HOSTN:/tmp/"
  ssh -n $SSH_OPTS "a25689@10.239.121.$HOSTN" \
    "mkdir -p $RDEST/experiments/aerial/scripts $RDEST/configs $RDEST/experiments/aerial/rl; \
     cp -f /tmp/$SCRIPT $RDEST/experiments/aerial/scripts/; \
     cp -f /tmp/aerial_rl_geomfix_near_wm.yaml $RDEST/configs/; \
     cp -f /tmp/wam_*.py $RDEST/experiments/aerial/scripts/; \
     cp -f /tmp/train_obstacle_cost_labels.py $RDEST/experiments/aerial/rl/; \
     chmod +x $RDEST/experiments/aerial/scripts/$SCRIPT"
  # ensure datasets + init wm
  for REL in "$WM_REL" "$DS_NEAR" "$DS_MERGED"; do
    if ! ssh -n $SSH_OPTS "a25689@10.239.121.$HOSTN" "test -e $RDEST/$REL && (test -f $RDEST/$REL/wm_step_8500.pt -o -f $RDEST/$REL/wm_obs.pt -o -d $RDEST/$REL) && ls $RDEST/$REL 2>/dev/null | grep -q ."; then
      echo "sync $REL → $HOSTN"
      ssh -n $SSH_OPTS "a25689@10.239.121.$HOSTN" "mkdir -p $RDEST/$REL"
      rsync -a "$R125/$REL/" -e "ssh $SSH_OPTS" "a25689@10.239.121.$HOSTN:$RDEST/$REL/" || echo SYNC_FAIL_$REL
    else
      echo "have $REL"
    fi
  done
  # enrich optional
  if [[ -d "$R125/$DS_ENRICH" ]]; then
    ssh -n $SSH_OPTS "a25689@10.239.121.$HOSTN" "test -d $RDEST/$DS_ENRICH && ls $RDEST/$DS_ENRICH|grep -q ." || \
      rsync -a "$R125/$DS_ENRICH/" -e "ssh $SSH_OPTS" "a25689@10.239.121.$HOSTN:$RDEST/$DS_ENRICH/" || true
  fi
  ssh $SSH_OPTS "a25689@10.239.121.$HOSTN" \
    "cd $RDEST; export PYTHONPATH=$RDEST STAMP=$STAMP HOST=$HOSTN ROOT=$RDEST;
     PY=$RDEST/.venv/bin/python; [[ -x \$PY ]] || PY=python3;
     nohup env HOST=$HOSTN STAMP=$STAMP ROOT=$RDEST PYTHON_BIN=\$PY \
       bash experiments/aerial/scripts/$SCRIPT >/dev/null 2>&1 &
     echo LAUNCH_${ARM}=\$!; sleep 2; pgrep -af run_geomfix_r | grep -v pgrep | head -3"
}

launch_h100 31126 R3 14 run_geomfix_r3_wm_near.sh
launch_h100 30627 R4 11 run_geomfix_r4_wm_probe.sh

# watch
nohup env STAMP="$STAMP" bash "$R125/experiments/aerial/scripts/watch_geomfix_quad.sh" \
  >/dev/null 2>&1 &
echo WATCH=$!
echo GEOMFIX_LAUNCH_DONE
