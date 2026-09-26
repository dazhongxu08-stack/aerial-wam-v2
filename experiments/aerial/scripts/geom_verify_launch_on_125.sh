#!/usr/bin/env bash
# Runs ON 125. Stop leftover C arms if any, sync DS to 84, launch G1–G4.
#   STAMP=20260925_geom bash /tmp/geom_deploy/geom_verify_launch_on_125.sh
set -euo pipefail
STAMP="${STAMP:?}"
MAX_SAMPLES="${MAX_SAMPLES:-400}"
echo "STAMP=$STAMP MAX_SAMPLES=$MAX_SAMPLES on $(hostname) $(date -Is)"
D=/tmp/geom_deploy
KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"
R125=/home/yao/aerial-wam-v2
WM_REL=experiments/aerial/rl/artifacts/wm_ckpt_obstacle_cost_gatefix_20260922_161641
DS_MERGED=experiments/aerial/rl/artifacts/dataset_v0_p45_merged_20260821
DS_NEAR=experiments/aerial/rl/artifacts/dataset_v0_three_zone_near_20260823fg
DS_ENRICH=experiments/aerial/rl/artifacts/dataset_v0_p45_near_enrich_20260820
GT_REL=experiments/aerial/rl/artifacts/obstacle_cost_gt_depth

install_into() {
  local ROOT=$1
  mkdir -p "$ROOT/experiments/aerial/scripts" "$ROOT/experiments/aerial/rl" \
           "$ROOT/docs/superpowers/plans"
  cp -f "$D"/wam_latent_depth_probe.py "$ROOT/experiments/aerial/scripts/" 2>/dev/null || true
  cp -f "$D"/wam_imagine_coll_rank.py "$ROOT/experiments/aerial/scripts/" 2>/dev/null || true
  cp -f "$D"/wam_obstacle_cost_cone_rank.py "$ROOT/experiments/aerial/scripts/"
  cp -f "$D"/run_geom_verify_arm.sh "$ROOT/experiments/aerial/scripts/"
  cp -f "$D"/summarize_geom_verify.py "$ROOT/experiments/aerial/scripts/" 2>/dev/null || true
  cp -f "$D"/train_obstacle_cost_labels.py "$ROOT/experiments/aerial/rl/" 2>/dev/null || true
  cp -f "$D"/2026-09-25-breakout-soft-hard-quad.md "$ROOT/docs/superpowers/plans/" 2>/dev/null || true
  chmod +x "$ROOT/experiments/aerial/scripts"/run_geom_verify_arm.sh \
           "$ROOT/experiments/aerial/scripts"/wam_*.py 2>/dev/null || true
}

# Ensure C arms dead (idempotent)
pkill -9 -f 'watch_c1234_oa_terminal' 2>/dev/null || true
pkill -9 -f 'chain_c1234_to_ab' 2>/dev/null || true
pkill -9 -f 'train_v4_ac' 2>/dev/null || true

install_into "$R125"
test -f "$R125/$WM_REL/wm_obs.pt"
mkdir -p "$R125/experiments/aerial/rl/artifacts/geom_verify_${STAMP}"

# ---------- 125 G1 ----------
cd "$R125"; export PYTHONPATH="$R125"
nohup env HOST=125 STAMP="$STAMP" ARM=G1 ROOT="$R125" MAX_SAMPLES="$MAX_SAMPLES" \
  bash experiments/aerial/scripts/run_geom_verify_arm.sh \
  >/dev/null 2>&1 &
echo G1_125_LAUNCH=$!

# ---------- sync helpers ----------
sync_tree() {
  local src=$1 dest_host=$2 dest_path=$3 ssh_opts=$4
  if ssh -n $ssh_opts "$dest_host" "test -d $dest_path && ls $dest_path | grep -q ."; then
    echo "skip sync (exists) $dest_path @ $dest_host"
    return 0
  fi
  echo "sync $src → $dest_host:$dest_path"
  ssh -n $ssh_opts "$dest_host" "mkdir -p $dest_path"
  rsync -a --info=stats1 "$src/" -e "ssh $ssh_opts" "$dest_host:$dest_path/"
}

# ---------- 84 G2 (need GT labels; optional DS for cone) ----------
R84=/data/aerial-wam-v2
scp -o BatchMode=yes -o ConnectTimeout=25 \
  "$D"/run_geom_verify_arm.sh \
  "$D"/wam_obstacle_cost_cone_rank.py \
  "$D"/wam_latent_depth_probe.py \
  "$D"/wam_imagine_coll_rank.py \
  ubantu@10.229.20.84:/tmp/ 2>/dev/null || true

# Prefer LAN 84; on timeout/fail fall back to G2 on 125 (after G1 starts — gate is light)
G2_HOST=84
if timeout 12 ssh -n -o BatchMode=yes -o ConnectTimeout=8 ubantu@10.229.20.84 'echo 84_ok'; then
  ssh -o BatchMode=yes -o ConnectTimeout=20 ubantu@10.229.20.84 \
    "mkdir -p $R84/experiments/aerial/scripts; \
     cp -f /tmp/run_geom_verify_arm.sh /tmp/wam_*.py $R84/experiments/aerial/scripts/; \
     chmod +x $R84/experiments/aerial/scripts/run_geom_verify_arm.sh"
  sync_tree "$R125/$GT_REL" ubantu@10.229.20.84 "$R84/$GT_REL" "-o BatchMode=yes -o ConnectTimeout=20"
  sync_tree "$R125/$DS_NEAR" ubantu@10.229.20.84 "$R84/$DS_NEAR" "-o BatchMode=yes -o ConnectTimeout=20" || true
  if ! ssh -n -o BatchMode=yes -o ConnectTimeout=12 ubantu@10.229.20.84 "test -f $R84/$WM_REL/wm_obs.pt"; then
    sync_tree "$R125/$WM_REL" ubantu@10.229.20.84 "$R84/$WM_REL" "-o BatchMode=yes -o ConnectTimeout=20"
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=20 ubantu@10.229.20.84 \
    "export STAMP='$STAMP' MAX_SAMPLES='$MAX_SAMPLES'; bash -s" <<'H84'
set -euo pipefail
ROOT=/data/aerial-wam-v2
pkill -9 -f train_v4_ac 2>/dev/null || true
pkill -9 -f run_mainchannel_c2 2>/dev/null || true
cd "$ROOT"; export PYTHONPATH="$ROOT" PYTHON_BIN=/data/venvs/sim_verify/bin/python
nohup env HOST=84 STAMP="$STAMP" ARM=G2 ROOT="$ROOT" MAX_SAMPLES="$MAX_SAMPLES" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash experiments/aerial/scripts/run_geom_verify_arm.sh >/dev/null 2>&1 &
echo G2_84_LAUNCH=$!
H84
else
  echo "WARN 84 unreachable — G2 falls back to 125 AFTER G1_DONE"
  G2_HOST=125
  (
    OUT="$R125/experiments/aerial/rl/artifacts/geom_verify_${STAMP}"
    for i in $(seq 1 240); do
      [[ -f "$OUT/G1_DONE.txt" || -f "$OUT/G1_FAIL.txt" ]] && break
      sleep 30
    done
    cd "$R125"; export PYTHONPATH="$R125"
    nohup env HOST=125 STAMP="$STAMP" ARM=G2 ROOT="$R125" MAX_SAMPLES="$MAX_SAMPLES" \
      bash experiments/aerial/scripts/run_geom_verify_arm.sh >/dev/null 2>&1 &
    echo G2_125_FALLBACK=$!
  ) &
fi
echo "G2_HOST=$G2_HOST"

launch_h100() {
  local PORT=$1 ARM=$2 HOSTN=$3
  local RDEST=/home/a25689/aerial-wam-v2
  local SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i $KEY -p $PORT"
  local SCP_OPTS="-o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i $KEY -P $PORT"
  echo "=== launch $ARM on $HOSTN :$PORT ==="
  # kill leftover
  ssh -n $SSH_OPTS a25689@10.239.121.$HOSTN \
    "pkill -9 -f train_v4_ac; pkill -9 -f run_mainchannel_c; true" || true
  # install scripts
  scp $SCP_OPTS \
    "$D"/run_geom_verify_arm.sh \
    "$D"/wam_obstacle_cost_cone_rank.py \
    "$D"/wam_latent_depth_probe.py \
    "$D"/wam_imagine_coll_rank.py \
    "a25689@10.239.121.$HOSTN:/tmp/"
  ssh -n $SSH_OPTS "a25689@10.239.121.$HOSTN" \
    "mkdir -p $RDEST/experiments/aerial/scripts; \
     cp -f /tmp/run_geom_verify_arm.sh /tmp/wam_*.py $RDEST/experiments/aerial/scripts/; \
     chmod +x $RDEST/experiments/aerial/scripts/run_geom_verify_arm.sh"
  # WM + datasets (skip if present)
  if ! ssh -n $SSH_OPTS "a25689@10.239.121.$HOSTN" "test -f $RDEST/$WM_REL/wm_obs.pt"; then
    sync_tree "$R125/$WM_REL" "a25689@10.239.121.$HOSTN" "$RDEST/$WM_REL" "$SSH_OPTS"
  fi
  if ! ssh -n $SSH_OPTS "a25689@10.239.121.$HOSTN" "test -d $RDEST/$DS_NEAR && ls $RDEST/$DS_NEAR|grep -q ."; then
    sync_tree "$R125/$DS_NEAR" "a25689@10.239.121.$HOSTN" "$RDEST/$DS_NEAR" "$SSH_OPTS"
  fi
  if ! ssh -n $SSH_OPTS "a25689@10.239.121.$HOSTN" "test -d $RDEST/$DS_MERGED && ls $RDEST/$DS_MERGED|grep -q ."; then
    sync_tree "$R125/$DS_MERGED" "a25689@10.239.121.$HOSTN" "$RDEST/$DS_MERGED" "$SSH_OPTS"
  fi
  ssh $SSH_OPTS "a25689@10.239.121.$HOSTN" \
    "export STAMP='$STAMP' MAX_SAMPLES='$MAX_SAMPLES' ARM='$ARM' HOSTN='$HOSTN'; bash -s" <<'HH'
set -euo pipefail
ROOT=/home/a25689/aerial-wam-v2
cd "$ROOT"; export PYTHONPATH="$ROOT"
PY=""
[[ -x $ROOT/.venv/bin/python ]] && PY=$ROOT/.venv/bin/python
nohup env HOST="$HOSTN" STAMP="$STAMP" ARM="$ARM" ROOT="$ROOT" MAX_SAMPLES="$MAX_SAMPLES" \
  ${PY:+PYTHON_BIN=$PY} \
  bash experiments/aerial/scripts/run_geom_verify_arm.sh >/dev/null 2>&1 &
echo "${ARM}_${HOSTN}_LAUNCH=$!"
HH
}

launch_h100 31126 G3 14
launch_h100 30627 G4 11

# local watch that only reports (no restart of C)
nohup bash -c "
OUT=$R125/experiments/aerial/rl/artifacts/geom_verify_${STAMP}
LOG=$R125/experiments/aerial/rl/artifacts/logs/watch_geom_verify_${STAMP}.log
while true; do
  {
    echo \"# geom_verify STATUS · ${STAMP}\"
    echo updated: \$(date -Is)
    echo
    for a in G1 G2 G3 G4; do
      if [[ -f \$OUT/\${a}_DONE.txt ]]; then st=DONE
      elif [[ -f \$OUT/\${a}_FAIL.txt ]]; then st=FAIL
      else st=RUNNING_OR_REMOTE; fi
      echo \"\$a \$st\"
    done
    ls -1 \$OUT 2>/dev/null | head -40
  } > \$OUT/STATUS.md
  # pull remote DONE markers best-effort
  scp -o BatchMode=yes -o ConnectTimeout=12 \
    ubantu@10.229.20.84:$R84/experiments/aerial/rl/artifacts/geom_verify_${STAMP}/G2_* \
    \$OUT/ 2>/dev/null || true
  scp -o BatchMode=yes -o ConnectTimeout=15 -i $KEY -P 31126 \
    a25689@10.239.121.14:/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/geom_verify_${STAMP}/G3_* \
    \$OUT/ 2>/dev/null || true
  scp -o BatchMode=yes -o ConnectTimeout=15 -i $KEY -P 30627 \
    a25689@10.239.121.11:/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/geom_verify_${STAMP}/G4_* \
    \$OUT/ 2>/dev/null || true
  n=0; for a in G1 G2 G3 G4; do [[ -f \$OUT/\${a}_DONE.txt || -f \$OUT/\${a}_FAIL.txt ]] && n=\$((n+1)); done
  echo \"[\$(date -Is)] done_markers=\$n/4\" >> \$LOG
  [[ \$n -eq 4 ]] && {
    $R125/.venv/bin/python $R125/experiments/aerial/scripts/summarize_geom_verify.py \
      --stamp $STAMP --root $R125 2>/dev/null \
    || /home/yao/sim_verify/.venv/bin/python $R125/experiments/aerial/scripts/summarize_geom_verify.py \
      --stamp $STAMP --root $R125
    echo ALL_GEOM_DONE >> \$OUT/ALL_DONE.txt
    exit 0
  }
  sleep 60
done
" >/dev/null 2>&1 &
echo WATCH_GEOM=$!
echo GEOM_LAUNCH_DONE stamp=$STAMP
