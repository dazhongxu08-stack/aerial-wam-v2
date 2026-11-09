#!/usr/bin/env bash
set +e
STAMP=${STAMP:-20260926_soft}
KEY=${H100_KEY:-$HOME/.ssh/id_ed25519_h100}; [[ -f $KEY ]] || KEY=$HOME/.ssh/id_ed25519
CK84=/data/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84
LOCAL=/home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84
L125=/home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h125
R14=/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84
R125=/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h125
LOG=/home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/sync_soft_ckpt_${STAMP}.log
mkdir -p "$LOCAL" "$(dirname "$LOG")"
while true; do
  scp -o BatchMode=yes -o ConnectTimeout=15 \
    "ubantu@10.229.20.84:$CK84/v4_ac_best.pt" \
    "ubantu@10.229.20.84:$CK84/v4_ac_latest.pt" \
    "$LOCAL/" >/dev/null 2>&1
  if [[ -f $LOCAL/v4_ac_best.pt || -f $LOCAL/v4_ac_latest.pt ]]; then
    ssh -o BatchMode=yes -o ConnectTimeout=12 -i "$KEY" -p 31126 a25689@10.239.121.14 "mkdir -p $R14" 2>/dev/null
    scp -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" -P 31126 \
      "$LOCAL/v4_ac_best.pt" "$LOCAL/v4_ac_latest.pt" \
      "a25689@10.239.121.14:$R14/" >/dev/null 2>&1
    echo "[$(date -Is)] synced Soft@84 →14" >>"$LOG"
  fi
  if ls "$L125"/v4_ac_best.pt >/dev/null 2>&1; then
    ssh -o BatchMode=yes -o ConnectTimeout=12 -i "$KEY" -p 31126 a25689@10.239.121.14 "mkdir -p $R125" 2>/dev/null
    scp -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" -P 31126 \
      "$L125"/v4_ac_best.pt "$L125"/v4_ac_latest.pt \
      "a25689@10.239.121.14:$R125/" >/dev/null 2>&1
    echo "[$(date -Is)] synced Soft@125 →14" >>"$LOG"
  fi
  sleep 180
done
