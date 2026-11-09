#!/usr/bin/env bash
# On 125: pull Hard@14 checkpoints and push them to the 84 gate.
set -uo pipefail
STAMP="${STAMP:-20260926_soft}"
ROOT="${ROOT:-/home/yao/aerial-wam-v2}"
ART="$ROOT/experiments/aerial/rl/artifacts"
DEST="$ART/v4_ac_ckpt_mainchannel_hard_${STAMP}_noshield_h14"
KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"
[[ -f "$KEY" ]] || KEY="$HOME/.ssh/id_ed25519"
mkdir -p "$DEST" "$ART/logs"
LOG="$ART/logs/sync_hard_ckpt_${STAMP}.log"
echo "[$(date -Is)] sync hard ckpt start" >>"$LOG"
ssh -o BatchMode=yes -o ConnectTimeout=15 ubantu@10.229.20.84 \
  "mkdir -p /data/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_hard_${STAMP}_noshield_h14" >>"$LOG" 2>&1 || true
while true; do
  scp -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" -P 31126 \
    "a25689@10.239.121.14:/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_hard_${STAMP}_noshield_h14/v4_ac_latest.pt" \
    "$DEST/v4_ac_latest.pt" >>"$LOG" 2>&1 || true
  scp -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" -P 31126 \
    "a25689@10.239.121.14:/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_hard_${STAMP}_noshield_h14/v4_ac_best.pt" \
    "$DEST/v4_ac_best.pt" >>"$LOG" 2>&1 || true
  if [[ -f "$DEST/v4_ac_latest.pt" ]]; then
    scp -o BatchMode=yes -o ConnectTimeout=20 \
      "$DEST/v4_ac_latest.pt" "$DEST/v4_ac_best.pt" \
      ubantu@10.229.20.84:/data/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_hard_${STAMP}_noshield_h14/ \
      >>"$LOG" 2>&1 || true
  fi
  sleep 120
done
