#!/usr/bin/env bash
# On 125: push 84's Soft latest to 14 and 11 only when the bytes change.
# 14 scores routes 0,1 and 11 scores routes 3,4, once per checksum.
# Does not train. Skips a host while its eval is reading the file.
set +e
STAMP="${STAMP:-20260926_soft}"
ROOT="${ROOT:-/home/yao/aerial-wam-v2}"
KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"
ART="$ROOT/experiments/aerial/rl/artifacts"
LOCAL="$ART/soft_assist_ckpt_${STAMP}"
SRC_HOST="${SRC_HOST:-ubantu@10.229.20.84}"
SRC="/data/aerial-wam-v2/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84/v4_ac_latest.pt"
DEST_REL="experiments/aerial/rl/artifacts/soft_assist_ckpt_${STAMP}"
LOG="$ART/logs/sync_soft_assist_${STAMP}.log"
mkdir -p "$LOCAL" "$ART/logs"
note() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

push_one() {
  local id="$1" port="$2" routes="$3"
  local host="a25689@10.239.121.${id}"
  local sshb=(ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i "$KEY" -p "$port" "$host")
  local scpb=(scp -o BatchMode=yes -o ConnectTimeout=60 -o StrictHostKeyChecking=accept-new -i "$KEY" -P "$port")
  local busy
  local have want
  want=$(cksum "$LOCAL/v4_ac_latest.pt" 2>/dev/null | awk '{print $1}')
  have=$("${sshb[@]}" "cksum /home/a25689/aerial-wam-v2/${DEST_REL}/v4_ac_latest.pt 2>/dev/null" | awk '{print $1}')
  if [[ -n "$want" && "$have" == "$want" ]]; then
    return 0
  fi
  busy=$("${sshb[@]}" "pgrep -f '[w]am_phase2_long_eval' >/dev/null && echo BUSY || echo IDLE" 2>/dev/null)
  if [[ "$busy" == *BUSY* ]]; then
    note "h${id} eval busy, defer ckpt"
    return 2
  fi
  "${sshb[@]}" "mkdir -p /home/a25689/aerial-wam-v2/${DEST_REL}" || return 1
  "${scpb[@]}" "$LOCAL/v4_ac_latest.pt" "${host}:/tmp/soft_assist_latest.pt" || return 1
  "${sshb[@]}" "mv -f /tmp/soft_assist_latest.pt /home/a25689/aerial-wam-v2/${DEST_REL}/v4_ac_latest.pt" || return 1
  note "pushed h${id} routes=${routes}"
}

ensure_gate() {
  local id="$1" port="$2" routes="$3"
  local host="a25689@10.239.121.${id}"
  local alive
  alive=$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" -p "$port" "$host" \
    "pgrep -f '[r]un_hard_noshield_gate' >/dev/null && echo UP || echo DOWN" 2>/dev/null)
  if [[ "$alive" == *UP* ]]; then
    return 0
  fi
  note "restart gate h${id} routes=${routes}"
  ssh -n -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" -p "$port" "$host" \
    "cd /home/a25689/aerial-wam-v2 && mkdir -p experiments/aerial/rl/artifacts/logs && \
     nohup env HOST=${id} STAMP=${STAMP} ROUTES=${routes} GATE_CFG=soft GATE_ONLY_CKPT_DIR=1 \
       CKPT_DIR=/home/a25689/aerial-wam-v2/${DEST_REL} INTERVAL=60 \
       PYTHON_BIN=/home/a25689/aerial-wam-v2/.venv/bin/python \
       bash experiments/aerial/scripts/run_hard_noshield_gate.sh \
       </dev/null >/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/soft_assist_h${id}_${STAMP}.out 2>&1 & echo PID=\$!"
}

note "sync soft assist start"
pending14=1
pending11=1
while true; do
  remote_sig=$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$SRC_HOST" "cksum '$SRC' 2>/dev/null" | awk '{print $1}')
  local_sig=$(cksum "$LOCAL/v4_ac_latest.pt" 2>/dev/null | awk '{print $1}')
  if [[ -n "$remote_sig" && "$remote_sig" != "$local_sig" ]]; then
    note "new soft ckpt sig=$remote_sig"
    if scp -o BatchMode=yes -o ConnectTimeout=60 "${SRC_HOST}:${SRC}" "$LOCAL/v4_ac_latest.pt.new"; then
      got=$(cksum "$LOCAL/v4_ac_latest.pt.new" | awk '{print $1}')
      if [[ "$got" == "$remote_sig" ]]; then
        mv -f "$LOCAL/v4_ac_latest.pt.new" "$LOCAL/v4_ac_latest.pt"
        pending14=1
        pending11=1
      else
        note "scp mismatch got=$got want=$remote_sig"
        rm -f "$LOCAL/v4_ac_latest.pt.new"
      fi
    else
      note "scp from 84 failed"
      rm -f "$LOCAL/v4_ac_latest.pt.new"
    fi
  fi
  if [[ -f "$LOCAL/v4_ac_latest.pt" && "$pending14" == 1 ]]; then
    push_one 14 31126 "0,1"
    [[ $? -eq 0 ]] && pending14=0
  fi
  if [[ -f "$LOCAL/v4_ac_latest.pt" && "$pending11" == 1 ]]; then
    push_one 11 30627 "3,4"
    [[ $? -eq 0 ]] && pending11=0
  fi
  if [[ -f "$LOCAL/v4_ac_latest.pt" ]]; then
    ensure_gate 14 31126 "0,1"
    ensure_gate 11 30627 "3,4"
  fi
  sleep "${SYNC_INTERVAL:-180}"
done
