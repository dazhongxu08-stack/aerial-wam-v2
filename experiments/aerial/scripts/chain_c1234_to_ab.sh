#!/usr/bin/env bash
# Wait for C1234 ALL_DONE, snapshot C gates, then launch A/B matrix.
# Runs on 125. Do NOT kill C while waiting.
#
#   C_STAMP=20260925_2100 AB_STAMP=20260926_after_c bash experiments/aerial/scripts/chain_c1234_to_ab.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
C_STAMP="${C_STAMP:-20260925_2100}"
AB_STAMP="${AB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
ITERS="${ITERS:-100}"
BC_ITERS="${BC_ITERS:-40}"
ART="$ROOT/experiments/aerial/rl/artifacts"
LOG="$ART/logs/chain_c1234_to_ab_${C_STAMP}_to_${AB_STAMP}.log"
DONE="$ART/logs/c1234_oa_terminal_${C_STAMP}_ALL_DONE.txt"
mkdir -p "$ART/logs"
exec >>"$LOG" 2>&1
echo "$$" >"$ART/logs/chain_c1234_to_ab.pid"
echo "[$(date -Is)] chain wait C_STAMP=$C_STAMP → AB_STAMP=$AB_STAMP"

KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"

while [[ ! -f "$DONE" ]]; do
  # also accept if all four DONE markers present
  ok=0
  grep -q C1_OL_BC_NOSHIELD_DONE "$ART/logs/mainchannel_c1_ol_bc_${C_STAMP}_h125.log" 2>/dev/null && ok=$((ok+1)) || true
  ssh -n -o BatchMode=yes -o ConnectTimeout=12 ubantu@10.229.20.84 \
    "grep -q C2_WM_ESCAPE_NOSHIELD_DONE /data/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/mainchannel_c2_wm_escape_${C_STAMP}_h84.log 2>/dev/null" \
    && ok=$((ok+1)) || true
  ssh -n -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" -p 31126 a25689@10.239.121.14 \
    "grep -q C3_ANTISTICK_NOSHIELD_DONE /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/mainchannel_c3_antistick_${C_STAMP}_h14.log 2>/dev/null" \
    && ok=$((ok+1)) || true
  ssh -n -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" -p 30627 a25689@10.239.121.11 \
    "grep -q C4_TERMINAL_NOSHIELD_DONE /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/mainchannel_c4_terminal_${C_STAMP}_h11.log 2>/dev/null" \
    && ok=$((ok+1)) || true
  echo "[$(date -Is)] waiting C done markers=$ok/4"
  if [[ "$ok" -eq 4 ]]; then
    echo C1234_ALL_DONE >"$DONE"
    break
  fi
  sleep 180
done

echo "[$(date -Is)] C round complete — snapshot gates"
SNAP="$ART/logs/c1234_gate_snapshot_${C_STAMP}.md"
{
  echo "# C1234 gate snapshot · $C_STAMP"
  echo "generated: $(date -Is)"
  echo
} >"$SNAP"
for p in \
  "$ART/eval_mainchannel_c1_ol_bc_${C_STAMP}_h125_gate_noshield/CAPABILITY_SUMMARY.json"
 do
  if [[ -f "$p" ]]; then
    echo "## C1" >>"$SNAP"
    cat "$p" >>"$SNAP"
    echo >>"$SNAP"
  fi
 done
ssh -n -o BatchMode=yes ubantu@10.229.20.84 \
  "cat /data/aerial-wam-v2/experiments/aerial/rl/artifacts/eval_mainchannel_c2_wm_escape_${C_STAMP}_h84_gate_noshield/CAPABILITY_SUMMARY.json 2>/dev/null" \
  >>"$SNAP" || true
echo >>"$SNAP"
ssh -n -o BatchMode=yes -i "$KEY" -p 31126 a25689@10.239.121.14 \
  "cat /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/eval_mainchannel_c3_antistick_${C_STAMP}_h14_gate_noshield/CAPABILITY_SUMMARY.json 2>/dev/null" \
  >>"$SNAP" || true
echo >>"$SNAP"
ssh -n -o BatchMode=yes -i "$KEY" -p 30627 a25689@10.239.121.11 \
  "cat /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/eval_mainchannel_c4_terminal_${C_STAMP}_h11_gate_noshield/CAPABILITY_SUMMARY.json 2>/dev/null" \
  >>"$SNAP" || true
echo "[$(date -Is)] snapshot → $SNAP"

# Ensure AB payloads
if [[ ! -f /tmp/ab1234_deploy/ab1234_launch_on_125.sh ]]; then
  echo "WARN /tmp/ab1234_deploy missing launcher — expect deploy from Mac already staged"
fi
export STAMP="$AB_STAMP" ITERS="$ITERS" BC_ITERS="$BC_ITERS"
bash /tmp/ab1234_deploy/ab1234_launch_on_125.sh
echo "[$(date -Is)] CHAIN_TO_AB_DONE c=$C_STAMP ab=$AB_STAMP"
