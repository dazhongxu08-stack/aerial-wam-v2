#!/usr/bin/env bash
# Stage AB payloads on 125 and arm chain (does not stop C round).
#   C_STAMP=20260925_2100 bash experiments/aerial/scripts/arm_chain_c_to_ab.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SSH125="${SSH125:-cursor-125-public}"
C_STAMP="${C_STAMP:-20260925_2100}"
AB_STAMP="${AB_STAMP:-$(date +%Y%m%d)_after_c}"
ITERS="${ITERS:-100}"
BC_ITERS="${BC_ITERS:-40}"
cd "$ROOT"

PAYLOAD=(
  configs/aerial_rl_urban_complex_a2_antistick_strong.yaml
  configs/aerial_rl_urban_complex_antistick_reward.yaml
  configs/aerial_rl_urban_complex_directional_oa.yaml
  experiments/aerial/scripts/run_mainchannel_a1_antistick_noshield.sh
  experiments/aerial/scripts/run_mainchannel_a2_antistick_strong_noshield.sh
  experiments/aerial/scripts/run_mainchannel_b1_pathexpert_bc_noshield.sh
  experiments/aerial/scripts/run_mainchannel_b2_pathexpert_deprivilege_noshield.sh
  experiments/aerial/scripts/ab1234_launch_on_125.sh
  experiments/aerial/scripts/chain_c1234_to_ab.sh
  docs/superpowers/plans/2026-09-25-ab-after-c-round.md
)

ssh -o ConnectTimeout=25 -o BatchMode=yes "$SSH125" 'mkdir -p /tmp/ab1234_deploy'
rsync -avz "${PAYLOAD[@]}" "$SSH125:/tmp/ab1234_deploy/"

# Stage a small arm script and run it (ssh -n: no stdin heredoc)
cat > /tmp/arm_chain_remote.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
C_STAMP='$C_STAMP'
AB_STAMP='$AB_STAMP'
ITERS='$ITERS'
BC_ITERS='$BC_ITERS'
R=/home/yao/aerial-wam-v2
D=/tmp/ab1234_deploy
cp -f "\$D"/chain_c1234_to_ab.sh "\$D"/ab1234_launch_on_125.sh "\$R/experiments/aerial/scripts/"
cp -f "\$D"/run_mainchannel_a*.sh "\$D"/run_mainchannel_b*.sh "\$R/experiments/aerial/scripts/"
cp -f "\$D"/*.yaml "\$R/configs/" 2>/dev/null || true
cp -f "\$D"/2026-09-25-ab-after-c-round.md "\$R/docs/superpowers/plans/" 2>/dev/null || true
chmod +x "\$R/experiments/aerial/scripts"/chain_c1234_to_ab.sh \
         "\$R/experiments/aerial/scripts"/ab1234_launch_on_125.sh \
         "\$R/experiments/aerial/scripts"/run_mainchannel_a*.sh \
         "\$R/experiments/aerial/scripts"/run_mainchannel_b*.sh
pkill -f 'chain_c1234_to_ab' 2>/dev/null || true
sleep 1
cd "\$R"
nohup env C_STAMP="\$C_STAMP" AB_STAMP="\$AB_STAMP" ITERS="\$ITERS" BC_ITERS="\$BC_ITERS" \
  bash experiments/aerial/scripts/chain_c1234_to_ab.sh >/dev/null 2>&1 &
echo CHAIN_PID=\$!
echo "armed C=\$C_STAMP → AB=\$AB_STAMP"
pgrep -af chain_c1234_to_ab | grep -v pgrep | head -2
pgrep -af 'run_mainchannel_c1|train_v4_ac.*c1_' | grep -v pgrep | head -1 || echo "C1 check: not in pgrep"
tail -3 "\$R/experiments/aerial/rl/artifacts/logs/chain_c1234_to_ab_\${C_STAMP}_to_\${AB_STAMP}.log" 2>/dev/null || true
EOF
scp -o BatchMode=yes /tmp/arm_chain_remote.sh "$SSH125:/tmp/arm_chain_remote.sh"
ssh -n -o ConnectTimeout=25 -o BatchMode=yes "$SSH125" 'bash /tmp/arm_chain_remote.sh'

mkdir -p "$ROOT/artifacts"
cat > "$ROOT/artifacts/c_to_ab_chain_ARMED.md" <<EOF
# Chain armed

- C round stamp: \`$C_STAMP\` (let finish)
- AB stamp after C: \`$AB_STAMP\`
- A1@125 spontaneous antistick
- A2@84 strong antistick
- B1@14 PathExpert→BC
- B2@11 PathExpert→deprivilege toward_g

C is **not** stopped. Chain waits for ALL_DONE then deploys AB.
EOF
echo "MAC_ARM_DONE c=$C_STAMP ab=$AB_STAMP"
