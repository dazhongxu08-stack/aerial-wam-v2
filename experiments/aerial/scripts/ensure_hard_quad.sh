#!/usr/bin/env bash
# One Hard only, on 125 (faster than 14/11). No eval loop, no extra copies.
# 14, 11, and 84 stay stopped. Restart resumes 125's own latest ckpt.
set +e
STAMP="${STAMP:-20260926_soft}"
ROOT="${ROOT:-/home/yao/aerial-wam-v2}"
LOG="$ROOT/experiments/aerial/rl/artifacts/logs/ensure_hard_quad_${STAMP}.log"
mkdir -p "$(dirname "$LOG")"

note() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

note "ensure single Hard@125"
cd "$ROOT"
if [[ -f "$ROOT/experiments/aerial/rl/artifacts/breakout_soft_${STAMP}/ACCEPT_N3_HOLD" ]]; then
  echo "HOLD accept n3"
  exit 0
fi
if pgrep -f '[t]rain_v4_ac' >/dev/null || pgrep -f '[r]un_mainchannel_hard_noshield' >/dev/null; then
  echo "UP h125"
  exit 0
fi
ART=experiments/aerial/rl/artifacts
HARD="$ART/v4_ac_ckpt_mainchannel_hard_${STAMP}_noshield_h125"
SOFT="$ART/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84/v4_ac_best.pt"
WM="$ART/wm_ckpt_soft_init_antisat_cost/wm_soft_init.pt"
INIT="$SOFT"
[[ -f "$HARD/v4_ac_latest.pt" ]] && INIT="$HARD/v4_ac_latest.pt"
[[ -f "$WM" && -f "$INIT" ]] || { echo "NOINIT h125"; exit 2; }
PY="$HOME/sim_verify/.venv/bin/python"
[[ -x "$PY" ]] || PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT"
nohup env HOST=125 STAMP="$STAMP" ITERS=80 INIT="$INIT" WM="$WM" PYTHON_BIN="$PY" \
  bash experiments/aerial/scripts/run_mainchannel_hard_noshield.sh \
  >"$ART/logs/hard125_${STAMP}.out" 2>&1 &
echo "RESTART Hard@125 pid=$! init=$INIT"
