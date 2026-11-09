#!/usr/bin/env bash
# Soft positive-example gate watcher @14 — NO Hard until K positives.
# Pulls Soft ckpts from 84 (or local OUT), runs noshield eval, counts R0/R1
# progress≥0.5 & not severe_collision. Writes READY_HARD.txt when K met.
#
#   HOST=14 STAMP=20260926_soft SOFT_HOST=84 K=8 \
#     bash experiments/aerial/scripts/run_soft_positive_gate_watch.sh
set -uo pipefail
HOST="${HOST:-14}"
STAMP="${STAMP:?}"
K="${SOFT_POS_K:-8}"
INTERVAL="${INTERVAL:-300}"
# Pull Soft ckpts from Soft host (84) so gate can fire without waiting for local train.
SOFT_REMOTE="${SOFT_REMOTE:-ubantu@10.229.20.84}"
SOFT_REMOTE_ROOT="${SOFT_REMOTE_ROOT:-/data/aerial-wam-v2}"
SOFT_REMOTE_CKPT_GLOB="${SOFT_REMOTE_CKPT_GLOB:-experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
OUT="$ART/breakout_soft_${STAMP}"
mkdir -p "$OUT" "$ART/logs"
LOG="$ART/logs/soft_positive_gate_${STAMP}_h${HOST}.log"
exec >>"$LOG" 2>&1

PY="${PYTHON_BIN:-}"
[[ -z "$PY" && -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
[[ -z "$PY" ]] && PY=python3

echo "[$(date -Is)] soft positive gate watch K=$K stamp=$STAMP host=$HOST"
LOCK="$OUT/gate_pos.lock"

while true; do
  if [[ -f "$OUT/READY_HARD.txt" ]]; then
    echo "[$(date -Is)] READY_HARD already — idle"
    sleep 300
    continue
  fi
  # Shared cum across parallel gate hosts (11+14 share ART).
  cum=0
  [[ -f "$OUT/SOFT_POS_COUNT.txt" ]] && cum=$(tr -d '[:space:]' <"$OUT/SOFT_POS_COUNT.txt" || echo 0)
  [[ "$cum" =~ ^[0-9]+$ ]] || cum=0

  # Prefer Soft ckpt dirs (refresh from Soft host each cycle; hub sync is fallback)
  mkdir -p "$ART/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84"
  scp -o BatchMode=yes -o ConnectTimeout=20 \
    "$SOFT_REMOTE:$SOFT_REMOTE_ROOT/$SOFT_REMOTE_CKPT_GLOB/v4_ac_best.pt" \
    "$ART/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84/" 2>/dev/null || \
  scp -o BatchMode=yes -o ConnectTimeout=20 \
    "$SOFT_REMOTE:$SOFT_REMOTE_ROOT/$SOFT_REMOTE_CKPT_GLOB/v4_ac_latest.pt" \
    "$ART/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84/" 2>/dev/null || true

  CKPT=""
  for d in \
    "$ART/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84" \
    "$ART/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h125" \
    $(ls -dt "$ART"/v4_ac_ckpt_mainchannel_soft_*_noshield_h* 2>/dev/null | head -3); do
    [[ -f "$d/v4_ac_best.pt" ]] && CKPT="$d/v4_ac_best.pt" && break
    [[ -f "$d/v4_ac_latest.pt" ]] && CKPT="$d/v4_ac_latest.pt" && break
  done
  if [[ -z "$CKPT" ]]; then
    echo "[$(date -Is)] no Soft ckpt yet; sleep $INTERVAL"
    sleep "$INTERVAL"
    continue
  fi
  echo "[$(date -Is)] h${HOST} eval Soft ckpt=$CKPT cum=$cum/$K"
  EOUT="$OUT/gate_eval_h${HOST}_$(date +%H%M%S)_$$"
  mkdir -p "$EOUT"
  CFG="$ROOT/configs/_tmp_soft_gate_h${HOST}.yaml"
  "$PY" - <<PY
import yaml
from pathlib import Path
def deep_merge(a,b):
    out=dict(a or {})
    for k,v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k]=deep_merge(out[k], v)
        else:
            out[k]=v
    return out
base=yaml.safe_load(Path("configs/aerial_rl_urban_complex_p2c.yaml").read_text())
oa=yaml.safe_load(Path("configs/aerial_rl_urban_complex_directional_oa.yaml").read_text())
soft=yaml.safe_load(Path("configs/aerial_rl_urban_complex_soft_noshield.yaml").read_text())
Path("$CFG").write_text(yaml.safe_dump(deep_merge(deep_merge(base, oa), soft), sort_keys=False))
PY
  # Match Soft train WM (antisat+cost). Never prefer bare gatefix wm_obs.
  WM="$ART/wm_ckpt_soft_init_antisat_cost/wm_soft_init.pt"
  if [[ ! -f "$WM" ]]; then
    WM="$ART/wm_ckpt_geomfix_strong_20260926_hardnear_h11/wm_step_2500.pt"
  fi
  if [[ ! -f "$WM" ]]; then
    echo "[$(date -Is)] FATAL no Soft/antisat WM for gate eval" | tee -a "$LOG"
    sleep "$INTERVAL"
    continue
  fi
  DEPTH="$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"
  TAU="$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt"
  ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$CKPT" \
    --depth-ckpt "$DEPTH" --tau-ckpt "$TAU" --goal-feat-mode meter \
    --routes 0,1 --traj-out "$EOUT/traj" --out "$EOUT/eval.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10 --max-steps 600 --success-dist 3 \
    --min-spawn-z 24 --min-spawn-clear-m 5 --no-shield \
    >"$EOUT/eval.log" 2>&1
  set -e
  ADD=$("$PY" - <<PY
import json
from pathlib import Path
p=Path("$EOUT/eval.json")
if not p.exists():
  print(0); raise SystemExit
rows=json.loads(p.read_text()).get("episodes") or []
by={int(r["route_idx"]):r for r in rows}
n=0
for ri in (0,1):
  r=by.get(ri) or {}
  d0=float(r.get("d0_m") or r.get("d_start_m") or 0) or 0.0
  df=float(r.get("d_final_m") or 0)
  prog=float(r.get("progress_frac") or ((d0-df)/d0 if d0>1e-3 else 0.0))
  if prog>=0.5 and not r.get("severe_collision"):
    n+=1
print(n)
PY
)
  # Atomic shared counter (parallel hosts 11+14).
  (
    flock -x 9
    cur=0
    [[ -f "$OUT/SOFT_POS_COUNT.txt" ]] && cur=$(tr -d '[:space:]' <"$OUT/SOFT_POS_COUNT.txt" || echo 0)
    [[ "$cur" =~ ^[0-9]+$ ]] || cur=0
    cum=$((cur + ADD))
    echo "[$(date -Is)] h${HOST} +$ADD → cum=$cum/$K" | tee -a "$OUT/SOFT_POS_CUM.txt"
    echo "$cum" >"$OUT/SOFT_POS_COUNT.txt"
    if [[ "$cum" -ge "$K" && ! -f "$OUT/READY_HARD.txt" ]]; then
      echo "READY_HARD cum=$cum K=$K ckpt=$CKPT host=$HOST $(date -Is)" | tee "$OUT/READY_HARD.txt"
      ssh -o BatchMode=yes -o ConnectTimeout=15 "$SOFT_REMOTE" \
        "mkdir -p $SOFT_REMOTE_ROOT/experiments/aerial/rl/artifacts/breakout_soft_${STAMP}" 2>/dev/null || true
      scp -o BatchMode=yes -o ConnectTimeout=15 "$OUT/READY_HARD.txt" \
        "$SOFT_REMOTE:$SOFT_REMOTE_ROOT/experiments/aerial/rl/artifacts/breakout_soft_${STAMP}/READY_HARD.txt" 2>/dev/null || true
      echo "Soft gate PASS — READY_HARD synced to Soft host for early-stop; Hard@14 may start"
    fi
  ) 9>"$LOCK"
  if [[ -f "$OUT/READY_HARD.txt" ]]; then
    break
  fi
  sleep "$INTERVAL"
done
