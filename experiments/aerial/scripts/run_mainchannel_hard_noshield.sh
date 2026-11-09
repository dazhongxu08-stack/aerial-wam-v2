#!/usr/bin/env bash
# Hard L-main · noshield · terminate_on_collision=true · w_collision=2 (OA full).
# INIT = Soft best. Cost head stays frozen (WM not updated in train_v4_ac).
# Bound iters at 80 so this does not repeat the +160 collapse run.
#
#   HOST=14 STAMP=20260926_soft bash experiments/aerial/scripts/run_mainchannel_hard_noshield.sh
set -euo pipefail
HOST="${HOST:?set HOST}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PYTHON_BIN:-}"
for c in "${PYTHON_BIN:-}" "$ROOT/.venv/bin/python" "${HOME}/sim_verify/.venv/bin/python" /data/venvs/sim_verify/bin/python; do
  [[ -n "$c" && -x "$c" ]] && PY="$c" && break
done
PY="${PY:-python3}"

ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-20260926_soft}"
ITERS="${ITERS:-80}"
EPI_PER_ITER="${EPI_PER_ITER:-4}"
NEAR_GOAL_FRAC="${NEAR_GOAL_FRAC:-0.25}"
IMAGINE_HORIZON="${IMAGINE_HORIZON:-40}"
IMAGINE_BATCH="${IMAGINE_BATCH:-16}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ANN_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
OUT="${OUT:-$ART/v4_ac_ckpt_mainchannel_hard_${STAMP}_noshield_h${HOST}}"
MASTER="$ART/logs/mainchannel_hard_${STAMP}_h${HOST}.log"
CFG="$ROOT/configs/_tmp_mainchannel_hard_h${HOST}.yaml"
mkdir -p "$ART/logs" "$OUT"
exec >>"$MASTER" 2>&1

echo "=== Hard noshield HOST=$HOST stamp=$STAMP iters=$ITERS ==="
hostname; date -Is
echo "POLICY: terminate_on_collision=true; w_collision=2; --no-planner --no-shield; freeze cost"

READY="$ART/breakout_soft_${STAMP}/READY_HARD.txt"
if [[ ! -f "$READY" ]]; then
  echo "FATAL no READY_HARD at $READY"
  exit 2
fi
echo "READY_HARD $(cat "$READY")"

pick_actor() {
  local d=$1
  [[ -f "$d/v4_ac_best.pt" ]] && { echo "$d/v4_ac_best.pt"; return 0; }
  [[ -f "$d/v4_ac_latest.pt" ]] && { echo "$d/v4_ac_latest.pt"; return 0; }
  return 1
}

INIT="${INIT:-}"
if [[ -z "$INIT" ]]; then
  INIT="$(pick_actor "$ART/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84" || true)"
fi
[[ -f "${INIT:-}" ]] || { echo FATAL no Soft INIT; exit 1; }

WM="${WM:-$ART/wm_ckpt_soft_init_antisat_cost/wm_soft_init.pt}"
[[ -f "$WM" ]] || { echo "FATAL missing Soft WM $WM"; exit 1; }
echo "INIT=$INIT WM=$WM"

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
hard=yaml.safe_load(Path("configs/aerial_rl_urban_complex_hard_noshield.yaml").read_text())
cfg=deep_merge(deep_merge(base, oa), hard)
cfg.setdefault("planner", {})["enable"]=False
r=cfg.get("reward") or {}
assert r.get("terminate_on_collision") is True, r
assert float(r.get("w_collision") or 0) == 2.0, r
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("cfg Hard OK", {k:r.get(k) for k in ("terminate_on_collision","w_collision","w_press_wall","use_learned_obstacle_cost")})
PY

echo "=== TRAIN Hard iters=$ITERS ==="
"$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
  --src "$ANN" --out "$ANN_FOCUS" --routes 0,1,3,4 || true
[[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 8; }
"$PY" -m experiments.aerial.rl.train_v4_ac \
  --config "$CFG" \
  --backend airsim --dynamics torch \
  --wm-ckpt "$WM" --init-actor-ckpt "$INIT" \
  --no-planner --no-shield \
  --iters "$ITERS" --episodes-per-iter "$EPI_PER_ITER" \
  --imagine-batch "$IMAGINE_BATCH" --imagine-horizon "$IMAGINE_HORIZON" \
  --device cuda --annotation "$ANN_FOCUS" \
  --ckpt-dir "$OUT" --save-every-iter \
  --min-steps-for-best 20 --near-goal-frac "$NEAR_GOAL_FRAC" \
  --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 6 \
  --min-spawn-clear-m 5.0
echo "HARD_NOSHIELD_DONE stamp=$STAMP out=$OUT"
date -Is
