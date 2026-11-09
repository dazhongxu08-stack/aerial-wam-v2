#!/usr/bin/env bash
# Closed-loop no-shield gate for Hard ckpts. Does not train.
# Pulls h14 Hard ckpts (local dir if already synced) and evals ROUTES.
#
#   HOST=125 STAMP=20260926_soft ROUTES=0,1,3,4 \
#     bash experiments/aerial/scripts/run_hard_noshield_gate.sh
set -uo pipefail
HOST="${HOST:?set HOST}"
STAMP="${STAMP:-20260926_soft}"
ROUTES="${ROUTES:-0,1,3,4}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PYTHON_BIN:-}"
for c in "${PYTHON_BIN:-}" "$ROOT/.venv/bin/python" "${HOME}/sim_verify/.venv/bin/python" /data/venvs/sim_verify/bin/python; do
  [[ -n "$c" && -x "$c" ]] && PY="$c" && break
done
PY="${PY:-python3}"

ART="$ROOT/experiments/aerial/rl/artifacts"
OUT="$ART/breakout_hard_${STAMP}"
CKPT_DIR="${CKPT_DIR:-$ART/v4_ac_ckpt_mainchannel_hard_${STAMP}_noshield_h14}"
SOFT_DIR="$ART/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h84"
GATE_CFG="${GATE_CFG:-hard}"
SEEN="$OUT/scored_h${HOST}.txt"
LOG="$ART/logs/hard_gate_${STAMP}_h${HOST}.log"
CFG="$ROOT/configs/_tmp_hard_gate_h${HOST}.yaml"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
WM="${WM:-$ART/wm_ckpt_soft_init_antisat_cost/wm_soft_init.pt}"
DEPTH="${DEPTH_CKPT:-$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU_CKPT:-$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
INTERVAL="${INTERVAL:-90}"
mkdir -p "$OUT" "$ART/logs" "$CKPT_DIR"
exec >>"$LOG" 2>&1

echo "[$(date -Is)] hard gate host=$HOST routes=$ROUTES stamp=$STAMP"
[[ -f "$WM" ]] || { echo "FATAL no WM $WM"; exit 1; }

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
name="hard" if "$GATE_CFG" == "hard" else "soft"
over=yaml.safe_load(Path(f"configs/aerial_rl_urban_complex_{name}_noshield.yaml").read_text())
cfg=deep_merge(deep_merge(base, oa), over)
cfg.setdefault("planner", {})["enable"]=False
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("hard gate cfg ok")
PY

last_sig=""
while true; do
  actor=""
  tag="hard"
  if [[ -f "$CKPT_DIR/v4_ac_latest.pt" ]]; then
    actor="$CKPT_DIR/v4_ac_latest.pt"
  elif [[ -f "$CKPT_DIR/v4_ac_best.pt" ]]; then
    actor="$CKPT_DIR/v4_ac_best.pt"
  elif [[ "${GATE_ONLY_CKPT_DIR:-0}" != "1" && -f "$SOFT_DIR/v4_ac_best.pt" ]]; then
    actor="$SOFT_DIR/v4_ac_best.pt"
    tag="soft_init"
  fi
  if [[ -z "$actor" ]]; then
    echo "[$(date -Is)] no actor yet"
    sleep "$INTERVAL"
    continue
  fi
  sig="$(cksum "$actor" | awk '{print $1}')"
  if [[ -f "$SEEN" ]] && grep -qx "$sig" "$SEEN"; then
    sleep "$INTERVAL"
    continue
  fi
  eout="$OUT/gate_h${HOST}_${tag}_$(date +%H%M%S)"
  mkdir -p "$eout/traj"
  echo "[$(date -Is)] EVAL $tag actor=$actor routes=$ROUTES"
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$actor" \
    --depth-ckpt "$DEPTH" --tau-ckpt "$TAU" --goal-feat-mode meter \
    --routes "$ROUTES" --traj-out "$eout/traj" --out "$eout/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10 --max-steps 600 --success-dist 3 --terminal-pin-rem-m 20 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5 \
    --no-shield
  rc=$?
  set -e
  echo "[$(date -Is)] EVAL_RC=$rc out=$eout/eval_all.json"
  TAG="$tag" JSON="$eout/eval_all.json" HOST="$HOST" "$PY" - <<'PY' || true
import json, os
from pathlib import Path
p=Path(os.environ["JSON"])
if not p.exists():
    print("[gate] MISSING json"); raise SystemExit(0)
rows=json.loads(p.read_text()).get("episodes") or []
arr=sum(1 for r in rows if r.get("arrived"))
stick=sum(1 for r in rows if int(r.get("route_idx",-1)) in (0,1) and r.get("severe_collision"))
print(f"[h{os.environ['HOST']} {os.environ['TAG']}] SR={arr}/{len(rows)} P0_stick={stick}")
PY
  echo "$sig" >>"$SEEN"
  sleep "$INTERVAL"
done
