#!/usr/bin/env bash
# R3@14 — near WM FT until probe_hard PASS (n≥30, R²≥0.3). Strict READY_SOFT contract.
set -euo pipefail
HOST="${HOST:-14}"
STAMP="${STAMP:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
OUT="$ART/geomfix_${STAMP}"
LOG="$ART/logs/geomfix_R3_${STAMP}_h${HOST}.log"
mkdir -p "$OUT" "$ART/logs"
exec > >(tee -a "$LOG") 2>&1

PY="${PYTHON_BIN:-}"
[[ -z "$PY" && -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
[[ -z "$PY" ]] && PY=python3

INIT="${INIT_CKPT:-$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
[[ -f "$INIT" ]] || INIT="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
# Prefer thickened near pack; fall back to three_zone + enrich merge path
DS="${DS:-$ART/dataset_geomfix_nearthick_${STAMP}}"
if [[ ! -d "$DS" ]] || [[ -z "$(ls -A "$DS"/*.npz 2>/dev/null || true)" ]]; then
  DS="$ART/dataset_v0_three_zone_near_20260823fg"
fi
DS_PROBE="${DS_PROBE:-$DS}"
CKPT_DIR="$ART/wm_ckpt_geomfix_near_${STAMP}_h${HOST}"
STEPS="${WM_STEPS:-2500}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"
CFG_BASE=configs/aerial_rl.yaml
CFG_OVER=configs/aerial_rl_geomfix_near_wm.yaml

echo "=== R3 hardnear FT stamp=$STAMP DS=$DS $(date -Is) ==="
test -f "$INIT"
test -d "$DS"

TMP_CFG="$OUT/R3_wm_cfg_h${HOST}.yaml"
"$PY" - <<PY
import yaml
from pathlib import Path
base=yaml.safe_load(Path("$CFG_BASE").read_text()) or {}
over=yaml.safe_load(Path("$CFG_OVER").read_text()) or {}
wm=dict(base.get("world_model") or {})
ov=dict(over.get("world_model") or {})
ls=dict(wm.get("loss_scales") or {})
ls.update(dict(ov.pop("loss_scales", None) or {}))
wm.update(ov)
wm["loss_scales"]=ls
wm["checkpoint_dir"]="$CKPT_DIR"
base["world_model"]=wm
Path("$TMP_CFG").write_text(yaml.safe_dump(base, sort_keys=False))
print("wrote", "$TMP_CFG")
PY

probe_ok() {
  local jf=$1
  "$PY" - <<PY
import json
from pathlib import Path
p=Path("$jf")
if not p.exists():
  print(0); raise SystemExit
j=json.loads(p.read_text())
fwd=j.get("forward_min_depth") or {}
n=int(j.get("n_samples") or 0)
ok=(fwd.get("verdict")=="has_geometry"
    and float(fwd.get("r2_holdout") or -1)>=0.3
    and n>=30)
print(1 if ok else 0)
print("n", n, "r2", fwd.get("r2_holdout"), "verdict", fwd.get("verdict"))
PY
}

CKPT="$INIT"
TOTAL=0
PASS=0
for round in $(seq 1 "$MAX_ROUNDS"); do
  echo "=== R3 FT round $round/$MAX_ROUNDS init=$CKPT ==="
  "$PY" -m experiments.aerial.rl._wm_train_validate \
    --dataset "$DS" \
    --config "$TMP_CFG" \
    --steps "$STEPS" \
    --wm-batch 16 --window 8 --horizon 15 \
    --device cuda --allow-v0-desync \
    --init-ckpt "$CKPT" --save-ckpt \
    --checkpoint-dir "$CKPT_DIR" \
    --save-step $((TOTAL + STEPS)) || true
  TOTAL=$((TOTAL + STEPS))
  CKPT=$(ls -t "$CKPT_DIR"/wm_step_*.pt 2>/dev/null | head -1 || true)
  [[ -n "${CKPT:-}" ]] || { echo FAIL >"$OUT/R3_FAIL.txt"; exit 2; }
  echo "$CKPT" | tee "$OUT/R3_WM_CKPT.txt"

  PROBE="$OUT/R3_latent_probe_three_zone_r${round}.json"
  "$PY" experiments/aerial/scripts/wam_latent_depth_probe.py \
    --dataset "$DS_PROBE" --wm-ckpt "$CKPT" --device cuda \
    --max-samples 800 --stride 1 --window 8 --max-center-depth-m 14.0 \
    --out "$PROBE" || true
  cp -f "$PROBE" "$OUT/R3_latent_probe_three_zone.json" 2>/dev/null || true
  cp -f "$PROBE" "$OUT/R3_latent_probe_three_zone_v2.json" 2>/dev/null || true

  OK=$(probe_ok "$PROBE" | head -1)
  echo "probe_ok=$OK"
  if [[ "$OK" == "1" ]]; then
    PASS=1
    echo "PROBE_HARD_PASS round=$round" | tee "$OUT/R3_PROBE_PASS.txt"
    break
  fi
done

if [[ "$PASS" -eq 1 ]]; then
  echo "GEOMFIX_R3_DONE $(date -Is)" | tee "$OUT/R3_DONE.txt"
  rm -f "$OUT/R3_FAIL.txt"
  exit 0
fi
echo "R3_PROBE_FAIL after $MAX_ROUNDS rounds" | tee "$OUT/R3_FAIL.txt"
# Do NOT leave a DONE marker on failure — auto_advance will relaunch.
rm -f "$OUT/R3_DONE.txt"
exit 2
