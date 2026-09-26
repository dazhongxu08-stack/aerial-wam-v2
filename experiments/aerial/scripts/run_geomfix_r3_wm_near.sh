#!/usr/bin/env bash
# R3@14 — near-field WM FT (depth-aux + coll hinge) then B′-1 probe.
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
DS="${DS:-$ART/dataset_v0_three_zone_near_20260823fg}"
DS2="${DS2:-$ART/dataset_v0_p45_merged_20260821}"
CKPT_DIR="$ART/wm_ckpt_geomfix_near_${STAMP}_h${HOST}"
STEPS="${WM_STEPS:-1500}"
CFG_BASE=configs/aerial_rl.yaml
CFG_OVER=configs/aerial_rl_geomfix_near_wm.yaml

echo "=== R3 WM near FT stamp=$STAMP $(date -Is) ==="
echo "INIT=$INIT DS=$DS"
test -f "$INIT"
test -d "$DS"

# Merge overlay keys into a temp yaml for _wm_train_validate (single --config)
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

set +e
"$PY" -m experiments.aerial.rl._wm_train_validate \
  --dataset "$DS" \
  --config "$TMP_CFG" \
  --steps "$STEPS" \
  --wm-batch 16 --window 8 --horizon 15 \
  --device cuda \
  --allow-v0-desync \
  --init-ckpt "$INIT" \
  --save-ckpt \
  --checkpoint-dir "$CKPT_DIR" \
  --save-step $((STEPS))
RC=$?
set -e
echo "wm_train_rc=$RC"

# pick latest ckpt
CKPT=$(ls -t "$CKPT_DIR"/wm_step_*.pt 2>/dev/null | head -1 || true)
if [[ -z "${CKPT:-}" ]]; then
  echo "FATAL no wm ckpt in $CKPT_DIR"
  echo FAIL >"$OUT/R3_FAIL.txt"
  exit 2
fi
echo "$CKPT" | tee "$OUT/R3_WM_CKPT.txt"

# B′-1 probes
"$PY" experiments/aerial/scripts/wam_latent_depth_probe.py \
  --dataset "$DS" --wm-ckpt "$CKPT" --device cuda \
  --max-samples 400 --stride 2 --window 8 \
  --out "$OUT/R3_latent_probe_three_zone.json" || true
"$PY" experiments/aerial/scripts/wam_latent_depth_probe.py \
  --dataset "$DS2" --wm-ckpt "$CKPT" --device cuda \
  --max-samples 400 --stride 2 --window 8 \
  --out "$OUT/R3_latent_probe_merged.json" || true

# If still weak on three_zone, keep training another chunk (machine busy)
FWD_OK=$("$PY" - <<PY
import json
from pathlib import Path
p=Path("$OUT/R3_latent_probe_three_zone.json")
if not p.exists():
  print(0); raise SystemExit
j=json.loads(p.read_text())
fwd=j.get("forward_min_depth") or {}
ok=fwd.get("verdict")=="has_geometry" and float(fwd.get("r2_holdout") or -1)>=0.3
print(1 if ok else 0)
PY
)
if [[ "$FWD_OK" != "1" ]]; then
  echo "=== R3 continue FT (hard pack still weak) ==="
  STEPS2="${WM_STEPS2:-1500}"
  "$PY" -m experiments.aerial.rl._wm_train_validate \
    --dataset "$DS" \
    --config "$TMP_CFG" \
    --steps "$STEPS2" \
    --wm-batch 16 --window 8 --horizon 15 \
    --device cuda --allow-v0-desync \
    --init-ckpt "$CKPT" --save-ckpt \
    --checkpoint-dir "$CKPT_DIR" \
    --save-step $((STEPS + STEPS2)) || true
  CKPT=$(ls -t "$CKPT_DIR"/wm_step_*.pt 2>/dev/null | head -1)
  echo "$CKPT" | tee "$OUT/R3_WM_CKPT.txt"
  "$PY" experiments/aerial/scripts/wam_latent_depth_probe.py \
    --dataset "$DS" --wm-ckpt "$CKPT" --device cuda \
    --max-samples 400 --out "$OUT/R3_latent_probe_three_zone_v2.json" || true
fi

echo "GEOMFIX_R3_DONE $(date -Is)" | tee "$OUT/R3_DONE.txt"
rm -f "$OUT/R3_FAIL.txt"
