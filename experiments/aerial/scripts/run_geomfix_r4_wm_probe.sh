#!/usr/bin/env bash
# R4@11 — strong hinge WM until imagine PASS (gap≥0.05) + probe aux. Strict contract.
set -euo pipefail
HOST="${HOST:-11}"
STAMP="${STAMP:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
OUT="$ART/geomfix_${STAMP}"
LOG="$ART/logs/geomfix_R4_${STAMP}_h${HOST}.log"
mkdir -p "$OUT" "$ART/logs"
exec > >(tee -a "$LOG") 2>&1

PY="${PYTHON_BIN:-}"
[[ -z "$PY" && -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
[[ -z "$PY" ]] && PY=python3

# Prefer clean depth-aux long — never continue from saturated geomfix_strong coll heads
# unless INIT_CKPT is explicitly set. FORCE_CLEAN_INIT=1 ignores geomfix_strong.
INIT="${INIT_CKPT:-}"
if [[ -z "$INIT" || "${FORCE_CLEAN_INIT:-0}" == "1" ]]; then
  INIT="$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt"
fi
[[ -f "$INIT" ]] || INIT="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
echo "R4 INIT_CKPT=$INIT FORCE_CLEAN_INIT=${FORCE_CLEAN_INIT:-0}"
DS="${DS:-$ART/dataset_geomfix_nearthick_${STAMP}}"
if [[ ! -d "$DS" ]] || [[ -z "$(ls -A "$DS"/*.npz 2>/dev/null || true)" ]]; then
  DS="$ART/dataset_v0_p45_near_enrich_20260820"
  [[ -d "$DS" ]] || DS="$ART/dataset_v0_three_zone_near_20260823fg"
fi
DS_NEAR="${DS_NEAR:-$DS}"
CKPT_DIR="$ART/wm_ckpt_geomfix_strong_${STAMP}_h${HOST}"
STEPS="${WM_STEPS:-2500}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"

echo "=== R4 strong WM stamp=$STAMP DS=$DS $(date -Is) ==="

TMP_CFG="$OUT/R4_wm_cfg_h${HOST}.yaml"
"$PY" - <<PY
import yaml
from pathlib import Path
base=yaml.safe_load(Path("configs/aerial_rl.yaml").read_text()) or {}
over=yaml.safe_load(Path("configs/aerial_rl_geomfix_near_wm.yaml").read_text()) or {}
wm=dict(base.get("world_model") or {})
wm.update(over.get("world_model") or {})
# Soft prox gate only (targets clamped [0,1] in dynamics_torch). Hinge does ranking.
wm["coll_fwd_depth_aux_weight"]=1.0
wm["coll_near_depth_m"]=8.0
wm["coll_pos_weight"]=1.5
wm["coll_rank_hinge_weight"]=10.0
wm["coll_rank_hinge_margin"]=0.25
wm["coll_hinge_fwd_max_m"]=6.0
ls=dict(wm.get("loss_scales") or {})
ls["depth"]=1.75
wm["loss_scales"]=ls
wm["checkpoint_dir"]="$CKPT_DIR"
base["world_model"]=wm
Path("$TMP_CFG").write_text(yaml.safe_dump(base, sort_keys=False))
PY

imagine_ok() {
  local jf=$1
  "$PY" - <<PY
import json
from pathlib import Path
p=Path("$jf")
if not p.exists():
  print(0); raise SystemExit
j=json.loads(p.read_text())
v=j.get("verdict") or {}
if not isinstance(v, dict):
  v={}
gap=v.get("median_p_coll_gap")
if gap is None:
  gap=j.get("median_p_coll_gap")
useful=bool(v.get("useful"))
ok=useful or (gap is not None and float(gap)>=0.05)
print(1 if ok else 0)
print("gap", gap, "useful", useful)
PY
}

CKPT="$INIT"
TOTAL=0
PASS=0
for round in $(seq 1 "$MAX_ROUNDS"); do
  echo "=== R4 FT round $round/$MAX_ROUNDS ==="
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
  [[ -n "${CKPT:-}" ]] || { echo FAIL >"$OUT/R4_FAIL.txt"; exit 2; }
  echo "$CKPT" | tee "$OUT/R4_WM_CKPT.txt"

  "$PY" experiments/aerial/scripts/wam_latent_depth_probe.py \
    --dataset "$DS_NEAR" --wm-ckpt "$CKPT" --device cuda \
    --max-samples 800 --stride 1 --window 8 --max-center-depth-m 14.0 \
    --out "$OUT/R4_latent_probe_three_zone_r${round}.json" || true
  cp -f "$OUT/R4_latent_probe_three_zone_r${round}.json" \
    "$OUT/R4_latent_probe_three_zone.json" 2>/dev/null || true

  IM="$OUT/R4_imagine_coll_rank_r${round}.json"
  "$PY" experiments/aerial/scripts/wam_imagine_coll_rank.py \
    --dataset "$DS_NEAR" --wm-ckpt "$CKPT" --device cuda \
    --max-samples 300 --stride 1 --encode-mode window --window 8 \
    --out "$IM" || true
  cp -f "$IM" "$OUT/R4_imagine_coll_rank.json" 2>/dev/null || true

  OK=$(imagine_ok "$IM" | head -1)
  echo "imagine_ok=$OK"
  if [[ "$OK" == "1" ]]; then
    PASS=1
    echo "IMAGINE_PASS round=$round" | tee "$OUT/R4_IMAGINE_PASS.txt"
    break
  fi
done

if [[ "$PASS" -eq 1 ]]; then
  echo "GEOMFIX_R4_DONE $(date -Is)" | tee "$OUT/R4_DONE.txt"
  rm -f "$OUT/R4_FAIL.txt"
  exit 0
fi
echo "R4_IMAGINE_FAIL after $MAX_ROUNDS rounds" | tee "$OUT/R4_FAIL.txt"
rm -f "$OUT/R4_DONE.txt"
exit 2
