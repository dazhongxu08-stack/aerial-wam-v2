#!/usr/bin/env bash
# R4@11 — stronger-hinge WM FT + continuous imagine/cone probes (no shield).
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

INIT="${INIT_CKPT:-$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
[[ -f "$INIT" ]] || INIT="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
DS="${DS:-$ART/dataset_v0_p45_near_enrich_20260820}"
[[ -d "$DS" ]] || DS="$ART/dataset_v0_three_zone_near_20260823fg"
DS_NEAR="$ART/dataset_v0_three_zone_near_20260823fg"
CKPT_DIR="$ART/wm_ckpt_geomfix_strong_${STAMP}_h${HOST}"
STEPS="${WM_STEPS:-2000}"

echo "=== R4 strong WM + probes stamp=$STAMP $(date -Is) ==="

TMP_CFG="$OUT/R4_wm_cfg_h${HOST}.yaml"
"$PY" - <<PY
import yaml
from pathlib import Path
base=yaml.safe_load(Path("configs/aerial_rl.yaml").read_text()) or {}
over=yaml.safe_load(Path("configs/aerial_rl_geomfix_near_wm.yaml").read_text()) or {}
wm=dict(base.get("world_model") or {})
wm.update(over.get("world_model") or {})
# stronger than R3
wm["coll_fwd_depth_aux_weight"]=3.0
wm["coll_rank_hinge_weight"]=5.0
ls=dict(wm.get("loss_scales") or {})
ls["depth"]=1.25
wm["loss_scales"]=ls
wm["checkpoint_dir"]="$CKPT_DIR"
base["world_model"]=wm
Path("$TMP_CFG").write_text(yaml.safe_dump(base, sort_keys=False))
PY

"$PY" -m experiments.aerial.rl._wm_train_validate \
  --dataset "$DS" \
  --config "$TMP_CFG" \
  --steps "$STEPS" \
  --wm-batch 16 --window 8 --horizon 15 \
  --device cuda --allow-v0-desync \
  --init-ckpt "$INIT" --save-ckpt \
  --checkpoint-dir "$CKPT_DIR" \
  --save-step "$STEPS" || true

CKPT=$(ls -t "$CKPT_DIR"/wm_step_*.pt 2>/dev/null | head -1 || true)
[[ -n "${CKPT:-}" ]] || { echo FAIL >"$OUT/R4_FAIL.txt"; exit 2; }
echo "$CKPT" | tee "$OUT/R4_WM_CKPT.txt"

"$PY" experiments/aerial/scripts/wam_latent_depth_probe.py \
  --dataset "$DS_NEAR" --wm-ckpt "$CKPT" --device cuda \
  --max-samples 400 --out "$OUT/R4_latent_probe_three_zone.json" || true
"$PY" experiments/aerial/scripts/wam_imagine_coll_rank.py \
  --dataset "$DS_NEAR" --wm-ckpt "$CKPT" --device cuda \
  --max-samples 200 --stride 2 --encode-mode window --window 8 \
  --out "$OUT/R4_imagine_coll_rank.json" || true

# If R1 cost ckpt already published on shared NFS / synced path, cone-rank it
OBS=""
[[ -f "$OUT/R1_WM_OBS.txt" ]] && OBS=$(cat "$OUT/R1_WM_OBS.txt")
[[ -z "$OBS" && -f "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt" ]] && \
  OBS=$(cat "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt")
if [[ -n "$OBS" && -f "$OBS" ]]; then
  "$PY" experiments/aerial/scripts/wam_obstacle_cost_cone_rank.py \
    --dataset "$DS_NEAR" --wm-ckpt "$OBS" --device cuda \
    --max-samples 400 --out "$OUT/R4_cone_rank_on_R1.json" || true
fi

echo "GEOMFIX_R4_DONE $(date -Is)" | tee "$OUT/R4_DONE.txt"
rm -f "$OUT/R4_FAIL.txt"
