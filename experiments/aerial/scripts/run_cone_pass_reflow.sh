#!/usr/bin/env bash
# Cone-pass reflow: window-encode three_zone_near + GT cone labels + same-scene
# ranking loss → cone rank until PASS. Mainchannel (no shield). Soft blocked.
set -euo pipefail
HOST="${HOST:-125}"
STAMP="${STAMP:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
GT="$ART/obstacle_cost_gt_depth"
OUT="$ART/geomfix_${STAMP}"
LOG="$ART/logs/cone_pass_${STAMP}_h${HOST}.log"
mkdir -p "$OUT" "$ART/logs" "$GT"
exec > >(tee -a "$LOG") 2>&1

if [[ -x "$HOME/sim_verify/.venv/bin/python" ]]; then
  PY="$HOME/sim_verify/.venv/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON_BIN:-python3}"
fi
[[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]] && PY="$PYTHON_BIN"

WM_BASE="${WM_BASE:-$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
[[ -f "$WM_BASE" ]] || WM_BASE="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
DS_NEAR="${DS_NEAR:-$ART/dataset_v0_three_zone_near_20260823fg}"
FRAMES="${FRAMES:-$GT/frames_multi2.npz}"
HEAD_STEPS="${HEAD_STEPS:-3000}"
DEVICE="${DEVICE:-cuda}"

echo "=== cone-pass reflow stamp=$STAMP $(date -Is) ==="
echo "PY=$PY WM_BASE=$WM_BASE DS_NEAR=$DS_NEAR"
test -f "$WM_BASE"
test -d "$DS_NEAR"

# 1) Window-encode hard near episodes with GT cone probe labels (eval-aligned).
LABELS_CONE="$GT/labels_cone_window_${STAMP}.npz"
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels pack-cone \
  --wm-ckpt "$WM_BASE" --dataset "$DS_NEAR" --out "$LABELS_CONE" \
  --device "$DEVICE" --window 8 --stride 2 --max-samples "${MAX_SAMPLES:-600}" \
  --max-forward-depth-m 14.0 --d-near 3.0 --d-far 22.0

"$PY" - <<PY
import numpy as np, sys
from collections import Counter
d=np.load("$LABELS_CONE", allow_pickle=True)
g=d["group"].astype(str)
c=dict(Counter(g.tolist()))
print("cone_pack_groups", c)
n_scenes=int(d["scene_id"].max())+1 if len(d["scene_id"]) else 0
print("n_rows", len(g), "n_scenes", n_scenes)
need_fwd=int(c.get("fwd_near", 0))
need_left=int(c.get("left_near", 0))
# Each scene expands to 6 probes; cone needs ≥20 frames per core slice.
if need_fwd < 20*6:
    print(f"WARN fwd_near rows={need_fwd} (<120); cone may INSUFFICIENT")
if need_left < 20*6:
    print(f"WARN left_near rows={need_left} (<120); cone may INSUFFICIENT")
if n_scenes < 30:
    print(f"FATAL n_scenes={n_scenes} < 30")
    sys.exit(3)
PY

# 2) Optional: merge multi2 frame pack (gate left_near mass) with cone pack.
LABELS="$LABELS_CONE"
if [[ -f "$FRAMES" && "${MERGE_MULTI2:-0}" == "1" ]]; then
  LABELS_F="$GT/labels_multi2_coneprobe_${STAMP}.npz"
  "$PY" -m experiments.aerial.rl.train_obstacle_cost_labels pack \
    --wm-ckpt "$WM_BASE" --frames "$FRAMES" --out "$LABELS_F" \
    --device "$DEVICE" --d-near 5.0 --d-far 18.0
  LABELS_MERGED="$GT/labels_cone_merged_${STAMP}.npz"
  "$PY" - <<PY
import numpy as np
from pathlib import Path
a=np.load("$LABELS_CONE", allow_pickle=True)
b=np.load("$LABELS_F", allow_pickle=True)
# Offset scene_id so pairwise rank stays within each pack's scenes.
sid_a=a["scene_id"]
sid_b=b["scene_id"] + (int(sid_a.max()) + 1 if len(sid_a) else 0)
keys=["feature","action","obstacle_label","clearance_m","group","scene_id"]
out={}
for k in keys:
  if k=="scene_id":
    out[k]=np.concatenate([sid_a, sid_b], axis=0)
  else:
    out[k]=np.concatenate([a[k], b[k]], axis=0)
Path("$LABELS_MERGED").parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed("$LABELS_MERGED", **out)
from collections import Counter
print("merged", len(out["group"]), dict(Counter(out["group"].astype(str).tolist())))
PY
  LABELS="$LABELS_MERGED"
fi

OBS_CKPT="$ART/wm_ckpt_obstacle_cost_conepass_${STAMP}/wm_obs.pt"
mkdir -p "$(dirname "$OBS_CKPT")"
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels train \
  --wm-ckpt "$WM_BASE" --labels "$LABELS" --out-ckpt "$OBS_CKPT" \
  --steps "$HEAD_STEPS" --batch 128 --device "$DEVICE" \
  --scene-rank-weight "${SCENE_RANK_WEIGHT:-3.0}" --scene-rank-margin 0.25

# Gate on multi2-ish groups if present; do not block cone on gate alone.
GATE_JSON="$OUT/cone_pass_gate.json"
set +e
"$PY" -m experiments.aerial.rl.train_obstacle_cost_labels gate \
  --wm-ckpt "$OBS_CKPT" --labels "$LABELS" --report "$GATE_JSON" --device "$DEVICE" \
  --require-left-near --min-left-near 20
GATE_RC=$?
set -e
echo "gate_rc=$GATE_RC"

CONE_JSON="$OUT/cone_pass_rank.json"
set +e
"$PY" experiments/aerial/scripts/wam_obstacle_cost_cone_rank.py \
  --dataset "$DS_NEAR" --wm-ckpt "$OBS_CKPT" --device "$DEVICE" \
  --max-samples 600 --stride 2 --window 8 --max-forward-depth-m 14.0 \
  --out "$CONE_JSON"
CONE_RC=$?
set -e
echo "cone_rc=$CONE_RC"
echo "$OBS_CKPT" > "$GT/LATEST_CONEPASS_CKPT.txt"
echo "$OBS_CKPT" > "$OUT/CONE_PASS_WM_OBS.txt"
cp -f "$CONE_JSON" "$OUT/" 2>/dev/null || true

VERDICT="$("$PY" - <<PY
import json
from pathlib import Path
p=Path("$CONE_JSON")
d=json.loads(p.read_text()) if p.is_file() else {}
print(d.get("verdict","MISSING"))
PY
)"
echo "cone_verdict=$VERDICT"
if [[ "$VERDICT" == "PASS" ]]; then
  echo "CONE_PASS $(date -Is)" | tee "$OUT/CONE_PASS.txt"
  rm -f "$OUT/CONE_FAIL.txt"
  exit 0
fi
echo "CONE_FAIL verdict=$VERDICT" | tee "$OUT/CONE_FAIL.txt"
exit 1
