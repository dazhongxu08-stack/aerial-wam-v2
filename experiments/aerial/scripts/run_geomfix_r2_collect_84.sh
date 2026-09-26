#!/usr/bin/env bash
# R2@84 — parallel left_near GT collect (AirSim) → sync path for 125 merge.
set -euo pipefail
HOST="${HOST:-84}"
STAMP="${STAMP:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
GT="$ART/obstacle_cost_gt_depth"
OUT="$ART/geomfix_${STAMP}"
LOG="$ART/logs/geomfix_R2_${STAMP}_h${HOST}.log"
mkdir -p "$OUT" "$ART/logs" "$GT"
exec > >(tee -a "$LOG") 2>&1

if [[ -x /data/venvs/sim_verify/bin/python ]]; then
  PY=/data/venvs/sim_verify/bin/python
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON_BIN:-python3}"
fi

ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
echo "=== R2 collect stamp=$STAMP $(date -Is) ==="

# Keep collecting in rounds so the machine never idles
ROUNDS="${ROUNDS:-3}"
for r in $(seq 1 "$ROUNDS"); do
  TOP="$GT/frames_geomfix_84_r${r}_${STAMP}.npz"
  echo "=== round $r/$ROUNDS → $TOP ==="
  set +e
  "$PY" -m experiments.aerial.rl.collect_obstacle_cost_gt_frames \
    --config configs/aerial_rl_urban_complex_p2c.yaml \
    --annotation "$ANN" \
    --out "$TOP" \
    --host 127.0.0.1 --port 41451 \
    --episodes "${EPISODES:-50}" \
    --steps-per-ep "${STEPS_PER_EP:-20}" \
    --d-near 5.0 --d-far 18.0 \
    --min-per-group 25 --max-frames 1000
  rc=$?
  set -e
  echo "round_$r rc=$rc"
  if [[ -f "$TOP" ]]; then
    "$PY" - <<PY
import numpy as np
from collections import Counter
d=np.load("$TOP", allow_pickle=True)
print("groups", dict(Counter(d["group"].astype(str).tolist())))
PY
    cp -f "$TOP" "$OUT/" || true
  fi
done

echo "GEOMFIX_R2_DONE $(date -Is)" | tee "$OUT/R2_DONE.txt"
