#!/usr/bin/env bash
# Near-field episode collect → thicken hard pack for probe/imagine (mainchannel).
# Used on 125/84. Writes episode_*.npz under dataset_geomfix_nearthick_${STAMP}.
set -euo pipefail
HOST="${HOST:-125}"
STAMP="${STAMP:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
OUT="$ART/geomfix_${STAMP}"
LOG="$ART/logs/geomfix_nearcollect_${STAMP}_h${HOST}.log"
mkdir -p "$OUT" "$ART/logs"
exec > >(tee -a "$LOG") 2>&1

if [[ -x "$HOME/sim_verify/.venv/bin/python" ]]; then
  PY="$HOME/sim_verify/.venv/bin/python"
elif [[ -x /data/venvs/sim_verify/bin/python ]]; then
  PY=/data/venvs/sim_verify/bin/python
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON_BIN:-python3}"
fi
[[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]] && PY="$PYTHON_BIN"

ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
BASE_NEAR="$ART/dataset_v0_three_zone_near_20260823fg"
ENRICH="$ART/dataset_v0_p45_near_enrich_20260820"
THICK="$ART/dataset_geomfix_nearthick_${STAMP}"
EPISODES="${EPISODES:-40}"
MAX_STEPS="${MAX_STEPS:-120}"
ROUNDS="${ROUNDS:-2}"

echo "=== near episode collect host=$HOST stamp=$STAMP $(date -Is) ==="
mkdir -p "$THICK"

# Seed thick pack once from existing near corpora
"$PY" - <<PY
from pathlib import Path
import shutil
thick=Path("$THICK")
existing=list(thick.glob("episode_*.npz"))
if existing:
  print("thick_already", len(existing)); raise SystemExit
n=0
for src in [Path("$BASE_NEAR"), Path("$ENRICH")]:
  if not src.is_dir():
    continue
  for p in sorted(src.glob("episode_*.npz")):
    dst=thick/f"seed_{src.name}_{p.name}"
    if not dst.exists():
      shutil.copy2(p, dst); n+=1
print("seeded", n, "→", thick)
PY

for r in $(seq 1 "$ROUNDS"); do
  ROUND_DIR="$THICK/round_h${HOST}_r${r}_$(date +%H%M%S)"
  mkdir -p "$ROUND_DIR"
  echo "=== round $r/$ROUNDS → $ROUND_DIR ==="
  set +e
  "$PY" -m experiments.aerial.rl.collect_dataset \
    --backend airsim \
    --annotation "$ANN" \
    --out "$ROUND_DIR" \
    --episodes "$EPISODES" \
    --max-steps "$MAX_STEPS" \
    --step-hz 5.0 \
    --grab-depth \
    --approach-bias \
    --approach-dist-m 18 \
    --host 127.0.0.1 --port 41451
  rc=$?
  set -e
  echo "round_$r rc=$rc"
  # flatten into thick
  i=$(ls "$THICK"/episode_*.npz "$THICK"/seed_*.npz "$THICK"/h*.npz 2>/dev/null | wc -l | tr -d ' ')
  for p in "$ROUND_DIR"/episode_*.npz; do
    [[ -f "$p" ]] || continue
    i=$((i+1))
    cp -f "$p" "$THICK/h${HOST}_$(printf '%05d' "$i").npz"
  done
  echo "thick_count=$(ls "$THICK"/*.npz 2>/dev/null | wc -l | tr -d ' ')"
done

echo "$THICK" > "$OUT/NEARTHICK_DS.txt"
echo "GEOMFIX_NEARCOLLECT_DONE host=$HOST $(date -Is)" | tee "$OUT/NEARCOLLECT_h${HOST}_DONE.txt"
