#!/usr/bin/env bash
# e6 control arm: SAME n=5 gate protocol as eval_urban_complex_depthaux_n5.sh,
# but on the PRE-e5 actor/WM (the exact checkpoint e5 warm-started from) so
# the e5-vs-baseline SR/IR comparison is apples-to-apples under identical
# nondeterminism conditions instead of comparing against remembered numbers
# from earlier sessions.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

export ACTOR="${ACTOR:-experiments/aerial/rl/artifacts/v4_ac_ckpt_phase2_toward_g_20260905_112006/v4_ac_latest.pt}"
export WM="${WM:-experiments/aerial/rl/artifacts/wm_ckpt_d_full_20260828/wm_step_3500.pt}"
N_REPS="${N_REPS:-5}"
REP_START="${REP_START:-1}"
BASE_LOG="artifacts/eval_urban_complex_baseline_n5.log"

echo "=== e6-control n=$N_REPS gate eval actor=$ACTOR wm=$WM (start rep=$REP_START) ===" | tee -a "$BASE_LOG"

for i in $(seq "$REP_START" "$N_REPS"); do
  echo "--- rep $i/$N_REPS ---" | tee -a "$BASE_LOG"
  STAMP="20260920_baseline_rep${i}" bash experiments/aerial/scripts/eval_urban_complex_toward_g_gate.sh \
    2>&1 | tee -a "$BASE_LOG"
  rc="${PIPESTATUS[0]}"
  echo "--- rep $i/$N_REPS exit=$rc ---" | tee -a "$BASE_LOG"
done

echo "=== aggregate over $N_REPS reps ===" | tee -a "$BASE_LOG"
python3 - "$N_REPS" <<'PY' | tee -a "$BASE_LOG"
import json, sys
from pathlib import Path
n_reps = int(sys.argv[1])
all_rows = []
for i in range(1, n_reps + 1):
    p = Path(f"artifacts/urban_complex_toward_g_gate_20260920_baseline_rep{i}/eval_all.json")
    if not p.is_file():
        print(f"[warn] missing {p}")
        continue
    rows = json.loads(p.read_text()).get("episodes", [])
    for r in rows:
        r["_rep"] = i
        all_rows.append(r)
n = len(all_rows)
arr = sum(1 for r in all_rows if r.get("arrived"))
print(f"[e6-control aggregate] SR {arr}/{n} = {arr/n:.1%}" if n else "[e6-control aggregate] no results")
by_route = {}
for r in all_rows:
    ridx = r.get("route_idx")
    by_route.setdefault(ridx, []).append(r)
for ridx in sorted(by_route):
    rows = by_route[ridx]
    a = sum(1 for r in rows if r.get("arrived"))
    prog = sum(float(r.get("progress_ratio", 0)) for r in rows) / len(rows) * 100
    print(f"  route={ridx}: arrived {a}/{len(rows)}  mean_progress={prog:.1f}%")
PY
