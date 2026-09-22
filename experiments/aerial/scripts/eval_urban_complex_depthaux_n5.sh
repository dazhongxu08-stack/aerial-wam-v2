#!/usr/bin/env bash
# e6: n>=5 gate eval of the e5 depth-aux+w_intervention actor on routes 0,1,3,4,
# using the SAME toward_g gate script/params as the historical baseline for a
# clean A/B (only difference: ACTOR + WM ckpt point at the e5 outputs).
set -uo pipefail
# NOTE: no -e here on purpose. wam_phase2_long_eval.py / the toward_g gate
# script exit nonzero on a FAIL verdict (single-rep gate < MIN_GATE_ARRIVED),
# which is an EXPECTED outcome for a hard-route rep, not a script bug. Per the
# workspace rule ("gate not met -> keep going, never stop"), a FAIL verdict on
# rep i must not abort reps i+1..N_REPS -- only run explosions the wrapper
# doesn't understand should stop the sweep.

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

export ACTOR="${ACTOR:-experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_20260920_depthaux/v4_ac_latest.pt}"
export WM="${WM:-experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt}"
N_REPS="${N_REPS:-5}"
REP_START="${REP_START:-1}"
BASE_LOG="artifacts/eval_urban_complex_depthaux_n5.log"

echo "=== e6 n=$N_REPS gate eval actor=$ACTOR wm=$WM (start rep=$REP_START) ===" | tee -a "$BASE_LOG"

for i in $(seq "$REP_START" "$N_REPS"); do
  echo "--- rep $i/$N_REPS ---" | tee -a "$BASE_LOG"
  STAMP="20260920_depthaux_rep${i}" bash experiments/aerial/scripts/eval_urban_complex_toward_g_gate.sh \
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
    p = Path(f"artifacts/urban_complex_toward_g_gate_20260920_depthaux_rep{i}/eval_all.json")
    if not p.is_file():
        print(f"[warn] missing {p}")
        continue
    rows = json.loads(p.read_text()).get("episodes", [])
    for r in rows:
        r["_rep"] = i
        all_rows.append(r)
n = len(all_rows)
arr = sum(1 for r in all_rows if r.get("arrived"))
print(f"[e6 aggregate] SR {arr}/{n} = {arr/n:.1%}" if n else "[e6 aggregate] no results")
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
