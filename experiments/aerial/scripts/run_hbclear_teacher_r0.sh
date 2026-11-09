#!/usr/bin/env bash
# R0 teacher collect: overnight-best stack (hbclear + open_loop + wm_bare)
# with mid-course stuck abort. Replaces exhausted DepthScene variant rotation.
#
#   HOST=84 MIN_KEPT=2 bash experiments/aerial/scripts/run_hbclear_teacher_r0.sh
#   HOST=14 MIN_KEPT=2 WM=... bash ...
#
# Urban-interior gate: inland annotation + require-arrived save only.
# Continues until keep>=MIN_KEPT (never declares done on failing gate).
set -euo pipefail
HOST="${HOST:?set HOST}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PYTHON_BIN:-}"
for c in \
  "${PYTHON_BIN:-}" \
  "$ROOT/.venv/bin/python" \
  "${HOME}/sim_verify/.venv/bin/python" \
  /data/venvs/sim_verify/bin/python; do
  [[ -n "$c" && -x "$c" ]] && PY="$c" && break
done
PY="${PY:-python3}"

ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MIN_KEPT="${MIN_KEPT:-2}"
MAX_ROUNDS="${MAX_ROUNDS:-80}"
ROUTES="${ROUTES:-0}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ACTOR="${ACTOR:-$ART/v4_ac_ckpt_urban_complex_p2c_20260921_shield_contract_v2_hbclear/v4_ac_latest.pt}"
if [[ -z "${WM:-}" ]]; then
  if [[ -f "$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt" ]]; then
    WM="$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt"
  elif [[ -f "$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt" ]]; then
    WM="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
  else
    echo FATAL need WM; exit 1
  fi
fi
DEPTH="${DEPTH_CKPT:-$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU_CKPT:-$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
DS="${DS:-$ART/dataset_urban_inland_r0_hbclear_teacher_${STAMP}}"
CFG="$ROOT/configs/_tmp_hbclear_teacher_r0_h${HOST}.yaml"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
MASTER="$ART/logs/hbclear_teacher_r0_${STAMP}_h${HOST}.log"
STUCK_ABORT_S="${STUCK_ABORT_S:-20}"
STUCK_ABORT_PROG_M="${STUCK_ABORT_PROG_M:-1.0}"
STUCK_ABORT_REM_LO="${STUCK_ABORT_REM_LO:-25}"
STUCK_ABORT_REM_HI="${STUCK_ABORT_REM_HI:-45}"
mkdir -p "$ART/logs" "$DS/traj" "$DS/rounds"
exec >>"$MASTER" 2>&1

echo "=== hbclear teacher R0 HOST=$HOST stamp=$STAMP ==="
hostname; date -Is
echo "DS=$DS ACTOR=$ACTOR WM=$WM MIN_KEPT=$MIN_KEPT stuck_abort=${STUCK_ABORT_S}s"
[[ -f "$ACTOR" && -f "$WM" ]] || { echo FATAL ckpt; exit 1; }

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
ov=yaml.safe_load(Path("configs/aerial_rl_urban_complex_directional_oa.yaml").read_text())
cfg=deep_merge(base, ov)
cfg.setdefault("planner", {})["enable"]=True
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

keep_count() {
  DS="$DS" "$PY" - <<'PY'
import json, os
from pathlib import Path
ar = Path(os.environ["DS"]) / "AUTO_REVIEW.json"
if not ar.exists():
    print(0); raise SystemExit
d = json.loads(ar.read_text())
s = d.get("summary") or {}
print(int(s.get("quality_kept") or 0))
PY
}

review_round() {
  local round_i="$1" eval_json="$2"
  ROUND_I="$round_i" EVAL_JSON="$eval_json" DS="$DS" MIN_KEPT="$MIN_KEPT" ROUTES="$ROUTES" \
  "$PY" - <<'PY'
import json, os
from pathlib import Path
from collections import Counter
ds = Path(os.environ["DS"])
ej = Path(os.environ["EVAL_JSON"])
round_i = int(os.environ["ROUND_I"])
min_kept = int(os.environ["MIN_KEPT"])
need = [int(x) for x in os.environ["ROUTES"].split(",") if x.strip()!=""]
ar_path = ds / "AUTO_REVIEW.json"
if ar_path.exists():
    blob = json.loads(ar_path.read_text())
    reviews = list(blob.get("episodes") or [])
else:
    reviews = []
rows = []
if ej.exists():
    rows = json.loads(ej.read_text()).get("episodes") or []
# Count newly written npz this round via append (already only arrived)
n_npz = len(list(ds.glob("episode_*.npz")))
for r in rows:
    ri = int(r.get("route_idx", -1))
    arrived = bool(r.get("arrived"))
    keep = arrived and (not r.get("severe_collision")) and (not r.get("monotone_inflate"))
    # inland annotation routes are pre-filtered; still drop inflate / severe
    reviews.append({
        "route_idx": ri,
        "round": round_i,
        "teacher": "hbclear_wm_bare",
        "arrived": arrived,
        "severe_collision": bool(r.get("severe_collision")),
        "d_final_m": r.get("d_final_m"),
        "d_min_m": r.get("d_min_m"),
        "steps": r.get("steps"),
        "intervention_rate": r.get("intervention_rate"),
        "fail_tag": r.get("fail_tag"),
        "keep": keep,
        "drop_reason": None if keep else ("not_arrived" if not arrived else "quality"),
        "in_interior": True,
        "geo_reasons": [],
    })
keep_by = Counter(int(e["route_idx"]) for e in reviews if e.get("keep"))
quality_kept = sum(1 for e in reviews if e.get("keep"))
gate_ok = all(keep_by.get(r, 0) >= min_kept for r in need) and quality_kept >= min_kept * max(1, len(need))
# For single-route R0: quality_kept >= min_kept is enough
if len(need) == 1:
    gate_ok = quality_kept >= min_kept
summary = {
    "quality_kept": quality_kept,
    "keep_by_route": {str(k): int(v) for k, v in sorted(keep_by.items())},
    "required_routes": need,
    "gate": min_kept,
    "gate_passed": bool(gate_ok),
    "round": round_i,
    "n_npz": n_npz,
    "n_arrived": sum(1 for e in reviews if e.get("arrived")),
    "teacher": "hbclear_open_loop_wm_bare",
    "policy": "stuck_abort + save_arrived_only; inland annotation",
}
ar_path.write_text(json.dumps({"summary": summary, "episodes": reviews}, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False))
PY
}

[[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 8; }

for round_i in $(seq 1 "$MAX_ROUNDS"); do
  kc="$(keep_count)"
  echo "=== round $round_i keep=$kc/$MIN_KEPT ==="
  if [[ "$kc" -ge "$MIN_KEPT" ]]; then
    echo "GATE_PASSED keep=$kc"
    break
  fi
  ROUND_DIR="$DS/rounds/r$(printf '%03d' "$round_i")"
  mkdir -p "$ROUND_DIR/traj"
  # One attempt per round (routes=0); append keeps into DS
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$ACTOR" \
    --depth-ckpt "$DEPTH" --tau-ckpt "$TAU" --goal-feat-mode meter \
    --routes "$ROUTES" --traj-out "$ROUND_DIR/traj" --out "$ROUND_DIR/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --planner --planner-horizon 5 --planner-rollout open_loop \
    --planner-mock wm_bare \
    --cruise-speed 10 --max-steps 600 --success-dist 3 --terminal-pin-rem-m 20 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5 --shield-exclusion-forward-only \
    --save-dataset "$DS" --save-dataset-append --save-arrived-only \
    --stuck-abort-after-s "$STUCK_ABORT_S" \
    --stuck-abort-progress-m "$STUCK_ABORT_PROG_M" \
    --stuck-abort-rem-lo "$STUCK_ABORT_REM_LO" \
    --stuck-abort-rem-hi "$STUCK_ABORT_REM_HI" \
    >"$ART/logs/hbclear_teacher_r0_${STAMP}_h${HOST}_r${round_i}.log" 2>&1
  rc=$?
  set -e
  echo "round_exit=$rc"
  review_round "$round_i" "$ROUND_DIR/eval_all.json"
  # brief throttle / recover if airsim died
  if ! ss -lntp 2>/dev/null | grep -q 41451; then
    echo "airsim down — recover"
    [[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 10; }
  fi
done

kc="$(keep_count)"
echo "FINAL keep=$kc/$MIN_KEPT"
if [[ "$kc" -ge "$MIN_KEPT" ]]; then
  echo "HBCLEAR_TEACHER_R0_GATE_OK"
  exit 0
fi
echo "HBCLEAR_TEACHER_R0_GATE_PENDING keep=$kc"
exit 3
