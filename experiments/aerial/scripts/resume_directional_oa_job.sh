#!/usr/bin/env bash
# Resume remaining directional-OA work after interrupt (eval-first; train only if needed).
# Usage:
#   JOB=actor_ft_125 STAMP=20260922_162118 bash experiments/aerial/scripts/resume_directional_oa_job.sh
#   JOB=fix3_84     STAMP=20260922_153449 bash experiments/aerial/scripts/resume_directional_oa_job.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
JOB="${JOB:?JOB=actor_ft_125|cl_ft_125|fix3_84}"
STAMP="${STAMP:?}"
PY="${PYTHON_BIN:-${HOME}/sim_verify/.venv/bin/python}"
[[ -x "$PY" ]] || PY="${PYTHON_BIN:-/data/venvs/sim_verify/bin/python}"
[[ -x "$PY" ]] || PY=python3
ANN_HARD="${ANN_HARD:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
DEPTH="${DEPTH:-experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU:-experiments/aerial/rl/artifacts/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"

case "$JOB" in
  actor_ft_125)
    DONE_MARK="ACTOR-FT-125 DONE"
    MASTER="$ART/logs/directional_oa_actor_ft_${STAMP}.log"
    OUT="$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}_actor_ft"
    CFG="$ROOT/configs/_tmp_directional_oa_actor_eval.yaml"
    EVAL_TAGS=(actor_noshield_a0 actor_noshield_a1 actor_noshield_a2)
    NEED_PLANNER=0
    LOG_PREFIX=directional_oa_actor_ft
    ;;
  cl_ft_125)
    DONE_MARK="CL-FT-125 DONE"
    MASTER="$ART/logs/directional_oa_cl_ft_${STAMP}.log"
    OUT="$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}_cl_ft"
    CFG="$ROOT/configs/_tmp_directional_oa_cl_eval.yaml"
    EVAL_TAGS=(cl_noshield_b0 cl_noshield_b1 cl_noshield_b2 cl_shield_b0 cl_shield_b1 cl_shield_b2 actor_noshield_best_a0 actor_noshield_best_a1 actor_noshield_best_a2 actor_noshield_latest_a0 actor_noshield_latest_a1 actor_noshield_latest_a2)
    NEED_PLANNER=1
    LOG_PREFIX=directional_oa_cl_ft
    ;;
  fix3_84)
    DONE_MARK="FIX3-84 DONE"
    MASTER="$ART/logs/directional_oa_fix3_84_${STAMP}.log"
    OUT="$ART/v4_ac_ckpt_urban_directional_oa_${STAMP}_fix3_84"
    CFG="$ROOT/configs/_tmp_directional_oa_eval_84.yaml"
    EVAL_TAGS=(cl_noshield_b0 cl_noshield_b1 cl_noshield_b2 actor_noshield_a0 actor_noshield_a1 actor_noshield_a2)
    NEED_PLANNER=1
    LOG_PREFIX=directional_oa_fix3_84
    ;;
  *)
    echo "FATAL unknown JOB=$JOB"; exit 2
    ;;
esac

if [[ -f "$MASTER" ]] && grep -q "$DONE_MARK" "$MASTER"; then
  echo "ALREADY_DONE job=$JOB stamp=$STAMP"
  exit 0
fi

# Alive?
if pgrep -af "$LOG_PREFIX|train_v4_ac.*${STAMP}|long_eval.*${STAMP}" | grep -vE 'pgrep|resume_directional|watch_directional' >/dev/null; then
  echo "ALREADY_RUNNING job=$JOB stamp=$STAMP"
  exit 0
fi

mkdir -p "$ART/logs"
exec >>"$ART/logs/${LOG_PREFIX}_resume_${STAMP}.log" 2>&1
echo "=== RESUME job=$JOB stamp=$STAMP $(date -Is) ==="

if [[ -f "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt" ]]; then
  WM="$(cat "$ART/obstacle_cost_gt_depth/LATEST_REFLOW_CKPT.txt")"
elif [[ -n "${WM:-}" ]]; then
  :
else
  # Prefer stamp-local / job-local WM from master log
  WM="$(grep -Eo '/[^ ]+/wm_obs\.pt' "$MASTER" 2>/dev/null | tail -1 || true)"
fi
[[ -n "${WM:-}" && -f "$WM" ]] || { echo "FATAL no WM"; exit 1; }
echo "WM=$WM"

if [[ -f "$OUT/v4_ac_best.pt" ]]; then ACTOR_BEST="$OUT/v4_ac_best.pt"
elif [[ -f "$OUT/v4_ac_latest.pt" ]]; then ACTOR_BEST="$OUT/v4_ac_latest.pt"
else
  echo "FATAL no actor in $OUT — need full retrain"; exit 1
fi
ACTOR_LATEST="$OUT/v4_ac_latest.pt"
[[ -f "$ACTOR_LATEST" ]] || ACTOR_LATEST="$ACTOR_BEST"
ACTOR="$ACTOR_BEST"
echo "ACTOR_BEST=$ACTOR_BEST ACTOR_LATEST=$ACTOR_LATEST"

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
cfg.setdefault("planner", {})
if int("$NEED_PLANNER"):
    cfg["planner"]["enable"]=True
    cfg["planner"]["rollout_mode"]="closed_loop"
else:
    cfg["planner"]["enable"]=False
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", "$CFG")
PY

run_eval() {
  local TAG=$1; shift
  local EOUT="$ART/eval_directional_oa_${STAMP}_${TAG}"
  if [[ -f "$EOUT/eval_all.json" ]]; then
    echo "SKIP completed $TAG"
    return 0
  fi
  local CKPT="$ACTOR_BEST"
  if [[ "$TAG" == *"_latest_"* ]]; then
    CKPT="$ACTOR_LATEST"
  fi
  mkdir -p "$EOUT/traj"
  echo "=== EVAL $TAG (resume) ckpt=$(basename "$CKPT") ==="
  local extra=()
  if [[ "$TAG" == cl_* ]]; then
    extra+=(--planner --planner-horizon 15 --planner-rollout closed_loop)
  fi
  local shield_args=(--no-shield)
  if [[ "$TAG" == cl_shield_* ]]; then
    shield_args=()  # yaml three_zone shield
  fi
  local depth_args=()
  [[ -f "$DEPTH" ]] && depth_args+=(--depth-ckpt "$DEPTH")
  [[ -f "$TAU" ]] && depth_args+=(--tau-ckpt "$TAU")
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN_HARD" --wm-ckpt "$WM" --actor-ckpt "$CKPT" \
    "${depth_args[@]}" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$EOUT/traj" --out "$EOUT/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10.0 --max-steps 600 --success-dist 3.0 --terminal-pin-rem-m 20 \
    --min-spawn-z 24.0 --spawn-z-retry-m 5.0 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5.0 \
    "${shield_args[@]}" "${extra[@]}" \
    > "$ART/logs/${LOG_PREFIX}_eval_${TAG}_${STAMP}.log" 2>&1 || true
  "$PY" - <<PY
import json
from pathlib import Path
p=Path("$EOUT/eval_all.json")
if not p.exists():
    print("[$TAG] MISSING")
else:
    rows=json.loads(p.read_text())["episodes"]
    arr=sum(1 for r in rows if r.get("arrived"))
    climbs=[round(100*float(r.get("oa_chose_climb_rate") or 0),1) for r in rows]
    print(f"[$TAG] SR {arr}/{len(rows)} climb%={climbs}")
PY
}

for TAG in "${EVAL_TAGS[@]}"; do
  run_eval "$TAG"
done

{
  echo "=== $DONE_MARK stamp=$STAMP actor=$ACTOR wm=$WM (resume) ==="
} | tee -a "$MASTER"
echo "RESUME_DONE job=$JOB stamp=$STAMP"
