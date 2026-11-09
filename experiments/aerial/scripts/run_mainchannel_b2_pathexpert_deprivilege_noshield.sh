#!/usr/bin/env bash
# B2 · PathExpert→deprivilege (toward_g only) (A*/polyline teacher) collect → BC → noshield toward_g gate.
# Interior/inland gates; AUTO_REVIEW; keep collecting until keep gate (do not stop early).
#
#   HOST=11 STAMP=... bash experiments/aerial/scripts/run_mainchannel_b2_pathexpert_deprivilege_noshield.sh
set -euo pipefail
HOST="${HOST:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${PYTHON_BIN:-}"
for c in "${PYTHON_BIN:-}" "$ROOT/.venv/bin/python" "${HOME}/sim_verify/.venv/bin/python" /data/venvs/sim_verify/bin/python; do
  [[ -n "$c" && -x "$c" ]] && PY="$c" && break
done
PY="${PY:-python3}"
ART="$ROOT/experiments/aerial/rl/artifacts"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROUTES="${ROUTES:-0,1,3,4}"
MIN_KEPT="${MIN_KEPT:-8}"
MIN_KEPT_PER_ROUTE="${MIN_KEPT_PER_ROUTE:-2}"
MAX_ROUNDS="${MAX_ROUNDS:-40}"
BC_ITERS="${BC_ITERS:-40}"
EP_PER_ROUND="${EP_PER_ROUND:-4}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_BC="${SKIP_BC:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ANN_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
DS="${DS:-$ART/dataset_urban_inland_r0134_pathexpert_${STAMP}_h${HOST}}"
OUT="${OUT:-$ART/v4_ac_ckpt_mainchannel_b2_pathexpert_${STAMP}_noshield_h${HOST}}"
MASTER="$ART/logs/mainchannel_b2_pathexpert_${STAMP}_h${HOST}.log"
CFG="$ROOT/configs/_tmp_mainchannel_b2_h${HOST}.yaml"
mkdir -p "$ART/logs" "$OUT" "$DS"
exec >>"$MASTER" 2>&1
echo "=== B2 PathExpert deprivilege→BC HOST=$HOST stamp=$STAMP ==="; hostname; date -Is

pick_actor() { local d=$1; [[ -f $d/v4_ac_best.pt ]] && echo $d/v4_ac_best.pt && return; [[ -f $d/v4_ac_latest.pt ]] && echo $d/v4_ac_latest.pt && return; return 1; }
INIT="${INIT:-}"
[[ -z "$INIT" ]] && INIT="$(pick_actor "$ART/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2" || true)"
[[ -f "${INIT:-}" ]] || { echo FATAL no INIT; exit 1; }
if [[ -n "${WM:-}" && -f "$WM" ]]; then :
elif [[ -f "$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt" ]]; then WM="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
elif [[ -f "$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt" ]]; then WM="$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt"
else echo FATAL WM; exit 1; fi
DEPTH="${DEPTH_CKPT:-$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU_CKPT:-$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
echo "INIT=$INIT WM=$WM DS=$DS"

"$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py --src "$ANN" --out "$ANN_FOCUS" --routes "$ROUTES" || true

review_gate() {
  DS="$DS" ROUTES="$ROUTES" MIN_KEPT="$MIN_KEPT" MIN_KEPT_PER_ROUTE="$MIN_KEPT_PER_ROUTE" "$PY" - <<'PY'
import json, os
from pathlib import Path
from collections import defaultdict
ds = Path(os.environ["DS"])
routes = [int(x) for x in os.environ["ROUTES"].split(",") if x.strip()!=""]
min_kept = int(os.environ["MIN_KEPT"])
min_per = int(os.environ["MIN_KEPT_PER_ROUTE"])
man = ds / "manifest.json"
eps = []
if man.exists():
    raw = json.loads(man.read_text())
    eps = raw.get("episodes") or raw if isinstance(raw, list) else raw.get("episodes") or []
keep_by = defaultdict(int)
kept = 0
drops = []
for i, e in enumerate(eps):
    arrived = bool(e.get("arrived"))
    ri = e.get("route_idx", e.get("base_route_idx", e.get("episode_idx", i)))
    try: ri = int(ri)
    except Exception: ri = -1
    # Prefer arrived keeps; fall back to usable path if flagged
    usable = arrived or (float(e.get("path_length_m") or 0) >= 35 and int(e.get("steps") or e.get("n_steps") or 0) >= 30 and not e.get("collided", False))
    inland_ok = e.get("in_interior", True) is not False and e.get("in_water", False) is not True
    if usable and inland_ok and arrived:
        keep_by[ri] += 1
        kept += 1
    else:
        drops.append({"i": i, "route": ri, "arrived": arrived, "reason": "not_arrived_or_geo"})
need = [r for r in routes if keep_by.get(r, 0) < min_per]
gate = kept >= min_kept and not need
summary = {
  "gate_passed": gate,
  "n_episodes": len(eps),
  "n_kept_arrived": kept,
  "keep_by_route": {str(k): v for k, v in sorted(keep_by.items())},
  "required_routes": routes,
  "min_kept": min_kept,
  "min_kept_per_route": min_per,
  "missing_routes": need,
  "n_drop_samples": len(drops),
}
(ds / "AUTO_REVIEW.json").write_text(json.dumps({"summary": summary, "note": "B2 PathExpert deprivilege arrived keeps"}, indent=2) + "\n")
print("AUTO_REVIEW", json.dumps(summary))
raise SystemExit(0 if gate else 1)
PY
}

if [[ "$SKIP_COLLECT" != "1" ]]; then
  if [[ -f "$DS/AUTO_REVIEW.json" ]] && review_gate; then
    echo "AUTO SKIP_COLLECT gate already passed"
  else
    for round in $(seq 1 "$MAX_ROUNDS"); do
      echo "=== COLLECT round $round/$MAX_ROUNDS ==="
      [[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 8; }
      append=()
      [[ "$round" -gt 1 || -f "$DS/manifest.json" ]] && append+=(--append)
      "$PY" -m experiments.aerial.rl.collect_path_expert_dataset \
        --backend airsim --host 127.0.0.1 --step-hz 5 --grab-depth \
        --episodes "$EP_PER_ROUND" --max-steps 600 --success-dist-m 3 \
        --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 \
        --min-usable-path-m 35 --min-usable-steps 30 \
        --min-spawn-inland-m 100 --min-path-inland-m 80 \
        --route-indices "$ROUTES" \
        --annotation "$ANN_FOCUS" --out "$DS" \
        --keep-failed --skip-min-ok-gate \
        "${append[@]}" || true
      if review_gate; then
        echo "COLLECT_GATE_OK round=$round"
        break
      fi
      echo "gate not passed — continue (rule: do not stop on failing gate)"
    done
    review_gate || echo "WARN collect gate still open after $MAX_ROUNDS — proceed with whatever arrived keeps exist"
  fi
fi

"$PY" - <<PY
import yaml
from pathlib import Path
def deep_merge(a,b):
    out=dict(a or {})
    for k,v in (b or {}).items():
        out[k]=deep_merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out
base=yaml.safe_load(Path("configs/aerial_rl_urban_complex_p2c.yaml").read_text())
oa=yaml.safe_load(Path("configs/aerial_rl_urban_complex_directional_oa.yaml").read_text())
cfg=deep_merge(base,oa); cfg.setdefault("planner",{})["enable"]=False
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

if [[ "$SKIP_BC" != "1" ]]; then
  [[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 6; }
  echo "=== BC from PathExpert DS ==="
  "$PY" -m experiments.aerial.rl.train_v4_ac \
    --config "$CFG" --config-overlay configs/aerial_rl_urban_complex_directional_oa.yaml \
    --backend airsim --dynamics torch \
    --wm-ckpt "$WM" --init-actor-ckpt "$INIT" --dataset "$DS" \
    --no-planner --no-shield \
    --enable-bc --bc-batch 64 --bc-updates-per-iter 12 --bc-loss-scale 1.0 \
    --iters "$BC_ITERS" --episodes-per-iter 2 \
    --imagine-batch 16 --imagine-horizon 15 \
    --device cuda --annotation "$ANN_FOCUS" --ckpt-dir "$OUT" --save-every-iter \
    --min-steps-for-best 20 --near-goal-frac 0.25 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 6 --min-spawn-clear-m 5.0
  echo BC_TRAIN_DONE
fi

ACTOR="$(pick_actor "$OUT" || echo "$INIT")"
echo "ACTOR=$ACTOR"

summarize() {
  TAG="$1" JSON="$2" "$PY" - <<'PY'
import json,os; from pathlib import Path
p=Path(os.environ["JSON"]); tag=os.environ["TAG"]
if not p.exists(): print(f"[{tag}] MISSING"); raise SystemExit(0)
rows=json.loads(p.read_text())["episodes"]; by={int(r["route_idx"]):r for r in rows}
stick=sum(1 for ri in (0,1) if by.get(ri,{}).get("severe_collision")); arr=sum(1 for r in rows if r.get("arrived"))
print(f"[{tag}] SR={arr}/{len(rows)} P0_stick={stick}/2")
Path(p).parent.joinpath("CAPABILITY_SUMMARY.json").write_text(json.dumps({
 "tag":tag,"sr":f"{arr}/{len(rows)}","p0_stick":stick,
 "capability_pass": bool(arr>=1 and stick<=1),
 "note":"B2 PathExpert deprivilege BC; B2 deprivilege: toward_g noshield ONLY (no polyline credit)"}, indent=2)+"\n")
PY
}

if [[ "$SKIP_EVAL" != "1" ]]; then
  eout="$ART/eval_mainchannel_b2_${STAMP}_h${HOST}_gate_noshield"
  mkdir -p "$eout/traj"; set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$ACTOR" \
    --depth-ckpt "$DEPTH" --tau-ckpt "$TAU" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$eout/traj" --out "$eout/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10 --max-steps 600 --success-dist 3 --terminal-pin-rem-m 20 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 --min-spawn-clear-m 5 \
    --no-shield >"$ART/logs/mainchannel_b2_${STAMP}_h${HOST}_gate.log" 2>&1
  set -e
  summarize gate_noshield "$eout/eval_all.json"
fi
echo "B2_PATHEXPERT_DEPRIVILEGE_NOSHIELD_DONE stamp=$STAMP ds=$DS out=$OUT"; date -Is
