#!/usr/bin/env bash
# Soft L-main · noshield · terminate_on_collision=false · freeze cost (WM frozen in AC).
# Soft gate: R0/R1 progress≥0.5 & not stuck ≥K before Hard.
#
#   HOST=84 STAMP=20260926_soft bash experiments/aerial/scripts/run_mainchannel_soft_noshield.sh
set -euo pipefail
HOST="${HOST:?set HOST}"
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
ITERS="${ITERS:-120}"
EPI_PER_ITER="${EPI_PER_ITER:-4}"
NEAR_GOAL_FRAC="${NEAR_GOAL_FRAC:-0.25}"
IMAGINE_HORIZON="${IMAGINE_HORIZON:-40}"
IMAGINE_BATCH="${IMAGINE_BATCH:-16}"
# Soft early-stop: stop when Soft positive gate writes READY_HARD (K met).
SOFT_EARLY_STOP="${SOFT_EARLY_STOP:-1}"
SOFT_GATE_DIR="${SOFT_GATE_DIR:-$ART/breakout_soft_${STAMP}}"
SKIP_BASELINE="${SKIP_BASELINE:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
ANN="${ANN:-experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json}"
ANN_FOCUS="$ROOT/experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json"
RECOVER="${RECOVER_SCRIPT:-$HOME/aerial_airsim_persistent/recover_renderer.sh}"
OUT="${OUT:-$ART/v4_ac_ckpt_mainchannel_soft_${STAMP}_noshield_h${HOST}}"
MASTER="$ART/logs/mainchannel_soft_${STAMP}_h${HOST}.log"
CFG="$ROOT/configs/_tmp_mainchannel_soft_h${HOST}.yaml"
mkdir -p "$ART/logs" "$OUT"
exec >>"$MASTER" 2>&1

echo "=== Soft noshield HOST=$HOST stamp=$STAMP iters=$ITERS ==="
hostname; date -Is
echo "POLICY: terminate_on_collision=false; w_collision light; --no-planner --no-shield; no Hard yet"

pick_actor() {
  local d=$1
  [[ -f "$d/v4_ac_best.pt" ]] && { echo "$d/v4_ac_best.pt"; return 0; }
  [[ -f "$d/v4_ac_latest.pt" ]] && { echo "$d/v4_ac_latest.pt"; return 0; }
  return 1
}

INIT="${INIT:-}"
if [[ -z "$INIT" ]]; then
  INIT="$(pick_actor "$ART/v4_ac_ckpt_urban_directional_oa_20260922_022358_best_sr2" || true)"
fi
[[ -f "${INIT:-}" ]] || { echo FATAL no INIT floor; exit 1; }

# Soft WM MUST be geomfix/antisat imagine-PASS (+ obstacle cost). Never prefer bare gatefix.
MERGED_DEFAULT="$ART/wm_ckpt_soft_init_antisat_cost/wm_soft_init.pt"
if [[ -n "${WM:-}" && -f "$WM" ]]; then :
elif [[ -f "$MERGED_DEFAULT" ]]; then
  WM="$MERGED_DEFAULT"
elif [[ -f "$ART/wm_ckpt_geomfix_strong_20260926_hardnear_h11/wm_step_2500.pt" ]]; then
  WM="$ART/wm_ckpt_geomfix_strong_20260926_hardnear_h11/wm_step_2500.pt"
  echo "WARN using antisat WM without merged cost head — expect obstacle_cost_trained=False"
elif [[ -f "$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt" ]]; then
  echo "FATAL refuse bare gatefix wm_obs for Soft — merge antisat+cost first (merge_soft_wm_init.py)"
  exit 2
else
  echo FATAL need antisat/geomfix or merged Soft WM; exit 1
fi
DEPTH="${DEPTH_CKPT:-$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt}"
TAU="${TAU_CKPT:-$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt}"
# Prefer absolute depth if relative missing (noshield Soft still builds predictor from yaml).
if [[ ! -f "$DEPTH" && -f "$ROOT/$DEPTH" ]]; then DEPTH="$ROOT/$DEPTH"; fi
if [[ ! -f "$DEPTH" ]]; then
  echo "WARN depth ckpt missing ($DEPTH) — Soft continues; shield unused under --no-shield"
  DEPTH=""
fi
echo "INIT=$INIT WM=$WM"
echo "SOFT_EARLY_STOP=$SOFT_EARLY_STOP GATE_DIR=$SOFT_GATE_DIR"

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
oa=yaml.safe_load(Path("configs/aerial_rl_urban_complex_directional_oa.yaml").read_text())
soft=yaml.safe_load(Path("configs/aerial_rl_urban_complex_soft_noshield.yaml").read_text())
cfg=deep_merge(deep_merge(base, oa), soft)
cfg.setdefault("planner", {})["enable"]=False
r=cfg.get("reward") or {}
assert r.get("terminate_on_collision") is False, r
assert float(r.get("w_collision") or 0) <= 2.0, r
assert r.get("peel_despite_progress") is True, r
Path("$CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("cfg Soft OK", {k:r.get(k) for k in ("terminate_on_collision","w_collision","w_press_wall","peel_despite_progress","use_learned_obstacle_cost")})
PY

summarize() {
  local tag=$1 json=$2
  TAG="$tag" JSON="$json" "$PY" - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ["JSON"]); tag=os.environ["TAG"]
if not p.exists():
    print(f"[{tag}] MISSING"); raise SystemExit(0)
rows=json.loads(p.read_text())["episodes"]
by={int(r["route_idx"]):r for r in rows}
stick=sum(1 for ri in (0,1) if by.get(ri,{}).get("severe_collision"))
arr=sum(1 for r in rows if r.get("arrived"))
# Soft positive: R0/R1 progress proxy via 1 - d_final/d0 if present, else path_progress
positives=0
for ri in (0,1):
    r=by.get(ri) or {}
    prog=r.get("progress_frac")
    if prog is None:
        d0=float(r.get("d0_m") or r.get("d_start_m") or 0) or 0.0
        df=float(r.get("d_final_m") or 0)
        prog=(d0-df)/d0 if d0>1e-3 else 0.0
    stuck=bool(r.get("severe_collision"))
    if float(prog)>=0.5 and not stuck:
        positives+=1
print(f"[{tag}] SR={arr}/{len(rows)} P0_stick={stick}/2 soft_pos={positives}/2 "
      f"R0df={round(float(by.get(0,{}).get('d_final_m') or 0),1)} "
      f"R1df={round(float(by.get(1,{}).get('d_final_m') or 0),1)}")
Path(p).parent.joinpath("SOFT_SUMMARY.json").write_text(json.dumps({
    "tag": tag, "sr": f"{arr}/{len(rows)}", "p0_stick": stick,
    "soft_positives_r01": positives,
    "note": "Soft noshield; Hard blocked until cumulative soft_positives>=K",
}, indent=2)+"\n")
PY
}

run_eval() {
  local tag=$1 actor=$2
  local eout="$ART/eval_mainchannel_soft_${STAMP}_h${HOST}_${tag}"
  mkdir -p "$eout/traj"
  echo "=== EVAL $tag ==="
  set +e
  "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$actor" \
    --depth-ckpt "$DEPTH" --tau-ckpt "$TAU" --goal-feat-mode meter \
    --routes 0,1,3,4 --traj-out "$eout/traj" --out "$eout/eval_all.json" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10 --max-steps 600 --success-dist 3 --terminal-pin-rem-m 20 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5 \
    --no-shield \
    >"$ART/logs/mainchannel_soft_${STAMP}_h${HOST}_${tag}.log" 2>&1
  set -e
  summarize "$tag" "$eout/eval_all.json"
}

[[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 8; }

if [[ "$SKIP_BASELINE" != "1" ]]; then
  run_eval "baseline_noshield" "$INIT"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "=== TRAIN Soft iters=$ITERS early_stop=$SOFT_EARLY_STOP ==="
  "$PY" experiments/aerial/scripts/build_outdoor_complex_focus_annotation.py \
    --src "$ANN" --out "$ANN_FOCUS" --routes 0,1,3,4 || true
  [[ -x "$RECOVER" ]] && { bash "$RECOVER" outdoor >/dev/null 2>&1 || true; sleep 6; }
  TRAIN_ARGS=(
    -m experiments.aerial.rl.train_v4_ac
    --config "$CFG"
    --backend airsim --dynamics torch
    --wm-ckpt "$WM" --init-actor-ckpt "$INIT"
    --no-planner --no-shield
    --iters "$ITERS" --episodes-per-iter "$EPI_PER_ITER"
    --imagine-batch "$IMAGINE_BATCH" --imagine-horizon "$IMAGINE_HORIZON"
    --device cuda --annotation "$ANN_FOCUS"
    --ckpt-dir "$OUT" --save-every-iter
    --min-steps-for-best 20 --near-goal-frac "$NEAR_GOAL_FRAC"
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 6
    --min-spawn-clear-m 5.0
  )
  if [[ "$SOFT_EARLY_STOP" == "1" ]]; then
    mkdir -p "$SOFT_GATE_DIR"
    "$PY" "${TRAIN_ARGS[@]}" &
    TRAIN_PID=$!
    echo "TRAIN_PID=$TRAIN_PID watching $SOFT_GATE_DIR/READY_HARD.txt"
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
      if [[ -f "$SOFT_GATE_DIR/READY_HARD.txt" ]]; then
        echo "SOFT_EARLY_STOP READY_HARD — killing Soft train $(date -Is)"
        kill "$TRAIN_PID" 2>/dev/null || true
        sleep 5
        kill -9 "$TRAIN_PID" 2>/dev/null || true
        echo "SOFT_STOPPED_ON_GATE" | tee "$OUT/SOFT_EARLY_STOP.txt"
        break
      fi
      sleep 60
    done
    wait "$TRAIN_PID" 2>/dev/null || true
  else
    "$PY" "${TRAIN_ARGS[@]}"
  fi
  echo TRAIN_DONE
fi

ACTOR="$(pick_actor "$OUT" || echo "$INIT")"
echo "ACTOR=$ACTOR"

if [[ "$SKIP_EVAL" != "1" ]]; then
  run_eval "gate_noshield" "$ACTOR"
fi

echo "SOFT_NOSHIELD_DONE stamp=$STAMP out=$OUT"
echo "HARD_BLOCKED_UNTIL_SOFT_POSITIVES — do not launch Hard from this script"
date -Is
