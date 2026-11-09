#!/usr/bin/env bash
# Three attempts to fly route 1's side passage. The frozen actor is not updated.
#   1. image residual, plus the frames from the flight that turned and then quit
#   2. the same correction from forward/left/right clearance
#   3. an 8-step stick read from the current picture
set -uo pipefail
cd /data/aerial-wam-v2
export PYTHONPATH=/data/aerial-wam-v2
export PYTHONUNBUFFERED=1
PY=/data/venvs/sim_verify/bin/python
ART=experiments/aerial/rl/artifacts
mkdir -p "$ART/logs"
exec >>"$ART/logs/overnight_route1_20261007.log" 2>&1
echo "START $(date -Is)"

CFG=configs/_tmp_mainchannel_soft_h84.yaml
WM=$ART/wm_ckpt_soft_init_antisat_cost/wm_soft_init.pt
ACTOR=$ART/v4_ac_ckpt_gap_bc_20261007/v4_ac_latest.pt
ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json
DEPTH=$ART/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt
TAU=$ART/tau_ckpt_foe_r60_20260815/tau_foe_calibrator.pt
R0=$ART/gap_nearwall_collect_20261007/episode_00000.npz
TEACHER=$ART/gap_passage_20261007f/episode_00000.npz

fly() {
  local tag=$1 route=$2
  shift 2
  local eout=$ART/overnight_r1_${tag}
  mkdir -p "$eout/traj"
  echo "=== FLY $tag route=$route $(date -Is) $* ==="
  env "$@" "$PY" -m experiments.aerial.scripts.wam_phase2_long_eval \
    --config "$CFG" --annotation "$ANN" --wm-ckpt "$WM" --actor-ckpt "$ACTOR" \
    --depth-ckpt "$DEPTH" --tau-ckpt "$TAU" --goal-feat-mode meter \
    --routes "$route" --traj-out "$eout/traj" --out "$eout/eval_all.json" \
    --save-dataset "$eout" \
    --subgoal-source toward_g --r-m-intent 100 \
    --cruise-speed 10 --max-steps 600 --success-dist 3 --terminal-pin-rem-m 20 \
    --min-spawn-z 24 --spawn-z-retry-m 5 --spawn-z-max-retries 4 \
    --min-spawn-clear-m 5 --no-shield --encode-current \
    >"$ART/logs/overnight_r1_${tag}.log" 2>&1 || true
  "$PY" - << PY
import json
p = "$eout/eval_all.json"
try:
    e = json.load(open(p))["episodes"][0]
except Exception as exc:
    print(f"RESULT {tag} MISSING {exc}")
else:
    print(f"RESULT {tag} arrived={e.get('arrived')} d_final={e.get('d_final_m')} steps={e.get('steps')} collided={e.get('collided')}")
PY
}

echo "=== A relabel + image residual v2 $(date -Is) ==="
"$PY" -m experiments.aerial.scripts.relabel_gap_actions \
  --src "$ART/visual_r1_dagger_20261007/episode_00000.npz" \
  --out-dir "$ART/visual_r1_dagger_labeled" || echo RELABEL_FAIL
"$PY" -m experiments.aerial.scripts.train_visual_residual \
  --route0 "$R0" \
  --teacher "$TEACHER" "$ART/visual_r1_dagger_labeled/episode_00000.npz" \
  --actor-ckpt "$ACTOR" --wm-ckpt "$WM" --config "$CFG" \
  --out "$ART/visual_residual_route1_v2.pt" \
  --steps 400 --turn-weight 40 --hold 12 || echo TRAIN_VISUAL_V2_FAIL
fly visual_v2 1 VISUAL_RESIDUAL=$ART/visual_residual_route1_v2.pt

if [[ -f $ART/overnight_r1_visual_v2/episode_00000.npz ]]; then
  echo "=== A2 second image round $(date -Is) ==="
  "$PY" -m experiments.aerial.scripts.relabel_gap_actions \
    --src "$ART/overnight_r1_visual_v2/episode_00000.npz" \
    --out-dir "$ART/visual_r1_dagger_v2_labeled" || echo RELABEL2_FAIL
  "$PY" -m experiments.aerial.scripts.train_visual_residual \
    --route0 "$R0" \
    --teacher "$TEACHER" \
      "$ART/visual_r1_dagger_labeled/episode_00000.npz" \
      "$ART/visual_r1_dagger_v2_labeled/episode_00000.npz" \
    --actor-ckpt "$ACTOR" --wm-ckpt "$WM" --config "$CFG" \
    --out "$ART/visual_residual_route1_v3.pt" \
    --steps 400 --turn-weight 40 --hold 12 || echo TRAIN_VISUAL_V3_FAIL
  fly visual_v3 1 VISUAL_RESIDUAL=$ART/visual_residual_route1_v3.pt
fi

echo "=== B clearance head $(date -Is) ==="
"$PY" -m experiments.aerial.scripts.train_clearance_head \
  --route0 "$R0" \
  --teacher "$TEACHER" "$ART/visual_r1_dagger_labeled/episode_00000.npz" \
  --actor-ckpt "$ACTOR" --wm-ckpt "$WM" --config "$CFG" \
  --out "$ART/clearance_head_route1_20261007.pt" || echo TRAIN_CLEAR_FAIL
fly clearance 1 CLEARANCE_HEAD=$ART/clearance_head_route1_20261007.pt

echo "=== C action chunk $(date -Is) ==="
"$PY" -m experiments.aerial.scripts.train_action_chunk \
  --route0 "$R0" \
  --teacher "$TEACHER" "$ART/visual_r1_dagger_labeled/episode_00000.npz" \
  --out "$ART/action_chunk_route1_20261007.pt" --steps 400 || echo TRAIN_CHUNK_FAIL
fly chunk 1 ACTION_CHUNK=$ART/action_chunk_route1_20261007.pt

echo "=== route 0 checks $(date -Is) ==="
if [[ -f $ART/visual_residual_route1_v3.pt ]]; then
  fly r0_visual 0 VISUAL_RESIDUAL=$ART/visual_residual_route1_v3.pt
elif [[ -f $ART/visual_residual_route1_v2.pt ]]; then
  fly r0_visual 0 VISUAL_RESIDUAL=$ART/visual_residual_route1_v2.pt
fi
fly r0_clear 0 CLEARANCE_HEAD=$ART/clearance_head_route1_20261007.pt
fly r0_chunk 0 ACTION_CHUNK=$ART/action_chunk_route1_20261007.pt

echo "END $(date -Is)"
grep RESULT "$ART/logs/overnight_route1_20261007.log" || true
