#!/usr/bin/env bash
# Record route 0 as the policy flew it, route 1 as teacher-then-policy,
# repeat the route-0 episode, clone from the checkpoint that flies route 0,
# then fly both routes with no teacher.
set -euo pipefail
cd /data/aerial-wam-v2
export PYTHONPATH=/data/aerial-wam-v2
ART=experiments/aerial/rl/artifacts
PY=/data/venvs/sim_verify/bin/python
CKPT=$ART/v4_ac_ckpt_gap_bc_20261007/v4_ac_latest.pt
WM=$ART/wm_ckpt_soft_init_antisat_cost/wm_soft_init.pt
CFG=configs/_tmp_mainchannel_soft_h84.yaml
ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json
R0=$ART/gap_anchor_r0_20261007
R1=$ART/gap_anchor_r1_20261007
mkdir -p "$R0" "$R1"

echo "COLLECT_R0 $(date -Is)"
"$PY" -m experiments.aerial.scripts.fly_gap_passage \
  --annotation "$ANN" --routes 0 --out "$R0" --max-steps 220 \
  --record-executed --learner-ckpt "$CKPT" --wm-ckpt "$WM" --config "$CFG" \
  > "$R0/fly.log" 2>&1

echo "COLLECT_R1 $(date -Is)"
"$PY" -m experiments.aerial.scripts.fly_gap_passage \
  --annotation "$ANN" --routes 1 --out "$R1" --max-steps 320 \
  --early-west --record-executed \
  --learner-ckpt "$CKPT" --wm-ckpt "$WM" --config "$CFG" \
  > "$R1/fly.log" 2>&1

"$PY" - << 'PY'
import json, pathlib, sys
r0 = json.loads(pathlib.Path("experiments/aerial/rl/artifacts/gap_anchor_r0_20261007/route00_summary.json").read_text())
r1 = json.loads(pathlib.Path("experiments/aerial/rl/artifacts/gap_anchor_r1_20261007/route01_summary.json").read_text())
print("summary", r0.get("arrived"), r0.get("d_final"), r1.get("arrived"), r1.get("d_final"), "n1", r1.get("n_intervene"))
sys.exit(0 if r0.get("arrived") and r1.get("arrived") else 2)
PY

DS=$ART/gap_anchor_mix_20261007
mkdir -p "$DS"
cp -f "$R0/episode_00000.npz" "$DS/episode_00000.npz"
cp -f "$R0/episode_00000.npz" "$DS/episode_00001.npz"
cp -f "$R0/episode_00000.npz" "$DS/episode_00002.npz"
cp -f "$R1/episode_00000.npz" "$DS/episode_00003.npz"
OUT=$ART/v4_ac_ckpt_gap_anchor_20261007
mkdir -p "$OUT"
echo "BC $(date -Is)"
"$PY" -m experiments.aerial.rl.train_v4_ac \
  --config "$CFG" --backend airsim --dynamics torch \
  --wm-ckpt "$WM" --init-actor-ckpt "$CKPT" \
  --no-planner --no-shield --bc-only --enable-bc \
  --dataset "$DS" --iters 6 --bc-updates-per-iter 4 --bc-batch 32 \
  --imagine-horizon 15 --device cuda \
  --ckpt-dir "$OUT" --save-every-iter \
  --annotation experiments/aerial/phase3_unified/annotations/outdoor_complex_focus134_inland.json \
  > "$ART/logs/gap_anchor_bc_20261007.log" 2>&1
echo "FLY $(date -Is)"
FLY=$ART/gap_anchor_fly_20261007
mkdir -p "$FLY"
"$PY" -m experiments.aerial.scripts.fly_gap_passage \
  --annotation "$ANN" --routes 0,1 --out "$FLY" --max-steps 250 \
  --learner-ckpt "$OUT/v4_ac_latest.pt" --wm-ckpt "$WM" --config "$CFG" \
  > "$FLY/fly.log" 2>&1
echo "DONE $(date -Is)"
grep result "$FLY/fly.log"
