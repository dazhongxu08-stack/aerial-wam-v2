#!/usr/bin/env bash
# Keep geomfix Phase-0 busy on 125/84/14/11 until READY_SOFT.
# Runs ON 125. Restarts collect / cost reflow / WM FT as arms go idle.
#
#   STAMP=20260925_geomfix bash experiments/aerial/scripts/auto_advance_geomfix.sh
set -uo pipefail
STAMP="${STAMP:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
OUT="$ART/geomfix_${STAMP}"
GT="$ART/obstacle_cost_gt_depth"
LOG="$ART/logs/auto_advance_geomfix_${STAMP}.log"
KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"
PY="${PYTHON_BIN:-}"
[[ -z "$PY" && -x "$HOME/sim_verify/.venv/bin/python" ]] && PY="$HOME/sim_verify/.venv/bin/python"
[[ -z "$PY" && -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
[[ -z "$PY" ]] && PY=python3
WM_BASE="$ART/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt"
[[ -f "$WM_BASE" ]] || WM_BASE="$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
DS_NEAR="$ART/dataset_v0_three_zone_near_20260823fg"
ANN=experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json
INTERVAL="${INTERVAL:-60}"
mkdir -p "$OUT" "$ART/logs" "$GT"
echo $$ >"$ART/logs/auto_advance_geomfix_${STAMP}.pid"
exec >>"$LOG" 2>&1
ts(){ date -Is; }
echo "[$(ts)] auto_advance_geomfix start stamp=$STAMP"

ready() { [[ -f "$OUT/READY_SOFT.txt" ]]; }

busy125() {
  pgrep -af 'collect_obstacle_cost_gt_frames|train_obstacle_cost_labels|wam_obstacle_cost_cone' \
    | grep -v pgrep >/dev/null
}
busy84() {
  timeout 10 ssh -n -o BatchMode=yes -o ConnectTimeout=8 ubantu@10.229.20.84 \
    "pgrep -af 'collect_obstacle|run_geomfix_r2' | grep -v pgrep >/dev/null" 2>/dev/null
}
busy14() {
  local n="" i
  for i in 1 2 3; do
    n=$(ssh -n -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new \
      -i "$KEY" -p 31126 a25689@10.239.121.14 \
      "pgrep -c -f '_wm_train_validate' || echo 0" 2>/dev/null || true)
    [[ -n "${n}" ]] && break
    sleep 1
  done
  # empty after retries → treat idle so we relaunch (SSH flake must not park GPUs)
  [[ -z "${n}" ]] && return 1
  [[ "${n}" -ge 1 ]]
}
busy11() {
  local n="" i
  for i in 1 2 3; do
    n=$(ssh -n -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new \
      -i "$KEY" -p 30627 a25689@10.239.121.11 \
      "pgrep -c -f '_wm_train_validate' || echo 0" 2>/dev/null || true)
    [[ -n "${n}" ]] && break
    sleep 1
  done
  [[ -z "${n}" ]] && return 1
  [[ "${n}" -ge 1 ]]
}

pull_84_frames() {
  scp -o BatchMode=yes -o ConnectTimeout=20 \
    "ubantu@10.229.20.84:/data/aerial-wam-v2/experiments/aerial/rl/artifacts/obstacle_cost_gt_depth/frames_geomfix_84_*_${STAMP}.npz" \
    "$GT/" 2>/dev/null || true
}

launch_125_cycle() {
  local tag="cycle_$(date +%H%M%S)"
  echo "[$(ts)] LAUNCH 125 $tag"
  nohup env STAMP="$STAMP" TAG="$tag" PY="$PY" ROOT="$ROOT" WM_BASE="$WM_BASE" \
    bash -c '
set -e
cd "$ROOT"; export PYTHONPATH="$ROOT"
ART=experiments/aerial/rl/artifacts
GT=$ART/obstacle_cost_gt_depth
OUT=$ART/geomfix_$STAMP
DS=$ART/dataset_v0_three_zone_near_20260823fg
# 1) collect top-up
COL=$GT/frames_geomfix_125_${TAG}.npz
"$PY" -m experiments.aerial.rl.collect_obstacle_cost_gt_frames \
  --config configs/aerial_rl_urban_complex_p2c.yaml \
  --annotation experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json \
  --out "$COL" --host 127.0.0.1 --port 41451 \
  --episodes 40 --steps-per-ep 20 --d-near 5.0 --d-far 18.0 \
  --min-per-group 25 --max-frames 900 || true
# 2) merge multi2 + recent geomfix frames only (avoid unbounded growth / pack stall)
"$PY" - <<PY
import numpy as np
from pathlib import Path
from collections import Counter
gt=Path("experiments/aerial/rl/artifacts/obstacle_cost_gt_depth")
parts=[gt/"frames_multi2.npz"]
parts += sorted(gt.glob("frames_geomfix_125_cycle_*.npz"))[-8:]
parts += sorted(gt.glob("frames_geomfix_84_*.npz"))[-6:]
parts += sorted(gt.glob("frames_geomfix_topup_*.npz"))[-2:]
parts=[p for p in parts if p.is_file()]
keys=["rgb","depth","action_xyz","proprio","group"]
chunks={k:[] for k in keys}
n=0
for p in parts:
  try:
    d=np.load(p, allow_pickle=True)
  except Exception as e:
    print("skip", p, e); continue
  if "group" not in d.files: continue
  for k in keys:
    if k not in d.files: break
  else:
    for k in keys: chunks[k].append(d[k])
    n+=1
    print("add", p.name, len(d["group"]))
out={k:np.concatenate(chunks[k],0) for k in keys}
path=gt/"frames_geomfix_merged_live.npz"
np.savez_compressed(path, **out)
print("MERGED", path, len(out["group"]), dict(Counter(out["group"].astype(str).tolist())), "from", n)
PY
  FR=$GT/frames_geomfix_merged_live.npz
  test -f "$FR"
  LAB=$GT/labels_geomfix_live_${TAG}.npz
  "$PY" -m experiments.aerial.rl.train_obstacle_cost_labels pack \
    --wm-ckpt "$WM_BASE" --frames "$FR" --out "$LAB" --device cuda --d-near 5.0 --d-far 18.0
  # require left_near in pack
  "$PY" - <<PY
import numpy as np, sys
from collections import Counter
g=np.load("$LAB", allow_pickle=True)["group"].astype(str)
c=dict(Counter(g.tolist())); print(c)
sys.exit(0 if c.get("left_near",0)>=20 else 3)
PY
  CKPT=$ART/wm_ckpt_obstacle_cost_geomfix_live_${TAG}/wm_obs.pt
  mkdir -p "$(dirname "$CKPT")"
  "$PY" -m experiments.aerial.rl.train_obstacle_cost_labels train \
    --wm-ckpt "$WM_BASE" --labels "$LAB" --out-ckpt "$CKPT" --steps 2500 --batch 128 --device cuda
  "$PY" -m experiments.aerial.rl.train_obstacle_cost_labels gate \
    --wm-ckpt "$CKPT" --labels "$LAB" --report "$OUT/R1_gate_live_${TAG}.json" --device cuda \
    --require-left-near --min-left-near 20
  "$PY" experiments/aerial/scripts/wam_obstacle_cost_cone_rank.py \
    --dataset "$DS" --wm-ckpt "$CKPT" --device cuda --max-samples 400 \
    --out "$OUT/R1_cone_live_${TAG}.json" || true
  echo "$CKPT" > "$GT/LATEST_REFLOW_CKPT.txt"
  echo "$CKPT" > "$OUT/R1_WM_OBS.txt"
  echo "125_CYCLE_DONE $TAG $(date -Is)"
' >/dev/null 2>&1 &
  echo "125_PID=$!"
}

launch_84() {
  echo "[$(ts)] LAUNCH 84 collect"
  ssh -o BatchMode=yes -o ConnectTimeout=15 ubantu@10.229.20.84 \
    "export STAMP='$STAMP'; bash -s" <<'H84' || true
set +e
ROOT=/data/aerial-wam-v2
cd "$ROOT"; export PYTHONPATH="$ROOT"
PY=/data/venvs/sim_verify/bin/python
pkill -f 'run_geomfix_r2|collect_obstacle_cost' 2>/dev/null
# fresh rounds without relying on DONE marker
nohup env HOST=84 STAMP="${STAMP}_keep" ROUNDS=6 PYTHON_BIN="$PY" \
  bash experiments/aerial/scripts/run_geomfix_r2_collect_84.sh >/dev/null 2>&1 &
echo 84_PID=$!
H84
}

launch_h100() {
  local PORT=$1 HOSTN=$2 ARM=$3 SCRIPT=$4
  local NEWSTAMP="${STAMP}_${ARM}_$(date +%H%M%S)"
  echo "[$(ts)] LAUNCH $ARM @$HOSTN stamp=$NEWSTAMP"
  # NOTE: never pkill -f the script name — it matches this ssh cmdline and suicides.
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -p "$PORT" "a25689@10.239.121.$HOSTN" \
    "cd /home/a25689/aerial-wam-v2; export PYTHONPATH=\$PWD;
     PY=\$PWD/.venv/bin/python;
     pkill -f '_wm_train_validate' 2>/dev/null || true;
     pkill -f 'wam_latent_depth_probe|wam_imagine_collision' 2>/dev/null || true;
     sleep 1;
     INIT='';
     for c in experiments/aerial/rl/artifacts/wm_ckpt_geomfix_*/wm_step_*.pt; do
       [[ -f \$c ]] && INIT=\$c
     done
     [[ -z \$INIT ]] && INIT=experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt
     mkdir -p experiments/aerial/rl/artifacts/logs;
     nohup env HOST=$HOSTN STAMP=$NEWSTAMP PYTHON_BIN=\$PY WM_STEPS=2000 INIT_CKPT=\$INIT \
       bash experiments/aerial/scripts/$SCRIPT \
       >experiments/aerial/rl/artifacts/logs/launch_${ARM}_${NEWSTAMP}.out 2>&1 &
     echo ${ARM}_PID=\$! INIT=\$INIT;
     sleep 2; pgrep -af '_wm_train_validate|run_geomfix_r' | grep -v pgrep | head -5 || echo ${ARM}_NO_PROC_YET" || true
}

update_verdict() {
  "$PY" - <<PY || true
import json
from pathlib import Path
out=Path("$OUT")
def ok_gate():
  for p in sorted(out.glob("R1_gate*.json"), reverse=True):
    j=json.loads(p.read_text())
    if j.get("passed") and int((j.get("numbers") or {}).get("n_left_near") or 0)>=20 \
       and not (j.get("checks") or {}).get("left_near_skipped"):
      return True, p.name
  return False, None
def ok_cone():
  for p in sorted(out.glob("*cone*.json"), reverse=True):
    try:
      j=json.loads(p.read_text())
    except Exception:
      continue
    if j.get("verdict")=="PASS":
      return True, p.name
  return False, None
def ok_probe():
  for p in list(out.glob("R3_latent*.json"))+list(out.glob("R4_latent*.json")):
    j=json.loads(p.read_text())
    fwd=j.get("forward_min_depth") or {}
    if fwd.get("verdict")=="has_geometry" and float(fwd.get("r2_holdout") or -1)>=0.3 and int(j.get("n_samples") or 0)>=30:
      return True, p.name
  return False, None
def ok_imagine():
  for p in out.glob("*imagine*.json"):
    j=json.loads(p.read_text())
    v=j.get("verdict") or {}
    if isinstance(v, dict) and (v.get("useful") or float(v.get("median_p_coll_gap") or 0)>=0.05):
      return True, p.name
  return False, None
g,gn=ok_gate(); c,cn=ok_cone(); p,pn=ok_probe(); i,inn=ok_imagine()
ready=g and c and p and i
summary={"gate":g,"gate_file":gn,"cone":c,"cone_file":cn,"probe_hard":p,"probe_file":pn,
         "imagine":i,"imagine_file":inn,"READY_SOFT":ready,"updated":"$(ts)"}
(out/"PHASE0_VERDICT.json").write_text(json.dumps(summary, indent=2)+"\n")
(out/"STATUS_LIVE.md").write_text(
  f"# geomfix live\\n\\n{json.dumps(summary, indent=2)}\\n")
if ready:
  (out/"READY_SOFT.txt").write_text("READY_SOFT\\n")
print(summary)
PY
}

# ensure watch scripts exist on remotes for r2
while true; do
  if ready; then
    echo "[$(ts)] READY_SOFT — auto_advance idle wait"
    update_verdict
    sleep 300
    continue
  fi
  pull_84_frames
  update_verdict

  # disk hygiene: keep only newest live cost ckpt (each ~657MB)
  (
    cd "$ART" || exit 0
    ls -dt wm_ckpt_obstacle_cost_geomfix_live_cycle_* 2>/dev/null | tail -n +2 | xargs -r rm -rf
  ) || true

  busy125 || launch_125_cycle
  busy84 || launch_84
  busy14 || launch_h100 31126 14 R3 run_geomfix_r3_wm_near.sh
  busy11 || launch_h100 30627 11 R4 run_geomfix_r4_wm_probe.sh

  # pull remote artifacts
  scp -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -P 31126 \
    "a25689@10.239.121.14:/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/geomfix_${STAMP}*/R3_*.json" \
    "$OUT/" 2>/dev/null || true
  scp -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -P 30627 \
    "a25689@10.239.121.11:/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/geomfix_${STAMP}*/R4_*.json" \
    "$OUT/" 2>/dev/null || true

  {
    echo "# geomfix auto-advance · $STAMP"
    echo "updated: $(ts)"
    echo
    echo "| host | busy |"
    echo "|------|------|"
    busy125 && echo "| 125 | YES |" || echo "| 125 | relaunched |"
    busy84 && echo "| 84 | YES |" || echo "| 84 | relaunched |"
    busy14 && echo "| 14 | YES |" || echo "| 14 | relaunched |"
    busy11 && echo "| 11 | YES |" || echo "| 11 | relaunched |"
    echo
    echo "READY_SOFT: $(ready && echo yes || echo no)"
  } >"$OUT/STATUS.md"

  echo "[$(ts)] tick"; cat "$OUT/STATUS.md"
  sleep "$INTERVAL"
done
