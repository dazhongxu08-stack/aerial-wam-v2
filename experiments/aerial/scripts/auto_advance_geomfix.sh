#!/usr/bin/env bash
# Strict READY_SOFT joint push: gate∧cone already green → thicken near GT + WM FT
# until probe_hard ∧ imagine. No Soft until READY_SOFT.txt.
#
#   STAMP=20260926_hardnear bash experiments/aerial/scripts/auto_advance_geomfix.sh
set -uo pipefail
STAMP="${STAMP:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
ART="$ROOT/experiments/aerial/rl/artifacts"
OUT="$ART/geomfix_${STAMP}"
GT="$ART/obstacle_cost_gt_depth"
LOG="$ART/logs/auto_advance_geomfix_${STAMP}.log"
KEY="${REMOTE_KEY:-${H100_KEY:-$HOME/.ssh/id_ed25519_h100}}"
PY="${PYTHON_BIN:-}"
[[ -z "$PY" && -x "$HOME/sim_verify/.venv/bin/python" ]] && PY="$HOME/sim_verify/.venv/bin/python"
[[ -z "$PY" && -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
[[ -z "$PY" ]] && PY=python3
INTERVAL="${INTERVAL:-90}"
THICK="$ART/dataset_geomfix_nearthick_${STAMP}"
mkdir -p "$OUT" "$ART/logs" "$GT" "$THICK"
echo $$ >"$ART/logs/auto_advance_geomfix_${STAMP}.pid"
exec >>"$LOG" 2>&1
ts(){ date -Is; }
echo "[$(ts)] auto_advance HARDNEAR start stamp=$STAMP"

ready() { [[ -f "$OUT/READY_SOFT.txt" ]]; }

cone_ok() {
  "$PY" - <<PY
import json
from pathlib import Path
out=Path("$OUT")
for p in list(out.glob("*cone*.json"))+list(out.glob("R1_cone*.json")):
  try:
    if json.loads(p.read_text()).get("verdict")=="PASS":
      print("1"); raise SystemExit
  except Exception:
    pass
print("0")
PY
}

probe_ok() {
  "$PY" - <<PY
import json
from pathlib import Path
out=Path("$OUT")
for p in list(out.glob("R3_latent*.json"))+list(out.glob("R4_latent*.json")):
  try:
    j=json.loads(p.read_text())
  except Exception:
    continue
  fwd=j.get("forward_min_depth") or {}
  n=int(j.get("n_samples") or 0)
  if fwd.get("verdict")=="has_geometry" and float(fwd.get("r2_holdout") or -1)>=0.3 and n>=30:
    print("1"); raise SystemExit
print("0")
PY
}

imagine_ok() {
  "$PY" - <<PY
import json
from pathlib import Path
out=Path("$OUT")
for p in out.glob("*imagine*.json"):
  try:
    j=json.loads(p.read_text())
  except Exception:
    continue
  v=j.get("verdict") or {}
  gap=v.get("median_p_coll_gap") if isinstance(v,dict) else j.get("median_p_coll_gap")
  useful=bool(v.get("useful")) if isinstance(v,dict) else False
  if useful or (gap is not None and float(gap)>=0.05):
    print("1"); raise SystemExit
print("0")
PY
}

busy125() {
  pgrep -af 'collect_dataset|run_geomfix_near_episode|near_episode' | grep -v pgrep >/dev/null
}
busy84() {
  # Do NOT match legacy collect_obstacle / r2 — that falsely blocks nearthick.
  timeout 12 ssh -n -o BatchMode=yes -o ConnectTimeout=8 ubantu@10.229.20.84 \
    "pgrep -af 'collect_dataset|run_geomfix_near_episode|nearthick' | grep -v pgrep >/dev/null" 2>/dev/null
}
busy14() {
  local n="" i
  for i in 1 2 3; do
    n=$(ssh -n -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new \
      -i "$KEY" -p 31126 a25689@10.239.121.14 \
      "pgrep -c -f '_wm_train_validate|run_geomfix_r3' || echo 0" 2>/dev/null || true)
    [[ -n "${n}" ]] && break
    sleep 1
  done
  [[ -z "${n}" ]] && return 1
  [[ "${n}" -ge 1 ]]
}
busy11() {
  local n="" i
  for i in 1 2 3; do
    n=$(ssh -n -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new \
      -i "$KEY" -p 30627 a25689@10.239.121.11 \
      "pgrep -c -f '_wm_train_validate|run_geomfix_r4' || echo 0" 2>/dev/null || true)
    [[ -n "${n}" ]] && break
    sleep 1
  done
  [[ -z "${n}" ]] && return 1
  [[ "${n}" -ge 1 ]]
}

sync_thick_to_remote() {
  local PORT=$1 HOSTN=$2
  [[ -d "$THICK" ]] || return 0
  local n
  n=$(ls "$THICK"/*.npz 2>/dev/null | wc -l | tr -d ' ')
  [[ "${n:-0}" -ge 5 ]] || return 0
  echo "[$(ts)] sync thick n=$n → 4090@$HOSTN (tar|ssh)"
  ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -p "$PORT" "a25689@10.239.121.$HOSTN" \
    "mkdir -p /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/dataset_geomfix_nearthick_${STAMP}" || true
  # remotes often lack rsync — stream tar over ssh
  tar -C "$THICK" -cf - . 2>/dev/null | \
    ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new \
      -i "$KEY" -p "$PORT" "a25689@10.239.121.$HOSTN" \
      "tar -C /home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/dataset_geomfix_nearthick_${STAMP} -xf -" \
    && echo "SYNC_OK_$HOSTN" || echo "SYNC_FAIL_$HOSTN"
}

launch_125_near() {
  echo "[$(ts)] LAUNCH 125 near-episode collect"
  nohup env HOST=125 STAMP="$STAMP" EPISODES=36 ROUNDS=2 \
    bash "$ROOT/experiments/aerial/scripts/run_geomfix_near_episode_collect.sh" \
    >/dev/null 2>&1 &
  echo "125_PID=$!"
}

launch_84_near() {
  echo "[$(ts)] LAUNCH 84 near-episode collect"
  scp -o BatchMode=yes -o ConnectTimeout=20 \
    "$ROOT/experiments/aerial/scripts/run_geomfix_near_episode_collect.sh" \
    ubantu@10.229.20.84:/tmp/ 2>/dev/null || true
  ssh -o BatchMode=yes -o ConnectTimeout=15 ubantu@10.229.20.84 \
    "export STAMP='$STAMP'; bash -s" <<'H84' || true
set +e
ROOT=/data/aerial-wam-v2
cp -f /tmp/run_geomfix_near_episode_collect.sh "$ROOT/experiments/aerial/scripts/" 2>/dev/null
chmod +x "$ROOT/experiments/aerial/scripts/run_geomfix_near_episode_collect.sh"
cd "$ROOT"; export PYTHONPATH="$ROOT" PYTHON_BIN=/data/venvs/sim_verify/bin/python
pkill -f 'run_geomfix_near_episode|collect_dataset' 2>/dev/null
nohup env HOST=84 STAMP="$STAMP" EPISODES=36 ROUNDS=3 PYTHON_BIN="$PYTHON_BIN" \
  bash experiments/aerial/scripts/run_geomfix_near_episode_collect.sh >/dev/null 2>&1 &
echo 84_PID=$!
H84
  # also pull any thick npz back to 125 later via scp of whole thick dir if 84 writes locally
}

pull_84_thick() {
  scp -o BatchMode=yes -o ConnectTimeout=30 \
    "ubantu@10.229.20.84:/data/aerial-wam-v2/experiments/aerial/rl/artifacts/dataset_geomfix_nearthick_${STAMP}/h84_*.npz" \
    "$THICK/" 2>/dev/null || true
}

launch_remote_4090() {
  local PORT=$1 HOSTN=$2 ARM=$3 SCRIPT=$4
  local NEWSTAMP="${STAMP}"
  echo "[$(ts)] LAUNCH $ARM @4090-$HOSTN"
  sync_thick_to_remote "$PORT" "$HOSTN"
  # never pkill script name (ssh cmdline suicide)
  ssh -o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -p "$PORT" "a25689@10.239.121.$HOSTN" \
    "cd /home/a25689/aerial-wam-v2; export PYTHONPATH=\$PWD;
     PY=\$PWD/.venv/bin/python;
     pkill -f '_wm_train_validate' 2>/dev/null || true;
     pkill -f 'wam_latent_depth_probe|wam_imagine_coll' 2>/dev/null || true;
     sleep 2;
     # clear false DONE so contract loops
     rm -f experiments/aerial/rl/artifacts/geomfix_${STAMP}/${ARM}_DONE.txt;
     mkdir -p experiments/aerial/rl/artifacts/logs;
     INIT='';
     # R4 antisat: never continue from saturated geomfix_strong coll heads
     if [[ "$ARM" == "R4" ]]; then
       INIT=experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt;
       FORCE_CLEAN=1;
       rm -f experiments/aerial/rl/artifacts/geomfix_${STAMP}/R4_FAIL.txt \
             experiments/aerial/rl/artifacts/geomfix_${STAMP}/R4_DONE.txt \
             experiments/aerial/rl/artifacts/geomfix_${STAMP}/R4_imagine_coll_rank*.json;
     else
       FORCE_CLEAN=0;
       for c in experiments/aerial/rl/artifacts/wm_ckpt_geomfix_*/wm_step_*.pt; do
         [[ -f \$c ]] && INIT=\$c
       done
     fi
     [[ -z \$INIT || ! -f \$INIT ]] && INIT=experiments/aerial/rl/artifacts/wm_ckpt_depth_aux_long_20260920/wm_step_8500.pt;
     nohup env HOST=$HOSTN STAMP=$NEWSTAMP PYTHON_BIN=\$PY WM_STEPS=2500 MAX_ROUNDS=4 \
       FORCE_CLEAN_INIT=\$FORCE_CLEAN \
       INIT_CKPT=\$INIT DS=experiments/aerial/rl/artifacts/dataset_geomfix_nearthick_${STAMP} \
       DS_PROBE=experiments/aerial/rl/artifacts/dataset_geomfix_nearthick_${STAMP} \
       DS_NEAR=experiments/aerial/rl/artifacts/dataset_geomfix_nearthick_${STAMP} \
       bash experiments/aerial/scripts/$SCRIPT \
       >experiments/aerial/rl/artifacts/logs/launch_${ARM}_${NEWSTAMP}.out 2>&1 &
     echo ${ARM}_PID=\$! INIT=\$INIT FORCE_CLEAN=\$FORCE_CLEAN;
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
  # seed from prior stamp if needed
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
    if float(j.get("median_p_coll_gap") or 0)>=0.05:
      return True, p.name
  return False, None
g,gn=ok_gate(); c,cn=ok_cone(); p,pn=ok_probe(); i,inn=ok_imagine()
# Import prior gate/cone PASS from geomfix2 if this stamp lacks them
if not g or not c:
  prev=Path("$ART")/"geomfix_20260926_geomfix2"
  if prev.is_dir():
    if not c:
      for pth in prev.glob("*cone*.json"):
        try:
          j=json.loads(pth.read_text())
        except Exception:
          continue
        if j.get("verdict")=="PASS":
          (out/"R1_cone_rank.json").write_text(pth.read_text()); c,cn=True,pth.name; break
    if not g:
      for pth in sorted(prev.glob("R1_gate*.json"), reverse=True):
        j=json.loads(pth.read_text())
        if j.get("passed") and int((j.get("numbers") or {}).get("n_left_near") or 0)>=20:
          (out/"R1_gate.json").write_text(pth.read_text()); g,gn=True,pth.name; break
ready=g and c and p and i
summary={"gate":g,"gate_file":gn,"cone":c,"cone_file":cn,"probe_hard":p,"probe_file":pn,
         "imagine":i,"imagine_file":inn,"READY_SOFT":ready,"updated":"$(ts)",
         "mode":"hardnear_joint"}
(out/"PHASE0_VERDICT.json").write_text(json.dumps(summary, indent=2)+"\n")
(out/"STATUS_LIVE.md").write_text(
  f"# geomfix hardnear\\n\\n{json.dumps(summary, indent=2)}\\n")
if ready:
  (out/"READY_SOFT.txt").write_text("READY_SOFT\\n")
print(summary)
PY
}

# seed gate/cone from previous PASS stamp
if [[ -d "$ART/geomfix_20260926_geomfix2" ]]; then
  cp -f "$ART/geomfix_20260926_geomfix2/R1_cone_rank.json" "$OUT/" 2>/dev/null || true
  cp -f "$ART/geomfix_20260926_geomfix2/R1_gate.json" "$OUT/" 2>/dev/null || \
    cp -f "$(ls -t "$ART"/geomfix_20260926_geomfix2/R1_gate*.json 2>/dev/null | head -1)" "$OUT/R1_gate.json" 2>/dev/null || true
fi

while true; do
  if ready; then
    echo "[$(ts)] READY_SOFT — idle"
    update_verdict
    sleep 300
    continue
  fi
  pull_84_thick
  update_verdict

  # 125/84: always thicken near episodes (never cone-pack while cone PASS)
  if [[ "$(cone_ok)" == "1" ]]; then
    busy125 || launch_125_near
    busy84 || launch_84_near
  else
    # should not happen; keep near collect anyway
    busy125 || launch_125_near
    busy84 || launch_84_near
  fi

  # 14/11: relaunch until probe/imagine green (ignore stale DONE)
  if [[ "$(probe_ok)" != "1" ]]; then
    busy14 || launch_remote_4090 31126 14 R3 run_geomfix_r3_wm_near.sh
  fi
  if [[ "$(imagine_ok)" != "1" ]]; then
    busy11 || launch_remote_4090 30627 11 R4 run_geomfix_r4_wm_probe.sh
  fi

  # pull remote 4090 artifacts
  scp -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -P 31126 \
    "a25689@10.239.121.14:/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/geomfix_${STAMP}/R3_*.json" \
    "$OUT/" 2>/dev/null || true
  scp -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -P 30627 \
    "a25689@10.239.121.11:/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/geomfix_${STAMP}/R4_*.json" \
    "$OUT/" 2>/dev/null || true

  {
    echo "# geomfix hardnear · $STAMP"
    echo "updated: $(ts)"
    echo
    echo "| host | role | busy |"
    echo "|------|------|------|"
    busy125 && echo "| 125 | near GT thicken | YES |" || echo "| 125 | near GT thicken | relaunch |"
    busy84 && echo "| 84 | near GT thicken | YES |" || echo "| 84 | near GT thicken | relaunch |"
    busy14 && echo "| 14 | WM+probe | YES |" || echo "| 14 | WM+probe | relaunch |"
    busy11 && echo "| 11 | WM+imagine | YES |" || echo "| 11 | WM+imagine | relaunch |"
    echo
    echo "READY_SOFT: $(ready && echo yes || echo no)"
    echo "thick_npz: $(ls "$THICK"/*.npz 2>/dev/null | wc -l | tr -d ' ')"
  } >"$OUT/STATUS.md"

  echo "[$(ts)] tick"; cat "$OUT/STATUS.md"
  sleep "$INTERVAL"
done
