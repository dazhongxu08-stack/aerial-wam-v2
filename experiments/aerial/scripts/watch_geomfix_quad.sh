#!/usr/bin/env bash
# Watch R1–R4 geomfix; pull markers; restart dead arms; write STATUS.
# Runs on 125.
set -euo pipefail
STAMP="${STAMP:?}"
INTERVAL="${INTERVAL:-90}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ART="$ROOT/experiments/aerial/rl/artifacts"
OUT="$ART/geomfix_${STAMP}"
LOG="$ART/logs/watch_geomfix_${STAMP}.log"
KEY="${H100_KEY:-$HOME/.ssh/id_ed25519_h100}"
mkdir -p "$OUT" "$ART/logs"
echo $$ >"$ART/logs/watch_geomfix_${STAMP}.pid"
exec >>"$LOG" 2>&1
ts(){ date -Is; }
echo "[$(ts)] watch_geomfix start"

is_done() {
  local a=$1
  [[ -f "$OUT/${a}_DONE.txt" || -f "$OUT/${a}_SKIP.txt" ]]
}
is_fail() { [[ -f "$OUT/$1_FAIL.txt" ]]; }

alive_125() { pgrep -af 'run_geomfix_r1' | grep -v pgrep >/dev/null; }
alive_84() {
  timeout 10 ssh -n -o BatchMode=yes -o ConnectTimeout=8 ubantu@10.229.20.84 \
    "pgrep -af run_geomfix_r2 | grep -v pgrep >/dev/null" 2>/dev/null
}
alive_14() {
  ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -p 31126 a25689@10.239.121.14 \
    "pgrep -af run_geomfix_r3 | grep -v pgrep >/dev/null" 2>/dev/null
}
alive_11() {
  ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -p 30627 a25689@10.239.121.11 \
    "pgrep -af run_geomfix_r4 | grep -v pgrep >/dev/null" 2>/dev/null
}

pull_remote() {
  scp -o BatchMode=yes -o ConnectTimeout=12 \
    ubantu@10.229.20.84:/data/aerial-wam-v2/experiments/aerial/rl/artifacts/geomfix_${STAMP}/R2_* \
    "$OUT/" 2>/dev/null || true
  scp -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -P 31126 \
    a25689@10.239.121.14:/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/geomfix_${STAMP}/R3_* \
    "$OUT/" 2>/dev/null || true
  scp -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
    -i "$KEY" -P 30627 \
    a25689@10.239.121.11:/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/geomfix_${STAMP}/R4_* \
    "$OUT/" 2>/dev/null || true
  # also pull 84 collect npz into GT for later merge
  scp -o BatchMode=yes -o ConnectTimeout=20 \
    ubantu@10.229.20.84:/data/aerial-wam-v2/experiments/aerial/rl/artifacts/obstacle_cost_gt_depth/frames_geomfix_84_*_${STAMP}.npz \
    "$ART/obstacle_cost_gt_depth/" 2>/dev/null || true
}

restart() {
  local a=$1
  echo "[$(ts)] RESTART $a"
  case "$a" in
    R1)
      cd "$ROOT"; export PYTHONPATH="$ROOT"
      nohup env HOST=125 STAMP="$STAMP" \
        bash experiments/aerial/scripts/run_geomfix_r1_cost_reflow.sh >/dev/null 2>&1 &
      ;;
    R2)
      ssh -o BatchMode=yes ubantu@10.229.20.84 \
        "cd /data/aerial-wam-v2; export PYTHONPATH=/data/aerial-wam-v2 STAMP=$STAMP PYTHON_BIN=/data/venvs/sim_verify/bin/python;
         nohup env HOST=84 STAMP=$STAMP PYTHON_BIN=\$PYTHON_BIN ROUNDS=2 \
           bash experiments/aerial/scripts/run_geomfix_r2_collect_84.sh >/dev/null 2>&1 &" || true
      ;;
    R3)
      ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$KEY" -p 31126 a25689@10.239.121.14 \
        "cd /home/a25689/aerial-wam-v2; export PYTHONPATH=\$PWD STAMP=$STAMP;
         PY=\$PWD/.venv/bin/python; nohup env HOST=14 STAMP=$STAMP PYTHON_BIN=\$PY \
           bash experiments/aerial/scripts/run_geomfix_r3_wm_near.sh >/dev/null 2>&1 &" || true
      ;;
    R4)
      ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$KEY" -p 30627 a25689@10.239.121.11 \
        "cd /home/a25689/aerial-wam-v2; export PYTHONPATH=\$PWD STAMP=$STAMP;
         PY=\$PWD/.venv/bin/python; nohup env HOST=11 STAMP=$STAMP PYTHON_BIN=\$PY \
           bash experiments/aerial/scripts/run_geomfix_r4_wm_probe.sh >/dev/null 2>&1 &" || true
      ;;
  esac
}

while true; do
  pull_remote
  {
    echo "# geomfix STATUS · $STAMP"
    echo "updated: $(ts)"
    echo
    echo "| arm | host | state |"
    echo "|-----|------|-------|"
  } >"$OUT/STATUS.md.tmp"

  all=1
  for a in R1 R2 R3 R4; do
    st=RUNNING
    if is_done "$a"; then st=DONE
    elif is_fail "$a"; then st=FAIL; all=0
    else
      all=0
      case "$a" in
        R1) alive_125 || { st=DEAD_RESTART; restart R1; } ;;
        R2) [[ -f "$OUT/R2_SKIP.txt" ]] && st=SKIP || { alive_84 || { st=DEAD_RESTART; restart R2; }; } ;;
        R3) alive_14 || { st=DEAD_RESTART; restart R3; } ;;
        R4) alive_11 || { st=DEAD_RESTART; restart R4; } ;;
      esac
    fi
    host=125; [[ $a == R2 ]] && host=84; [[ $a == R3 ]] && host=14; [[ $a == R4 ]] && host=11
    echo "| $a | $host | $st |" >>"$OUT/STATUS.md.tmp"
  done
  mv "$OUT/STATUS.md.tmp" "$OUT/STATUS.md"
  echo "[$(ts)] tick"; cat "$OUT/STATUS.md"

  # Soft unlock marker (never auto-start Soft here — agent decides)
  if [[ -f "$OUT/R1_DONE.txt" && -f "$OUT/R3_DONE.txt" && -f "$OUT/R4_DONE.txt" ]]; then
    echo "PHASE0_ARMS_DONE $(ts)" >"$OUT/PHASE0_ARMS_DONE.txt"
    # summarize pass/fail for Soft gate
    /home/yao/sim_verify/.venv/bin/python - <<PY || true
import json
from pathlib import Path
out=Path("$OUT")
def ok_probe(p):
  if not p.exists(): return False
  j=json.loads(p.read_text())
  fwd=j.get("forward_min_depth") or {}
  return fwd.get("verdict")=="has_geometry" and float(fwd.get("r2_holdout") or -1)>=0.3
def ok_gate(p):
  if not p.exists(): return False
  j=json.loads(p.read_text())
  return bool(j.get("passed")) and not j.get("checks",{}).get("left_near_skipped") and int((j.get("numbers") or {}).get("n_left_near") or 0)>=20
def ok_cone(p):
  return p.exists() and json.loads(p.read_text()).get("verdict")=="PASS"
def ok_imagine(p):
  if not p.exists(): return False
  j=json.loads(p.read_text())
  v=j.get("verdict") or {}
  gap=v.get("median_p_coll_gap") if isinstance(v,dict) else j.get("median_p_coll_gap")
  useful=v.get("useful") if isinstance(v,dict) else False
  return bool(useful) or (gap is not None and float(gap)>=0.05)
g1=ok_gate(out/"R1_gate_v2.json") or ok_gate(out/"R1_gate.json")
c1=ok_cone(out/"R1_cone_rank_v2.json") or ok_cone(out/"R1_cone_rank.json") or ok_cone(out/"R4_cone_rank_on_R1.json")
p3=ok_probe(out/"R3_latent_probe_three_zone_v2.json") or ok_probe(out/"R3_latent_probe_three_zone.json")
p4=ok_probe(out/"R4_latent_probe_three_zone.json")
im=ok_imagine(out/"R4_imagine_coll_rank.json")
ready=g1 and c1 and (p3 or p4) and im
summary={"gate":g1,"cone":c1,"probe_hard":bool(p3 or p4),"imagine":im,"READY_SOFT":ready}
(out/"PHASE0_VERDICT.json").write_text(json.dumps(summary,indent=2)+"\n")
(out/"PHASE0_VERDICT.md").write_text("# Phase0 verdict\\n\\n"+json.dumps(summary,indent=2)+"\\n")
print(summary)
if ready:
  (out/"READY_SOFT.txt").write_text("READY_SOFT\\n")
PY
    if [[ "$all" -eq 1 ]]; then
      echo "[$(ts)] all arms terminal — watch exit"
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
