#!/usr/bin/env bash
# Mac: pull V1–V4 status from 125; optionally --kick resume dead 125 V1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ART="$ROOT/artifacts"
STAMP="${V_STAMP:-20260925_1007}"
SSH125="${SSH125:-cursor-125-public}"
KICK="${1:-}"
mkdir -p "$ART"

# sync snap script + handoff, run on 125
scp -o ConnectTimeout=20 -o BatchMode=yes \
  "$ROOT/experiments/aerial/scripts/snap_v1234_status.py" \
  "$ROOT/artifacts/HANDOFF_125_V1234.md" \
  "$SSH125:/tmp/" >/dev/null
ssh -o ConnectTimeout=25 -o BatchMode=yes "$SSH125" \
  "mkdir -p ~/aerial-wam-v2/artifacts ~/aerial-wam-v2/experiments/aerial/scripts; \
   cp /tmp/snap_v1234_status.py ~/aerial-wam-v2/experiments/aerial/scripts/; \
   cp /tmp/HANDOFF_125_V1234.md ~/aerial-wam-v2/artifacts/; \
   V_STAMP=$STAMP python3 ~/aerial-wam-v2/experiments/aerial/scripts/snap_v1234_status.py" \
  >"$ART/v1234_pull_raw.json"

scp -o ConnectTimeout=20 -o BatchMode=yes \
  "$SSH125:aerial-wam-v2/artifacts/v1234_STATUS.json" \
  "$ART/v1234_STATUS.json" >/dev/null
scp -o ConnectTimeout=20 -o BatchMode=yes \
  "$SSH125:aerial-wam-v2/artifacts/v1234_RESULTS.jsonl" \
  "$ART/v1234_RESULTS.jsonl" 2>/dev/null || true

python3 - <<'PY'
import json
from pathlib import Path
p = Path("/Users/xudazhong/Projects/aerial-wam-v2/artifacts/v1234_STATUS.json")
d = json.loads(p.read_text())
print(f"STATUS {d.get('updated')} stamp={d.get('stamp')}")
need = []
for h in ("125", "84", "14", "11"):
    s = d["hosts"][h]
    g = s.get("gate") or {}
    print(
        f"  {h} {s.get('lane')}: alive={s.get('alive')} iter={s.get('iter_max')}/80 "
        f"done={s.get('done')} resume={s.get('need_resume')} SR={g.get('sr')}"
    )
    if s.get("need_resume"):
        need.append(h)
        print(f"    last={(s.get('last') or '')[-100:]}")
Path("/Users/xudazhong/Projects/aerial-wam-v2/artifacts/v1234_NEED_RESUME.txt").write_text(
    ",".join(need) + ("\n" if need else "")
)
if need:
    print("NEED_RESUME", ",".join(need))
PY

# Merge new gates into local outcome-style results
if [[ -f "$ART/v1234_RESULTS.jsonl" ]]; then
  touch "$ART/outcome_a_RESULTS.jsonl"
  # append unique lines
  python3 - <<'PY'
from pathlib import Path
src = Path("/Users/xudazhong/Projects/aerial-wam-v2/artifacts/v1234_RESULTS.jsonl")
dst = Path("/Users/xudazhong/Projects/aerial-wam-v2/artifacts/outcome_a_RESULTS.jsonl")
ex = dst.read_text() if dst.exists() else ""
for line in src.read_text().splitlines():
    if not line.strip():
        continue
    if line not in ex:
        with dst.open("a") as f:
            f.write(line + "\n")
        print("APPENDED_RESULT", line[:120])
PY
fi

if [[ "$KICK" == "--kick" ]]; then
  NEED=$(cat "$ART/v1234_NEED_RESUME.txt" 2>/dev/null || true)
  ssh -o ConnectTimeout=25 -o BatchMode=yes "$SSH125" \
    "STAMP='$STAMP' NEED='$NEED' bash -s" <<'KICK'
set -euo pipefail
STAMP=${STAMP:-20260925_1007}
NEED=${NEED:-}
kick125() {
  ROOT=/home/yao/aerial-wam-v2
  CK=$ROOT/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_b_curiosity_${STAMP}_noshield_h125
  G=$ROOT/experiments/aerial/rl/artifacts/eval_mainchannel_b_curiosity_${STAMP}_h125_gate_noshield/eval_all.json
  [[ -f "$G" ]] && return 0
  pgrep -af 'b_curiosity|train_v4_ac.*curiosity' | grep -vq pgrep && return 0
  L=$(ls -1t "$CK"/v4_ac_iter_*.pt 2>/dev/null | head -1 || true)
  [[ -n "$L" ]] || return 0
  n=${L##*_}; n=${n%.pt}; n=$((10#$n)); rem=$((80-n-1)); ((rem<1))&&rem=1
  bash "$HOME/aerial_airsim_persistent/recover_renderer.sh" outdoor >/dev/null 2>&1 || true
  sleep 6
  cd "$ROOT"; export PYTHONPATH="$ROOT"
  nohup env HOST=125 STAMP="$STAMP" SKIP_BASELINE=1 INIT="$L" ITERS="$rem" \
    bash experiments/aerial/scripts/run_mainchannel_b_curiosity_noshield.sh >/dev/null 2>&1 &
  echo KICKED_125 from=$L rem=$rem
}
# 84
kick84() {
  ssh -o BatchMode=yes ubantu@10.229.20.84 "STAMP=$STAMP bash -s" <<'H'
ROOT=/data/aerial-wam-v2; STAMP=${STAMP}
CK=$ROOT/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_v2_floor_b_${STAMP}_noshield_h84
G=$ROOT/experiments/aerial/rl/artifacts/eval_mainchannel_v2_floor_b_${STAMP}_h84_gate_noshield/eval_all.json
[[ -f "$G" ]] && exit 0
pgrep -af 'v2_floor|train_v4_ac.*v2_floor' | grep -vq pgrep && exit 0
L=$(ls -1t "$CK"/v4_ac_iter_*.pt 2>/dev/null | head -1 || true)
# if only baseline done / no ckpt yet, cold restart full script
cd "$ROOT"; export PYTHONPATH="$ROOT" PYTHON_BIN=/data/venvs/sim_verify/bin/python
if [[ -n "$L" ]]; then
  n=${L##*_}; n=${n%.pt}; n=$((10#$n)); rem=$((80-n-1)); ((rem<1))&&rem=1
  nohup env HOST=84 STAMP="$STAMP" PYTHON_BIN="$PYTHON_BIN" SKIP_BASELINE=1 INIT="$L" ITERS="$rem" \
    bash experiments/aerial/scripts/run_mainchannel_v2_floor_then_b.sh >/dev/null 2>&1 &
  echo KICKED_84 from=$L rem=$rem
else
  nohup env HOST=84 STAMP="$STAMP" PYTHON_BIN="$PYTHON_BIN" \
    bash experiments/aerial/scripts/run_mainchannel_v2_floor_then_b.sh >/dev/null 2>&1 &
  echo KICKED_84 cold
fi
H
}
kick14() {
  ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_h100 -p 31126 a25689@10.239.121.14 "STAMP=$STAMP bash -s" <<'H'
ROOT=/home/a25689/aerial-wam-v2; STAMP=${STAMP}
CK=$ROOT/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_b_openside_${STAMP}_noshield_h14
G=$ROOT/experiments/aerial/rl/artifacts/eval_mainchannel_b_openside_${STAMP}_h14_gate_noshield/eval_all.json
[[ -f "$G" ]] && exit 0
pgrep -af 'b_openside|train_v4_ac.*b_openside' | grep -vq pgrep && exit 0
L=$(ls -1t "$CK"/v4_ac_iter_*.pt 2>/dev/null | head -1 || true)
cd "$ROOT"; export PYTHONPATH="$ROOT"
if [[ -n "$L" ]]; then
  n=${L##*_}; n=${n%.pt}; n=$((10#$n)); rem=$((80-n-1)); ((rem<1))&&rem=1
  nohup env HOST=14 STAMP="$STAMP" SKIP_BASELINE=1 INIT="$L" ITERS="$rem" \
    bash experiments/aerial/scripts/run_mainchannel_b_openside_noshield.sh >/dev/null 2>&1 &
  echo KICKED_14 from=$L rem=$rem
else
  nohup env HOST=14 STAMP="$STAMP" SKIP_BASELINE=1 \
    bash experiments/aerial/scripts/run_mainchannel_b_openside_noshield.sh >/dev/null 2>&1 &
  echo KICKED_14 cold
fi
H
}
kick11() {
  ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_h100 -p 30627 a25689@10.239.121.11 "STAMP=$STAMP bash -s" <<'H'
ROOT=/home/a25689/aerial-wam-v2; STAMP=${STAMP}
CK=$ROOT/experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_v4_neargoal_${STAMP}_noshield_h11
G=$ROOT/experiments/aerial/rl/artifacts/eval_mainchannel_v4_neargoal_${STAMP}_h11_gate_noshield/eval_all.json
[[ -f "$G" ]] && exit 0
pgrep -af 'v4_neargoal|train_v4_ac.*v4_neargoal' | grep -vq pgrep && exit 0
L=$(ls -1t "$CK"/v4_ac_iter_*.pt 2>/dev/null | head -1 || true)
cd "$ROOT"; export PYTHONPATH="$ROOT"
if [[ -n "$L" ]]; then
  n=${L##*_}; n=${n%.pt}; n=$((10#$n)); rem=$((80-n-1)); ((rem<1))&&rem=1
  nohup env HOST=11 STAMP="$STAMP" SKIP_BASELINE=1 INIT="$L" ITERS="$rem" \
    bash experiments/aerial/scripts/run_mainchannel_v4_neargoal_noshield.sh >/dev/null 2>&1 &
  echo KICKED_11 from=$L rem=$rem
else
  nohup env HOST=11 STAMP="$STAMP" SKIP_BASELINE=1 \
    bash experiments/aerial/scripts/run_mainchannel_v4_neargoal_noshield.sh >/dev/null 2>&1 &
  echo KICKED_11 cold
fi
H
}
[[ "$NEED" == *125* ]] && kick125 || true
[[ "$NEED" == *84* ]] && kick84 || true
[[ "$NEED" == *14* ]] && kick14 || true
[[ "$NEED" == *11* ]] && kick11 || true
# if NEED empty but file empty, kick none
echo KICK_PASS need="$NEED"
KICK
fi

echo "AGENT_LOOP_TICK_v1234_pull $(date +%Y-%m-%dT%H:%M:%S%z) pulled"
