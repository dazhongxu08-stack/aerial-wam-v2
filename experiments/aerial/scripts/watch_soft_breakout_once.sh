#!/usr/bin/env bash
# Mac-side Soft watch: prefer hub STATUS on 125 (avoids nested hop flakes).
set +e
STAMP="${STAMP:-20260926_soft}"
STATE="${STATE:-$HOME/Projects/aerial-wam-v2/artifacts/soft_breakout_watch_state.json}"
ROOT="${ROOT:-$HOME/Projects/aerial-wam-v2}"
TMP="$ROOT/artifacts/soft_breakout_watch_raw.txt"
mkdir -p "$(dirname "$STATE")" "$(dirname "$TMP")"

ssh -o ConnectTimeout=25 -o BatchMode=yes cursor-125-public "bash -s" <<REMOTE >"$TMP"
set +e
cat /home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts/soft_breakout_STATUS.json 2>/dev/null
echo ---MD---
cat /home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts/soft_breakout_STATUS.md 2>/dev/null
echo ---ALERTS---
tail -8 /home/yao/aerial-wam-v2/experiments/aerial/rl/artifacts/logs/soft_breakout_ALERTS_${STAMP}.log 2>/dev/null
echo ---MON---
pgrep -af soft_breakout_monitor | grep -v pgrep | head -2
REMOTE

python3 - <<PY
import json, re, time
from pathlib import Path
raw = Path("$TMP").read_text(errors="replace")
state_path = Path("$STATE")
prev = {}
if state_path.exists():
    try:
        prev = json.loads(state_path.read_text())
    except Exception:
        prev = {}

hub = {}
m = re.search(r"\{[\s\S]*?\n\}", raw)
if m:
    try:
        hub = json.loads(m.group(0))
    except Exception:
        hub = {}

alive = int(hub.get("soft84") or 0)
gate = 1 if int(hub.get("gate14") or 0) or int(hub.get("gate11") or 0) else 0
nckpt = int(hub.get("nckpt84") or prev.get("nckpt") or 0)
pos = hub.get("pos", prev.get("pos", -1))
ready = int(hub.get("ready") or 0)
early = int(prev.get("early") or 0)
mon = 1 if "soft_breakout_monitor" in raw else 0
events_hub = hub.get("events") or []

cur = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "alive": alive,
    "wm_ok": 1,
    "nckpt": nckpt,
    "latest": hub.get("soft_iter") or "",
    "pos": pos,
    "ready": ready,
    "early": early,
    "gate": gate,
    "gate11": int(hub.get("gate11") or 0),
    "gate14": int(hub.get("gate14") or 0),
    "monitor125": mon,
    "events": events_hub,
    "raw_tail": "\\n".join(raw.strip().splitlines()[-20:]),
}

events = []
if prev:
    if alive and not prev.get("alive"):
        events.append("Soft train 恢复")
    if not alive and prev.get("alive"):
        events.append("Soft train 停了/崩了")
    if nckpt > int(prev.get("nckpt") or 0):
        events.append(f"新 Soft ckpt（n={nckpt}）")
    try:
        if int(pos) > int(prev.get("pos") if prev.get("pos") is not None else -1):
            events.append(f"Soft 正例累计 pos={pos}/8")
    except Exception:
        pass
    if ready and not prev.get("ready"):
        events.append("READY_HARD — Soft 门闩过，可开 Hard")
    if gate and not prev.get("gate"):
        events.append("门闩进程恢复")
    if not gate and prev.get("gate"):
        events.append("门闩进程停了")
    if mon and not prev.get("monitor125"):
        events.append("125 hub monitor 上线")
    if not mon and prev.get("monitor125"):
        events.append("125 hub monitor 停了")
    for e in events_hub:
        if e and e not in (prev.get("events") or []):
            events.append(f"hub:{e}")
else:
    events.append(
        f"watch 基线: Soft_alive={alive} nckpt={nckpt} pos={pos} gate={gate} mon125={mon} ready={ready}"
    )

state_path.write_text(json.dumps(cur, indent=2) + "\\n")
if events:
    print("SOFT_WATCH_NEW_RESULT " + " | ".join(events))
    print(json.dumps(cur, ensure_ascii=False))
else:
    print("SOFT_WATCH_IDLE " + json.dumps(
        {"alive": alive, "nckpt": nckpt, "pos": pos, "gate": gate, "ready": ready, "mon125": mon},
        ensure_ascii=False,
    ))
PY
