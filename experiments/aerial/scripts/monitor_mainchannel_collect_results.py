#!/usr/bin/env python3
"""Stable collect-results poller for 84+125."""
from __future__ import annotations
import json, os, re, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INTERVAL = int(os.environ.get("INTERVAL", "180"))
STAMP = os.environ.get("STAMP", "20260923_1347")
SSH84 = os.environ.get("SSH84", "cursor-84-public")
SSH125 = os.environ.get("SSH125", "cursor-125-public")
ROOT84 = os.environ.get("ROOT84", "/data/aerial-wam-v2")
ROOT125 = os.environ.get("ROOT125", "/home/yao/aerial-wam-v2")
ART = ROOT / "artifacts"
STATUS = ART / "daytime_mainchannel_STATUS.md"
LOG = ART / "monitor_mainchannel_collect_results.log"
STATE = ART / "monitor_mainchannel_collect_results.state"
PIDFILE = ART / "monitor_mainchannel_collect_results.pid"
ART.mkdir(parents=True, exist_ok=True)
PIDFILE.write_text(str(os.getpid()))

REMOTE = r'''
ROOT="{root}"; HOST="{host}"; STAMP="{stamp}"
ART="$ROOT/experiments/aerial/rl/artifacts"
DS="$ART/dataset_urban_inland_r01_depth_scene_$STAMP"
MASTER="$ART/logs/mainchannel_fullfix_${{STAMP}}_h${{HOST}}.log"
alive=0; pgrep -af "collect_depth_scene_expert_dataset.*$STAMP" | grep -v pgrep >/dev/null && alive=1
air=0; ss -lntp 2>/dev/null | grep -q 41451 && air=1
python3 - "$DS/AUTO_REVIEW.json" "$MASTER" "$alive" "$air" "$HOST" <<'P'
import json,re,sys
from pathlib import Path
ar,master,alive,air,host=sys.argv[1:6]
n_rev=n_keep=n_arr=0; by={{}}; last=[]; round_n=0; gate=4
if Path(ar).exists():
  d=json.load(open(ar)); s=d.get("summary") or {{}}; eps=d.get("episodes") or []
  n_rev=len(eps)
  n_keep=int(s.get("quality_kept") or sum(1 for e in eps if e.get("keep")))
  n_arr=int(s.get("n_arrived") or sum(1 for e in eps if e.get("arrived")))
  gate=int(s.get("min_kept_gate") or s.get("gate") or 4)
  round_n=int(s.get("rounds") or s.get("round") or 0)
  for e in eps:
    if e.get("keep"):
      r=str(e.get("route_idx")); by[r]=by.get(r,0)+1
  for e in eps[-3:]:
    last.append("R%s r%s arr=%s keep=%s steps=%s d=%.1f %s"%(e.get("route_idx"),e.get("round"),e.get("arrived"),e.get("keep"),e.get("steps"),float(e.get("final_dist_to_goal_m") or 0),e.get("drop_reason") or ""))
steps=[]; hz=[]
if Path(master).exists():
  text=Path(master).read_text(errors="ignore")
  for m in re.finditer(r"collector achieved ([0-9.]+) Hz.*?over ([0-9]+) steps", text):
    hz.append(float(m.group(1))); steps.append(int(m.group(2)))
avg_hz=sum(hz[-8:])/len(hz[-8:]) if hz else 0.0
avg_steps=sum(steps[-8:])/len(steps[-8:]) if steps else 0.0
sec=int(avg_steps/avg_hz) if avg_hz and avg_steps else 0
print(json.dumps({{"host":host,"ok":True,"alive":int(alive),"air":int(air),"n_rev":n_rev,"n_keep":n_keep,"n_arr":n_arr,"gate":gate,"by_route":by,"round":round_n,"avg_hz":round(avg_hz,2),"avg_steps":int(avg_steps),"sec_per_ep":sec,"last":last}},ensure_ascii=False))
P
'''

def log(msg: str) -> None:
    line = time.strftime("[%Y-%m-%dT%H:%M:%S%z] ") + msg
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")

def snap(ssh: str, root: str, host: str) -> dict:
    script = REMOTE.format(root=root, host=host, stamp=STAMP)
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", ssh, "bash", "-s"],
            input=script, text=True, capture_output=True, timeout=60,
        )
        out = (r.stdout or "").strip().splitlines()
        for line in reversed(out):
            line=line.strip()
            if line.startswith("{"):
                return json.loads(line)
    except Exception as e:
        return {"host": host, "ok": False, "error": str(e)}
    return {"host": host, "ok": False, "error": (r.stderr or r.stdout or "")[-200:]}

def fmt(d: dict) -> str:
    if not d.get("ok"):
        return f"{d.get('host')} SSH_FAIL {d.get('error','')}"
    last = " · ".join(d.get("last") or []) or "—"
    return (
        f"reviewed={d['n_rev']} keep={d['n_keep']}/{d['gate']} arr={d['n_arr']} "
        f"~{d['sec_per_ep']}s/ep ({d['avg_steps']}steps@{d['avg_hz']}Hz) "
        f"r{d['round']} by={json.dumps(d.get('by_route') or {}, separators=(',',':'))} "
        f"alive={d['alive']} air={d['air']} | {last}"
    )

prev = {"84": 0, "125": 0}
if STATE.exists():
    try:
        prev.update(json.loads(STATE.read_text()))
    except Exception:
        pass

log(f"collect-results py-monitor start stamp={STAMP} every {INTERVAL}s")
while True:
    try:
        a = snap(SSH84, ROOT84, "84")
        b = snap(SSH125, ROOT125, "125")
        ra = int(a.get("n_rev") or -1); rb = int(b.get("n_rev") or -1)
        ka = int(a.get("n_keep") or 0); kb = int(b.get("n_keep") or 0)
        ga = int(a.get("gate") or 4); gb = int(b.get("gate") or 4)
        sa = int(a.get("sec_per_ep") or 0); sb = int(b.get("sec_per_ep") or 0)
        secs = [x for x in (sa, sb) if x > 0]
        pace = int(sum(secs) / len(secs)) if secs else 0
        da, db = max(0, ra - prev["84"]), max(0, rb - prev["125"])
        note = f"NEW +84:{da} +125:{db}" if (da or db) else ""
        la, lb = fmt(a), fmt(b)
        log(f"tick {note}".rstrip())
        log(f"  84:  {la}")
        log(f"  125: {lb}")
        STATUS.write_text(
            f"# Daytime main-channel LIVE — collect results\n\n"
            f"**updated:** {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
            f"**stamp:** `{STAMP}` · 只采 R0/R1 · 监控每 {INTERVAL}s\n"
            f"**单条耗时（不论 keep）:** 约 **{pace}s**（84≈{sa}s · 125≈{sb}s；满步≈6min）\n\n"
            f"| Host | 快照 |\n|------|------|\n"
            f"| **84** | `{la}` |\n| **125** | `{lb}` |\n\n"
            f"{note}\n\n日志: `artifacts/monitor_mainchannel_collect_results.log`\n"
        )
        if note and os.uname().sysname == "Darwin":
            subprocess.run(
                ["osascript", "-e", f'display notification "84={ra} 125={rb} keep {ka}+{kb}" with title "collect results"'],
                check=False, capture_output=True,
            )
        prev = {"84": max(0, ra), "125": max(0, rb)}
        STATE.write_text(json.dumps(prev))
        if a.get("ok") and b.get("ok") and ka >= ga and kb >= gb and ga > 0 and gb > 0:
            log("both keep gates met — exit")
            PIDFILE.unlink(missing_ok=True)
            break
    except Exception as e:
        log(f"tick_error {e!r}")
    time.sleep(INTERVAL)
