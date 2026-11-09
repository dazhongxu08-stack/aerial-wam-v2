#!/usr/bin/env python3
"""Four-host mainchannel collect monitor + auto-resume (84/125/14/11).

Ensures each host's watch/collect is alive; refreshes
artifacts/daytime_mainchannel_STATUS.md every INTERVAL seconds.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INTERVAL = int(os.environ.get("INTERVAL", "90"))
ART = ROOT / "artifacts"
STATUS = ART / "daytime_mainchannel_STATUS.md"
LOG = ART / "monitor_mainchannel_quad.log"
STATE = ART / "monitor_mainchannel_quad.state"
PIDFILE = ART / "monitor_mainchannel_quad.pid"
ART.mkdir(parents=True, exist_ok=True)
PIDFILE.write_text(str(os.getpid()))

# Direct LAN hosts
HOSTS = {
    "84": {
        "via": None,
        "ssh": os.environ.get("SSH84", "cursor-84-public"),
        "root": os.environ.get("ROOT84", "/data/aerial-wam-v2"),
        "stamp": os.environ.get("STAMP84", "20260923_1853"),
        "tag": "84",
        "wm": "",
        "python_bin": "",
    },
    "125": {
        "via": None,
        "ssh": os.environ.get("SSH125", "cursor-125-public"),
        "root": os.environ.get("ROOT125", "/home/yao/aerial-wam-v2"),
        "stamp": os.environ.get("STAMP125", "20260923_1347"),
        "tag": "125",
        "wm": "",
        "python_bin": "",
    },
    # Hop via 125 → H100 jump hosts
    "14": {
        "via": os.environ.get("SSH125", "cursor-125-public"),
        "ssh": "a25689@10.239.121.14",
        "port": int(os.environ.get("PORT14", "31126")),
        "key": os.environ.get("H100_KEY", "~/.ssh/id_ed25519_h100"),
        "root": os.environ.get("ROOT14", "/home/a25689/aerial-wam-v2"),
        "stamp": os.environ.get("STAMP14", "20260923_1853"),
        "tag": "14",
        "wm": (
            "/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/"
            "wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
        ),
        "python_bin": "/home/a25689/aerial-wam-v2/.venv/bin/python",
    },
    "11": {
        "via": os.environ.get("SSH125", "cursor-125-public"),
        "ssh": "a25689@10.239.121.11",
        "port": int(os.environ.get("PORT11", "30627")),
        "key": os.environ.get("H100_KEY", "~/.ssh/id_ed25519_h100"),
        "root": os.environ.get("ROOT11", "/home/a25689/aerial-wam-v2"),
        "stamp": os.environ.get("STAMP11", "20260923_1853"),
        "tag": "11",
        "wm": (
            "/home/a25689/aerial-wam-v2/experiments/aerial/rl/artifacts/"
            "wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
        ),
        "python_bin": "/home/a25689/aerial-wam-v2/.venv/bin/python",
    },
}

ENSURE_REMOTE = r'''
set +e
ROOT="{root}"; HOST="{tag}"; STAMP="{stamp}"
WM="{wm}"; PYTHON_BIN="{python_bin}"
ART="$ROOT/experiments/aerial/rl/artifacts"
PIDFILE="$ART/logs/watch_mainchannel_fullfix_${{STAMP}}_h${{HOST}}.pid"
MASTER="$ART/logs/mainchannel_fullfix_${{STAMP}}_h${{HOST}}.log"
DS="$ART/dataset_urban_inland_r01_depth_scene_$STAMP"
mkdir -p "$ART/logs"
# Prefer live collect for this stamp
if pgrep -af "collect_depth_scene_expert_dataset.*$STAMP" | grep -v pgrep >/dev/null; then
  echo COLLECT_ALIVE; exit 0
fi
if pgrep -af "run_mainchannel_fullfix|watch_mainchannel_fullfix" | grep -v pgrep >/dev/null; then
  # watch alone is ok — it will resume
  if [[ -f "$PIDFILE" ]]; then
    wpid=$(tr -d " \n" <"$PIDFILE" 2>/dev/null)
    if [[ -n "$wpid" ]] && kill -0 "$wpid" 2>/dev/null; then
      echo WATCH_ALIVE; exit 0
    fi
  fi
  if pgrep -af "watch_mainchannel_fullfix.sh" | grep -vE "pgrep|bash -s" >/dev/null; then
    echo WATCH_ALIVE; exit 0
  fi
fi
# AirSim down → recover if script exists
if ! ss -lntp 2>/dev/null | grep -q 41451; then
  R1="$HOME/aerial_airsim_persistent/recover_renderer_h${{HOST}}.sh"
  R2="$HOME/aerial_airsim_persistent/recover_renderer.sh"
  if [[ -x "$R1" ]]; then bash "$R1" >/dev/null 2>&1 || true
  elif [[ -x "$R2" ]]; then bash "$R2" outdoor >/dev/null 2>&1 || true
  fi
  sleep 8
fi
# Prefer project .venv
if [[ -z "$PYTHON_BIN" && -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi
mkdir -p "$HOME/sim_verify"
[[ -x "$ROOT/.venv/bin/python" ]] && ln -sfn "$ROOT/.venv" "$HOME/sim_verify/.venv" 2>/dev/null || true
cat > "$ART/logs/mainchannel_fullfix_${{STAMP}}_h${{HOST}}.env" <<ENV
ROUTES=0,1
MIN_KEPT=4
MIN_KEPT_PER_ROUTE=2
MAX_DYAW=0.45
MAX_DDYAW=0.12
REPLAN_PERIOD_S=2.0
W_JUMP=0.05
SKIP_BC=1
SKIP_EVAL=1
ENV
[[ -n "$WM" ]] && echo "WM=$WM" >> "$ART/logs/mainchannel_fullfix_${{STAMP}}_h${{HOST}}.env"
[[ -n "$PYTHON_BIN" ]] && echo "PYTHON_BIN=$PYTHON_BIN" >> "$ART/logs/mainchannel_fullfix_${{STAMP}}_h${{HOST}}.env"
chmod +x "$ROOT/experiments/aerial/scripts/watch_mainchannel_fullfix.sh" \
  "$ROOT/experiments/aerial/scripts/run_mainchannel_fullfix.sh" 2>/dev/null
nohup env HOST="$HOST" STAMP="$STAMP" INTERVAL=90 \
  ROUTES=0,1 MIN_KEPT=4 MIN_KEPT_PER_ROUTE=2 \
  MAX_DYAW=0.45 MAX_DDYAW=0.12 REPLAN_PERIOD_S=2.0 W_JUMP=0.05 \
  SKIP_BC=1 SKIP_EVAL=1 APPEND=1 \
  ${{WM:+WM="$WM"}} ${{PYTHON_BIN:+PYTHON_BIN="$PYTHON_BIN"}} \
  /bin/bash "$ROOT/experiments/aerial/scripts/watch_mainchannel_fullfix.sh" \
  >>"$ART/logs/watch_mainchannel_fullfix_${{STAMP}}_h${{HOST}}_nohup.log" 2>&1 &
echo "WATCH_STARTED pid=$!"
'''

SNAP_REMOTE = r'''
ROOT="{root}"; HOST="{tag}"; STAMP="{stamp}"
ART="$ROOT/experiments/aerial/rl/artifacts"
DS="$ART/dataset_urban_inland_r01_depth_scene_$STAMP"
MASTER="$ART/logs/mainchannel_fullfix_${{STAMP}}_h${{HOST}}.log"
alive=0; pgrep -af "collect_depth_scene_expert_dataset.*$STAMP" | grep -v pgrep >/dev/null && alive=1
watch=0; pgrep -af "watch_mainchannel_fullfix" | grep -v pgrep >/dev/null && watch=1
air=0; ss -lntp 2>/dev/null | grep -q 41451 && air=1
python3 - "$DS/AUTO_REVIEW.json" "$MASTER" "$alive" "$watch" "$air" "$HOST" "$STAMP" <<'P'
import json, re, sys
from pathlib import Path
ar, master, alive, watch, air, host, stamp = sys.argv[1:8]
n_rev = n_keep = n_arr = 0
by = {{}}; last = []; round_n = 0; gate = 4; variant = ""
if Path(ar).exists():
    d = json.load(open(ar)); s = d.get("summary") or {{}}; eps = d.get("episodes") or []
    n_rev = len(eps)
    n_keep = int(s.get("quality_kept") or sum(1 for e in eps if e.get("keep")))
    n_arr = int(s.get("n_arrived") or sum(1 for e in eps if e.get("arrived")))
    gate = int(s.get("min_kept_gate") or s.get("gate") or 4)
    round_n = int(s.get("rounds") or s.get("round") or 0)
    for e in eps:
        if e.get("keep"):
            r = str(e.get("route_idx")); by[r] = by.get(r, 0) + 1
    for e in eps[-2:]:
        last.append(
            "R%s d=%.1f keep=%s %s"
            % (
                e.get("route_idx"),
                float(e.get("final_dist_to_goal_m") or 0),
                e.get("keep"),
                e.get("drop_reason") or e.get("retry_variant") or "",
            )
        )
if Path(master).exists():
    text = Path(master).read_text(errors="ignore")
    ms = list(re.finditer(r"(?:retry-)?variant=([A-Za-z0-9_]+)", text))
    if ms:
        variant = ms[-1].group(1)
print(json.dumps({{
    "host": host, "ok": True, "stamp": stamp,
    "alive": int(alive), "watch": int(watch), "air": int(air),
    "n_rev": n_rev, "n_keep": n_keep, "n_arr": n_arr, "gate": gate,
    "by_route": by, "round": round_n, "variant": variant, "last": last,
}}, ensure_ascii=False))
P
'''


def log(msg: str) -> None:
    line = time.strftime("[%Y-%m-%dT%H:%M:%S%z] ") + msg
    # File only — supervisor redirects stdout to the same LOG.
    with LOG.open("a") as f:
        f.write(line + "\n")
        f.flush()


def _ssh_base(cfg: dict) -> list[str]:
    if cfg.get("via"):
        # hop: ssh via 'ssh -i key -p port user@host bash -s'
        inner = (
            f"ssh -o BatchMode=yes -o ConnectTimeout=20 "
            f"-i {cfg['key']} -p {cfg['port']} {cfg['ssh']} bash -s"
        )
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", cfg["via"], inner]
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", cfg["ssh"], "bash", "-s"]


def run_remote(cfg: dict, script: str, timeout: int = 90) -> tuple[int, str, str]:
    cmd = _ssh_base(cfg)
    try:
        r = subprocess.run(
            cmd, input=script, text=True, capture_output=True, timeout=timeout
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except Exception as e:
        return 1, "", str(e)


def ensure(cfg: dict) -> str:
    script = ENSURE_REMOTE.format(
        root=cfg["root"],
        tag=cfg["tag"],
        stamp=cfg["stamp"],
        wm=cfg.get("wm") or "",
        python_bin=cfg.get("python_bin") or "",
    )
    rc, out, err = run_remote(cfg, script, timeout=120)
    line = (out or err or "").strip().splitlines()
    return line[-1] if line else f"ENSURE_FAIL rc={rc} {err[-120:]}"


def snap(cfg: dict) -> dict:
    script = SNAP_REMOTE.format(root=cfg["root"], tag=cfg["tag"], stamp=cfg["stamp"])
    rc, out, err = run_remote(cfg, script, timeout=70)
    for line in reversed((out or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {"host": cfg["tag"], "ok": False, "error": (err or out or "")[-200:]}


def fmt(d: dict) -> str:
    if not d.get("ok"):
        return f"SSH_FAIL {d.get('error', '')}"
    last = " · ".join(d.get("last") or []) or "—"
    return (
        f"rev={d['n_rev']} keep={d['n_keep']}/{d['gate']} arr={d['n_arr']} "
        f"r{d['round']} var={d.get('variant') or '?'} "
        f"alive={d['alive']} watch={d['watch']} air={d['air']} | {last}"
    )


prev: dict[str, int] = {h: 0 for h in HOSTS}
if STATE.exists():
    try:
        prev.update({k: int(v) for k, v in json.loads(STATE.read_text()).items()})
    except Exception:
        pass

log(
    "quad monitor start hosts="
    + ",".join(f"{k}:{HOSTS[k]['stamp']}" for k in HOSTS)
    + f" every {INTERVAL}s"
)

while True:
    try:
        ensures = {}
        snaps = {}
        for name, cfg in HOSTS.items():
            ensures[name] = ensure(cfg)
            snaps[name] = snap(cfg)

        deltas = []
        for name, d in snaps.items():
            n = int(d.get("n_rev") or -1)
            if n >= 0:
                delta = max(0, n - prev.get(name, 0))
                if delta:
                    deltas.append(f"+{name}:{delta}")
                prev[name] = max(prev.get(name, 0), n)

        note = "NEW " + " ".join(deltas) if deltas else ""
        log(f"tick {note}".rstrip())
        for name in HOSTS:
            log(f"  ensure[{name}]={ensures[name]}")
            log(f"  {name}: {fmt(snaps[name])}")

        rows = "\n".join(
            f"| **{n}** | `{HOSTS[n]['stamp']}` | `{fmt(snaps[n])}` |" for n in HOSTS
        )
        keeps = sum(int(snaps[n].get("n_keep") or 0) for n in HOSTS if snaps[n].get("ok"))
        alives = sum(1 for n in HOSTS if snaps[n].get("alive"))
        STATUS.write_text(
            f"# Daytime main-channel LIVE — 四机\n\n"
            f"**updated:** {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
            f"**monitor:** quad watchdog（中断自动恢复 watch/collect）every {INTERVAL}s\n"
            f"**alive collect:** {alives}/4 · **total keep:** {keeps}\n\n"
            f"| Host | stamp | 快照 |\n|------|-------|------|\n{rows}\n\n"
            f"{note}\n\n"
            f"日志: `artifacts/monitor_mainchannel_quad.log`\n"
        )
        STATE.write_text(json.dumps(prev))

        if os.uname().sysname == "Darwin" and (deltas or alives < 4):
            title = "mainchannel 四机"
            body = f"alive={alives}/4 keep={keeps} {note}".strip()
            subprocess.run(
                ["osascript", "-e", f'display notification "{body}" with title "{title}"'],
                check=False,
                capture_output=True,
            )

        # Exit only if all keep gates met (rare); usually run forever.
        if all(
            snaps[n].get("ok")
            and int(snaps[n].get("n_keep") or 0) >= int(snaps[n].get("gate") or 4)
            and int(snaps[n].get("gate") or 0) > 0
            for n in HOSTS
        ):
            log("all keep gates met — exit")
            PIDFILE.unlink(missing_ok=True)
            break
    except Exception as e:
        log(f"tick_error {e!r}")
    time.sleep(INTERVAL)
