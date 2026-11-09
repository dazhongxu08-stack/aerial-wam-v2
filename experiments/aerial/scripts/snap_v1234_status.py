#!/usr/bin/env python3
"""Snap V1–V4 status on 125 (hops to 84/14/11). Write artifacts/v1234_STATUS.json + append RESULTS."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

STAMP = os.environ.get("V_STAMP", "20260925_1007")
HOME = Path.home()
LOCAL_ROOT = HOME / "aerial-wam-v2"
ART = LOCAL_ROOT / "experiments/aerial/rl/artifacts"
OUT_DIR = LOCAL_ROOT / "artifacts"
TZ = timezone(timedelta(hours=8))


def sh(cmd: str, timeout: int = 45) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"ERR:{e}"


def iter_max(ckpt: Path) -> int:
    its = []
    if not ckpt.exists():
        return -1
    for p in ckpt.glob("v4_ac_iter_*.pt"):
        try:
            its.append(int(p.stem.split("_")[-1]))
        except Exception:
            pass
    return max(its) if its else -1


def parse_gate(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    eps = d.get("episodes") or []
    arr = sum(1 for e in eps if e.get("arrived"))
    stick = sum(
        1
        for e in eps
        if int(e.get("route_idx", -1)) in (0, 1) and e.get("severe_collision")
    )
    return {
        "sr": f"{arr}/{len(eps)}",
        "p0_stick": f"{stick}/2",
        "arrived_routes": [e.get("route_idx") for e in eps if e.get("arrived")],
        "capability_pass": bool(arr >= 1 and stick <= 1),
        "vs_floor_2_4": "ge" if arr >= 2 else "lt",
        "routes": [
            {
                "route": e.get("route_idx"),
                "arrived": bool(e.get("arrived")),
                "sev": bool(e.get("severe_collision")),
                "d_final": round(float(e.get("d_final_m") or 0), 1),
                "steps": e.get("steps") or e.get("n_steps"),
            }
            for e in eps
        ],
    }


def last_line(log: Path) -> str:
    if not log.exists():
        return ""
    for line in reversed(log.read_text(errors="ignore").splitlines()[-120:]):
        if any(k in line for k in ("iter ", "RuntimeError", "TRAIN_DONE", "EVAL", "IMU", "baseline")):
            return line[-180:]
    return ""


def snap_local(lane: str, ckpt: str, gate: str, log: str, proc: str, script: str) -> dict:
    ck = ART / ckpt
    g = ART / gate
    lg = ART / log
    alive = bool(sh(f"pgrep -af '{proc}' | grep -v pgrep").strip())
    gate_d = parse_gate(g)
    done = gate_d is not None
    return {
        "lane": lane,
        "alive": alive,
        "iter_max": iter_max(ck),
        "iters_target": 80,
        "gate": gate_d,
        "done": done,
        "need_resume": (not alive) and (not done),
        "last": last_line(lg),
        "script": script,
    }


def snap_hop(ssh: str, root: str, lane: str, ckpt: str, gate: str, log: str, proc: str, script: str) -> dict:
    remote = f"""
python3 - <<'PY'
import json, subprocess
from pathlib import Path
art = Path({root!r}) / "experiments/aerial/rl/artifacts"
ck, g, lg = art / {ckpt!r}, art / {gate!r}, art / {log!r}
alive = bool(subprocess.getoutput("pgrep -af {proc!r} | grep -v pgrep").strip())
its = []
if ck.exists():
    for p in ck.glob("v4_ac_iter_*.pt"):
        try: its.append(int(p.stem.split("_")[-1]))
        except Exception: pass
im = max(its) if its else -1
gate = None
if g.exists():
    d = json.loads(g.read_text()); eps = d.get("episodes") or []
    arr = sum(1 for e in eps if e.get("arrived"))
    stick = sum(1 for e in eps if int(e.get("route_idx", -1)) in (0, 1) and e.get("severe_collision"))
    gate = {{
        "sr": f"{{arr}}/{{len(eps)}}",
        "p0_stick": f"{{stick}}/2",
        "arrived_routes": [e.get("route_idx") for e in eps if e.get("arrived")],
        "capability_pass": bool(arr >= 1 and stick <= 1),
        "vs_floor_2_4": "ge" if arr >= 2 else "lt",
    }}
last = ""
if lg.exists():
    for line in reversed(lg.read_text(errors="ignore").splitlines()[-120:]):
        if any(k in line for k in ("iter ", "RuntimeError", "TRAIN_DONE", "EVAL", "IMU", "baseline")):
            last = line[-180:]; break
done = gate is not None
print(json.dumps({{
    "lane": {lane!r}, "alive": alive, "iter_max": im, "iters_target": 80,
    "gate": gate, "done": done, "need_resume": (not alive) and (not done),
    "last": last, "script": {script!r},
}}))
PY
"""
    raw = sh(f"{ssh} bash -s <<'EOS'\n{remote}\nEOS\n", timeout=60)
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return {
        "lane": lane,
        "alive": False,
        "iter_max": -1,
        "gate": None,
        "done": False,
        "need_resume": True,
        "last": raw[-200:],
        "script": script,
    }


def main() -> None:
    stamp = STAMP
    out = {
        "updated": datetime.now(TZ).isoformat(timespec="seconds"),
        "stamp": stamp,
        "hosts": {
            "125": snap_local(
                "V1",
                f"v4_ac_ckpt_mainchannel_b_curiosity_{stamp}_noshield_h125",
                f"eval_mainchannel_b_curiosity_{stamp}_h125_gate_noshield/eval_all.json",
                f"logs/mainchannel_b_curiosity_{stamp}_h125.log",
                "b_curiosity|train_v4_ac.*curiosity|wam_phase2.*curiosity",
                "run_mainchannel_b_curiosity_noshield.sh",
            ),
            "84": snap_hop(
                "ssh -o BatchMode=yes -o ConnectTimeout=12 ubantu@10.229.20.84",
                "/data/aerial-wam-v2",
                "V2",
                f"v4_ac_ckpt_mainchannel_v2_floor_b_{stamp}_noshield_h84",
                f"eval_mainchannel_v2_floor_b_{stamp}_h84_gate_noshield/eval_all.json",
                f"logs/mainchannel_v2_floor_b_{stamp}_h84.log",
                "v2_floor|train_v4_ac.*v2_floor|wam_phase2.*v2_floor",
                "run_mainchannel_v2_floor_then_b.sh",
            ),
            "14": snap_hop(
                "ssh -o BatchMode=yes -o ConnectTimeout=12 -i ~/.ssh/id_ed25519_h100 -p 31126 a25689@10.239.121.14",
                "/home/a25689/aerial-wam-v2",
                "V3",
                f"v4_ac_ckpt_mainchannel_b_openside_{stamp}_noshield_h14",
                f"eval_mainchannel_b_openside_{stamp}_h14_gate_noshield/eval_all.json",
                f"logs/mainchannel_b_openside_{stamp}_h14.log",
                "b_openside|train_v4_ac.*b_openside|wam_phase2.*b_openside",
                "run_mainchannel_b_openside_noshield.sh",
            ),
            "11": snap_hop(
                "ssh -o BatchMode=yes -o ConnectTimeout=12 -i ~/.ssh/id_ed25519_h100 -p 30627 a25689@10.239.121.11",
                "/home/a25689/aerial-wam-v2",
                "V4",
                f"v4_ac_ckpt_mainchannel_v4_neargoal_{stamp}_noshield_h11",
                f"eval_mainchannel_v4_neargoal_{stamp}_h11_gate_noshield/eval_all.json",
                f"logs/mainchannel_v4_neargoal_{stamp}_h11.log",
                "v4_neargoal|train_v4_ac.*v4_neargoal|wam_phase2.*v4_neargoal",
                "run_mainchannel_v4_neargoal_noshield.sh",
            ),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "v1234_STATUS.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    res = OUT_DIR / "v1234_RESULTS.jsonl"
    existing = res.read_text() if res.exists() else ""
    with res.open("a") as f:
        for h, s in out["hosts"].items():
            g = s.get("gate")
            if not g:
                continue
            marker = f'"stamp": "{stamp}", "host": "{h}", "sr": "{g["sr"]}"'
            if marker in existing:
                continue
            row = {"ts": out["updated"], "stamp": stamp, "host": h, "lane": s.get("lane"), **g, "iter_max": s.get("iter_max")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing += marker
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
