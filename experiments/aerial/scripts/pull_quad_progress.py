#!/usr/bin/env python3
"""Pull 84/125/14/11 mainchannel collect progress into daytime_mainchannel_STATUS.md."""
from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts"
STATUS = ART / "daytime_mainchannel_STATUS.md"
ART.mkdir(parents=True, exist_ok=True)

HOSTS = {
    "84": {
        "via": None,
        "ssh": os.environ.get("SSH84", "cursor-84-public"),
        "root": "/data/aerial-wam-v2",
        "stamp": "20260923_2252",
        "role": "hbclear_teacher_r0",
    },
    "125": {
        "via": None,
        "ssh": os.environ.get("SSH125", "cursor-125-public"),
        "root": "/home/yao/aerial-wam-v2",
        "stamp": "20260923_1347",
    },
    "14": {
        "via": os.environ.get("SSH125", "cursor-125-public"),
        "ssh": "a25689@10.239.121.14",
        "port": 31126,
        # Key lives on the hop host (125), not this Mac.
        "key": os.environ.get("H100_KEY", "~/.ssh/id_ed25519_h100"),
        "root": "/home/a25689/aerial-wam-v2",
        "stamp": "20260923_2252",
    },
    "11": {
        "via": os.environ.get("SSH125", "cursor-125-public"),
        "ssh": "a25689@10.239.121.11",
        "port": 30627,
        "key": os.environ.get("H100_KEY", "~/.ssh/id_ed25519_h100"),
        "root": "/home/a25689/aerial-wam-v2",
        "stamp": "20260923_1853",
    },
}

REMOTE = r"""
import json, subprocess
from pathlib import Path
from collections import defaultdict
stamp = {stamp!r}
root = Path({root!r})
procs = subprocess.getoutput("pgrep -af 'collect_depth_scene|wam_phase2_long_eval|hbclear_teacher_r0|run_hbclear_teacher'")
alive = (
    (stamp in procs and ("collect_depth_scene" in procs or "hbclear_teacher" in procs or "wam_phase2_long_eval" in procs))
    or ("run_hbclear_teacher_r0" in procs and stamp in procs)
)
if not alive and "train_v4_ac" in subprocess.getoutput("pgrep -af train_v4_ac"):
    alive = True
# also alive if any hbclear teacher for this host stamp without exact match in cmdline
if not alive and "run_hbclear_teacher_r0" in procs:
    alive = True
art = root / "experiments/aerial/rl/artifacts"
cands = [
    art / f"dataset_urban_inland_r0_hbclear_teacher_{{stamp}}/AUTO_REVIEW.json",
    art / f"dataset_urban_inland_r01_depth_scene_{{stamp}}/AUTO_REVIEW.json",
    art / f"dataset_urban_inland_r0_depth_scene_{{stamp}}/AUTO_REVIEW.json",
    art / f"dataset_urban_inland_r1_depth_scene_{{stamp}}/AUTO_REVIEW.json",
]
p = next((c for c in cands if c.exists()), None)
eps = json.loads(p.read_text())["episodes"] if p is not None else []
best = {{}}
for e in eps:
    df = e.get("final_dist_to_goal_m", e.get("d_final_m"))
    ri = e.get("route_idx")
    if df is None:
        continue
    v = float(df)
    if ri not in best or v < best[ri]:
        best[ri] = v
keeps = [e for e in eps if e.get("keep")]
last = eps[-2:] if eps else []
print(json.dumps({{
    "alive": bool(alive),
    "n": len(eps),
    "keep": len(keeps),
    "arr": sum(1 for e in eps if e.get("arrived")),
    "b0": None if 0 not in best else round(best[0], 1),
    "b1": None if 1 not in best else round(best[1], 1),
    "ds": None if p is None else p.parent.name,
    "keeps": [
        {{
            "r": e.get("route_idx"),
            "final": round(float(e.get("final_dist_to_goal_m") or e.get("d_final_m") or -1), 2),
            "lat": round(float(e.get("max_lateral_offset_m") or 0), 2),
            "steps": e.get("steps"),
            "var": e.get("retry_variant") or e.get("teacher"),
        }}
        for e in keeps[-4:]
    ],
    "last": [
        {{
            "r": e.get("route_idx"),
            "final": round(float(e.get("final_dist_to_goal_m") or e.get("d_final_m") or -1), 1),
            "steps": e.get("steps"),
            "interv": round(float(e.get("intervention_rate") or e.get("intervention_frac") or 0), 2),
            "var": e.get("retry_variant") or e.get("teacher") or e.get("fail_tag"),
            "keep": bool(e.get("keep")),
        }}
        for e in last
    ],
}}))
"""


def _ssh_cmd(cfg: dict) -> list[str]:
    remote_py = REMOTE.format(stamp=cfg["stamp"], root=cfg["root"])
    if cfg.get("via"):
        inner = (
            f"ssh -o BatchMode=yes -o ConnectTimeout=20 "
            f"-i {cfg['key']} -p {cfg['port']} {cfg['ssh']} python3 -"
        )
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=35",
            cfg["via"],
            inner,
        ], remote_py
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        cfg["ssh"],
        "python3",
        "-",
    ], remote_py


def snap(name: str, cfg: dict) -> dict:
    cmd, payload = _ssh_cmd(cfg)
    try:
        r = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as e:
        return {"ok": False, "err": str(e)}
    if r.returncode != 0:
        return {"ok": False, "err": (r.stderr or r.stdout)[-300:]}
    for line in reversed(r.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            d["ok"] = True
            return d
    return {"ok": False, "err": "no json", "raw": r.stdout[-200:]}


def main() -> None:
    snaps = {h: snap(h, cfg) for h, cfg in HOSTS.items()}
    alive = sum(1 for s in snaps.values() if s.get("alive"))
    keep = sum(int(s.get("keep") or 0) for s in snaps.values() if s.get("ok"))
    lines = [
        "# 四机主采 LIVE（不停）",
        "",
        f"**updated:** {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "**policy:** 125 DepthScene R0+R1 · 84/14 hbclear教师R0 · 11 DepthScene R0；每 **3 分钟**拉结果（SSH public）",
        "",
        f"**alive:** {alive}/4 · **total keep:** {keep}",
        "",
        "| Host | stamp | n | keep | arr | best R0/R1 | alive | 近况 |",
        "|------|-------|---|------|-----|------------|-------|------|",
    ]
    for h, cfg in HOSTS.items():
        s = snaps[h]
        if not s.get("ok"):
            lines.append(
                f"| **{h}** | `{cfg['stamp']}` | ? | ? | ? | ? | 0 | err:{(s.get('err') or '')[:40]} |"
            )
            continue
        last = s.get("last") or []
        near = " · ".join(
            f"R{x['r']} d={x['final']} s={x['steps']} ir={x['interv']}" for x in last
        ) or "-"
        lines.append(
            f"| **{h}** | `{cfg['stamp']}` | {s.get('n')} | {s.get('keep')} | "
            f"{s.get('arr')} | {s.get('b0')}/{s.get('b1')} | {int(bool(s.get('alive')))} | {near} |"
        )
    lines.append("")
    lines.append("### Keeps")
    any_k = False
    for h, s in snaps.items():
        for k in s.get("keeps") or []:
            any_k = True
            lines.append(
                f"- **{h}** R{k['r']} final={k['final']} lat={k['lat']} "
                f"steps={k['steps']} var=`{k['var']}`"
            )
    if not any_k:
        lines.append("- (none yet beyond prior)")
    lines.append("")
    STATUS.write_text("\n".join(lines) + "\n")
    print(STATUS.read_text())


if __name__ == "__main__":
    main()
