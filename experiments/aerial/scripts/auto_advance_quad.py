#!/usr/bin/env python3
"""Keep 84/125/14/11 on outcome_A noshield FT. Never restart a finished stamp.

Also aggregates **results**: gate SR/P0/per-route + train crumbs into
artifacts/auto_advance_LAST.json, daytime_mainchannel_STATUS.md, and
artifacts/outcome_a_RESULTS.jsonl (append-only when a new gate appears).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts"
STATUS = ART / "daytime_mainchannel_STATUS.md"
LAST = ART / "auto_advance_LAST.json"
RESULTS = ART / "outcome_a_RESULTS.jsonl"
LEDGER = ART / "MAINCHANNEL_LEDGER.md"
STAMP = os.environ.get("OUTCOME_A_STAMP", os.environ.get("ANTISTICK_STAMP", "20260924_1605"))
# Host-specific defaults applied inside run script; ensure only sets STAMP + SKIP
ITERS = os.environ.get("ITERS", "")  # empty → script per-host default
EPI_PER_ITER = os.environ.get("EPI_PER_ITER", "4")
NEAR_GOAL_FRAC = os.environ.get("NEAR_GOAL_FRAC", "0.4")
IMAGINE_HORIZON = os.environ.get("IMAGINE_HORIZON", "")
IMAGINE_BATCH = os.environ.get("IMAGINE_BATCH", "16")
SCRIPT = "run_mainchannel_outcome_a_noshield.sh"
DONE_MARK = "OUTCOME_A_NOSHIELD_DONE"
# Floor to beat (ledger)
FLOOR_SR = "2/4"
FLOOR_NOTE = "directional_oa 022358 R3+R4"

HOST_VARIANT = {
    "125": "baseline p2/c6/s20",
    "84": "soft_prog p1/c10",
    "14": "success_pull s40",
    "11": "mid_prog p3.5/c6",
}


def _iters_target_for(host: str) -> int:
    if ITERS:
        try:
            return int(ITERS)
        except ValueError:
            pass
    return 80  # A‴: all hosts 80×H40


def sh(cmd: list[str], timeout: int = 100, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=timeout)


def ensure(*, root: str, host: str, ssh: list[str], extra_env: str = "") -> str:
    # Per-host: train→gate already sequential in SCRIPT. Here we only (re)start
    # that host; never wait on siblings. If TRAIN_DONE but gate missing, kick
    # SKIP_TRAIN=1 eval-only so the first finished host evals immediately.
    iters_tgt = _iters_target_for(host)
    remote = f"""set +e
ROOT={root}; STAMP={STAMP}; HOST={host}; ITERS_TGT={iters_tgt}
ART=$ROOT/experiments/aerial/rl/artifacts
LOG=$ART/logs/mainchannel_outcome_a_${{STAMP}}_h${{HOST}}.log
GATE=$ART/eval_mainchannel_outcome_a_${{STAMP}}_h${{HOST}}_gate_noshield/eval_all.json
CKPT=$ART/v4_ac_ckpt_mainchannel_outcome_a_${{STAMP}}_noshield_h${{HOST}}
{extra_env}
if [[ -f "$GATE" ]] || grep -q '{DONE_MARK}' "$LOG" 2>/dev/null; then
  echo DONE
  ls -1t $CKPT/v4_ac_iter_*.pt 2>/dev/null | head -1
  exit 0
fi
# train or gate-eval already running on this host → leave it
if pgrep -af "run_mainchannel_outcome_a_noshield|train_v4_ac.*mainchannel_outcome_a.*${{STAMP}}|wam_phase2_long_eval.*outcome_a_${{STAMP}}.*h${{HOST}}" | grep -v pgrep >/dev/null; then
  echo OK
  ls -1t $CKPT/v4_ac_iter_*.pt 2>/dev/null | head -1
  exit 0
fi
# train finished (mark or last iter ckpt) but no gate → eval only, do not retrain
ITER_MAX=-1
for f in $CKPT/v4_ac_iter_*.pt; do
  [[ -f "$f" ]] || continue
  n=${{f##*_}}; n=${{n%.pt}}; n=$((10#$n))
  (( n > ITER_MAX )) && ITER_MAX=$n
done
TRAIN_DONE=0
grep -q 'TRAIN_DONE' "$LOG" 2>/dev/null && TRAIN_DONE=1
# last iter index is ITERS-1
if [[ "$TRAIN_DONE" = 1 ]] || (( ITER_MAX >= ITERS_TGT - 1 )); then
  echo EVAL_ONLY
  if ! ss -lntp | grep -q 41451; then
    bash $HOME/aerial_airsim_persistent/recover_renderer.sh outdoor >/dev/null 2>&1 || true
    sleep 8
  fi
  nohup env HOST=$HOST STAMP=$STAMP SKIP_BASELINE=1 SKIP_TRAIN=1 \\
    EPI_PER_ITER={EPI_PER_ITER} NEAR_GOAL_FRAC={NEAR_GOAL_FRAC} IMAGINE_BATCH={IMAGINE_BATCH} \\
    ${{PYTHON_BIN:+PYTHON_BIN=$PYTHON_BIN}} ${{WM:+WM=$WM}} \\
    bash $ROOT/experiments/aerial/scripts/{SCRIPT} \\
    >>$ART/logs/mainchannel_outcome_a_${{STAMP}}_h${{HOST}}.log 2>&1 < /dev/null &
  echo EVAL_STARTED
  exit 0
fi
echo RESTART
pkill -f run_mainchannel_antistick_noshield 2>/dev/null || true
pkill -f run_hbclear_teacher_r0 2>/dev/null || true
pkill -f collect_depth_scene_expert_dataset 2>/dev/null || true
pkill -f 'train_v4_ac.*mainchannel_antistick' 2>/dev/null || true
pkill -f run_mainchannel_outcome_a_noshield 2>/dev/null || true
pkill -f 'train_v4_ac.*mainchannel_outcome_a' 2>/dev/null || true
sleep 2
if ! ss -lntp | grep -q 41451; then
  bash $HOME/aerial_airsim_persistent/recover_renderer.sh outdoor >/dev/null 2>&1 || true
  sleep 8
fi
# Resume from latest stamp ckpt if any (avoid cold hbclear restart mid-run)
RESUME_INIT=""; RESUME_ITERS=""
LATEST=$(ls -1t $CKPT/v4_ac_iter_*.pt 2>/dev/null | head -1 || true)
if [[ -n "$LATEST" ]]; then
  n=${{LATEST##*_}}; n=${{n%.pt}}; n=$((10#$n))
  rem=$((ITERS_TGT - n - 1))
  if (( rem < 1 )); then rem=1; fi
  RESUME_INIT="$LATEST"
  RESUME_ITERS="$rem"
  echo "RESUME_FROM=$LATEST rem=$rem"
fi
nohup env HOST=$HOST STAMP=$STAMP SKIP_BASELINE=1 \\
  EPI_PER_ITER={EPI_PER_ITER} NEAR_GOAL_FRAC={NEAR_GOAL_FRAC} IMAGINE_BATCH={IMAGINE_BATCH} \\
  ${{RESUME_INIT:+INIT=$RESUME_INIT}} ${{RESUME_ITERS:+ITERS=$RESUME_ITERS}} \\
  ${{PYTHON_BIN:+PYTHON_BIN=$PYTHON_BIN}} ${{WM:+WM=$WM}} \\
  bash $ROOT/experiments/aerial/scripts/{SCRIPT} \\
  >>$ART/logs/mainchannel_outcome_a_${{STAMP}}_h${{HOST}}.log 2>&1 < /dev/null &
echo RESTARTED
"""
    r = sh(ssh + ["bash", "-s"], input_text=remote, timeout=110)
    out = (r.stdout or "").strip().splitlines()
    return out[-1] if out else f"ERR:{(r.stderr or '')[-160:]}"


def snap_progress(ssh: list[str], root: str, host: str) -> dict:
    remote = f"""
import json, re, subprocess
from pathlib import Path
root=Path({root!r}); host={host!r}; stamp={STAMP!r}
art=root/"experiments/aerial/rl/artifacts"
log=art/f"logs/mainchannel_outcome_a_{{stamp}}_h{{host}}.log"
ckpt=art/f"v4_ac_ckpt_mainchannel_outcome_a_{{stamp}}_noshield_h{{host}}"
alive=bool(subprocess.getoutput(
    f"pgrep -af 'train_v4_ac.*outcome_a_{{stamp}}|run_mainchannel_outcome_a|"
    f"wam_phase2_long_eval.*outcome_a_{{stamp}}'").strip())
iters=[]
if ckpt.exists():
  for p in ckpt.glob("v4_ac_iter_*.pt"):
    try: iters.append(int(p.stem.split("_")[-1]))
    except Exception: pass
gate=None
routes=[]
ge=art/f"eval_mainchannel_outcome_a_{{stamp}}_h{{host}}_gate_noshield/eval_all.json"
if ge.exists():
  rows=json.loads(ge.read_text())["episodes"]
  stick=sum(1 for r in rows if int(r.get("route_idx",-1)) in (0,1) and r.get("severe_collision"))
  arr=sum(1 for r in rows if r.get("arrived"))
  arrived_routes=sorted(int(r["route_idx"]) for r in rows if r.get("arrived"))
  for r in rows:
    routes.append({{
      "route": int(r.get("route_idx",-1)),
      "arrived": bool(r.get("arrived")),
      "d_final": round(float(r.get("d_final_m") or 0),1),
      "d_min": round(float(r.get("d_min_m") or 0),1),
      "prog": round(float(r.get("progress_ratio") or 0),3),
      "sev": bool(r.get("severe_collision")),
      "steps": int(r.get("steps") or 0),
      "ir": round(float(r.get("intervention_rate") or 0),2),
    }})
  gate={{
    "sr": f"{{arr}}/{{len(rows)}}",
    "p0_stick": f"{{stick}}/2",
    "arrived_routes": arrived_routes,
    "capability_pass": bool(arr >= 1 and stick <= 1),
    "vs_floor": "ge" if arr >= 2 else ("lt" if arr < 2 else "eq"),
  }}
last=""; done=False; gate_line=""; train_hz=None
if log.exists():
  text=log.read_text(errors="ignore")
  done=("{DONE_MARK}" in text) or ge.exists()
  for L in reversed(text.splitlines()[-200:]):
    if not gate_line and "gate_noshield" in L and "SR=" in L:
      gate_line=L.strip()[-200:]
    if not last and "iter " in L and "steps" in L:
      last=L[-140:]
      m=re.search(r"@ ([0-9.]+) Hz", L)
      if m: train_hz=float(m.group(1))
    if last and (gate_line or not ge.exists()):
      break
print(json.dumps({{
  "alive": alive,
  "iter_max": (max(iters) if iters else -1),
  "iters_target": {_iters_target_for(host)},
  "variant": {HOST_VARIANT.get(str(host), "?")!r},
  "gate": gate,
  "routes": routes,
  "gate_line": gate_line,
  "done": done,
  "train_hz": train_hz,
  "last": last,
}}))
"""
    r = sh(ssh + ["python3", "-"], input_text=remote, timeout=50)
    try:
        return json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:
        return {"alive": False, "err": (r.stderr or r.stdout or "")[-160:]}


def _seen_result_keys() -> set[str]:
    keys: set[str] = set()
    if not RESULTS.exists():
        return keys
    for line in RESULTS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            keys.add(f"{d.get('stamp')}|{d.get('host')}")
        except Exception:
            continue
    return keys


def append_results(snaps: dict) -> list[dict]:
    """Append new gate rows to RESULTS jsonl; return newly added."""
    ART.mkdir(parents=True, exist_ok=True)
    seen = _seen_result_keys()
    added = []
    for h, s in snaps.items():
        g = (s or {}).get("gate")
        if not g:
            continue
        key = f"{STAMP}|{h}"
        if key in seen:
            continue
        row = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "stamp": STAMP,
            "host": h,
            "policy": "outcome_A",
            "sr": g.get("sr"),
            "p0_stick": g.get("p0_stick"),
            "arrived_routes": g.get("arrived_routes"),
            "capability_pass": g.get("capability_pass"),
            "vs_floor_2_4": g.get("vs_floor"),
            "routes": (s or {}).get("routes") or [],
            "iter_max": (s or {}).get("iter_max"),
            "floor": FLOOR_SR,
            "floor_note": FLOOR_NOTE,
        }
        with RESULTS.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        added.append(row)
        seen.add(key)
    return added


def append_ledger(added: list[dict]) -> None:
    if not added or not LEDGER.exists():
        return
    lines = [
        "",
        f"### Gate 结果追加 · {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')}",
    ]
    for r in added:
        arr = ",".join(f"R{x}" for x in (r.get("arrived_routes") or [])) or "无"
        lines.append(
            f"- **h{r['host']}** outcome_A `{r['stamp']}`: SR={r.get('sr')} "
            f"P0_stick={r.get('p0_stick')} 到达=[{arr}] "
            f"pass={r.get('capability_pass')} vs下限2/4={r.get('vs_floor_2_4')}"
        )
    with LEDGER.open("a") as f:
        f.write("\n".join(lines) + "\n")


def _fmt_routes(routes: list) -> str:
    if not routes:
        return "-"
    parts = []
    for r in sorted(routes, key=lambda x: x.get("route", -1)):
        mark = "✓" if r.get("arrived") else ("✗sev" if r.get("sev") else "✗")
        parts.append(f"R{r.get('route')}{mark} df={r.get('d_final')} p={r.get('prog')}")
    return "; ".join(parts)


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    ssh125 = os.environ.get("SSH125", "cursor-125")
    ssh84 = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=35", ssh125,
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "ubantu@10.229.20.84",
    ]
    ssh125_direct = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", ssh125]
    ssh14 = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=35", ssh125,
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-i", "~/.ssh/id_ed25519_h100", "-p", "31126", "a25689@10.239.121.14",
    ]
    ssh11 = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=35", ssh125,
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-i", "~/.ssh/id_ed25519_h100", "-p", "30627", "a25689@10.239.121.11",
    ]
    actions = {
        "84": ensure(root="/data/aerial-wam-v2", host="84", ssh=ssh84),
        "125": ensure(root="/home/yao/aerial-wam-v2", host="125", ssh=ssh125_direct),
        "14": ensure(
            root="/home/a25689/aerial-wam-v2", host="14", ssh=ssh14,
            extra_env=(
                "PYTHON_BIN=$ROOT/.venv/bin/python\n"
                "WM=$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt\n"
            ),
        ),
        "11": ensure(
            root="/home/a25689/aerial-wam-v2", host="11", ssh=ssh11,
            extra_env=(
                "PYTHON_BIN=$ROOT/.venv/bin/python\n"
                "WM=$ART/wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt\n"
            ),
        ),
    }
    snaps = {
        "84": snap_progress(ssh84, "/data/aerial-wam-v2", "84"),
        "125": snap_progress(ssh125_direct, "/home/yao/aerial-wam-v2", "125"),
        "14": snap_progress(ssh14, "/home/a25689/aerial-wam-v2", "14"),
        "11": snap_progress(ssh11, "/home/a25689/aerial-wam-v2", "11"),
    }
    added = append_results(snaps)
    append_ledger(added)

    # Aggregate scoreboard
    gates_done = [h for h, s in snaps.items() if (s or {}).get("gate")]
    best_sr = None
    for h in gates_done:
        g = snaps[h]["gate"]
        n = int(str(g["sr"]).split("/")[0])
        if best_sr is None or n > best_sr[0]:
            best_sr = (n, h, g)

    payload = {
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stamp": STAMP,
        "policy": "outcome_A noshield; results tracked in outcome_a_RESULTS.jsonl",
        "floor": {"sr": FLOOR_SR, "note": FLOOR_NOTE},
        "actions": actions,
        "progress": snaps,
        "results_new": added,
        "scoreboard": {
            "hosts_with_gate": gates_done,
            "best": (
                {"host": best_sr[1], "sr": best_sr[2].get("sr"), "arrived": best_sr[2].get("arrived_routes")}
                if best_sr
                else None
            ),
        },
    }
    LAST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# 四机 · 路线 A 结果奖励关罩（含结果统计）",
        "",
        f"**updated:** {payload['updated']} · stamp `{STAMP}` · 四机分档",
        f"**下限对照:** {FLOOR_SR}（{FLOOR_NOTE}）",
        "",
        "| host | 档 |",
        "|------|----|",
        "| 125 | baseline **p=2 c=6 s=20** H=40 |",
        "| 84 | soft_prog **p=1 c=10** s=20 |",
        "| 14 | success_pull p=2 c=6 **s=40** |",
        "| 11 | mid_prog **p=3.5** c=6 s=20 |",
        "",
        "账本: `MAINCHANNEL_LEDGER.md` · 结果流水: `outcome_a_RESULTS.jsonl`",
        "",
        "## 进度",
        "",
        "| Host | variant | ensure | alive | iter | Hz | done |",
        "|------|---------|--------|-------|------|----|------|",
    ]
    for h in ("84", "125", "14", "11"):
        s = snaps.get(h) or {}
        hz = s.get("train_hz")
        tgt = s.get("iters_target") or _iters_target_for(h)
        lines.append(
            f"| **{h}** | {s.get('variant') or HOST_VARIANT.get(h,'?')} | {actions.get(h)} | "
            f"{int(bool(s.get('alive')))} | {s.get('iter_max', '?')}/{tgt} | "
            f"{hz if hz is not None else '-'} | {int(bool(s.get('done')))} |"
        )
    lines += ["", "## 关罩 gate 结果", ""]
    if not gates_done:
        lines.append("_尚无 gate；训完后自动写入。_")
    else:
        lines += [
            "| Host | SR | P0_stick | 到达 | vs下限2/4 | 分路 |",
            "|------|----|----------|------|-----------|------|",
        ]
        for h in ("84", "125", "14", "11"):
            s = snaps.get(h) or {}
            g = s.get("gate")
            if not g:
                lines.append(f"| **{h}** | — | — | — | — | 训练中 |")
                continue
            arr = ",".join(f"R{x}" for x in (g.get("arrived_routes") or [])) or "无"
            lines.append(
                f"| **{h}** | {g.get('sr')} | {g.get('p0_stick')} | {arr} | "
                f"{g.get('vs_floor')} | `{_fmt_routes(s.get('routes') or [])}` |"
            )
    if payload["scoreboard"]["best"]:
        b = payload["scoreboard"]["best"]
        lines += ["", f"**当前最好:** h{b['host']} SR={b['sr']} 到达={b.get('arrived')}"]
    if added:
        lines += ["", f"**本轮新入库:** {len(added)} 条 → `outcome_a_RESULTS.jsonl`"]
    STATUS.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
