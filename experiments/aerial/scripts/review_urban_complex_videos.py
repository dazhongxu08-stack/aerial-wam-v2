#!/usr/bin/env python3
"""Forensic review of urban-complex eval trajectories (video proxy via jsonl)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def analyze_traj(rows: list[dict]) -> dict:
    if not rows:
        return {"steps": 0, "verdict": "EMPTY"}
    positions = []
    d_goals = []
    d_fwds = []
    chosen = []
    for r in rows:
        p = r.get("pos") or r.get("position")
        if p is not None:
            positions.append([float(x) for x in p[:3]])
        for key in ("d_to_goal", "d_to_g", "d_goal"):
            if key in r and r[key] is not None:
                d_goals.append(float(r[key]))
                break
        if "d_fwd" in r:
            d_fwds.append(float(r["d_fwd"]))
        if r.get("chosen_idx") is not None:
            chosen.append(int(r["chosen_idx"]))

    steps = len(rows)
    start = positions[0] if positions else None
    end = positions[-1] if positions else None
    displacement = 0.0
    path_len = 0.0
    if len(positions) >= 2:
        for a, b in zip(positions, positions[1:]):
            seg = math.dist(a, b)
            path_len += seg
        displacement = math.dist(positions[0], positions[-1])

    d0 = d_goals[0] if d_goals else None
    d_min = min(d_goals) if d_goals else None
    d_fin = d_goals[-1] if d_goals else None
    d_fwd_min = min(d_fwds) if d_fwds else None

    # stall: last 20% steps d_min not improving
    stall = False
    if d_goals and len(d_goals) >= 10:
        tail = d_goals[int(len(d_goals) * 0.8) :]
        stall = max(tail) - min(tail) < 1.0 and (d_min or 999) > 15.0

    # spawn issue: <5 steps and d_min ~ d0 ~ 88m
    spawn_stuck = steps <= 5 and d_min is not None and d_min > 80.0

    # backward: net displacement tiny vs path
    spinning = path_len > 20.0 and displacement < 5.0

    if spawn_stuck:
        verdict = "SPAWN_STUCK"
    elif steps <= 3:
        verdict = "INSTANT_END"
    elif stall and (d_min or 0) > 20.0:
        verdict = "TERMINAL_STALL"
    elif spinning:
        verdict = "LOCAL_SPIN"
    elif (d_min or 999) <= 3.0:
        verdict = "ARRIVED"
    elif (d_min or 999) <= 25.0:
        verdict = "NEAR_MISS"
    elif (d_min or 999) <= 50.0:
        verdict = "MID_ROUTE"
    else:
        verdict = "NO_PROGRESS"

    return {
        "steps": steps,
        "path_m": round(path_len, 1),
        "disp_m": round(displacement, 1),
        "d0": round(d0, 2) if d0 is not None else None,
        "d_min": round(d_min, 2) if d_min is not None else None,
        "d_fin": round(d_fin, 2) if d_fin is not None else None,
        "d_fwd_min": round(d_fwd_min, 2) if d_fwd_min is not None else None,
        "stall": stall,
        "verdict": verdict,
        "shield_frac": round(sum(1 for c in chosen if c != 0) / max(len(chosen), 1), 3) if chosen else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=str(_REPO))
    args = p.parse_args()
    root = Path(args.repo)

    runs = [
        ("V12_complex_R04", root / "artifacts/wam_phase2_v12_complex_outdoor_videos_20260916/traj/R04/route00.jsonl", "phase2_toward_g+V12 poly"),
        ("V12_complex_R17", root / "artifacts/wam_phase2_v12_complex_outdoor_videos_20260916/traj/R17/route01.jsonl", "phase2_toward_g+V12 poly"),
        ("V12_complex_R19", root / "artifacts/wam_phase2_v12_complex_outdoor_videos_20260916/traj/R19/route02.jsonl", "phase2_toward_g+V12 poly"),
        ("focus134_poly", root / "artifacts/wam_phase2_v11_interior_ft_sr_eval_20260916_night_focus134", "focus134+V11 poly"),
        ("focus134_term", root / "artifacts/wam_phase2_v11_interior_terminal_eval_20260916_night_focus134", "focus134+toward_g"),
        ("night_poly", root / "artifacts/wam_phase2_v11_interior_ft_sr_eval_20260916_night_night", "night+V11 poly"),
        ("r0_v11_base", root / "artifacts/wam_phase2_focus134_route0_poly_tune_20260917/v11_base/traj/route00.jsonl", "focus134 poly tune base"),
        ("r0_cs8_direct", root / "artifacts/wam_phase2_focus134_route0_poly_tune_20260917/cs8_direct30/traj/route00.jsonl", "focus134 cs8+direct"),
    ]

    print("=" * 90)
    print("URBAN COMPLEX VIDEO / TRAJ FORENSICS")
    print("=" * 90)

    for run_name, path, stack in runs:
        print(f"\n## {run_name} ({stack})")
        if path.is_file():
            a = analyze_traj(load_jsonl(path))
            ri = path.stem.replace("route", "")
            print(f"  route={ri} {json.dumps(a, ensure_ascii=False)}")
            continue
        if not path.is_dir():
            print(f"  MISSING {path}")
            continue
        eval_json = path / "eval_all.json"
        eval_map = {}
        if eval_json.is_file():
            for ep in json.loads(eval_json.read_text()).get("episodes", []):
                eval_map[int(ep.get("route_idx", -1))] = ep
        traj_dir = path / "traj"
        for tp in sorted(traj_dir.glob("route*.jsonl")):
            ri = int(tp.stem.replace("route", ""))
            a = analyze_traj(load_jsonl(tp))
            ep = eval_map.get(ri, {})
            prog = float(ep.get("progress_ratio", 0)) * 100 if ep else None
            extra = f" eval_prog={prog:.1f}%" if prog is not None else ""
            print(f"  route={ri:02d} {a['verdict']:14s} steps={a['steps']:4d} d_min={a['d_min']} d_fin={a['d_fin']} path={a['path_m']}m disp={a['disp_m']}m{extra}")

    # interior20 smoke videos (phase2_toward_g, 20 routes)
    smoke = root / "artifacts/wam_phase2_v11_interior20_smoke_20260916"
    if smoke.is_dir():
        print(f"\n## interior20_smoke_videos (phase2_toward_g V11 poly, 20 routes)")
        for i in range(20):
            tp = smoke / "traj" / f"interior{i:02d}" / f"route{i:02d}.jsonl"
            vp = smoke / "videos" / f"V11_interior_interior{i:02d}_dual_view_dashboard.mp4"
            if not tp.is_file():
                print(f"  route={i:02d} NO_TRAJ video={vp.exists()}")
                continue
            a = analyze_traj(load_jsonl(tp))
            vid = "has_video" if vp.is_file() else "no_video"
            print(f"  route={i:02d} {a['verdict']:14s} steps={a['steps']:4d} d_min={a['d_min']} d_fin={a['d_fin']} {vid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
