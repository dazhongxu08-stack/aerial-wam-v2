#!/usr/bin/env python3
"""Print per-video forensic table for urban-complex benchmark."""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def load_summary(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else _REPO
    rows: list[dict] = []

    base = root / "artifacts/wam_phase2_v11_interior20_smoke_20260916"
    for i in range(20):
        p = base / "videos" / f"V11_interior_interior{i:02d}_trajectory_summary.json"
        if not p.exists():
            continue
        s = load_summary(p)
        r = {
            "tag": f"smoke20_r{i}",
            "stack": "phase2_toward_g+V11poly",
            "route_idx": i,
            "arrived": s.get("arrived"),
            "prog": round(float(s.get("progress_ratio", 0)) * 100, 1),
            "d_fin": round(float(s.get("final_distance_m", 0)), 1),
            "d_min": round(float(s.get("final_distance_m", 0)), 1),
            "steps": s.get("n_flown_steps"),
            "z_start": round(float(s["start_pos"][2]), 1) if s.get("start_pos") else None,
            "dur_s": s.get("duration_seconds"),
            "video": str(base / "videos" / f"V11_interior_interior{i:02d}_dual_view_dashboard.mp4"),
        }
        ev = base / f"eval_interior{i:02d}.json"
        if ev.is_file():
            ep = json.loads(ev.read_text(encoding="utf-8"))["episodes"][0]
            r["d_min"] = round(float(ep.get("d_min_m", 0)), 1)
            r["prog"] = round(float(ep.get("progress_ratio", 0)) * 100, 1)
            r["arrived"] = ep.get("arrived")
        rows.append(r)

    for label in ("R04", "R17"):
        p = root / f"artifacts/wam_phase2_v12_complex_outdoor_videos_20260916/videos/V12_complex_{label}_trajectory_summary.json"
        if p.is_file():
            s = load_summary(p)
            r = {
                "tag": f"v12complex_{label}",
                "stack": "phase2_toward_g+V12poly",
                "route_idx": s.get("route_idx"),
                "arrived": s.get("arrived"),
                "prog": round(float(s.get("progress_ratio", 0)) * 100, 1),
                "d_fin": round(float(s.get("final_distance_m", 0)), 1),
                "d_min": round(float(s.get("final_distance_m", 0)), 1),
                "steps": s.get("n_flown_steps"),
                "z_start": round(float(s["start_pos"][2]), 1) if s.get("start_pos") else None,
                "dur_s": s.get("duration_seconds"),
                "video": str(p.parent / f"V12_complex_{label}_dual_view_dashboard.mp4"),
            }
            ev = root / f"artifacts/wam_phase2_v12_complex_outdoor_videos_20260916/eval_{label}.json"
            if ev.is_file():
                ep = json.loads(ev.read_text(encoding="utf-8"))["episodes"][0]
                r["d_min"] = round(float(ep.get("d_min_m", 0)), 1)
                r["prog"] = round(float(ep.get("progress_ratio", 0)) * 100, 1)
            rows.append(r)

    for p in sorted(glob.glob(str(root / "artifacts/wam_phase2_v12_urban_smoke_r*_20260916/videos/*_trajectory_summary.json"))):
        s = load_summary(Path(p))
        tag = Path(p).stem.replace("_trajectory_summary", "")
        rows.append({
            "tag": tag,
            "stack": "urban_smoke",
            "route_idx": s.get("route_idx"),
            "arrived": s.get("arrived"),
            "prog": round(float(s.get("progress_ratio", 0)) * 100, 1),
            "d_fin": round(float(s.get("final_distance_m", 0)), 1),
            "d_min": round(float(s.get("final_distance_m", 0)), 1),
            "steps": s.get("n_flown_steps"),
            "z_start": round(float(s["start_pos"][2]), 1) if s.get("start_pos") else None,
            "dur_s": s.get("duration_seconds"),
            "video": str(Path(p).parent / f"{tag}_dual_view_dashboard.mp4"),
        })

    for name, sub in (
        ("f134_poly", "ft_sr_eval_20260916_night_focus134"),
        ("f134_term", "terminal_eval_20260916_night_focus134"),
    ):
        ev = root / f"artifacts/wam_phase2_v11_interior_{sub}/eval_all.json"
        if not ev.is_file():
            continue
        stack = "focus134+V11poly" if "poly" in name else "focus134+toward_g"
        for ep in json.loads(ev.read_text(encoding="utf-8")).get("episodes", []):
            ri = int(ep["route_idx"])
            rows.append({
                "tag": f"{name}_r{ri}",
                "stack": stack,
                "route_idx": ri,
                "arrived": ep.get("arrived"),
                "prog": round(float(ep.get("progress_ratio", 0)) * 100, 1),
                "d_fin": round(float(ep.get("d_final_m", 0)), 1),
                "d_min": round(float(ep.get("d_min_m", 0)), 1),
                "steps": ep.get("steps"),
                "z_start": None,
                "dur_s": None,
                "video": "(no video)",
            })

    def classify(r: dict) -> str:
        if r.get("steps") == 1 or (r.get("d_min") or 0) > 85:
            return "A_SPAWN_DEAD"
        if r.get("arrived"):
            return "Z_ARRIVED"
        if (r.get("d_min") or 999) <= 20:
            return "B_NEAR_MISS"
        if (r.get("prog") or 0) >= 70:
            return "C_GOOD_STALL"
        if (r.get("prog") or 0) >= 30:
            return "D_MID_STALL"
        if (r.get("prog") or 0) >= 5:
            return "E_LOW_PROG"
        return "F_NO_PROG"

    print("route | class | stack | arr | prog% | d_min | d_fin | steps | z0 | tag")
    by_route: dict[int, list[dict]] = {}
    for r in rows:
        ri = int(r.get("route_idx", -1))
        by_route.setdefault(ri, []).append(r)

    for ri in sorted(by_route):
        print(f"\n### ROUTE {ri}")
        for r in sorted(by_route[ri], key=lambda x: x["tag"]):
            c = classify(r)
            print(
                f"  {c} | {r['stack']:22s} | arr={str(r.get('arrived')):5s} | "
                f"prog={r.get('prog'):5} | d_min={r.get('d_min')} | d_fin={r.get('d_fin')} | "
                f"steps={r.get('steps')} | z0={r.get('z_start')} | {r['tag']}"
            )

    modes = Counter(classify(r) for r in rows)
    print("\n=== GLOBAL FAILURE MODES ===")
    for k in sorted(modes):
        print(f"  {k}: {modes[k]}")

    arrived = [r for r in rows if r.get("arrived")]
    print(f"\nTOTAL rows={len(rows)} arrived={len(arrived)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
