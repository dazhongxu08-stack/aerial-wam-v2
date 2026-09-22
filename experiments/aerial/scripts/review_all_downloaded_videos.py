#!/usr/bin/env python3
"""Forensic table for every dual-view MP4 under a local download root."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


def route_from_tag(tag: str) -> int | None:
    m = re.search(r"(?:interior|R)(\d{2})\b", tag, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"route(\d+)", tag, re.I)
    if m:
        return int(m.group(1))
    return None


def classify(row: dict) -> str:
    steps = row.get("steps")
    d_min = row.get("d_min")
    d_fin = row.get("d_fin")
    prog = row.get("prog") or 0
    dur = row.get("dur_s") or 0
    visual = row.get("visual")

    if row.get("arrived"):
        return "ARRIVED"
    if steps == 1 or (d_min or 0) > 85 or (d_fin or 0) > 85 and prog < 5:
        return "SPAWN_DEAD"
    if visual == "STATIC_SPAWN_FAIL" or (dur < 2 and prog < 5):
        return "SPAWN_DEAD"
    if (d_min or 999) <= 20 or prog >= 70:
        return "NEAR_MISS"
    if prog >= 20:
        return "PARTIAL"
    if prog >= 5:
        return "WEAK"
    return "NO_PROGRESS"


def diagnosis(row: dict) -> str:
    cls = row["class"]
    notes: list[str] = []
    if cls == "ARRIVED":
        return "到达目标"
    if cls == "SPAWN_DEAD":
        if (row.get("steps") or 0) <= 1:
            notes.append("1步结束")
        if (row.get("d_fin") or 0) > 80:
            notes.append("spawn距目标~89m")
        if row.get("visual") == "STATIC_SPAWN_FAIL":
            notes.append("画面几乎无运动")
        return "spawn碰撞/无效起点: " + ", ".join(notes) if notes else "spawn失败"
    if cls == "NEAR_MISS":
        d_min = row.get("d_min")
        if d_min is not None and d_min <= 20:
            notes.append(f"最近{d_min:.1f}m")
        if (row.get("prog") or 0) >= 70:
            notes.append(f"prog {row['prog']:.0f}%")
        notes.append("末段stall/超时")
        return ", ".join(notes)
    if cls == "PARTIAL":
        return f"能飞但远未到达 (prog {row.get('prog', 0):.0f}%)"
    if cls == "WEAK":
        return f"微弱进展 (prog {row.get('prog', 0):.0f}%)"
    return "几乎无进展"


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/urban_complex_videos_from_125")
    analysis_path = root / "_review_frames/analysis.json"
    visual_by_mp4: dict[str, dict] = {}
    if analysis_path.is_file():
        for item in json.loads(analysis_path.read_text(encoding="utf-8")):
            visual_by_mp4[item.get("mp4", "")] = item

    rows: list[dict] = []
    for mp4 in sorted(root.rglob("*dual_view_dashboard.mp4")):
        tag = mp4.stem.replace("_dual_view_dashboard", "")
        batch = mp4.parent.parent.name if mp4.parent.name == "videos" else mp4.parent.name
        summary = mp4.parent / f"{tag}_trajectory_summary.json"
        meta: dict = {}
        if summary.is_file():
            meta = json.loads(summary.read_text(encoding="utf-8"))
        rel = str(mp4.relative_to(root))
        vis = visual_by_mp4.get(rel, {})
        d_fin = float(meta.get("final_distance_m", 0)) if meta else None
        prog = round(float(meta.get("progress_ratio", 0)) * 100, 1) if meta else vis.get("prog")
        row = {
            "batch": batch,
            "tag": tag,
            "route_idx": meta.get("route_idx") if meta else route_from_tag(tag),
            "arrived": meta.get("arrived") if meta else vis.get("arrived"),
            "prog": prog,
            "d_fin": round(d_fin, 1) if d_fin is not None else vis.get("d_fin"),
            "d_min": round(d_fin, 1) if d_fin is not None else vis.get("d_fin"),
            "steps": meta.get("n_flown_steps") if meta else None,
            "z_start": round(float(meta["start_pos"][2]), 1) if meta.get("start_pos") else None,
            "dur_s": meta.get("duration_seconds") if meta else vis.get("dur_s"),
            "visual": vis.get("visual"),
            "mp4": rel,
        }
        row["class"] = classify(row)
        row["diagnosis"] = diagnosis(row)
        rows.append(row)

    out = root / "_review_frames/forensic_report.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    urban_batches = (
        "interior",
        "urban_smoke",
        "complex_outdoor",
        "ft_smoke",
        "ft_sr",
        "terminal_eval",
    )
    urban = [r for r in rows if any(k in r["batch"] for k in urban_batches)]
    noise = [r for r in rows if r not in urban]

    print(f"TOTAL {len(rows)} videos | urban-relevant {len(urban)} | other {len(noise)}\n")
    print(f"{'batch':42s} {'tag':28s} {'cls':12s} {'arr':5s} {'prog':>6s} {'d_fin':>6s} {'stp':>4s} {'z0':>5s} diagnosis")
    for r in sorted(urban, key=lambda x: (x.get("route_idx") if x.get("route_idx") is not None else 99, x["batch"], x["tag"])):
        print(
            f"{r['batch']:42s} {r['tag']:28s} {r['class']:12s} {str(r.get('arrived')):5s} "
            f"{str(r.get('prog','')):>6s} {str(r.get('d_fin','')):>6s} {str(r.get('steps','')):>4s} "
            f"{str(r.get('z_start','')):>5s} {r['diagnosis']}"
        )

    print("\n--- OTHER (route10/full16 etc) ---")
    for r in noise:
        print(
            f"{r['batch']:42s} {r['tag']:28s} {r['class']:12s} "
            f"prog={r.get('prog')} d_fin={r.get('d_fin')} | {r['diagnosis']}"
        )

    print("\nURBAN CLASS COUNTS:", dict(Counter(r["class"] for r in urban)))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
