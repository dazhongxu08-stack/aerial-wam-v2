#!/usr/bin/env python3
"""Visual-motion heuristics on extracted dual-view frames."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def split_panels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    mid = w // 2
    return img[:, :mid], img[:, mid:]


def motion_score(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(np.mean(cv2.absdiff(a, b)))


def classify(m: dict, ego_m: float, map_m: float, dur: float) -> str:
    prog = m.get("prog")
    arrived = m.get("arrived")
    d_fin = m.get("d_fin")
    if arrived:
        return "ARRIVED"
    if dur < 2.0 or (ego_m < 3.0 and map_m < 3.0):
        return "STATIC_SPAWN_FAIL"
    if (d_fin or 0) > 80 or (prog is not None and prog < 5 and map_m < 15):
        return "NO_PROGRESS"
    if (d_fin or 0) <= 20 or (prog is not None and prog >= 70):
        return "GOOD_FLY_STALL"
    if (prog or 0) >= 20:
        return "PARTIAL_FLY"
    return "WEAK_FLY"


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/urban_complex_videos_from_125")
    idx = json.loads((root / "_review_frames/index.json").read_text(encoding="utf-8"))
    rows = []
    for item in idx:
        frames = item.get("frames") or []
        if len(frames) < 2:
            rows.append({**item, "visual": "NO_FRAMES", "ego_motion": 0, "map_motion": 0})
            continue
        f0 = cv2.imread(str(root / frames[0]))
        fl = cv2.imread(str(root / frames[-1]))
        e0, m0 = split_panels(f0)
        el, ml = split_panels(fl)
        ego_m = motion_score(e0, el)
        map_m = motion_score(m0, ml)
        vis = classify(item, ego_m, map_m, float(item.get("dur_s") or 0))
        rows.append({**item, "visual": vis, "ego_motion": round(ego_m, 1), "map_motion": round(map_m, 1)})

    out = root / "_review_frames/analysis.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"{'tag':40s} {'vis':16s} {'arr':5s} {'prog':>6s} {'d_fin':>6s} {'dur':>5s} {'map_m':>6s}")
    for r in sorted(rows, key=lambda x: (x.get("route_idx") if x.get("route_idx") is not None else 99, x["tag"])):
        print(
            f"{r['tag']:40s} {r.get('visual','?'):16s} {str(r.get('arrived')):5s} "
            f"{str(r.get('prog','')):>6s} {str(r.get('d_fin','')):>6s} "
            f"{str(r.get('dur_s','')):>5s} {str(r.get('map_motion','')):>6s}"
        )
    from collections import Counter
    print("\nVISUAL CLASSES:", dict(Counter(r.get("visual") for r in rows)))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
