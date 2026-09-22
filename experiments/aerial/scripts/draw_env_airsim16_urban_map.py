#!/usr/bin/env python3
"""Draw env_airsim_16 urban fly zones, water exclusions, and all m1a20 spawns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DEFAULT_REGIONS = "experiments/aerial/phase3_unified/env_airsim16_regions.json"
DEFAULT_M1A20 = "artifacts/seen_airsim16_m1a20.json"


def _load_routes(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "routes" in data:
        return list(data["routes"])
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported format: {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--regions", default=DEFAULT_REGIONS)
    p.add_argument("--routes", default=DEFAULT_M1A20)
    p.add_argument("--out", default="artifacts/env_airsim16_urban_map.png")
    p.add_argument("--also-desktop", action="store_true", default=True)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from experiments.aerial.phase3_unified.region_geometry import classify_spawn_xy, load_regions

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    regions = load_regions(root / args.regions)
    routes = _load_routes(root / args.routes)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=140)

    for w in regions.get("water_exclusion", []):
        poly = Polygon(
            w["polygon_xy"],
            closed=True,
            facecolor="#4a90d9",
            edgecolor="#1a5a9a",
            alpha=0.45,
            linewidth=1.5,
            label=f"water: {w['id']}" if w == regions["water_exclusion"][0] else None,
        )
        ax.add_patch(poly)
        cx = np.mean([pt[0] for pt in w["polygon_xy"]])
        cy = np.mean([pt[1] for pt in w["polygon_xy"]])
        ax.text(cx, cy, w["id"], ha="center", va="center", fontsize=7, color="#0a3060")

    for u in regions.get("urban_regions", []):
        poly = Polygon(
            u["polygon_xy"],
            closed=True,
            facecolor="#8fbc8f",
            edgecolor="#2d6a2d",
            alpha=0.35,
            linewidth=1.5,
            label="urban fly zone" if u == regions["urban_regions"][0] else None,
        )
        ax.add_patch(poly)
        cx = np.mean([pt[0] for pt in u["polygon_xy"]])
        cy = np.mean([pt[1] for pt in u["polygon_xy"]])
        ax.text(cx, cy, u["id"], ha="center", va="center", fontsize=7, color="#1a4020")

    for u in regions.get("urban_interior", []):
        poly = Polygon(
            u["polygon_xy"],
            closed=True,
            facecolor="#c4e8c4",
            edgecolor="#1a6a1a",
            alpha=0.55,
            linewidth=2.0,
            linestyle="--",
            label="urban interior" if u == regions["urban_interior"][0] else None,
        )
        ax.add_patch(poly)

    ok_x, ok_y, bad_x, bad_y, labels_ok, labels_bad = [], [], [], [], [], []
    for i, r in enumerate(routes):
        pt = np.asarray(r["pos"], dtype=np.float64)[0]
        meta = classify_spawn_xy(float(pt[0]), float(pt[1]), regions)
        label = f"R{i+1:02d}"
        if meta["spawn_ok"]:
            ok_x.append(pt[0])
            ok_y.append(pt[1])
            labels_ok.append(label)
        else:
            bad_x.append(pt[0])
            bad_y.append(pt[1])
            why = "water" if meta["in_water"] else "non-urban"
            labels_bad.append(f"{label} ({why})")

    ax.scatter(ok_x, ok_y, c="#1a8f1a", s=80, zorder=5, label="spawn OK (urban)")
    ax.scatter(bad_x, bad_y, c="#d62728", s=100, marker="x", zorder=5, label="spawn REJECT")

    for x, y, lab in zip(ok_x, ok_y, labels_ok):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color="#0a4a0a")
    for x, y, lab in zip(bad_x, bad_y, labels_bad):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color="#8b0000")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("env_airsim_16 — urban (green), interior (dashed), water (blue)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"  spawn OK: {len(ok_x)}  REJECT: {len(bad_x)}")

    if args.also_desktop:
        desktop = Path.home() / "Desktop/aerial-wam-v2-仿真视频佐证/complex_routes/env_airsim16_urban_map.png"
        desktop.parent.mkdir(parents=True, exist_ok=True)
        desktop.write_bytes(out.read_bytes())
        print(f"Copied -> {desktop}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
