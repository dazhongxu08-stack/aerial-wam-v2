#!/usr/bin/env python3
"""Render per-route previews for the outdoor-complex subset (XY + altitude)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _plot_route(
    ep: Dict[str, Any],
    out_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(ep["pos"], dtype=np.float64).reshape(-1, 3)
    meta = ep.get("complexity_meta") or ep.get("complexity") or {}
    label = meta.get("route_label", ep.get("route_id", "?"))
    route_id = ep.get("route_id", "route")
    cat = meta.get("astar_category", "?")
    score = meta.get("complexity_score", "?")
    instr = str(ep.get("gpt_instruction", "")).strip()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=140)

    # XY top-down
    ax = axes[0]
    ax.plot(pts[:, 0], pts[:, 1], "k-", lw=2.0, zorder=2)
    ax.scatter(pts[0, 0], pts[0, 1], c="#2ca02c", s=80, zorder=3, label="start")
    ax.scatter(pts[-1, 0], pts[-1, 1], c="#d62728", s=120, marker="*", zorder=3, label="goal")
    for i in range(1, len(pts) - 1):
        ax.scatter(pts[i, 0], pts[i, 1], c="#888888", s=12, zorder=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{label} | {cat} | score={score}")
    ax.legend(loc="best", fontsize=8)

    # Altitude along arc length
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ax2 = axes[1]
    ax2.plot(s, pts[:, 2], color="#1f77b4", lw=2.0)
    ax2.fill_between(s, pts[:, 2], alpha=0.15, color="#1f77b4")
    ax2.set_xlabel("path length (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title(
        f"L={meta.get('nominal_length_m', '?')}m  "
        f"spawn_z={meta.get('spawn_z_m', '?')}m  "
        f"basis={meta.get('selection_basis', 'spawn_only')}"
    )
    ax2.grid(True, alpha=0.3)

    fig.suptitle(instr[:120] + ("…" if len(instr) > 120 else ""), fontsize=9, y=1.02)
    fig.tight_layout()
    out_png = out_dir / f"{route_id}_{label}_preview.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _catalog_md(episodes: List[Dict[str, Any]], pngs: List[Path]) -> str:
    lines = [
        "# Outdoor Complex Route Catalog",
        "",
        "6 routes selected for building-proximity / low-altitude complexity (V12 eval panel).",
        "",
    ]
    for ep, png in zip(episodes, pngs):
        meta = ep.get("complexity_meta") or ep.get("complexity") or {}
        pts = np.asarray(ep["pos"], dtype=np.float64).reshape(-1, 3)
        lines.extend(
            [
                f"## {meta.get('route_label', '?')} — `{ep.get('route_id')}`",
                "",
                f"![{ep.get('route_id')}]({png.name})",
                "",
                f"- **A* category:** `{meta.get('astar_category', '')}`",
                f"- **Complexity score:** {meta.get('complexity_score')}",
                f"- **Length:** {meta.get('nominal_length_m')} m | **tortuosity:** {meta.get('tortuosity')}",
                f"- **Spawn z:** {meta.get('spawn_z_m')} m at `{meta.get('spawn_pos', ep['pos'][0])}`",
                f"- **Urban zone:** `{meta.get('urban_region_ids', [])}`",
                f"- **Selection:** {meta.get('selection_basis', 'urban_spawn_only')}",
                f"- **Excluded from full16:** {meta.get('excluded_from_long16')}",
                f"- **Start:** `{pts[0].tolist()}`",
                f"- **Goal:** `{pts[-1].tolist()}`",
                f"- **Instruction:** {ep.get('gpt_instruction', '')}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--annotation",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json",
    )
    p.add_argument("--out-dir", default="artifacts/wam_phase2_complex_route_previews")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    anno = json.loads((root / args.annotation).read_text(encoding="utf-8"))
    episodes = list(anno.get("episodes", []))
    out_dir = root / args.out_dir

    pngs: List[Path] = []
    for ep in episodes:
        png = _plot_route(ep, out_dir)
        pngs.append(png)
        meta = ep.get("complexity_meta") or {}
        print(f"  {png.name}  {meta.get('route_label')}  {meta.get('astar_category')}")

    catalog = out_dir / "ROUTE_CATALOG.md"
    catalog.write_text(_catalog_md(episodes, [p.relative_to(out_dir) for p in pngs]), encoding="utf-8")
    print(f"Wrote {catalog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
