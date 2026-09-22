#!/usr/bin/env python3
"""Post-process V12 outdoor-complex eval into a markdown report + JSON summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _traj_stats(traj_path: Path) -> Dict[str, Any]:
    rows = _load_jsonl(traj_path)
    if not rows:
        return {}
    d_fwd = [float(r["d_fwd"]) for r in rows if r.get("d_fwd") is not None]
    d_left = [float(r["d_left"]) for r in rows if r.get("d_left") is not None]
    d_right = [float(r["d_right"]) for r in rows if r.get("d_right") is not None]
    intervened = sum(1 for r in rows if r.get("intervened"))
    out: Dict[str, Any] = {
        "n_steps": len(rows),
        "intervention_steps": intervened,
        "intervention_step_rate": round(intervened / max(1, len(rows)), 4),
    }
    if d_fwd:
        arr = np.asarray(d_fwd, dtype=np.float64)
        out.update(
            {
                "d_fwd_min_m": round(float(np.min(arr)), 2),
                "d_fwd_p10_m": round(float(np.percentile(arr, 10)), 2),
                "d_fwd_median_m": round(float(np.median(arr)), 2),
                "d_fwd_mean_m": round(float(np.mean(arr)), 2),
                "d_fwd_lt_8m_frac": round(float(np.mean(arr < 8.0)), 4),
                "d_fwd_lt_12m_frac": round(float(np.mean(arr < 12.0)), 4),
            }
        )
    if d_left and d_right:
        lat = np.minimum(np.asarray(d_left), np.asarray(d_right))
        out["d_lateral_min_m"] = round(float(np.min(lat)), 2)
        out["d_lateral_median_m"] = round(float(np.median(lat)), 2)
    return out


def build_report(
    eval_json: Path,
    traj_dir: Path,
    annotation: Path,
    *,
    baseline_json: Optional[Path] = None,
) -> Dict[str, Any]:
    eval_data = json.loads(eval_json.read_text(encoding="utf-8"))
    anno = json.loads(annotation.read_text(encoding="utf-8"))
    meta_by_idx = {int(ep["route_idx"]): ep for ep in anno.get("episodes", [])}

    baseline_by_base: Dict[int, Dict[str, Any]] = {}
    if baseline_json and baseline_json.is_file():
        base_data = json.loads(baseline_json.read_text(encoding="utf-8"))
        for ep in base_data.get("episodes", []):
            b = ep.get("base_route_idx")
            if b is not None:
                baseline_by_base[int(b)] = ep

    episodes_out: List[Dict[str, Any]] = []
    for ep in sorted(eval_data.get("episodes", []), key=lambda x: int(x["route_idx"])):
        ri = int(ep["route_idx"])
        meta = meta_by_idx.get(ri, {})
        cm = meta.get("complexity_meta") or meta.get("complexity") or {}
        traj_path = traj_dir / f"route{ri:02d}.jsonl"
        if not traj_path.is_file():
            alt = list(traj_dir.glob(f"**/route{ri:02d}.jsonl"))
            traj_path = alt[0] if alt else traj_path
        traj_stats = _traj_stats(traj_path)
        base_idx = ep.get("base_route_idx", meta.get("base_route_idx"))
        base_ep = baseline_by_base.get(int(base_idx)) if base_idx is not None else None
        episodes_out.append(
            {
                "route_idx": ri,
                "route_label": cm.get("route_label", f"R{int(base_idx)+1:02d}" if base_idx is not None else f"R{ri+1:02d}"),
                "base_route_idx": base_idx,
                "astar_category": cm.get("astar_category", ""),
                "complexity_score": cm.get("complexity_score"),
                "z_mean_m": cm.get("z_mean_m"),
                "excluded_from_long16": cm.get("excluded_from_long16"),
                "arrived": ep.get("arrived"),
                "spl": ep.get("spl"),
                "d_min_m": ep.get("d_min_m"),
                "d_final_m": ep.get("d_final_m"),
                "progress_ratio": ep.get("progress_ratio"),
                "intervention_rate": ep.get("intervention_rate"),
                "fail_tag": ep.get("fail_tag"),
                "traj": traj_stats,
                "full16_baseline": (
                    {
                        "arrived": base_ep.get("arrived"),
                        "spl": base_ep.get("spl"),
                        "d_min_m": base_ep.get("d_min_m"),
                        "intervention_rate": base_ep.get("intervention_rate"),
                    }
                    if base_ep
                    else None
                ),
            }
        )

    n = len(episodes_out)
    arrived = sum(1 for e in episodes_out if e.get("arrived"))
    d_fwd_mins = [e["traj"].get("d_fwd_min_m") for e in episodes_out if e["traj"].get("d_fwd_min_m") is not None]
    irs = [float(e["intervention_rate"]) for e in episodes_out if e.get("intervention_rate") is not None]

    summary = {
        "protocol": "wam_phase2_outdoor_complex_v1",
        "eval_json": str(eval_json),
        "n_routes": n,
        "success_rate": round(arrived / max(1, n), 4),
        "n_arrived": arrived,
        "mean_spl": round(float(np.mean([e["spl"] for e in episodes_out])), 4) if episodes_out else 0.0,
        "mean_ir": round(float(np.mean(irs)), 4) if irs else None,
        "mean_d_fwd_min_m": round(float(np.mean(d_fwd_mins)), 2) if d_fwd_mins else None,
        "episodes": episodes_out,
        "verdict": eval_data.get("verdict"),
        "metrics": eval_data.get("metrics"),
    }
    return summary


def _md_table(summary: Dict[str, Any]) -> str:
    lines = [
        "# WAM Phase-2 Outdoor Complex Scenario Report",
        "",
        f"- Routes: **{summary['n_routes']}** | SR: **{summary['n_arrived']}/{summary['n_routes']}** "
        f"({summary['success_rate']*100:.1f}%) | mean SPL: **{summary['mean_spl']:.3f}**",
    ]
    if summary.get("mean_d_fwd_min_m") is not None:
        lines.append(
            f"- Mean d_fwd min: **{summary['mean_d_fwd_min_m']:.1f} m** "
            f"| mean IR: **{summary.get('mean_ir', 0)*100:.1f}%**"
        )
    lines.extend(
        [
            "",
            "| Route | Cat | z_mean | d_fwd min | d_fwd p10 | IR | SPL | arr | vs full16 |",
            "|-------|-----|--------|-----------|-----------|----|-----|-----|-----------|",
        ]
    )
    for e in summary["episodes"]:
        t = e.get("traj") or {}
        b = e.get("full16_baseline")
        vs = "-"
        if b:
            vs = f"{'OK' if b.get('arrived') else 'FAIL'} d_min={b.get('d_min_m', 0):.1f}"
        lines.append(
            f"| {e.get('route_label','?')} | {e.get('astar_category','')} "
            f"| {e.get('z_mean_m','?')} | {t.get('d_fwd_min_m','-')} "
            f"| {t.get('d_fwd_p10_m','-')} | {float(e.get('intervention_rate',0))*100:.0f}% "
            f"| {float(e.get('spl',0)):.3f} | {'Y' if e.get('arrived') else 'N'} | {vs} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **d_fwd < 8–12 m** for a meaningful fraction of steps ⇒ drone is engaging nearby structure, not open water.",
            "- **IR > 40%** on complex routes ⇒ Shield is actively braking; compare with full16 open-corridor IR ~20%.",
            "- Routes marked **excluded_from_long16** were dropped by the longest-16 benchmark (often tighter / lower-alt).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", required=True)
    p.add_argument("--traj-dir", required=True)
    p.add_argument(
        "--annotation",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json",
    )
    p.add_argument("--baseline-json", default="", help="Optional full16 eval JSON for comparison")
    p.add_argument("--out-json", default="")
    p.add_argument("--out-md", default="")
    args = p.parse_args()

    eval_json = Path(args.eval_json)
    traj_dir = Path(args.traj_dir)
    annotation = Path(args.annotation)
    baseline = Path(args.baseline_json) if str(args.baseline_json).strip() else None

    summary = build_report(eval_json, traj_dir, annotation, baseline_json=baseline)
    out_json = Path(args.out_json) if args.out_json else eval_json.parent / "complex_report.json"
    out_md = Path(args.out_md) if args.out_md else eval_json.parent / "COMPLEX_REPORT.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_md.write_text(_md_table(summary), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(
        f"SR {summary['n_arrived']}/{summary['n_routes']} "
        f"mean_d_fwd_min={summary.get('mean_d_fwd_min_m')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
