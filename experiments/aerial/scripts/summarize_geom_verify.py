#!/usr/bin/env python3
"""Summarize geom_verify_${STAMP} JSON outputs into SUMMARY.md + SUMMARY.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        return {"error": f"parse_fail: {e}", "path": str(p)}


def _probe_line(j: Optional[Dict[str, Any]]) -> str:
    if not j or "error" in j and "n_samples" not in j:
        return f"missing/error: {j}"
    ctr = (j.get("center_depth") or {}).get("encode_window") or {}
    fwd = j.get("forward_min_depth") or {}
    return (
        f"n={j.get('n_samples')} readout={j.get('readout')} "
        f"center_win R2={ctr.get('r2_holdout')} MAE={ctr.get('mae_holdout_m')} "
        f"verdict={ctr.get('verdict')} | "
        f"fwd_min R2={fwd.get('r2_holdout')} MAE={fwd.get('mae_holdout_m')} "
        f"verdict={fwd.get('verdict')}"
    )


def _imagine_line(j: Optional[Dict[str, Any]]) -> str:
    if not j:
        return "missing"
    v = j.get("verdict") or {}
    return (
        f"n={j.get('n_samples')} median_gap={j.get('median_p_coll_gap')} "
        f"useful={v.get('useful') if isinstance(v, dict) else v}"
    )


def _cone_line(j: Optional[Dict[str, Any]]) -> str:
    if not j:
        return "missing"
    rates = j.get("rates") or {}
    bits = []
    for k, r in rates.items():
        bits.append(f"{k}:n={r.get('n')} rate={r.get('rate')} pass={r.get('pass')}")
    return f"verdict={j.get('verdict')} " + "; ".join(bits)


def _gate_line(j: Optional[Dict[str, Any]]) -> str:
    if not j:
        return "missing"
    checks = j.get("checks") or j.get("gate") or j
    return json.dumps(checks, ensure_ascii=False)[:500]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "experiments/aerial/rl/artifacts" / f"geom_verify_{args.stamp}"
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "G1_merged": out / f"G1_latent_probe_merged_n400.json",
        "G1_near_enrich": None,
        "G3_three_zone": None,
        "G3_merged": None,
        "G4_imagine_near": None,
        "G4_cone": None,
        "G2_gate": out / "G2_obstacle_cost_gate.json",
        "G2_cone": None,
    }
    # glob flexible n
    for p in sorted(out.glob("G1_latent_probe_merged_n*.json")):
        files["G1_merged"] = p
    for p in sorted(out.glob("G1_latent_probe_near_enrich_n*.json")):
        files["G1_near_enrich"] = p
    for p in sorted(out.glob("G3_latent_probe_three_zone_near_n*.json")):
        files["G3_three_zone"] = p
    for p in sorted(out.glob("G3_latent_probe_merged_n*.json")):
        files["G3_merged"] = p
    for p in sorted(out.glob("G4_imagine_coll_rank_n*.json")):
        files["G4_imagine_near"] = p
    for p in sorted(out.glob("G4_cone_rank_n*.json")):
        files["G4_cone"] = p
    for p in sorted(out.glob("G2_cone_rank_n*.json")):
        files["G2_cone"] = p

    g1m = _load(files["G1_merged"]) if files["G1_merged"] else None
    g1n = _load(files["G1_near_enrich"]) if files["G1_near_enrich"] else None
    g3z = _load(files["G3_three_zone"]) if files["G3_three_zone"] else None
    g3m = _load(files["G3_merged"]) if files["G3_merged"] else None
    g4i = _load(files["G4_imagine_near"]) if files["G4_imagine_near"] else None
    g4c = _load(files["G4_cone"]) if files["G4_cone"] else None
    g2g = _load(files["G2_gate"]) if files["G2_gate"] else None
    g2c = _load(files["G2_cone"]) if files["G2_cone"] else None

    # Pass criteria (tonight gate)
    fwd_ok = False
    for j in (g1m, g1n, g3z, g3m):
        if not j:
            continue
        fwd = j.get("forward_min_depth") or {}
        if fwd.get("verdict") == "has_geometry" and float(fwd.get("r2_holdout") or -1) >= 0.3:
            fwd_ok = True
            break

    cone_ok = False
    for j in (g4c, g2c):
        if j and j.get("verdict") == "PASS":
            cone_ok = True
            break

    gate_ok = False
    if g2g:
        checks = g2g.get("checks") or {}
        if isinstance(checks, dict):
            gate_ok = all(
                bool(checks.get(k))
                for k in ("fwd_empty_wins", "fwd_near_loses", "left_near_prefers_fwd")
                if k in checks
            ) and bool(checks)

    imagine_ok = False
    if g4i:
        v = g4i.get("verdict") or {}
        gap = g4i.get("median_p_coll_gap")
        if isinstance(v, dict) and v.get("useful"):
            imagine_ok = True
        elif gap is not None and float(gap) >= 0.05:
            imagine_ok = True

    overall = "PASS_GEOMETRY" if (fwd_ok and (cone_ok or gate_ok) and imagine_ok) else "FAIL_GEOMETRY"
    if fwd_ok and not (cone_ok or gate_ok):
        overall = "PARTIAL_LATENT_OK_COST_WEAK"
    if not fwd_ok and (cone_ok or gate_ok):
        overall = "PARTIAL_COST_OK_LATENT_WEAK"

    summary = {
        "stamp": args.stamp,
        "overall": overall,
        "fwd_min_geometry_ok": fwd_ok,
        "cone_rank_ok": cone_ok,
        "label_gate_ok": gate_ok,
        "imagine_ok": imagine_ok,
        "lines": {
            "G1_merged": _probe_line(g1m),
            "G1_near_enrich": _probe_line(g1n),
            "G3_three_zone": _probe_line(g3z),
            "G3_merged": _probe_line(g3m),
            "G4_imagine": _imagine_line(g4i),
            "G4_cone": _cone_line(g4c),
            "G2_gate": _gate_line(g2g),
            "G2_cone": _cone_line(g2c),
        },
        "decision": (
            "Allow Soft→Hard only if overall==PASS_GEOMETRY. "
            "If PARTIAL_COST_OK_LATENT_WEAK: fix WM depth-aux / forward hinge before actor. "
            "If PARTIAL_LATENT_OK_COST_WEAK: reflow obstacle_cost head on near-wall GT."
        ),
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    md = [
        f"# Geometry falsification · `{args.stamp}`",
        "",
        f"**Overall: `{overall}`**",
        "",
        "| Check | OK |",
        "|-------|----|",
        f"| forward_min latent geometry | {fwd_ok} |",
        f"| obstacle cone rank | {cone_ok} |",
        f"| label three-sort gate | {gate_ok} |",
        f"| imagine coll rank | {imagine_ok} |",
        "",
        "## Details",
        "",
    ]
    for k, v in summary["lines"].items():
        md.append(f"- **{k}**: `{v}`")
    md.append("")
    md.append(f"## Decision\n\n{summary['decision']}")
    md.append("")
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"out": str(out), "overall": overall}, indent=2))
    return 0 if overall == "PASS_GEOMETRY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
