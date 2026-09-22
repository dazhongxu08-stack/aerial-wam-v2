#!/usr/bin/env python3
"""Analyze urban post-train artifacts and recommend + optionally run next step."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _episodes(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(d.get("episodes") or [])


def _route_row(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "route_idx": int(e.get("route_idx", -1)),
        "arrived": bool(e.get("arrived")),
        "progress_ratio": float(e.get("progress_ratio", 0.0)),
        "intervention_rate": float(e.get("intervention_rate", 0.0)),
        "actual_length_m": float(e.get("actual_length_m", 0.0)),
        "d_min_m": e.get("d_min_m"),
        "d_final_m": e.get("d_final_m"),
        "collided": bool(e.get("collided")),
    }


def _by_route(eps: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(e["route_idx"]): _route_row(e) for e in eps if "route_idx" in e}


def _classify(row: Dict[str, Any]) -> str:
    if not row:
        return "MISSING"
    if row.get("spawn_fail"):
        return "SPAWN_DEAD"
    if row["arrived"]:
        return "ARRIVED"
    prog = row["progress_ratio"] * 100
    dmin = float(row.get("d_min_m") or 999)
    if prog >= 70 or dmin <= 20:
        return "NEAR_MISS"
    if prog >= 20:
        return "PARTIAL"
    if prog >= 5:
        return "WEAK"
    return "NO_PROGRESS"


def analyze(
    *,
    baseline20: Path,
    gate_ckpt: Path,
    shield_dir: Optional[Path],
    focus_routes: List[int],
) -> Dict[str, Any]:
    base = _by_route(_episodes(_load(baseline20)))
    gate = _by_route(_episodes(_load(gate_ckpt)))
    shield: Dict[str, Dict[int, Dict[str, Any]]] = {}
    if shield_dir and shield_dir.is_dir():
        for tag in ("shield_off", "shield_fwd_only"):
            p = shield_dir / f"eval_{tag}.json"
            if p.is_file():
                shield[tag] = _by_route(_episodes(_load(p)))

    rows: List[Dict[str, Any]] = []
    for ri in focus_routes:
        b = base.get(ri, {})
        g = gate.get(ri, {})
        entry: Dict[str, Any] = {
            "route_idx": ri,
            "baseline_class": _classify(b) if b else "MISSING",
            "baseline": b,
            "gate_ckpt": g,
            "gate_delta_prog_pp": round((g.get("progress_ratio", 0) - b.get("progress_ratio", 0)) * 100, 1)
            if b and g
            else None,
            "gate_delta_ir": round(g.get("intervention_rate", 0) - b.get("intervention_rate", 0), 3)
            if b and g
            else None,
        }
        for tag, tbl in shield.items():
            if ri in tbl:
                entry[tag] = tbl[ri]
                if b:
                    entry[f"{tag}_delta_ir"] = round(tbl[ri]["intervention_rate"] - b["intervention_rate"], 3)
                    entry[f"{tag}_delta_prog_pp"] = round(
                        (tbl[ri]["progress_ratio"] - b["progress_ratio"]) * 100, 1
                    )
        rows.append(entry)

    gate_arr = sum(1 for r in rows if r.get("gate_ckpt", {}).get("arrived"))
    base_arr = sum(1 for r in rows if r.get("baseline", {}).get("arrived"))

    # Shield: mean IR drop on focus routes (baseline shield-on vs ablation arms)
    ir_fwd_drop = [
        r["shield_fwd_only_delta_ir"]
        for r in rows
        if r.get("shield_fwd_only_delta_ir") is not None and r["shield_fwd_only_delta_ir"] < 0
    ]
    mean_ir_fwd_drop = sum(ir_fwd_drop) / len(ir_fwd_drop) if ir_fwd_drop else 0.0

    route1 = next((r for r in rows if r["route_idx"] == 1), None)
    route9 = base.get(9, {})
    route9_class = _classify(route9) if route9 else "MISSING"

    next_action: Dict[str, Any]
    if route1 and route1.get("gate_ckpt", {}).get("arrived"):
        next_action = {
            "action": "gate_focus_fwd_shield",
            "reason": "route1 arrived after r1s2r FT — verify focus134 with forward-only exclusion",
            "routes": focus_routes,
            "shield_exclusion_forward_only": True,
        }
    elif mean_ir_fwd_drop <= -0.15 and gate_arr <= base_arr:
        next_action = {
            "action": "gate_ckpt_fwd_shield",
            "reason": f"forward-only shield drops IR ~{mean_ir_fwd_drop:.2f} on focus routes; re-gate FT ckpt",
            "routes": focus_routes,
            "shield_exclusion_forward_only": True,
        }
    elif route1 and (route1.get("gate_delta_prog_pp") or 0) >= 5:
        next_action = {
            "action": "continue_train_route1",
            "reason": f"route1 prog +{route1.get('gate_delta_prog_pp')}pp but no arrival — extend FT",
            "focus_routes": "1",
            "iters": 32,
        }
    elif route9_class == "NEAR_MISS":
        next_action = {
            "action": "train_near_miss_route9",
            "reason": "route1 FT insufficient; baseline route9 is NEAR_MISS (65.8% prog)",
            "focus_routes": "9",
            "iters": 64,
        }
    else:
        trainable = [
            r["route_idx"]
            for r in rows
            if r.get("baseline_class") in ("NEAR_MISS", "PARTIAL") or (r.get("gate_delta_prog_pp") or 0) > 0
        ]
        if not trainable:
            trainable = [r for r in focus_routes if r != 1]
        next_action = {
            "action": "train_humanlike_depth_scene",
            "reason": (
                "pivot to depth-scene expert demos + efficiency FT "
                "(PathExpert/toward_g alone does not teach human-like OA)"
            ),
            "focus_routes": ",".join(str(x) for x in sorted(set(trainable))),
            "iters": 32,
            "collect_expert": True,
        }

    return {
        "focus_routes": focus_routes,
        "summary": {
            "baseline_arrived": base_arr,
            "gate_ckpt_arrived": gate_arr,
            "mean_ir_fwd_drop": round(mean_ir_fwd_drop, 3),
            "route1_baseline_prog_pct": round(base.get(1, {}).get("progress_ratio", 0) * 100, 1),
            "route1_gate_prog_pct": round(gate.get(1, {}).get("progress_ratio", 0) * 100, 1),
            "route9_baseline_class": route9_class,
        },
        "per_route": rows,
        "next_action": next_action,
    }


def _run_next(repo: Path, report: Dict[str, Any], *, train_stamp: str, base_stamp: str, dry_run: bool) -> None:
    import os

    act = report["next_action"]
    action = act["action"]
    print(f"\n[next] {action}: {act.get('reason')}", flush=True)
    if dry_run:
        print(json.dumps(act, indent=2))
        return

    env = dict(os.environ)
    scripts = repo / "experiments/aerial" / "scripts"
    ckpt_rel = f"experiments/aerial/rl/artifacts/v4_ac_ckpt_urban_complex_p2c_{train_stamp}/v4_ac_latest.pt"

    if action in ("gate_focus_fwd_shield", "gate_ckpt_fwd_shield"):
        routes = act.get("routes", [0, 1, 3, 4])
        stamp = f"{base_stamp}_fwdgate"
        run_env = {
            **env,
            "STAMP": stamp,
            "ACTOR": ckpt_rel,
            "ANNO": "experiments/aerial/phase3_unified/annotations/outdoor_complex_only.json",
            "ROUTES": ",".join(str(r) for r in routes),
            "MIN_SPAWN_Z": "38.0",
            "SPAWN_RETRY_M": "14.0",
            "SPAWN_Z_MAX_RETRIES": "3",
            "SHIELD_EXCLUSION_FORWARD_ONLY": "1",
        }
        subprocess.run(
            ["bash", str(scripts / "eval_urban_complex_toward_g_gate.sh")],
            cwd=str(repo),
            env=run_env,
            check=True,
        )

    elif action in ("continue_train_route1", "train_near_miss_route9", "train_focus_partial", "train_humanlike_depth_scene"):
        if action == "train_humanlike_depth_scene":
            stamp = f"{base_stamp}_humanlike"
            run_env = {
                **env,
                "STAMP": stamp,
                "FOCUS_ROUTES": str(act.get("focus_routes", "0,1,3,4")),
                "ITERS": str(act.get("iters", 32)),
                "MIN_SPAWN_Z": "38.0",
                "COLLECT_EXPERT": "1" if act.get("collect_expert", True) else "0",
                "RESUME_CKPT": ckpt_rel,
                "W_EFF_STRAFE": "0.08",
                "W_EFF_HEADING": "0.08",
                "W_EFF_IDLE": "0.03",
            }
            subprocess.run(
                ["bash", str(scripts / "train_urban_humanlike_p2c.sh")],
                cwd=str(repo),
                env=run_env,
                check=True,
            )
            return
        suffix = {
            "continue_train_route1": "r1s3",
            "train_near_miss_route9": "r9s1",
            "train_focus_partial": "focus_s1",
        }[action]
        stamp = f"{base_stamp}_{suffix}"
        run_env = {
            **env,
            "STAMP": stamp,
            "FOCUS_ROUTES": str(act["focus_routes"]),
            "ITERS": str(act.get("iters", 64)),
            "NEAR_FRAC": "0.5",
            "MIN_SPAWN_Z": "38.0",
            "SPAWN_RETRY_M": "14.0",
            "SPAWN_Z_MAX_RETRIES": "3",
            "RESUME_CKPT": ckpt_rel,
            "ENABLE_PLANNER": "1",
            "PLANNER_HORIZON": "1",
            "SHIELD_EXCLUSION_FORWARD_ONLY": "1",
            "TTI_COEFF": "2.5",
        }
        subprocess.run(
            ["bash", str(scripts / "train_urban_complex_p2c.sh")],
            cwd=str(repo),
            env=run_env,
            check=True,
        )
    else:
        print(f"unknown action {action}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline20", type=Path, required=True)
    p.add_argument("--gate-ckpt", type=Path, required=True)
    p.add_argument("--shield-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--focus-routes", default="0,1,3,4")
    p.add_argument("--train-stamp", default="20260917_r1s2r")
    p.add_argument("--base-stamp", default="20260917")
    p.add_argument("--run-next", action="store_true")
    p.add_argument("--dry-run-next", action="store_true")
    args = p.parse_args()

    focus = [int(x) for x in args.focus_routes.split(",") if x.strip()]
    report = analyze(
        baseline20=args.baseline20,
        gate_ckpt=args.gate_ckpt,
        shield_dir=args.shield_dir,
        focus_routes=focus,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    s = report["summary"]
    print(
        f"[analyze] baseline_arr={s['baseline_arrived']}/{len(focus)} "
        f"gate_arr={s['gate_ckpt_arrived']}/{len(focus)} "
        f"route1 {s['route1_baseline_prog_pct']}% -> {s['route1_gate_prog_pct']}% "
        f"ir_fwd_drop={s['mean_ir_fwd_drop']}"
    )
    print(f"[analyze] next={report['next_action']['action']}: {report['next_action']['reason']}")
    print(f"[analyze] wrote {args.out}")

    if args.run_next or args.dry_run_next:
        repo = Path(__file__).resolve().parents[3]
        _run_next(repo, report, train_stamp=args.train_stamp, base_stamp=args.base_stamp, dry_run=args.dry_run_next)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
