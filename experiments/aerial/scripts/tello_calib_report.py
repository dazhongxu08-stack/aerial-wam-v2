#!/usr/bin/env python3
"""Analyse Tello rc calibration and DA3 depth probe runs.

  PYTHONPATH=. python3 experiments/aerial/scripts/tello_calib_report.py rc     [RUN_DIR ...]
  PYTHONPATH=. python3 experiments/aerial/scripts/tello_calib_report.py depth  [RUN_DIR ...] [--device mps]

Without RUN_DIRs, scans artifacts/tello_gs/run_* by meta ``intended_use``.
rc   → artifacts/tello_gs/rc_calib_report.json   (per-stick speed / yaw rate / latency, fit at stick 100)
depth→ artifacts/tello_gs/depth_probe_report.json (DA3 vs tape per target)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[3]
GS = REPO / "artifacts/tello_gs"
DEPTH_CKPT = REPO / "experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _meta(run: Path) -> Dict[str, Any]:
    p = run / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _runs(args_runs: List[str], use: str) -> List[Path]:
    if args_runs:
        return [Path(r) for r in args_runs]
    return [r for r in sorted(GS.glob("run_*")) if _meta(r).get("intended_use") == use]


def _segments(tel: List[Dict[str, Any]], idx: int) -> List[Dict[str, Any]]:
    """Maximal runs of constant non-zero rc[idx] with the other sticks (except c) zero."""
    segs: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for i, r in enumerate(tel):
        rc = r.get("rc") or [0, 0, 0, 0]
        v = int(rc[idx])
        others = [int(rc[j]) for j in (0, 1, 3) if j != idx]
        if v != 0 and not any(others):
            if cur is not None and cur["stick"] == v:
                cur["i1"] = i
                continue
            if cur is not None:
                segs.append(cur)
            cur = {"stick": v, "i0": i, "i1": i}
        elif cur is not None:
            segs.append(cur)
            cur = None
    if cur is not None:
        segs.append(cur)
    return [s for s in segs if tel[s["i1"]]["t_s"] - tel[s["i0"]]["t_s"] >= 0.8]


def _tf(r: Dict[str, Any], k: str) -> float:
    try:
        return float((r.get("tello") or {}).get(k))
    except (TypeError, ValueError):
        return float("nan")


def _fit_at_100(points: List[tuple]) -> Optional[float]:
    """Least squares through origin: value = k * stick/100 → k."""
    pts = [(s / 100.0, v) for s, v in points if math.isfinite(v)]
    if not pts:
        return None
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    return float((x * y).sum() / max(1e-9, (x * x).sum()))


def _fit_dead_zone(points: List[tuple]) -> Optional[Dict[str, float]]:
    """value = slope * (stick - dead_zone); needs ≥2 distinct sticks."""
    pts = [(float(s), v) for s, v in points if math.isfinite(v)]
    if len({p[0] for p in pts}) < 2:
        return None
    slope, icpt = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
    return {"per_stick": round(float(slope), 4), "dead_zone_stick": round(float(-icpt / slope), 1) if slope else None,
            "at_100": round(float(slope * 100 + icpt), 3)}


def report_rc(runs: List[Path], out_path: Path) -> Dict[str, Any]:
    fwd_rows, yaw_rows = [], []
    for run in runs:
        meta = _meta(run)
        cal = meta.get("rc_calib") or {}
        tel = _jsonl(run / "telemetry.jsonl")
        if not tel:
            continue
        t = np.array([r["t_s"] for r in tel], np.float64)
        if cal.get("axis") == "fwd":
            segs = _segments(tel, 1)
            if int(cal.get("stick", 0)) == 0:
                fwd_rows.append({"run": run.name, "stick": 0, "tape_m": cal.get("tape_m"),
                                 "lateral_m": cal.get("lateral_m"), "note": "drift baseline"})
                continue
            # vgx is not body-forward on this firmware; horizontal |vg| is frame-free.
            spd = 0.1 * np.hypot(np.array([_tf(r, "vgx") for r in tel]), np.array([_tf(r, "vgy") for r in tel]))
            for s in segs:
                i0, i1 = s["i0"], s["i1"]
                dur = t[i1] - t[i0]
                tail = [i for i in range(i0, i1 + 1) if t[i] >= t[i0] + 0.4 * dur]
                v_ss = float(np.nanmedian(spd[tail])) if tail else float("nan")
                coast_end = int(np.searchsorted(t, t[i1] + 2.0))
                seg = slice(i0, max(i1 + 1, coast_end))
                integ = float(np.nansum(spd[seg][:-1] * np.diff(t[seg])))
                lat = next((t[i] - t[i0] for i in range(i0, min(len(t), i1 + 20)) if spd[i] - spd[i0] >= 0.1), None)
                tape = cal.get("tape_m")
                # tape and dist_vgx both span stick-on → 2 s coast, so their ratio rescales |vg|.
                scale = None if tape is None or integ < 0.05 else float(tape) / integ
                fwd_rows.append({
                    "run": run.name, "stick": s["stick"], "hold_s": round(dur, 2),
                    "v_ss_vgx_mps": round(v_ss, 3), "dist_vgx_m": round(integ, 3),
                    "tape_m": tape, "lateral_m": cal.get("lateral_m"),
                    "vgx_scale_tape": None if scale is None else round(scale, 3),
                    "v_tape_mps": None if tape is None else round(v_ss * scale if scale else float(tape) / max(0.1, dur), 3),
                    "v_tape_method": None if tape is None else ("vg_scaled" if scale else "tape_over_hold"),
                    "latency_s": None if lat is None else round(float(lat), 2),
                    "stopped_early": cal.get("stopped_early"),
                })
        elif cal.get("axis") == "yaw":
            yaw = np.degrees(np.unwrap(np.radians(np.array([_tf(r, "yaw") for r in tel], np.float64))))
            for s in _segments(tel, 3):
                i0, i1 = s["i0"], s["i1"]
                dur = t[i1] - t[i0]
                idx = [i for i in range(i0, i1 + 1) if t[i] >= t[i0] + 0.3 * dur]
                rate = float(np.polyfit(t[idx], yaw[idx], 1)[0]) if len(idx) >= 3 else float("nan")
                lat = next((t[i] - t[i0] for i in range(i0, min(len(t), i1 + 20)) if abs(yaw[i] - yaw[i0]) >= 2.0), None)
                end = int(np.searchsorted(t, t[i1] + 1.0))
                total = float(yaw[min(end, len(yaw) - 1)] - yaw[i0])
                yaw_rows.append({
                    "run": run.name, "stick": s["stick"], "hold_s": round(dur, 2),
                    "rate_deg_s": round(rate, 1), "total_deg": round(total, 1),
                    "latency_s": None if lat is None else round(float(lat), 2),
                })
    # Tello d is CW+, telemetry yaw grows CW, so rate/stick should be positive.
    fwd_pts_tape = [(r["stick"], r["v_tape_mps"]) for r in fwd_rows if r.get("v_tape_mps") is not None]
    fwd_pts_vgx = [(r["stick"], r["v_ss_vgx_mps"]) for r in fwd_rows if r["stick"] and "v_ss_vgx_mps" in r]
    yaw_pts = [(abs(r["stick"]), abs(r["rate_deg_s"])) for r in yaw_rows]
    v100_tape = _fit_at_100(fwd_pts_tape)
    v100_vgx = _fit_at_100(fwd_pts_vgx)
    y100 = _fit_at_100(yaw_pts)
    by_stick: Dict[int, List[float]] = {}
    for s, v in yaw_pts:
        by_stick.setdefault(s, []).append(v)
    out = {
        "fwd": fwd_rows,
        "yaw": yaw_rows,
        "yaw_rate_by_stick": {str(k): round(statistics.median(v), 1) for k, v in sorted(by_stick.items())},
        "fit": {
            "max_v_mps_at_100_tape": None if v100_tape is None else round(v100_tape, 3),
            "max_v_mps_at_100_vgx": None if v100_vgx is None else round(v100_vgx, 3),
            "max_yaw_deg_s_at_100": None if y100 is None else round(y100, 1),
            "max_yaw_rad_s_at_100": None if y100 is None else round(math.radians(y100), 4),
            "fwd_dead_zone_tape": _fit_dead_zone(fwd_pts_tape),
            "yaw_dead_zone_deg_s": _fit_dead_zone(yaw_pts),
        },
        "code_assumes": {
            "tello_consequence_policy": {"MAX_V_MPS": 0.35, "MAX_YAW_RAD_S": round(math.pi / 6, 4)},
            "tello_env": {"max_v_mps": 0.6, "max_yaw_rad_s": round(math.pi / 4, 4)},
        },
        "note": "Linear through origin; check per-stick rows for a dead zone / non-linearity before replacing constants.",
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    return out


def report_depth(runs: List[Path], device: str, ckpt: Path, out_path: Path) -> Dict[str, Any]:
    from PIL import Image

    from experiments.aerial.rl.collect_consequence_dataset import pool_min
    from experiments.aerial.rl.consequence_model import cones3_from_depth56
    from experiments.aerial.rl.depth_geometry import forward_min_depth
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    pred = DepthMinPredictor.from_checkpoint(str(ckpt), device=device)
    rows = []
    for run in runs:
        probe = _meta(run).get("depth_probe") or {}
        jpgs = sorted(p for p in (run / "calib_stills").glob("*.jpg") if not p.name.startswith("._"))
        if not probe or not jpgs:
            continue
        per = []
        for p in jpgs:
            rgb = np.asarray(Image.open(p).convert("RGB"))
            pred.reset()
            d = None
            for _ in range(4):
                d = pred.predict_depth(Observation(rgb=rgb, state=np.zeros(7, np.float32)))
            d = np.asarray(d, np.float32).squeeze()
            h, w = d.shape
            crop = d[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
            crop = crop[np.isfinite(crop) & (crop > 0)]
            cones = cones3_from_depth56(np.asarray(pool_min(d), np.float32))
            per.append({
                "file": p.name,
                "center_med": float(np.median(crop)),
                "center_p10": float(np.percentile(crop, 10)),
                "fwd_min": float(forward_min_depth(d, center_frac=0.5)),
                "cone_fwd": float(cones[0]), "cone_left": float(cones[1]), "cone_right": float(cones[2]),
            })
        tape = float(probe["tape_m"])
        land = probe.get("tape_land_m")
        gt = tape if land is None else 0.5 * (tape + float(land))
        med = statistics.median(x["center_med"] for x in per)
        fmin = statistics.median(x["fwd_min"] for x in per)
        cfwd = statistics.median(x["cone_fwd"] for x in per)
        rows.append({
            "run": run.name, "kind": probe.get("target_kind"), "desc": probe.get("target_desc"),
            "tape_m": tape, "tape_land_m": land, "gt_m": round(gt, 3),
            "drift_m": None if land is None else round(float(land) - tape, 3),
            "drift_flag": land is not None and abs(float(land) - tape) > 0.15,
            "n": len(per),
            "center_med_m": round(med, 3), "ratio_center_med": round(med / gt, 3),
            "fwd_min_m": round(fmin, 3), "ratio_fwd_min": round(fmin / gt, 3),
            "cone_fwd_m": round(cfwd, 3), "ratio_cone_fwd": round(cfwd / gt, 3),
            "spread_center_med": [round(min(x["center_med"] for x in per), 3), round(max(x["center_med"] for x in per), 3)],
            "stills": per,
        })
    out = {
        "ckpt": str(ckpt),
        "note": "Thin targets (chair_leg/pole): judge by fwd_min / cone_fwd, not center median. ratio>1 = reads farther than real (unsafe).",
        "runs": rows,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("rc", "depth"))
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt", default=str(DEPTH_CKPT))
    ap.add_argument("--out", default="", help="report path (default artifacts/tello_gs/<what>_report.json)")
    args = ap.parse_args()
    if args.what == "rc":
        out = report_rc(_runs(args.runs, "tello_rc_calib"), Path(args.out or GS / "rc_calib_report.json"))
        for r in out["fwd"]:
            print("fwd", {k: r.get(k) for k in ("run", "stick", "hold_s", "v_ss_vgx_mps", "dist_vgx_m", "tape_m", "vgx_scale_tape", "v_tape_mps", "latency_s", "stopped_early")})
        for r in out["yaw"]:
            print("yaw", r)
        print("yaw_rate_by_stick", out["yaw_rate_by_stick"])
        print("fit", out["fit"])
    else:
        out = report_depth(_runs(args.runs, "tello_depth_probe"), args.device, Path(args.ckpt),
                           Path(args.out or GS / "depth_probe_report.json"))
        for r in out["runs"]:
            print({k: r[k] for k in r if k != "stills"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
