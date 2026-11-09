#!/usr/bin/env python3
"""Check Tello runs against tello-consequence-collect.md §3 (per run) and §4 (batch).

  PYTHONPATH=. python3 experiments/aerial/scripts/tello_posttrain_qc.py artifacts/tello_gs/run_A artifacts/tello_gs/run_B ...

Writes ``QC.json`` into each run dir and prints the batch verdict. Grades follow §5:
后训练可用 only when every §3 item passes; anything else is at most 域差可用.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

AIR_H_M = 0.30
BAND = (0.80, 1.50)
LAND_TRIM_S = 3.0


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _f(d: Dict[str, Any], k: str):
    try:
        return float(d[k])
    except (KeyError, TypeError, ValueError):
        return None


def qc_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) if (run_dir / "meta.json").is_file() else {}
    tel = _load_jsonl(run_dir / "telemetry.jsonl")
    events = _load_jsonl(run_dir / "events.jsonl")
    checks: Dict[str, Dict[str, Any]] = {}

    def chk(name: str, ok: bool, value: Any, need: str) -> None:
        checks[name] = {"ok": bool(ok), "value": value, "need": need}

    air = [r for r in tel if r.get("airborne") and (_f(r.get("tello") or {}, "h") or 0.0) / 100.0 >= AIR_H_M]
    if air:
        t_lo, t_hi = air[0]["t_s"] + LAND_TRIM_S, air[-1]["t_s"] - LAND_TRIM_S
        air = [r for r in air if t_lo <= r["t_s"] <= t_hi] or air
    air_s = (air[-1]["t_s"] - air[0]["t_s"]) if len(air) > 1 else 0.0
    hs = [(_f(r.get("tello") or {}, "h") or 0.0) / 100.0 for r in air]
    in_band = float(np.mean([BAND[0] <= h <= BAND[1] for h in hs])) if hs else 0.0
    n_edge = max(1, len(hs) // 10)
    net_dz = (statistics.median(hs[-n_edge:]) - statistics.median(hs[:n_edge])) if hs else 0.0
    rcs = [r.get("rc") or [0, 0, 0, 0] for r in air]
    n = max(1, len(rcs))
    back = sum(1 for a, b, c, d in rcs if b <= -10) / n
    strafe = sum(1 for a, b, c, d in rcs if abs(a) >= 10) / n
    combined = sum(1 for a, b, c, d in rcs if b >= 8 and abs(d) >= 8) / n
    bats = [_f(r.get("tello") or {}, "bat") for r in tel]
    bats = [b for b in bats if b is not None]
    rssi = [r.get("wifi_rssi_dbm") for r in air if r.get("wifi_rssi_dbm") is not None]
    modes = {str(r.get("control_mode")) for r in air}
    near = [e for e in events if e.get("event") == "near_miss"]
    hits = [e for e in events if e.get("event") == "collision"]

    frames = sorted(p for p in (run_dir / "frames").glob("*.jpg") if not p.name.startswith("._"))
    sample = frames[:: max(1, len(frames) // 40)]
    bad = 0
    for p in sample:
        try:
            g = np.asarray(Image.open(p).convert("L"), np.float32)
            if g.mean() < 12.0 or g.std() < 4.0:
                bad += 1
        except OSError:
            bad += 1
    bad_frac = bad / max(1, len(sample))

    goal_ok = bool(meta.get("goal_desc")) and meta.get("goal_approx_m") is not None and meta.get("goal_bearing_deg") is not None
    chk("goal_in_meta", goal_ok, {k: meta.get(k) for k in ("goal_desc", "goal_approx_m", "goal_bearing_deg")}, "goal_desc/approx_m/bearing_deg")
    chk("air_60_120s", 60.0 <= air_s <= 120.0, round(air_s, 1), "60–120 s (takeoff/land ±3 s trimmed)")
    chk("height_band_70pct", in_band >= 0.70, round(in_band, 3), "≥0.70 of air time in 0.8–1.5 m")
    goal_dz = meta.get("goal_dz_m")
    dz_ok = abs(net_dz) <= 0.3 or (goal_dz is not None and abs(float(goal_dz)) > 1.0 and net_dz * float(goal_dz) > 0)
    chk("no_net_climb_descent", dz_ok, round(net_dz, 2), "|net dz| ≤ 0.3 m unless goal >1 m above/below")
    chk("near_miss_ge3", len(near) >= 3, len(near), "≥3 near_miss events")
    chk("no_backward", back <= 0.02, round(back, 3), "≤2% air samples with b ≤ −10")
    chk("no_strafe", strafe <= 0.02, round(strafe, 3), "≤2% air samples with |a| ≥ 10")
    chk("no_fwd_plus_yaw", combined <= 0.05, round(combined, 3), "≤5% samples forward+yaw together")
    chk("battery_ge20", (min(bats) if bats else 0) >= 20, min(bats) if bats else None, "min bat ≥20%")
    chk("rssi", (statistics.median(rssi) if rssi else -99) >= -70, statistics.median(rssi) if rssi else None, "median RSSI ≥ −70 dBm")
    chk("frames_clean", bad_frac <= 0.05 and len(frames) > 0, {"n": len(frames), "bad_frac": round(bad_frac, 3)}, "≤5% black/flat frames")
    chk("control_mode_consistent", len(modes) == 1 and str(meta.get("control")) in modes, sorted(modes), "telemetry control_mode == meta.control")
    ok = all(c["ok"] for c in checks.values())
    out = {
        "run": run_dir.name,
        "grade": "后训练可用" if ok else "域差可用(待查)",
        "pass": ok,
        "failed": [k for k, c in checks.items() if not c["ok"]],
        "near_miss_n": len(near),
        "collision_n": len(hits),
        "checks": checks,
    }
    (run_dir / "QC.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def batch_verdict(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    good = [r for r in results if r["pass"]]
    with_hit = [r for r in good if r["collision_n"] >= 1]
    return {
        "n_runs": len(results),
        "n_pass": len(good),
        "n_pass_with_collision": len(with_hit),
        "batch_ok": len(good) >= 6 and len(with_hit) >= 2,
        "need": "≥6 passing runs, ≥2 of them with a light collision / clear near-collision event",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    res = []
    for r in args.runs:
        q = qc_run(Path(r))
        res.append(q)
        print(f"{q['run']}: {q['grade']} near={q['near_miss_n']} hit={q['collision_n']} failed={q['failed']}")
    print(json.dumps(batch_verdict(res), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
