#!/usr/bin/env python3
"""Tello stick → real motion calibration (fills MAX_V_MPS / MAX_YAW_RAD_S with measured values).

Forward (one trial per flight, tape-measured takeoff mark → landing spot):

  PYTHONPATH=. python3 experiments/aerial/scripts/tello_rc_calib.py fwd --stick 16 --hold-s 3 --runway-m 3
  PYTHONPATH=. python3 experiments/aerial/scripts/tello_rc_calib.py fwd --stick 0 --hold-s 3   # drift baseline

vgx is integer dm/s and not body-forward on this firmware (stick 22 read −0.2 m/s on
2026-10-12), so the tape distance is the ground truth for forward speed.

Yaw (all sticks in one flight, angle from telemetry yaw, CW then CCW back):

  PYTHONPATH=. python3 experiments/aerial/scripts/tello_rc_calib.py yaw --sticks 26,50 --reps 3 --hold-s 2

Fill a tape value later:  tello_rc_calib.py tape RUN_DIR 1.23 [--lateral-m 0.1]
Analyse:                  tello_calib_report.py rc
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.aerial.scripts import tello_consequence_collect as gs
from experiments.aerial.scripts import tello_flight_util as fu

PRE_HOVER_S = 2.0
POST_HOVER_S = 2.0


def _fwd(args: argparse.Namespace) -> int:
    stick = int(args.stick)
    if not 0 <= stick <= 60:
        print("--stick must be 0..60 (indoor)", file=sys.stderr)
        return 2
    run_dir: Optional[Path] = None
    events: List[Dict[str, Any]] = []
    calib: Dict[str, Any] = {
        "axis": "fwd",
        "stick": stick,
        "hold_s": float(args.hold_s),
        "runway_m": float(args.runway_m),
        "tape_m": None,
        "lateral_m": None,
        "stopped_early": None,
    }
    try:
        gs._preflight()
        run_dir, t_rec0 = fu.start_record(
            {"intended_use": "tello_rc_calib", "control": "auto", "rc_calib": calib}, mode="auto"
        )
        print("takeoff", gs.cmd("takeoff"), flush=True)
        h0 = fu.settle()
        print(f"settled {h0:.2f} m", flush=True)
        fu.hold(0, 0, 0, 0, PRE_HOVER_S, alt_trim=True)
        dist = {"m": 0.0, "t": None}

        def stop(st: dict, t: float) -> Optional[str]:
            # vgx sign/frame is not body-forward on this firmware; use horizontal speed magnitude.
            tel = gs.last_telemetry(run_dir).get("tello") or {}
            v = 0.1 * math.hypot(float(tel.get("vgx") or 0.0), float(tel.get("vgy") or 0.0))
            if dist["t"] is not None:
                dist["m"] += v * (t - dist["t"])
            dist["t"] = t
            if dist["m"] >= 0.7 * float(args.runway_m):
                return f"runway: vgx-integrated {dist['m']:.2f} m"
            hit = fu.imu_hit(run_dir)
            return "imu_collision" if hit == "collision" else None

        t_on = time.time() - t_rec0
        _, why = fu.hold(0, stick, 0, 0, float(args.hold_s), stop=stop)
        t_off = time.time() - t_rec0
        events.append({"t_s": round(t_on, 3), "event": "calib_on", "axis": "fwd", "stick": stick})
        events.append({"t_s": round(t_off, 3), "event": "calib_off", "axis": "fwd", "stick": stick, "stop": why})
        calib["stopped_early"] = why
        calib["vgx_integrated_m"] = round(dist["m"], 3)
        fu.hold(0, 0, 0, 0, POST_HOVER_S)
        gs.land_if_airborne()
        time.sleep(2.0)
    except Exception as exc:
        print("ABORT", exc, file=sys.stderr)
        events.append({"t_s": None, "event": "abort", "note": str(exc)})
        try:
            gs.land_if_airborne()
        except Exception:
            pass
    finally:
        fu.finish(run_dir, events, {"rc_calib": calib})
    if run_dir is None:
        return 1
    print("量：起飞标记 → 落地点（机头方向）的卷尺距离；侧向偏移另量。")
    calib["tape_m"] = fu.ask_float("前进距离 m（回车跳过，之后用 tape 子命令补）: ")
    calib["lateral_m"] = fu.ask_float("侧向偏移 m（可空）: ")
    gs.patch_meta(run_dir, {"rc_calib": calib})
    print("rc_calib", calib, "dir", run_dir)
    return 0


def _yaw(args: argparse.Namespace) -> int:
    sticks = [int(s) for s in str(args.sticks).split(",") if s.strip()]
    if not sticks or any(not 5 <= s <= 80 for s in sticks):
        print("--sticks must be in 5..80", file=sys.stderr)
        return 2
    run_dir: Optional[Path] = None
    events: List[Dict[str, Any]] = []
    calib: Dict[str, Any] = {"axis": "yaw", "sticks": sticks, "reps": int(args.reps), "hold_s": float(args.hold_s)}
    try:
        gs._preflight()
        run_dir, t_rec0 = fu.start_record(
            {"intended_use": "tello_rc_calib", "control": "auto", "rc_calib": calib}, mode="auto"
        )
        print("takeoff", gs.cmd("takeoff"), flush=True)
        print(f"settled {fu.settle():.2f} m", flush=True)
        fu.hold(0, 0, 0, 0, PRE_HOVER_S, alt_trim=True)
        for s in sticks:
            for rep in range(int(args.reps)):
                for sign in (1, -1):
                    d = sign * s
                    t_on = time.time() - t_rec0
                    fu.hold(0, 0, 0, d, float(args.hold_s))
                    t_off = time.time() - t_rec0
                    events.append({"t_s": round(t_on, 3), "event": "calib_on", "axis": "yaw", "stick": d, "rep": rep})
                    events.append({"t_s": round(t_off, 3), "event": "calib_off", "axis": "yaw", "stick": d, "rep": rep})
                    fu.hold(0, 0, 0, 0, 1.5, alt_trim=True)
                    print(f"yaw stick {d:+d} rep {rep} done", flush=True)
        gs.land_if_airborne()
        time.sleep(2.0)
    except Exception as exc:
        print("ABORT", exc, file=sys.stderr)
        events.append({"t_s": None, "event": "abort", "note": str(exc)})
        try:
            gs.land_if_airborne()
        except Exception:
            pass
    finally:
        fu.finish(run_dir, events, {"rc_calib": calib})
    print("dir", run_dir)
    return 0 if run_dir is not None else 1


def _tape(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    meta = fu.read_meta(run_dir)
    calib = dict(meta.get("rc_calib") or {})
    if calib.get("axis") != "fwd":
        print("not a fwd rc_calib run", file=sys.stderr)
        return 2
    calib["tape_m"] = float(args.tape_m)
    if args.lateral_m is not None:
        calib["lateral_m"] = float(args.lateral_m)
    gs.patch_meta(run_dir, {"rc_calib": calib})
    print(calib)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    sub = ap.add_subparsers(dest="axis", required=True)
    f = sub.add_parser("fwd")
    f.add_argument("--stick", type=int, required=True)
    f.add_argument("--hold-s", type=float, default=3.0)
    f.add_argument("--runway-m", type=float, default=3.0, help="clear floor ahead; stops at 70%% by |vg|")
    y = sub.add_parser("yaw")
    y.add_argument("--sticks", default="26,50")
    y.add_argument("--reps", type=int, default=3)
    y.add_argument("--hold-s", type=float, default=2.0)
    t = sub.add_parser("tape")
    t.add_argument("run_dir")
    t.add_argument("tape_m", type=float)
    t.add_argument("--lateral-m", type=float, default=None)
    args = ap.parse_args()
    fu.set_base(args.base)
    return {"fwd": _fwd, "yaw": _yaw, "tape": _tape}[args.axis](args)


if __name__ == "__main__":
    raise SystemExit(main())
