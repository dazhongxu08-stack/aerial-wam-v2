#!/usr/bin/env python3
"""Tello DA3 depth probe at deployment height: hover ~1 m facing a target at a tape distance.

Auto (script takes off, holds a=b=d=0, grabs stills, lands):

  PYTHONPATH=. python3 experiments/aerial/scripts/tello_depth_probe.py \\
    --target-kind wall --target-desc '白墙' --tape-m 1.5

Manual (H12 hand-fly; stills only while sticks are zero and height in band):

  PYTHONPATH=. python3 experiments/aerial/scripts/tello_depth_probe.py --mode manual \\
    --target-kind chair_leg --target-desc '椅子腿' --tape-m 1.0

``--tape-m`` is camera (nose) → nearest target surface with the drone on the floor at
takeoff. After landing the script asks for the same distance again to expose drift.
Analyse: tello_calib_report.py depth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.aerial.scripts import tello_consequence_collect as gs
from experiments.aerial.scripts import tello_flight_util as fu

KINDS = ("wall", "desk_edge", "chair_back", "door_frame", "chair_leg", "pole", "glass", "other")
BAND = (0.85, 1.30)
STILL_EVERY_S = 0.5


def _grab(run_dir: Path, idx: int, st: dict, t: float, log) -> None:
    out = run_dir / "calib_stills"
    out.mkdir(exist_ok=True)
    (out / f"{idx:02d}.jpg").write_bytes(fu.snapshot_jpeg())
    log.write(json.dumps({"file": f"{idx:02d}.jpg", "t": round(t, 3), "height_cm": st.get("height_cm"),
                          "yaw": st.get("yaw"), "frame_seq": st.get("frame_seq")}) + "\n")
    log.flush()


def _auto(args: argparse.Namespace, run_dir: Path) -> int:
    print("takeoff", gs.cmd("takeoff"), flush=True)
    print(f"settled {fu.settle(band=BAND):.2f} m", flush=True)
    n = 0
    t0 = time.time()
    last = 0.0
    with (run_dir / "calib_stills.jsonl").open("w", encoding="utf-8") as log:
        while n < int(args.stills) and time.time() - t0 < float(args.hold_s) + 10.0:
            t_loop = time.time()
            st = gs.status()
            gs._gate(st)
            if fu.imu_hit(run_dir) == "collision":
                raise RuntimeError("imu collision during probe hover")
            h = fu.height_m(st)
            gs.rc(0, 0, fu.altitude_rc(h), 0)
            if t_loop - t0 >= 1.0 and t_loop - last >= STILL_EVERY_S and h is not None and BAND[0] <= h <= BAND[1]:
                _grab(run_dir, n, st, t_loop - t0, log)
                n += 1
                last = t_loop
            fu._sleep_rest(t_loop)
    gs.land_if_airborne()
    time.sleep(2.0)
    return n


def _manual(args: argparse.Namespace, run_dir: Path) -> int:
    print("H12 手飞：飞到目标前约 1 m 高悬停、松杆；稳定悬停时自动拍。落地后结束。", flush=True)
    n = 0
    last = 0.0
    zero_since: Optional[float] = None
    was_air = False
    t0 = time.time()
    with (run_dir / "calib_stills.jsonl").open("w", encoding="utf-8") as log:
        while time.time() - t0 < 300.0:
            t_loop = time.time()
            st = gs.status()
            air = bool(st.get("airborne"))
            if was_air and not air:
                break
            was_air = was_air or air
            h = fu.height_m(st)
            sticks_zero = list(st.get("rc") or [1]) == [0, 0, 0, 0]
            steady = air and sticks_zero and h is not None and BAND[0] <= h <= BAND[1]
            zero_since = (zero_since or t_loop) if steady else None
            if (zero_since is not None and t_loop - zero_since >= 1.0 and t_loop - last >= STILL_EVERY_S
                    and n < int(args.stills)):
                _grab(run_dir, n, st, t_loop - t0, log)
                n += 1
                last = t_loop
                print(f"still {n}/{args.stills} h={h:.2f}", flush=True)
            fu._sleep_rest(t_loop)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("auto", "manual"), default="auto")
    ap.add_argument("--target-kind", choices=KINDS, required=True)
    ap.add_argument("--target-desc", required=True)
    ap.add_argument("--tape-m", type=float, required=True)
    ap.add_argument("--stills", type=int, default=12)
    ap.add_argument("--hold-s", type=float, default=10.0)
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    args = ap.parse_args()
    fu.set_base(args.base)
    probe: Dict[str, Any] = {
        "target_kind": args.target_kind,
        "target_desc": args.target_desc,
        "tape_m": float(args.tape_m),
        "tape_land_m": None,
        "height_band_m": list(BAND),
        "mode": args.mode,
    }
    run_dir: Optional[Path] = None
    events: List[Dict[str, Any]] = []
    n = 0
    try:
        gs._preflight()
        run_dir, _ = fu.start_record(
            {"intended_use": "tello_depth_probe", "control": args.mode, "depth_probe": probe}, mode=args.mode
        )
        n = (_auto if args.mode == "auto" else _manual)(args, run_dir)
    except Exception as exc:
        print("ABORT", exc, file=sys.stderr)
        events.append({"t_s": None, "event": "abort", "note": str(exc)})
        if args.mode == "auto":
            try:
                gs.land_if_airborne()
            except Exception:
                pass
    finally:
        probe["n_stills"] = n
        fu.finish(run_dir, events, {"depth_probe": probe})
    if run_dir is None:
        return 1
    probe["tape_land_m"] = fu.ask_float("落地后机头 → 目标的卷尺距离 m（回车跳过）: ")
    if probe["tape_land_m"] is not None and abs(probe["tape_land_m"] - probe["tape_m"]) > 0.15:
        print(f"WARNING 漂移 {probe['tape_land_m'] - probe['tape_m']:+.2f} m，报告里会标出")
    gs.patch_meta(run_dir, {"depth_probe": probe})
    print("depth_probe", probe, "dir", run_dir)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
