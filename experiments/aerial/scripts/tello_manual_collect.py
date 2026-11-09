#!/usr/bin/env python3
"""H12 hand-flown consequence post-train collect (tello-consequence-collect.md §3/§8 manual).

  PYTHONPATH=. python3 experiments/aerial/scripts/tello_manual_collect.py \\
    --goal-desc '房间对面的门框' --goal-approx-m 4 --goal-bearing-deg 0

Sticks come from H12 (control_mode=manual), so the recorded ``rc`` is what the Tello
actually got. While flying, type at the Mac terminal:
  n<Enter>  near miss (obstacle filled the view, pilot stopped/turned)
  c<Enter>  light collision
IMU spikes are added automatically (source=imu). Recording ends on landing;
QC.json (§3) is written into the run dir.
"""
from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.aerial.scripts import tello_consequence_collect as gs
from experiments.aerial.scripts import tello_flight_util as fu
from experiments.aerial.scripts.tello_posttrain_qc import qc_run

IMU_DEBOUNCE_S = 1.5


def _key() -> str:
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().lower()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal-desc", required=True)
    ap.add_argument("--goal-approx-m", type=float, required=True)
    ap.add_argument("--goal-bearing-deg", type=float, required=True, help="relative to nose at takeoff, CCW+")
    ap.add_argument("--goal-dz-m", type=float, default=0.0, help="goal height minus takeoff hover height")
    ap.add_argument("--note", default="")
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    args = ap.parse_args()
    fu.set_base(args.base)

    run_dir: Optional[Path] = None
    events: List[Dict[str, Any]] = []
    try:
        gs._preflight()
        run_dir, t_rec0 = fu.start_record(
            {
                "goal_desc": args.goal_desc,
                "goal_approx_m": float(args.goal_approx_m),
                "goal_bearing_deg": float(args.goal_bearing_deg),
                "goal_dz_m": float(args.goal_dz_m),
                "control": "manual",
                "operator": "H12",
                "intended_use": "consequence_posttrain",
                "near_miss_n": 0,
                "collision_n": 0,
                "note": args.note,
            },
            mode="manual",
        )
        print("H12 起飞。n+回车=近障  c+回车=轻碰。落地自动结束。", flush=True)
        t_air0: Optional[float] = None
        last_imu = 0.0
        warned = set()
        out_band_since: Optional[float] = None
        while True:
            t_loop = time.time()
            st = gs.status()
            air = bool(st.get("airborne"))
            if t_air0 is None and air:
                t_air0 = t_loop
                print("airborne", flush=True)
            if t_air0 is not None and not air:
                print("landed", flush=True)
                break
            if t_air0 is None and t_loop - t_rec0 > 120.0:
                raise RuntimeError("no takeoff within 120 s")
            t_rec = round(t_loop - t_rec0, 3)
            k = _key()
            if k in ("n", "c"):
                ev = "near_miss" if k == "n" else "collision"
                events.append({"t_s": t_rec, "event": ev, "source": "operator"})
                print(f"{ev} @ {t_rec}s", flush=True)
            if t_air0 is not None:
                air_s = t_loop - t_air0
                hit = fu.imu_hit(run_dir)
                if hit and t_loop - last_imu >= IMU_DEBOUNCE_S:
                    # An IMU spike is not a §3 near miss (obstacle filling the view); only the operator marks those.
                    ev = "collision" if hit == "collision" else "imu_spike"
                    events.append({"t_s": t_rec, "event": ev, "source": "imu"})
                    last_imu = t_loop
                    print(f"IMU {ev} @ {t_rec}s", flush=True)
                h = fu.height_m(st)
                if h is not None and air_s > 5.0 and not (0.8 <= h <= 1.5):
                    out_band_since = out_band_since or t_loop
                    if t_loop - out_band_since > 2.0 and "band" not in warned:
                        print(f"WARNING 高度 {h:.2f} m 不在 0.8–1.5", flush=True)
                        warned.add("band")
                else:
                    out_band_since = None
                    warned.discard("band")
                for mark, msg in ((60.0, "已满 60 s，可以降落"), (110.0, "接近 120 s，请降落")):
                    if air_s >= mark and mark not in warned:
                        print(msg, flush=True)
                        warned.add(mark)
                bat = int(st.get("battery") or 100)
                if bat < 25 and "bat" not in warned:
                    print(f"WARNING 电量 {bat}%，请降落", flush=True)
                    warned.add("bat")
            fu._sleep_rest(t_loop)
        time.sleep(1.5)
    except KeyboardInterrupt:
        print("stopped by Ctrl-C", flush=True)
    except Exception as exc:
        print("ABORT", exc, file=sys.stderr)
        events.append({"t_s": None, "event": "abort", "note": str(exc)})
    finally:
        near = sum(1 for e in events if e["event"] == "near_miss")
        hits = sum(1 for e in events if e["event"] == "collision")
        fu.finish(run_dir, events, {"near_miss_n": near, "collision_n": hits})
    if run_dir is None:
        return 1
    q = qc_run(run_dir)
    print(f"QC {q['grade']} near={q['near_miss_n']} hit={q['collision_n']} failed={q['failed']}")
    print("dir", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
