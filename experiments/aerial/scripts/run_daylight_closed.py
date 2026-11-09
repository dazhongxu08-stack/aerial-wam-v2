#!/usr/bin/env python3
"""Re-run every office closed loop under the saved daylight scene.

Keeps exposure at 7. Uses the yaw_o depth head and the angle_f consequence head.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")

import apply_office_daylight as day
import collect_office_train as col
import sim_verify_complex as cx
import sim_verify_gap as sim
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

DEPTH = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt"
CONS = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/indoor_office_mix_angle_f_20261102/c_s218/consequence_best.pt"
OUT = Path("/data/simworld/run/daylight_closed.json")


def line(name: str, verdict: dict, row: dict) -> None:
    gt = row.get("gt", [None, None, None])
    da = row.get("da", [None, None, None])
    print(
        f"{name} ok={verdict['ok']} {verdict['why']} end={verdict.get('end')} "
        f"steps={verdict.get('steps', len([]))} "
        f"end_xy={row.get('x')},{row.get('y')} gt={gt[0]} da={da[0]} "
        f"surv={row.get('fwd_surv')} n_pass={row.get('n_pass')}",
        flush=True,
    )


def main() -> int:
    cfg = day.load_cfg()
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    day.apply(ucv, cfg)
    cams = str(col.req(ucv, "vget /cameras")).split()
    if "FusionCameraSensor" not in cams:
        col.req(ucv, "vset /cameras/spawn")
        cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.ensure_dog(ucv)
    print("loading", flush=True)
    depth_model = DepthMinPredictor.from_checkpoint(DEPTH, device="cuda")
    model = sim.load_checkpoint(CONS, "cuda")
    models = [model]
    report = []
    for route in sim.ROUTES:
        rows = sim.rollout(ucv, cam, depth_model, models, route, False)
        verdict = sim.judge(route, rows)
        verdict["name"] = route["name"]
        verdict["suite"] = "gap"
        report.append({"verdict": verdict, "rows": rows})
        line(route["name"], verdict, rows[-1])
    for route in cx.ROUTES:
        rows = cx.rollout(ucv, cam, depth_model, models, route)
        verdict = cx.judge(route, rows)
        verdict["name"] = route["name"]
        verdict["suite"] = "complex"
        verdict["expect"] = route["expect"]
        report.append({"verdict": verdict, "rows": rows})
        line(route["name"], verdict, rows[-1])
    passed = sum(1 for item in report if item["verdict"]["ok"])
    print(f"PASS {passed}/{len(report)}", flush=True)
    OUT.write_text(json.dumps(report))
    print("WROTE", OUT, flush=True)
    return 0 if passed == len(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
