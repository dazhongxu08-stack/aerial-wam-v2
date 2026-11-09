#!/usr/bin/env python3
"""Closed-loop routes with the frozen depth head plus the thin-strip correction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")

import collect_office_train as col
import fit_near_strip as fit
import sim_verify_complex as cx
import sim_verify_gap as gap

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
CFG = json.loads((ART / "near_strip_20261102/near_strip.json").read_text())["best"]
DEPTH = ART / "da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt"
CONS = ART / "indoor_office_mix_angle_f_20261102/c_s218/consequence_best.pt"
OUT = Path("/data/simworld/run/near_strip_closed.json")

_orig = gap.metric_depth


def metric_depth(depth_model, image, seen, gt_warp):
    hid, scale, raw_c, _da = _orig(depth_model, image, seen, gt_warp)
    hid = np.asarray(hid, np.float32)
    raw = col.cones3(hid)
    hid = fit.correct_thin(hid, CFG["alpha"], CFG["lo"], CFG["hi"], int(CFG["max_run"]))
    return hid, scale, raw, col.cones3(hid)


gap.metric_depth = metric_depth


def place(ucv) -> None:
    props = json.loads(Path("/data/simworld/run/aisle_props.json").read_text())
    for item in props:
        name, x, y = item["name"], float(item["x"]), float(item["y"])
        col.req(ucv, f"vset /object/{name}/location {x} {y} -48")
        print("placed", name, col.req(ucv, f"vget /object/{name}/location"), flush=True)


def main() -> None:
    print("strip", CFG, flush=True)
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    col.req(ucv, "vrun r.Fog 0")
    col.req(ucv, "vrun r.VolumetricFog 0")
    col.req(ucv, "vset /object/ExponentialHeightFog_1/hide")
    col.req(ucv, "vrun r.ExposureOffset 8")
    cams = str(col.req(ucv, "vget /cameras")).split()
    if "FusionCameraSensor" not in cams:
        col.req(ucv, "vset /cameras/spawn")
        cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.ensure_dog(ucv)
    place(ucv)
    print("loading depth", flush=True)
    depth_model = gap.DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    print("loading consequence", flush=True)
    models = [gap.load_checkpoint(str(CONS), "cuda")]
    report = []
    for route in list(gap.ROUTES) + list(cx.ROUTES):
        if "kind" in route:
            rows = gap.rollout(ucv, cam, depth_model, models, route, False)
            verdict = gap.judge(route, rows)
        else:
            rows = cx.rollout(ucv, cam, depth_model, models, route)
            verdict = cx.judge(route, rows)
        print(route["name"], json.dumps(verdict), flush=True)
        report.append({"name": route["name"], "expect": route.get("expect", route.get("kind")), "verdict": verdict, "rows": rows})
        OUT.write_text(json.dumps(report, indent=1))
    n_ok = sum(1 for item in report if item["verdict"].get("ok"))
    print(f"DONE {n_ok}/{len(report)}", flush=True)


if __name__ == "__main__":
    main()
