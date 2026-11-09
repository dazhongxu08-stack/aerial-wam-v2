#!/usr/bin/env python3
"""One tall barrel in the forward cone, with the right side open.

The four office gates do not put a single obstacle on the nose. This places
SM_BarelPlastic_A_21 on the lane at x=40, shifted 25 cm to image-left, scaled
tall. From x=-170 the forward cone is about a meter and the right side stays
open, so the walker has to turn right and pass the barrel. A forward step
with any cone under 0.55 m still fails. The barrel is restored afterwards.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/aerial-wam-v2/experiments/aerial/scripts")

import run_office_cons_loop as loop

BARREL = "SM_BarelPlastic_A_21"
# Lane center is y=-200. Negative Y is image-left at yaw 0, so the opening
# is to the right and the away turn is positive yaw.
PLACE = (40.0, -225.0, -48.0)
START = (-170.0, -200.0, 0.0)
GOAL = (300.0, -200.0)
STEPS = 48


def place_front_barrel(ucv) -> None:
    loop.col.req(ucv, f"vset /object/{BARREL}/object_mobility true")
    loop.col.req(ucv, f"vset /object/{BARREL}/physics false")
    loop.col.req(ucv, f"vset /object/{BARREL}/scale 1 1 3")
    loop.col.req(ucv, f"vset /object/{BARREL}/rotation 0 0 0")
    loop.col.req(ucv, f"vset /object/{BARREL}/location {PLACE[0]} {PLACE[1]} {PLACE[2]}")
    print("front_barrel", loop.col.req(ucv, f"vget /object/{BARREL}/bounds"), flush=True)


def judge_front(rows) -> bool:
    if not rows:
        print("GATE front_barrel FAIL no_frames", flush=True)
        return False
    drove_into = any(r["dx"] > 0.05 and min(r["gt"]) < loop.HIT_M for r in rows)
    yaw0 = float(rows[0]["yaw"])
    turned = any(float(r["yaw"]) - yaw0 >= 8.0 for r in rows)
    # The chosen spin is logged before that step is integrated.
    spin_at = [
        i for i, r in enumerate(rows)
        if "spin" in str(r["family"]) and float(r["dyaw"]) > 0.05
    ]
    turned_right = any(
        float(rows[j]["yaw"]) - yaw0 >= 8.0
        for i in spin_at
        for j in range(i, len(rows))
    )
    passed = any(float(r["x"]) > 80.0 and float(r["yaw"]) - yaw0 >= 8.0 for r in rows)
    ok = (not drove_into) and turned_right and passed
    last = rows[-1]
    min_fwd = min(r["gt"][0] for r in rows)
    print(
        f"GATE front_barrel {'PASS' if ok else 'FAIL'} "
        f"why={'passed' if ok else 'missed'} steps={len(rows)} end={last['family']} "
        f"x {rows[0]['x']:.0f}->{last['x']:.0f} yaw {yaw0:.0f}->{last['yaw']:.0f} "
        f"turned={turned} min_fwd={min_fwd:.2f} drove_into={drove_into}",
        flush=True,
    )
    return ok


def main() -> int:
    import numpy as np

    import sim_verify_gap as sim
    from experiments.aerial.rl.consequence_model import load_checkpoint
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor

    art = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts"
    loop.DEPTH = os.environ.get(
        "OFFICE_DEPTH",
        f"{art}/da3_office_align_20261104/depth_office_cone_align.pt",
    )
    loop.CONS = os.environ.get(
        "OFFICE_CONS",
        f"{art}/indoor_office_align_20261104/c_s218/consequence_open.pt",
    )

    def score_next(models, depth56, state, goal, cands, prev, da_f=0.0):
        curve, prog = sim._curves(models, depth56, state, goal, cands, prev)
        return curve[:, 0], prog

    def pick_learned(da, low, names, cands, surv, allowed, bearing):
        surv = np.asarray(surv, np.float64)
        names = list(names)
        cands = np.asarray(cands, np.float64)
        ok = np.flatnonzero(surv >= loop.SURV_MIN)
        if ok.size:
            moving = [int(i) for i in ok if "spin" not in names[int(i)]]
            if moving:
                dx = cands[moving, 0, 0]
                chosen = int(moving[int(np.argmax(dx))])
            else:
                chosen = int(ok[int(np.argmax(surv[ok]))])
            return loop.first_cmd(cands[chosen]), str(names[chosen]), float(surv[chosen])
        return np.zeros(4, np.float64), "stop", float(np.max(surv))

    loop.sim.score = score_next
    loop.pick = pick_learned
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(loop.col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    loop.boxes.save_restore(ucv)
    depth_model = DepthMinPredictor.from_checkpoint(loop.DEPTH, device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    model = load_checkpoint(loop.CONS, "cuda")
    model.eval()
    print(f"DEPTH {loop.DEPTH}\nCONS {loop.CONS}", flush=True)
    loop.restore_boxes(ucv)
    place_front_barrel(ucv)
    try:
        rows = loop.walk(
            ucv, cam, depth_model, model, "front_barrel", "front",
            START[0], START[1], START[2], GOAL, STEPS,
        )
        ok = judge_front(rows)
    finally:
        loop.restore_boxes(ucv)
        print("front_restored", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
