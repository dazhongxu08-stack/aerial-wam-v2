#!/usr/bin/env python3
"""Step-1 office closed loop: lowbox DA3 + cone rule, no consequence.

Four gates: empty aisle, side crate, facing wall, gap. Forward cones use
``office_bands`` (above horizon). Short crates use ``low_band``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_office_train as col
import collect_office_lowbox as boxes
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation

from office_bands import cones3_forward, horizon_row_56, short_clearance

DEPTH = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_lowbox_20261103/depth_office_cone_lowbox.pt"
NEAR_M = 1.25
LOW_NEAR_M = 1.40
SLOW_M = 2.05
FAST_DX = 0.35
SLOW_DX = 0.25
ARC_DX = 0.30
ARC_DYAW = 0.3141592653589793
HIT_M = 0.55
OUT = Path("/data/simworld/run/office_cone_loop_20261103")


def integrate(x, y, yaw, dx, dyaw):
    yaw2 = yaw + math.degrees(float(dyaw))
    rad = math.radians(yaw2)
    return x + float(dx) * 100.0 * math.cos(rad), y + float(dx) * 100.0 * math.sin(rad), yaw2


def sense(ucv, cam, depth_model, x, y, yaw):
    col.place(ucv, x, y, yaw)
    col.look(ucv, cam, x, y, yaw)
    packed = col.grab(ucv, cam)
    if packed is None:
        return None
    rgb, depth = packed
    image, seen, warped = col.warp_pair(rgb, depth)
    gt56 = col.pool_min(col.hide(warped, seen))
    gt_old = col.cones3(gt56)
    gt = cones3_forward(gt56)
    depth_model.reset()
    pred = depth_model.predict_depth(
        Observation(rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False)
    )
    pred = np.asarray(pred, np.float32)
    if pred.shape[:2] != warped.shape[:2]:
        import cv2

        pred = cv2.resize(pred, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
    hid = col.pool_min(col.hide(pred, seen))
    da_old = col.cones3(hid)
    da = cones3_forward(hid)
    # Short crates live in horizon→hide. DA does not paint them yet (3 m vs
    # GT 0.88 m); the closed loop reads GT on that strip so the row choice
    # can be tested. Forward still uses DA.
    low_gt = short_clearance(warped, hide=col.HIDE_ROW)
    low_da = short_clearance(pred, hide=col.HIDE_ROW)
    return {
        "gt": [float(v) for v in gt],
        "gt_old": [float(v) for v in gt_old],
        "da": [float(v) for v in da],
        "da_old": [float(v) for v in da_old],
        "low": low_gt,
        "low_da": low_da,
        "mean": float(rgb.mean()),
    }


def choose(da, low):
    fwd, left, right = (float(v) for v in da)
    left_l, center_l, right_l = low if low is not None else (None, None, None)
    left_close = left_l is not None and float(left_l) < LOW_NEAR_M
    right_close = right_l is not None and float(right_l) < LOW_NEAR_M
    center_close = center_l is not None and float(center_l) < LOW_NEAR_M
    blocked = fwd < NEAR_M or center_close
    left_open = (not left_close) and left >= NEAR_M
    right_open = (not right_close) and right >= NEAR_M
    if blocked or left_close or right_close:
        if left_open and not right_open:
            return ARC_DX, -ARC_DYAW, "arc_left"
        if right_open and not left_open:
            return ARC_DX, ARC_DYAW, "arc_right"
        if left_open and right_open:
            if left >= right:
                return ARC_DX, -ARC_DYAW, "arc_left"
            return ARC_DX, ARC_DYAW, "arc_right"
        return 0.0, 0.0, "stop"
    dx = SLOW_DX if fwd < SLOW_M else FAST_DX
    return dx, 0.0, "fwd"


def walk(ucv, cam, depth_model, name, kind, x, y, yaw, goal_xy, steps):
    print(
        f"WALK {name} kind={kind} start {x:.0f},{y:.0f} yaw {yaw:.0f} goal {goal_xy} "
        f"horizon56={horizon_row_56()}",
        flush=True,
    )
    rows = []
    gx, gy = goal_xy
    for step in range(steps):
        seen = sense(ucv, cam, depth_model, x, y, yaw)
        if seen is None:
            print(f"  s={step} grab_fail", flush=True)
            break
        dx, dyaw, family = choose(seen["da"], seen["low"])
        gt, da = seen["gt"], seen["da"]
        dist = math.hypot(gx - x, gy - y)
        print(
            f"  s={step} x={x:.0f} y={y:.0f} yaw={yaw:.1f} dist={dist:.0f} "
            f"gt={gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f} "
            f"da={da[0]:.2f},{da[1]:.2f},{da[2]:.2f} "
            f"old_da={seen['da_old'][0]:.2f},{seen['da_old'][1]:.2f},{seen['da_old'][2]:.2f} "
            f"low={seen['low']} low_da={seen['low_da']} {family} dx={dx:.2f} dyaw={dyaw:.3f} mean={seen['mean']:.1f}",
            flush=True,
        )
        rows.append({
            "x": x, "y": y, "yaw": yaw, "gt": gt, "da": da,
            "da_old": seen["da_old"], "low": list(seen["low"]), "low_da": list(seen["low_da"]),
            "family": family, "dx": dx, "dyaw": dyaw, "dist": dist,
        })
        if family == "stop" or min(gt) < HIT_M or dist < 40.0:
            break
        x, y, yaw = integrate(x, y, yaw, dx, dyaw)
    return rows


def judge(name, kind, rows):
    if not rows:
        print(f"GATE {name} FAIL no_frames", flush=True)
        return False
    last = rows[-1]
    min_fwd = min(r["gt"][0] for r in rows)
    drove_into = any(r["dx"] > 0.05 and min(r["gt"]) < HIT_M for r in rows)
    moved = any(r["dx"] > 0.05 for r in rows)
    yaw_delta = abs(rows[-1]["yaw"] - rows[0]["yaw"])
    start_dist, end_dist = rows[0]["dist"], last["dist"]
    if kind == "aisle":
        ok = moved and (not drove_into) and end_dist < start_dist - 80 and last["family"] != "stop"
        # empty aisle may stop at a real far wall; still require progress and no floor-stop at start
        if last["family"] == "stop" and last["gt"][0] < 1.4 and end_dist < start_dist - 150:
            ok = not drove_into
        why = "progress_no_hit" if ok else "stuck_or_hit"
    elif kind == "side":
        turned = yaw_delta >= 8.0 or any(r["family"].startswith("arc") for r in rows)
        ok = (not drove_into) and (turned or last["family"] == "stop")
        why = "yielded" if ok else "drove_into_or_ignored"
    elif kind == "wall":
        ok = (not drove_into) and last["family"] == "stop" and min_fwd > HIT_M
        why = "stopped_clear" if ok else "hit_or_ran"
    else:  # through
        progressed = last["x"] > rows[0]["x"] + 120
        ok = (not drove_into) and progressed and last["family"] != "stop"
        if last["family"] == "stop" and progressed and last["gt"][0] > 2.0:
            ok = True
        why = "through" if ok else "stopped_or_hit"
    flag = "PASS" if ok else "FAIL"
    print(
        f"GATE {name} {flag} why={why} steps={len(rows)} end={last['family']} "
        f"min_fwd={min_fwd:.2f} dist {start_dist:.0f}->{end_dist:.0f} "
        f"yaw {rows[0]['yaw']:.1f}->{last['yaw']:.1f} drove_into={drove_into}",
        flush=True,
    )
    return ok


def apply_daylight(ucv):
    sys.path.insert(0, "/data/simworld/run")
    import apply_office_daylight as day

    day.apply(ucv, day.load_cfg())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.ensure_dog(ucv)
    print("DAYLIGHT", flush=True)
    try:
        apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    boxes.save_restore(ucv)
    depth_model = DepthMinPredictor.from_checkpoint(DEPTH, device="cuda")
    print(f"DEPTH {DEPTH} horizon56={horizon_row_56()}", flush=True)
    summary = []

    def run_job(name, kind, x, y, yaw, goal, steps):
        rows = walk(ucv, cam, depth_model, name, kind, x, y, yaw, goal, steps)
        ok = judge(name, kind, rows)
        summary.append({"name": name, "kind": kind, "ok": ok, "n": len(rows), "end": rows[-1] if rows else None})

    print("LOWBOX_RESTORE empty aisle / wall / gap", flush=True)
    rows = json.loads((boxes.OUT / "lowbox_restore.json").read_text())
    for row in rows:
        name = row["name"]
        col.req(ucv, f"vset /object/{name}/object_mobility true")
        col.req(ucv, f"vset /object/{name}/physics false")
        col.req(ucv, f"vset /object/{name}/scale {row['scale']}")
        col.req(ucv, f"vset /object/{name}/rotation {row['rotation']}")
        col.req(ucv, f"vset /object/{name}/location {row['location']}")
    print("RESTORED", len(rows), flush=True)
    run_job("center_lane", "aisle", -480.0, -200.0, 0.0, (200.0, -200.0), 36)
    run_job("block_west", "wall", -520.0, -250.0, 0.0, (80.0, -250.0), 24)
    run_job("gap", "through", -160.0, -200.0, 0.0, (200.0, -200.0), 36)
    print("LOWBOX_PLACE side crate", flush=True)
    boxes.place_boxes(ucv)
    run_job("right_box", "side", -480.0, -170.0, 0.0, (220.0, -170.0), 32)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    n_ok = sum(1 for s in summary if s["ok"])
    print(f"DONE {n_ok}/{len(summary)} {OUT}", flush=True)
    if n_ok < len(summary):
        sys.exit(2)


if __name__ == "__main__":
    main()
