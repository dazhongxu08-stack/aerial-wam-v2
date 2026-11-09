#!/usr/bin/env python3
"""Office closed loop: lowbox DA3 + bands + new consequence. No robot."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_office_train as col
import collect_office_lowbox as boxes
import sim_verify_gap as sim
from experiments.aerial.rl.consequence_model import load_checkpoint, survival
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation
from office_bands import cones3_forward, horizon_row_56, short_clearance

DEPTH = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_far_20261103/depth_office_cone_far.pt"
CONS = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/indoor_office_mix_far_20261103/c_s218/consequence_best.pt"
SURV_MIN = 0.95
NEAR_M = 1.25
# Side crate median is 1.69 m. Gap smear at 1.74 m is rejected by span, not
# by this cutoff.
LOW_NEAR_M = 1.90
HIT_M = 0.55
ARC_DX = 0.30
ARC_DYAW = 0.3141592653589793
# Barrel tops sit just above the camera, so the forward cone still reads 7 m
# while the strip is 1.5 m. Survival stops there. Creep until the strip is
# inside this range; that is where the nose cone actually fills.
CREEP_DX = 0.60
# At x=-339 the center median is 0.61 m and GT forward is still 6.9 m
# (the nose is looking over the barrels). One more step, x=-330, is where
# GT forward falls to 1.0 m and the DA median is 0.55 m.
CREEP_UNTIL_M = 0.55
OUT = Path("/data/simworld/run/office_cons_loop_20261103")


def integrate(x, y, yaw, dx, dyaw):
    # Collect units: dx=1 → 0.35 m/s × 0.5 s = 17.5 cm, not 1 m.
    yaw2 = yaw + math.degrees(float(dyaw))
    rad = math.radians(yaw2)
    step = max(0.0, float(dx)) * 17.5
    return x + step * math.cos(rad), y + step * math.sin(rad), yaw2


def sense(ucv, cam, depth_model, x, y, yaw):
    col.place(ucv, x, y, yaw)
    col.look(ucv, cam, x, y, yaw)
    packed = col.grab(ucv, cam)
    if packed is None:
        return None
    rgb, depth = packed
    image, seen, warped = col.warp_pair(rgb, depth)
    gt56 = col.pool_min(col.hide(warped, seen))
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
    da = cones3_forward(hid)
    low_gt = short_clearance(warped, hide=col.HIDE_ROW)
    low_da = short_clearance(pred, hide=col.HIDE_ROW)
    return {
        "gt": [float(v) for v in gt],
        "da": [float(v) for v in da],
        "depth": hid,
        "low": low_da,
        "low_gt": low_gt,
        "mean": float(rgb.mean()),
    }


@torch.no_grad()
def score(model, depth56, state, goal, cands, prev):
    from experiments.aerial.rl.consequence_model import depth_input, goal_features

    d = torch.as_tensor(depth_input(depth56)[None], dtype=torch.float32, device="cuda")
    g = torch.as_tensor(goal_features(state, goal)[None], dtype=torch.float32, device="cuda")
    pa = torch.as_tensor(np.asarray(prev, np.float32)[None], dtype=torch.float32, device="cuda")
    a = torch.as_tensor(np.asarray(cands, np.float32), dtype=torch.float32, device="cuda")
    n = a.shape[0]
    out = model(d.expand(n, *d.shape[1:]), g.expand(n, -1), pa.expand(n, -1), a)
    s = survival(out["hazard_logit"])[:, -1].detach().cpu().numpy()
    prog = (-out["dgoal"][:, -1]).detach().cpu().numpy()
    return s, prog


def first_cmd(seq) -> np.ndarray:
    a = np.asarray(seq, np.float64)
    if a.ndim == 2:
        a = a[0]
    return np.asarray(a, np.float64).reshape(-1)[:4]


def short_arc(da, low):
    """Yield on the horizon→hide strip. Survival does not see the crate."""
    fwd, left, right = (float(v) for v in da)
    left_l, center_l, right_l = low if low is not None else (None, None, None)
    left_close = left_l is not None and float(left_l) < LOW_NEAR_M
    right_close = right_l is not None and float(right_l) < LOW_NEAR_M
    center_close = center_l is not None and float(center_l) < LOW_NEAR_M
    blocked = fwd < NEAR_M or center_close
    left_open = (not left_close) and left >= NEAR_M
    right_open = (not right_close) and right >= NEAR_M
    if not (blocked or left_close or right_close):
        return None
    # Gap: short objects on both sides, center and nose still open.
    if left_close and right_close and (not center_close) and fwd >= NEAR_M:
        return None
    # Wide face ahead. Yawing treats the barrel lip as a side obstacle.
    if center_close and not left_close and not right_close:
        return None
    if left_open and not right_open:
        return np.array([ARC_DX, 0.0, 0.0, -ARC_DYAW], np.float64), "arc_left"
    if right_open and not left_open:
        return np.array([ARC_DX, 0.0, 0.0, ARC_DYAW], np.float64), "arc_right"
    if left_open and right_open:
        if left >= right:
            return np.array([ARC_DX, 0.0, 0.0, -ARC_DYAW], np.float64), "arc_left"
        return np.array([ARC_DX, 0.0, 0.0, ARC_DYAW], np.float64), "arc_right"
    return np.zeros(4, np.float64), "stop"


def _side_block(low, da) -> bool:
    """Side strip blocks a creep only when that side's forward cone is also shut.

    A crate sits in the strip while the nose cone is still open; short_arc
    owns that case. A 1.2 m aisle-wall base with the left cone still at 1.8 m
    is the wall beside the lane, not a reason to stop.
    """
    if low is None:
        return False
    pairs = ((low[0], float(da[1])), (low[2], float(da[2])))
    return any(
        v is not None and float(v) < LOW_NEAR_M and cone < NEAR_M for v, cone in pairs
    )


def creep_over(low):
    """Center face still farther than the nose. One slow step, not a full stop."""
    if low is None or low[1] is None:
        return None
    center = float(low[1])
    if center <= CREEP_UNTIL_M or center > 1.6:
        return None
    return np.array([CREEP_DX, 0.0, 0.0, 0.0], np.float64), "creep"


def pick(da, low, names, cands, surv, allowed, bearing):
    surv = np.asarray(surv)
    yield_cmd = short_arc(da, low)
    # Short-strip yield is only for an otherwise surviving lane. If the
    # shield is already down, stop; do not arc around floor texture.
    if yield_cmd is not None and float(np.max(surv)) >= SURV_MIN:
        action, family = yield_cmd
        return action, family, float(np.max(surv))
    safe = np.asarray(allowed) & (surv >= SURV_MIN)
    names = list(names)
    cands = np.asarray(cands, np.float64)
    idx = np.where(safe)[0]
    moving = [i for i in idx if "spin" not in str(names[i])]
    # A spin that clears 0.95 is not a way past a wide barrel. Creep.
    if not moving:
        creep = creep_over(low)
        if creep is not None:
            action, family = creep
            return action, family, float(np.max(surv))
        fwd_m, left_m, right_m = (float(v) for v in da)
        # The lower map trips the shield while the nose cone is still open
        # (west barrels: surv 0.44 at GT forward 7 m). Creep until a cone
        # actually closes or a side face is inside LOW_NEAR_M.
        if min(fwd_m, left_m, right_m) >= NEAR_M and not _side_block(low, da):
            return np.array([CREEP_DX, 0.0, 0.0, 0.0], np.float64), "creep", float(np.max(surv))
        if not safe.any():
            return np.zeros(4, np.float64), "stop", float(np.max(surv))
        pool = list(idx)
    else:
        pool = moving
    dx = cands[pool, 0, 0]
    i = int(pool[int(np.argmax(dx))])
    return first_cmd(cands[i]), str(names[i]), float(surv[i])


def walk(ucv, cam, depth_model, model, name, kind, x, y, yaw, goal_xy, steps):
    print(f"WALK {name} kind={kind} start {x:.0f},{y:.0f} yaw {yaw:.0f}", flush=True)
    rng = np.random.default_rng(0)
    prev = np.zeros(4, np.float32)
    rows = []
    gx, gy = goal_xy
    goal = np.array([gx / 100.0, gy / 100.0, 0.52], np.float64)
    for step in range(steps):
        seen = sense(ucv, cam, depth_model, x, y, yaw)
        if seen is None:
            print(f"  s={step} grab_fail", flush=True)
            break
        state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, math.radians(yaw)], np.float64)
        bearing = sim.wrap(math.atan2(goal[1] - state[1], goal[0] - state[0]) - state[6])
        names, cands, allowed = sim.candidates(rng, bearing)
        surv, _prog = sim.score([model], seen["depth"], state, goal, cands, prev, seen["da"][0])
        action, family, s_pick = pick(seen["da"], seen["low"], names, cands, surv, allowed, bearing)
        gt, da = seen["gt"], seen["da"]
        dist = math.hypot(gx - x, gy - y)
        print(
            f"  s={step} x={x:.0f} y={y:.0f} yaw={yaw:.1f} dist={dist:.0f} "
            f"gt={gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f} da={da[0]:.2f},{da[1]:.2f},{da[2]:.2f} "
            f"low_da={seen['low']} low_gt={seen['low_gt']} surv_max={float(np.max(surv)):.3f} pick={s_pick:.3f} "
            f"{family} dx={float(action[0]):.2f} dyaw={float(action[3]):.3f}",
            flush=True,
        )
        rows.append({
            "x": x, "y": y, "yaw": yaw, "gt": gt, "da": da, "low": list(seen["low"]),
            "low_gt": list(seen["low_gt"]),
            "family": family, "dx": float(action[0]), "dyaw": float(action[3]),
            "dist": dist, "surv_max": float(np.max(surv)), "surv_pick": s_pick,
        })
        if family == "stop" or min(gt) < HIT_M or dist < 40.0:
            break
        x, y, yaw = integrate(x, y, yaw, float(action[0]), float(action[3]))
        prev = np.asarray(action, np.float32)
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
        ok = moved and (not drove_into) and end_dist < start_dist - 80
        why = "progress_no_hit" if ok else "stuck_or_hit"
    elif kind == "side":
        turned = yaw_delta >= 8.0 or any("yaw" in str(r["family"]) or str(r["family"]).startswith("arc") for r in rows)
        crate_block = any(r["surv_max"] < SURV_MIN or r["family"] == "stop" for r in rows)
        ok = (not drove_into) and (turned or crate_block)
        why = "yielded" if ok else "drove_into_or_ignored"
    elif kind == "wall":
        # A stop at the start pose still has GT forward ~7 m. The barrels
        # are the wall; the body has to close about 1.8 m before stopping.
        approached = end_dist < start_dist - 180
        ok = (not drove_into) and last["family"] == "stop" and min_fwd > HIT_M and approached
        why = "stopped_clear" if ok else "hit_or_ran"
    else:
        progressed = last["x"] > rows[0]["x"] + 120
        ok = (not drove_into) and progressed and last["surv_pick"] >= SURV_MIN
        why = "through" if ok else "stopped_or_hit"
    flag = "PASS" if ok else "FAIL"
    print(
        f"GATE {name} {flag} why={why} steps={len(rows)} end={last['family']} "
        f"min_fwd={min_fwd:.2f} dist {start_dist:.0f}->{end_dist:.0f} drove_into={drove_into}",
        flush=True,
    )
    return ok


def apply_daylight(ucv):
    import apply_office_daylight as day

    day.apply(ucv, day.load_cfg())


def restore_boxes(ucv):
    rows = json.loads((boxes.OUT / "lowbox_restore.json").read_text())
    for row in rows:
        name = row["name"]
        col.req(ucv, f"vset /object/{name}/object_mobility true")
        col.req(ucv, f"vset /object/{name}/physics false")
        col.req(ucv, f"vset /object/{name}/scale {row['scale']}")
        col.req(ucv, f"vset /object/{name}/rotation {row['rotation']}")
        col.req(ucv, f"vset /object/{name}/location {row['location']}")
    print("RESTORED", len(rows), flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not Path(CONS).is_file():
        print("MISSING_CONS", CONS, flush=True)
        return 1
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.ensure_dog(ucv)
    try:
        apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    boxes.save_restore(ucv)
    depth_model = DepthMinPredictor.from_checkpoint(DEPTH, device="cuda")
    model = load_checkpoint(CONS, "cuda")
    model.eval()
    print(f"DEPTH {DEPTH}\nCONS {CONS}\nhorizon56={horizon_row_56()} SURV {SURV_MIN}", flush=True)
    summary = []

    def run_job(name, kind, x, y, yaw, goal, steps):
        rows = walk(ucv, cam, depth_model, model, name, kind, x, y, yaw, goal, steps)
        ok = judge(name, kind, rows)
        summary.append({"name": name, "kind": kind, "ok": ok, "n": len(rows), "end": rows[-1] if rows else None})

    restore_boxes(ucv)
    run_job("center_lane", "aisle", -480.0, -200.0, 0.0, (200.0, -200.0), 36)
    run_job("block_west", "wall", -520.0, -250.0, 0.0, (80.0, -250.0), 40)
    run_job("gap", "through", -160.0, -200.0, 0.0, (200.0, -200.0), 36)
    boxes.place_boxes(ucv)
    run_job("right_box", "side", -480.0, -170.0, 0.0, (220.0, -170.0), 32)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    n_ok = sum(1 for s in summary if s["ok"])
    print(f"DONE {n_ok}/{len(summary)} {OUT}", flush=True)
    return 0 if n_ok == len(summary) else 2


if __name__ == "__main__":
    raise SystemExit(main())
