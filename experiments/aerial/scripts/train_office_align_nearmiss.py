#!/usr/bin/env python3
"""Nudge the aligned consequence only where a clear view is just under 0.95.

The west-wall face and the lane blockage stay forbidden. Frames the model
already treats as a wall are left alone. A new consequence file is written.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/aerial-wam-v2/experiments/aerial/scripts")

import run_office_cons_loop as loop
import train_office_align as align
from office_bands import cones3_forward
from experiments.aerial.rl.consequence_model import (
    depth_input,
    goal_features,
    load_checkpoint,
    save_checkpoint,
)
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
DEPTH = ART / "da3_office_align_20261104/depth_office_cone_align.pt"
INIT = ART / "indoor_office_align_20261104/c_s218/consequence_nearmiss.pt"
OUT = ART / "indoor_office_align_20261104/c_s218/consequence_block.pt"
CACHE = Path("/data/simworld/run/office_align_nearmiss_20261104/poses.npz")
YAW = 0.3141592653589793
STEPS = 25


def pose_row(depth_model, ucv, cam, x, y, yaw, goal_xy, kind):
    seen = loop.sense(ucv, cam, depth_model, float(x), float(y), float(yaw))
    if seen is None:
        print(f"grab_fail {kind} {x:.0f},{y:.0f}", flush=True)
        return None
    state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, math.radians(yaw)], np.float64)
    goal = np.array([goal_xy[0] / 100.0, goal_xy[1] / 100.0, 0.52], np.float64)
    gt = np.asarray(seen["gt"], np.float32)
    print(
        f"{kind} x={x:.0f} y={y:.0f} yaw={yaw:.0f} "
        f"gt={gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f} da={seen['da'][0]:.2f},{seen['da'][1]:.2f},{seen['da'][2]:.2f}",
        flush=True,
    )
    return {
        "depth": depth_input(np.asarray(seen["depth"], np.float32)),
        "goal": goal_features(state, goal),
        "cones": gt,
        "kind": kind,
        "x": float(x),
        "y": float(y),
    }


def sample(depth_model) -> dict[str, np.ndarray]:
    if CACHE.is_file():
        z = np.load(CACHE, allow_pickle=False)
        print(f"cache {len(z['depth'])}", flush=True)
        return {k: z[k] for k in z.files}
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(loop.col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    loop.boxes.save_restore(ucv)
    loop.restore_boxes(ucv)
    rows = []
    for x in np.arange(-180.0, 161.0, 10.0):
        rows.append((float(x), -200.0, 0.0, (200.0, -200.0), "gap"))
    for x in np.arange(-390.0, -20.0, 10.0):
        y = -200.0 + (float(x) + 393.0) * math.tan(math.radians(18.0))
        rows.append((float(x), float(y), 18.0, (200.0, -200.0), "offset"))
    for x in (-332.0, -324.0, -316.0, -308.0, -300.0):
        rows.append((x, -250.0, 0.0, (80.0, -250.0), "face"))
    for x in (-310.0, -290.0, -270.0, -250.0):
        rows.append((x, -200.0, 0.0, (200.0, -200.0), "block"))
    got = []
    for x, y, yaw, goal, kind in rows:
        row = pose_row(depth_model, ucv, cam, x, y, yaw, goal, kind)
        if row is not None:
            got.append(row)
    loop.boxes.place_boxes(ucv)
    for x in np.arange(-360.0, -250.0, 15.0):
        row = pose_row(depth_model, ucv, cam, float(x), -170.0, 0.0, (220.0, -170.0), "side")
        if row is not None:
            got.append(row)
    loop.restore_boxes(ucv)
    packed = {
        "depth": np.stack([r["depth"] for r in got]).astype(np.float32),
        "goal": np.stack([r["goal"] for r in got]).astype(np.float32),
        "cones": np.stack([r["cones"] for r in got]).astype(np.float32),
        "kind": np.asarray([r["kind"] for r in got]),
        "x": np.asarray([r["x"] for r in got], np.float32),
        "y": np.asarray([r["y"] for r in got], np.float32),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **packed)
    print(f"sampled {len(got)}", flush=True)
    return packed


def logits(model, depth, goal, prev, dx, dyaw):
    actions = torch.zeros(depth.shape[0], 10, 4, device=depth.device)
    actions[:, 0, 0] = dx
    actions[:, 0, 3] = dyaw
    return model(depth, goal, prev, actions)["hazard_logit"][:, 0]


def surv_fwd(model, depth, goal, prev):
    return 1.0 - torch.sigmoid(logits(model, depth, goal, prev, 0.8, 0.0))


def main() -> int:
    dev = torch.device("cuda")
    depth_model = DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    sim = sample(depth_model)
    real_rgb, real_gt = align.load_real()
    live = np.load(align.SIM_LIVE)
    real_d, real_c, real_g = align.encode_split(depth_model, real_rgb, real_gt)
    live_d, live_c, live_g = align.encode_split(depth_model, live["rgb"], live["gt56"].astype(np.float32))
    del depth_model
    torch.cuda.empty_cache()

    depth = torch.as_tensor(sim["depth"], device=dev)
    goal = torch.as_tensor(sim["goal"], device=dev)
    cones = sim["cones"]
    kind = sim["kind"]
    da_m = 0.5 * np.exp(sim["depth"] * math.log(100.0 / 0.5))
    da_cones = np.stack([cones3_forward(m) for m in da_m])
    clear = (cones[:, 0] > 3.0) & (cones[:, 1] > 1.5) & (cones[:, 2] > 1.5)
    clear &= (kind == "gap") | (kind == "offset")
    hit = ((kind == "face") | (kind == "block")) & (cones[:, 0] < 1.15) & (da_cones[:, 0] < 2.0)
    side_l = (kind == "side") & (cones[:, 1] < 1.0) & (cones[:, 0] > 2.0) & (cones[:, 2] > 1.2)
    side_r = (kind == "side") & (cones[:, 2] < 1.0) & (cones[:, 0] > 2.0) & (cones[:, 1] > 1.2)
    model = load_checkpoint(str(INIT), "cuda")
    model.eval()
    with torch.no_grad():
        s0 = surv_fwd(model, depth, goal, torch.zeros(depth.shape[0], 4, device=dev))
        s1 = surv_fwd(model, depth, goal, torch.tensor([0.8, 0, 0, 0], device=dev).expand(depth.shape[0], 4))
    both = torch.minimum(s0, s1).detach().cpu().numpy()
    near = clear.copy()
    print(
        f"clear {int(clear.sum())} near {int(near.sum())} hit {int(hit.sum())} "
        f"side {int(side_l.sum() + side_r.sum())}",
        flush=True,
    )
    order = np.argsort(both)
    shown = 0
    for i in order:
        if not clear[i]:
            continue
        print(
            f"  low {kind[i]} x={sim['x'][i]:.0f} y={sim['y'][i]:.0f} "
            f"gt={cones[i, 0]:.2f},{cones[i, 1]:.2f},{cones[i, 2]:.2f} surv={both[i]:.3f}",
            flush=True,
        )
        shown += 1
        if shown >= 12:
            break
    if int(near.sum()) < 1:
        print("NEARMISS_EMPTY", flush=True)
        return 2
    near_t = torch.as_tensor(near, device=dev)
    hit_t = torch.as_tensor(hit, device=dev)
    side_l_t = torch.as_tensor(side_l, device=dev)
    side_r_t = torch.as_tensor(side_r, device=dev)
    real_depth = torch.as_tensor(real_d, device=dev)
    real_goal = torch.as_tensor(real_g, device=dev)
    live_depth = torch.as_tensor(live_d, device=dev)
    live_goal = torch.as_tensor(live_g, device=dev)
    real_hit = torch.as_tensor(real_c[:, 0] < 1.15, device=dev)
    real_open = torch.as_tensor((real_c[:, 0] > 3.0) & (real_c[:, 1] > 1.5) & (real_c[:, 2] > 1.5), device=dev)
    live_hit = torch.as_tensor(live_c[:, 0] < 1.15, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # Face anchors are images the head already reads as a barrel. Their survival
    # has to stay low. A carried command uses the same images.
    zero = torch.zeros(depth.shape[0], 4, device=dev)
    carry = torch.zeros_like(zero)
    carry[:, 0] = 0.8
    best = None

    def report(step: int) -> dict:
        model.eval()
        with torch.no_grad():
            a = surv_fwd(model, depth, goal, zero)
            b = surv_fwd(model, depth, goal, carry)
            face = torch.maximum(
                a[hit_t].max() if bool(hit_t.any()) else a.new_zeros(()),
                b[hit_t].max() if bool(hit_t.any()) else b.new_zeros(()),
            )
            near_min = float(torch.minimum(a[near_t], b[near_t]).min())
            rz = torch.zeros(real_depth.shape[0], 4, device=dev)
            ro = surv_fwd(model, real_depth, real_goal, rz)
            real_open_min = float(ro[real_open].min()) if bool(real_open.any()) else 1.0
            real_hit_max = float(ro[real_hit].max()) if bool(real_hit.any()) else 0.0
            lz = torch.zeros(live_depth.shape[0], 4, device=dev)
            lo = surv_fwd(model, live_depth, live_goal, lz)
            live_hit_max = float(lo[live_hit].max()) if bool(live_hit.any()) else 0.0
            side = side_l_t | side_r_t
            side_max = float(torch.maximum(a, b)[side].max()) if bool(side.any()) else 0.0
        print(
            f"step {step} near {near_min:.3f} face {float(face):.3f} "
            f"real_open {real_open_min:.3f} real_hit {real_hit_max:.3f} "
            f"live_hit {live_hit_max:.3f} side {side_max:.3f}",
            flush=True,
        )
        return {
            "near": near_min,
            "face": float(face),
            "real_open": real_open_min,
            "real_hit": real_hit_max,
            "live_hit": live_hit_max,
            "side": side_max,
        }

    def ok(row: dict) -> bool:
        return (
            row["near"] >= 0.955
            and row["face"] <= 0.20
            and row["real_open"] >= 0.93
            and row["real_hit"] <= 0.25
            and row["live_hit"] <= 0.20
        )

    report(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, STEPS + 1):
        model.train()
        loss = depth.new_zeros(())
        for prev in (zero, carry):
            z = logits(model, depth, goal, prev, 0.8, 0.0)
            loss = loss + torch.relu(z[near_t] + 3.2).pow(2).mean()
            if bool(hit_t.any()):
                loss = loss + torch.relu(2.0 - z[hit_t]).pow(2).mean()
        rz = torch.zeros(real_depth.shape[0], 4, device=dev)
        z_real = logits(model, real_depth, real_goal, rz, 0.8, 0.0)
        if bool(real_open.any()):
            loss = loss + torch.relu(z_real[real_open] + 3.0).pow(2).mean()
        if bool(real_hit.any()):
            loss = loss + torch.relu(2.0 - z_real[real_hit]).pow(2).mean()
        lz = torch.zeros(live_depth.shape[0], 4, device=dev)
        z_live = logits(model, live_depth, live_goal, lz, 0.8, 0.0)
        if bool(live_hit.any()):
            loss = loss + torch.relu(2.0 - z_live[live_hit]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        if step % 5 == 0 or step == STEPS:
            row = report(step)
            if row["real_open"] < 0.90 or row["near"] < 0.90:
                print("NEARMISS_ABORT", flush=True)
                break
            if ok(row) and (best is None or row["near"] >= best):
                best = row["near"]
                save_checkpoint(OUT, model, {"step": step, "near": best})
                print("NEARMISS_SAVE", step, flush=True)
    if best is None:
        print("NEARMISS_NO_SAVE", flush=True)
        return 2
    print("NEARMISS_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
