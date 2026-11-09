#!/usr/bin/env python3
"""Teach the aligned consequence the open office corridor without dropping the wall.

Samples the aisle, the gap, the west wall, and the side crates through the
same hide the loop uses. Labels come from simulator depth. The goal vector
and the previous command match the closed loop, so a carried forward step
cannot veto a clear aisle. Writes a new consequence file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/aerial-wam-v2/experiments/aerial/scripts")

import run_office_cons_loop as loop
import train_office_align as align
from experiments.aerial.rl.consequence_model import (
    depth_input,
    goal_features,
    load_checkpoint,
    save_checkpoint,
)
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
DEPTH = ART / "da3_office_align_20261104/depth_office_cone_align.pt"
INIT = ART / "indoor_office_align_20261104/c_s218/consequence_best.pt"
OUT = ART / "indoor_office_align_20261104/c_s218/consequence_corridor.pt"
CACHE = Path("/data/simworld/run/office_align_corridor_20261104/poses.npz")
YAW = 0.3141592653589793
STEPS = 60


def labels(cones: np.ndarray) -> dict[str, np.ndarray]:
    fwd, left, right = cones[:, 0], cones[:, 1], cones[:, 2]
    side_l = (left < 1.0) & (fwd > 2.0) & (right > 1.2)
    side_r = (right < 1.0) & (fwd > 2.0) & (left > 1.2)
    return {
        "open": (fwd > 3.0) & (left > 2.0) & (right > 2.0),
        "hit": fwd < 1.15,
        "side_l": side_l,
        "side_r": side_r,
    }


def sample_sim(depth_model) -> dict[str, np.ndarray]:
    if CACHE.is_file():
        z = np.load(CACHE)
        print(f"cache {len(z['depth'])} {CACHE}", flush=True)
        return {k: z[k] for k in z.files}
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(loop.col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    loop.boxes.save_restore(ucv)
    routes = [("lane", -200.0, (200.0, -200.0), np.arange(-500.0, 61.0, 15.0))]
    depth, goal, cones = [], [], []
    loop.restore_boxes(ucv)
    for _name, y, goal_xy, xs in routes:
        for x in xs:
            seen = loop.sense(ucv, cam, depth_model, float(x), float(y), 0.0)
            if seen is None:
                print(f"grab_fail {x:.0f},{y:.0f}", flush=True)
                continue
            state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, 0.0], np.float64)
            goal_m = np.array([goal_xy[0] / 100.0, goal_xy[1] / 100.0, 0.52], np.float64)
            depth.append(depth_input(np.asarray(seen["depth"], np.float32)))
            goal.append(goal_features(state, goal_m))
            cones.append(np.asarray(seen["gt"], np.float32))
            print(
                f"lane x={x:.0f} gt={seen['gt'][0]:.2f},{seen['gt'][1]:.2f},{seen['gt'][2]:.2f} "
                f"da={seen['da'][0]:.2f},{seen['da'][1]:.2f},{seen['da'][2]:.2f}",
                flush=True,
            )
    wall_goal = (80.0, -250.0)
    for x in np.arange(-520.0, -300.0, 12.0):
        seen = loop.sense(ucv, cam, depth_model, float(x), -250.0, 0.0)
        if seen is None:
            continue
        state = np.array([x / 100.0, -2.50, 0.52, 0, 0, 0, 0.0], np.float64)
        goal_m = np.array([wall_goal[0] / 100.0, wall_goal[1] / 100.0, 0.52], np.float64)
        depth.append(depth_input(np.asarray(seen["depth"], np.float32)))
        goal.append(goal_features(state, goal_m))
        cones.append(np.asarray(seen["gt"], np.float32))
        print(
            f"wall x={x:.0f} gt={seen['gt'][0]:.2f},{seen['gt'][1]:.2f},{seen['gt'][2]:.2f} "
            f"da={seen['da'][0]:.2f},{seen['da'][1]:.2f},{seen['da'][2]:.2f}",
            flush=True,
        )
    loop.boxes.place_boxes(ucv)
    side_goal = (220.0, -170.0)
    for x in np.arange(-480.0, -250.0, 15.0):
        seen = loop.sense(ucv, cam, depth_model, float(x), -170.0, 0.0)
        if seen is None:
            continue
        state = np.array([x / 100.0, -1.70, 0.52, 0, 0, 0, 0.0], np.float64)
        goal_m = np.array([side_goal[0] / 100.0, side_goal[1] / 100.0, 0.52], np.float64)
        depth.append(depth_input(np.asarray(seen["depth"], np.float32)))
        goal.append(goal_features(state, goal_m))
        cones.append(np.asarray(seen["gt"], np.float32))
        print(
            f"side x={x:.0f} gt={seen['gt'][0]:.2f},{seen['gt'][1]:.2f},{seen['gt'][2]:.2f} "
            f"da={seen['da'][0]:.2f},{seen['da'][1]:.2f},{seen['da'][2]:.2f}",
            flush=True,
        )
    loop.restore_boxes(ucv)
    packed = {
        "depth": np.stack(depth).astype(np.float32),
        "goal": np.stack(goal).astype(np.float32),
        "cones": np.stack(cones).astype(np.float32),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **packed)
    print(f"sampled {len(depth)}", flush=True)
    return packed


def pack(depth, goal, device):
    d = torch.as_tensor(depth, dtype=torch.float32, device=device)
    g = torch.as_tensor(goal, dtype=torch.float32, device=device)
    prev = torch.zeros(d.shape[0], 4, device=device)
    return d, g, prev


def logits(model, depth, goal, prev, dx, dyaw):
    actions = torch.zeros(depth.shape[0], 10, 4, device=depth.device)
    actions[:, 0, 0] = dx
    actions[:, 0, 3] = dyaw
    return model(depth, goal, prev, actions)["hazard_logit"][:, 0]


def group_loss(model, depth, goal, prev, mask) -> torch.Tensor:
    loss = depth.new_zeros(())
    if bool(mask["open"].any()):
        zf = logits(model, depth, goal, prev, 0.8, 0.0)
        zl = logits(model, depth, goal, prev, 0.0, -YAW)
        zr = logits(model, depth, goal, prev, 0.0, YAW)
        m = mask["open"]
        loss = loss + torch.relu(zf[m] + 4.0).pow(2).mean()
        loss = loss + torch.relu(2.0 - zl[m]).pow(2).mean()
        loss = loss + torch.relu(2.0 - zr[m]).pow(2).mean()
    if bool(mask["hit"].any()):
        zf = logits(model, depth, goal, prev, 0.8, 0.0)
        loss = loss + 4.0 * torch.relu(2.0 - zf[mask["hit"]]).pow(2).mean()
    side = mask["side_l"] | mask["side_r"]
    if bool(side.any()):
        zf = logits(model, depth, goal, prev, 0.8, 0.0)
        loss = loss + 4.0 * torch.relu(2.0 - zf[side]).pow(2).mean()
    if bool(mask["side_l"].any()):
        zr = logits(model, depth, goal, prev, 0.0, YAW)
        loss = loss + 4.0 * torch.relu(zr[mask["side_l"]] + 2.0).pow(2).mean()
    if bool(mask["side_r"].any()):
        zl = logits(model, depth, goal, prev, 0.0, -YAW)
        loss = loss + 4.0 * torch.relu(zl[mask["side_r"]] + 2.0).pow(2).mean()
    return loss


def stats_of(model, depth, goal, prev, mask) -> dict[str, float]:
    zf = logits(model, depth, goal, prev, 0.8, 0.0)
    zl = logits(model, depth, goal, prev, 0.0, -YAW)
    zr = logits(model, depth, goal, prev, 0.0, YAW)
    sf = 1.0 - torch.sigmoid(zf)
    sl = 1.0 - torch.sigmoid(zl)
    sr = 1.0 - torch.sigmoid(zr)
    side = mask["side_l"] | mask["side_r"]
    away = []
    if bool(mask["side_l"].any()):
        away.append(sr[mask["side_l"]])
    if bool(mask["side_r"].any()):
        away.append(sl[mask["side_r"]])
    return {
        "open": float(sf[mask["open"]].min()) if bool(mask["open"].any()) else 1.0,
        "spin": float(torch.maximum(sl, sr)[mask["open"]].max()) if bool(mask["open"].any()) else 0.0,
        "hit": float(sf[mask["hit"]].max()) if bool(mask["hit"].any()) else 0.0,
        "side": float(sf[side].max()) if bool(side.any()) else 0.0,
        "away": float(torch.cat(away).min()) if away else 1.0,
    }


def main() -> int:
    dev = torch.device("cuda")
    depth_model = DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    sim = sample_sim(depth_model)
    real_rgb, real_gt = align.load_real()
    live = np.load(align.SIM_LIVE)
    real_d, real_c, real_g = align.encode_split(depth_model, real_rgb, real_gt)
    sim_d, sim_c, sim_g = align.encode_split(depth_model, live["rgb"], live["gt56"].astype(np.float32))
    del depth_model
    torch.cuda.empty_cache()
    replay = {
        "real": pack(real_d, real_g, dev) + ({k: torch.as_tensor(v, device=dev) for k, v in labels(real_c).items()},),
        "live": pack(sim_d, sim_g, dev) + ({k: torch.as_tensor(v, device=dev) for k, v in labels(sim_c).items()},),
    }
    sampled_mask = {k: torch.as_tensor(v, device=dev) for k, v in labels(sim["cones"]).items()}
    sampled = (
        torch.as_tensor(sim["depth"], device=dev),
        torch.as_tensor(sim["goal"], device=dev),
        sampled_mask,
    )
    carried = torch.zeros(sim["depth"].shape[0], 4, device=dev)
    carried[:, 0] = 0.8
    n_open = int(sampled_mask["open"].sum())
    n_hit = int(sampled_mask["hit"].sum())
    n_side = int((sampled_mask["side_l"] | sampled_mask["side_r"]).sum())
    print(f"sampled open {n_open} hit {n_hit} side {n_side}", flush=True)
    for name, (_d, _g, _p, mask) in replay.items():
        print(
            f"{name} open {int(mask['open'].sum())} hit {int(mask['hit'].sum())} "
            f"side {int((mask['side_l'] | mask['side_r']).sum())}",
            flush=True,
        )
    model = load_checkpoint(str(INIT), "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best = None

    def report(step: int) -> dict:
        model.eval()
        out = {}
        with torch.no_grad():
            depth, goal, mask = sampled
            zero = torch.zeros_like(carried)
            for tag, prev in (("still", zero), ("carry", carried)):
                row = stats_of(model, depth, goal, prev, mask)
                out[tag] = row
                print(
                    f"step {step} corridor {tag} open {row['open']:.3f} spin {row['spin']:.3f} "
                    f"hit {row['hit']:.3f} side {row['side']:.3f} away {row['away']:.3f}",
                    flush=True,
                )
            for name, (depth, goal, prev, mask) in replay.items():
                row = stats_of(model, depth, goal, prev, mask)
                out[name] = row
                print(
                    f"step {step} {name} open {row['open']:.3f} spin {row['spin']:.3f} "
                    f"hit {row['hit']:.3f} side {row['side']:.3f} away {row['away']:.3f}",
                    flush=True,
                )
        return out

    def ok(row: dict) -> bool:
        return (
            row["still"]["open"] >= 0.96
            and row["carry"]["open"] >= 0.96
            and row["still"]["spin"] <= 0.50
            and row["carry"]["spin"] <= 0.50
            and row["still"]["hit"] <= 0.20
            and row["carry"]["hit"] <= 0.20
            and row["still"]["side"] <= 0.35
            and row["carry"]["side"] <= 0.35
            and row["real"]["open"] >= 0.93
            and row["real"]["hit"] <= 0.25
            and row["live"]["hit"] <= 0.25
        )

    report(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, STEPS + 1):
        model.train()
        loss = torch.zeros((), device=dev)
        for _name, (depth, goal, prev, mask) in replay.items():
            loss = loss + group_loss(model, depth, goal, prev, mask)
        depth, goal, mask = sampled
        loss = loss + 4.0 * group_loss(model, depth, goal, torch.zeros_like(carried), mask)
        loss = loss + 4.0 * group_loss(model, depth, goal, carried, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 or step == STEPS:
            row = report(step)
            save_checkpoint(OUT.with_name("consequence_corridor_last.pt"), model, {"step": step})
            if ok(row) and (best is None or row["carry"]["open"] >= best):
                best = row["carry"]["open"]
                save_checkpoint(OUT, model, {"step": step, "carry_open": best})
                print("CORRIDOR_SAVE", step, flush=True)
    if best is None:
        save_checkpoint(OUT, model, {"step": STEPS, "fallback": True})
        print("CORRIDOR_SAVE_LAST", flush=True)
    print("CORRIDOR_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
