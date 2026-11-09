#!/usr/bin/env python3
"""Turn toward the open side of a real shelf, then drive.

The nose at the end of the gap faces boxes about 2 m away. A right turn
opens the depth to about 3 m; a left turn closes it. The walker is unchanged.
Init is consequence_cam60_around_open.pt. Writes consequence_cam60_shelf.pt.
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

import collect_office_train as col
import run_office_cons_loop as loop
import train_office_align as align
import train_office_cam60_around as base
from experiments.aerial.rl.consequence_model import load_checkpoint, save_checkpoint
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "indoor_office_cam60_20261105/c_s218/consequence_cam60_around_open.pt"
OUT = ART / "indoor_office_cam60_20261105/c_s218/consequence_cam60_shelf.pt"
BLOCK = (
    (0.80, 0.0),
    (0.35, 0.0),
    (1.00, 0.0),
    (0.80, -0.10),
    (0.20, -base.YAW),
    (0.80, -base.YAW),
    (0.16, -base.YAW),
)


def step_pose(x, y, yaw, dx=0.8):
    rad = math.radians(yaw)
    moved = max(0.0, dx) * 17.5
    return x + moved * math.cos(rad), y + moved * math.sin(rad), yaw


def main() -> int:
    if OUT.resolve() in {INIT.resolve(), base.INIT.resolve(), base.OUT.resolve()}:
        raise RuntimeError("refusing to overwrite an existing consequence file")
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    loop.restore_boxes(ucv)
    nx, ny, nyaw = step_pose(205.0, -106.0, 30.3)
    poses = [
        (205.0, -106.0, 12.3, "shelf"),
        (205.0, -106.0, 30.3, "opened"),
        (nx, ny, nyaw, "opened"),
    ]
    grabbed = []
    try:
        for x, y, yaw, name in poses:
            col.place(ucv, x, y, yaw)
            col.look(ucv, cam, x, y, yaw)
            packed = col.grab(ucv, cam)
            if packed is None:
                print(f"grab_fail {name}", flush=True)
                continue
            image, seen, warped = col.warp_pair(*packed)
            gt = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
            grabbed.append((name, x, y, yaw, np.ascontiguousarray(image), gt))
    finally:
        loop.restore_boxes(ucv)
        print("RESTORED", flush=True)

    sim = np.load(base.SIM_NPZ)
    extra = np.load(Path("/data/simworld/run/office_cam60_20261105/open_hold.npz"))
    wrap = DepthMinPredictor.from_checkpoint(str(base.DEPTH), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    depth, da, goals, kind = [], [], [], []

    def add(rgb, gt56, x, y, yaw, gx, gy, name):
        d, a, g, goal = base.encode_rgb(wrap, rgb, gt56, x, y, yaw, gx, gy)
        depth.append(d)
        da.append(a)
        goals.append(goal)
        kind.append(name)
        print(f"{name} x={x:.0f} yaw={yaw:.1f} da={a[0]:.2f},{a[1]:.2f},{a[2]:.2f} gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f}", flush=True)

    for i, rgb in enumerate(sim["rgb"]):
        x, y, yaw = (float(v) for v in sim["xy"][i])
        name = str(sim["kind"][i])
        gx, gy = (80.0, -250.0) if name == "wall" else (220.0, -170.0) if name == "side" else base.GOAL_XY
        add(rgb, sim["gt56"][i], x, y, yaw, gx, gy, name)
    for i, rgb in enumerate(extra["rgb"]):
        x, y, yaw = (float(v) for v in extra["xy"][i])
        gx, gy = (float(v) for v in extra["goal"][i])
        add(rgb, extra["gt56"][i], x, y, yaw, gx, gy, "hold")
    for name, x, y, yaw, rgb, gt56 in grabbed:
        add(rgb, gt56, x, y, yaw, *base.GOAL_XY, name)
    del wrap
    torch.cuda.empty_cache()

    da_a = np.stack(da)
    kind_a = np.asarray(kind)
    turn = np.flatnonzero((kind_a == "gap") & (da_a[:, 1] < 0.85) & (da_a[:, 2] > 2.5) & (da_a[:, 0] > 1.5))
    opened = np.flatnonzero(
        ((da_a.min(axis=1) > 2.0) & (da_a[:, 0] > 2.5) & np.isin(kind_a, ["aisle", "side", "gap", "hold"]))
        | ((kind_a == "opened") & (da_a[:, 0] > 2.0))
    )
    nose = np.flatnonzero((kind_a == "front") & (da_a[:, 0] < 0.8) & (da_a[:, 2] > 2.5))
    passed = np.flatnonzero((kind_a == "front_yaw") & (da_a[:, 0] > 2.0))
    shelf = np.flatnonzero(kind_a == "shelf")
    print(f"turn {len(turn)} open {len(opened)} nose {len(nose)} pass {len(passed)} shelf {len(shelf)}", flush=True)
    if min(len(turn), len(opened), len(nose), len(passed), len(shelf)) == 0:
        print("SHELF_EMPTY", flush=True)
        return 1

    dev = torch.device("cuda")
    depth_t = torch.as_tensor(np.stack(depth), device=dev)
    goal_t = torch.as_tensor(np.stack(goals), device=dev)
    groups = {
        "turn": torch.as_tensor(turn, device=dev),
        "open": torch.as_tensor(opened, device=dev),
        "nose": torch.as_tensor(nose, device=dev),
        "pass": torch.as_tensor(passed, device=dev),
        "shelf": torch.as_tensor(shelf, device=dev),
    }
    real_rgb, real_gt = align.load_real()
    real_model = DepthMinPredictor.from_checkpoint(str(base.DEPTH), device="cuda")
    real_d, real_c, real_g = align.encode_split(real_model, real_rgb, real_gt)
    del real_model
    torch.cuda.empty_cache()
    real_depth = torch.as_tensor(real_d, device=dev)
    real_goal = torch.as_tensor(real_g, device=dev)
    hit_i = torch.as_tensor(np.flatnonzero(real_c[:, 0] < 1.15), device=dev)

    model = load_checkpoint(str(INIT), "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    def logits(index, dx, dyaw, prev_dx, depth, goal):
        prev = torch.zeros(index.shape[0], 4, device=dev)
        prev[:, 0] = prev_dx
        return align.logits(model, depth[index], goal[index], prev, dx, dyaw)

    def surv(z):
        return 1.0 - torch.sigmoid(z)

    def report(step: int) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            blocked = torch.stack([
                surv(logits(groups["shelf"], dx, dyaw, 0.8, depth_t, goal_t)) for dx, dyaw in BLOCK
            ])
            stats = {
                "shelf": float(surv(logits(groups["shelf"], 0.0, base.YAW, 0.8, depth_t, goal_t)).min()),
                "into": float(surv(logits(groups["shelf"], 0.0, -base.YAW, 0.8, depth_t, goal_t)).max()),
                "block": float(blocked.max()),
                "drive": float(surv(logits(groups["open"], 0.8, 0.0, 0.0, depth_t, goal_t)).min()),
                "turn": float(surv(logits(groups["turn"], 0.0, base.YAW, 0.8, depth_t, goal_t)).min()),
                "tfwd": float(torch.stack([
                    surv(logits(groups["turn"], dx, dyaw, 0.8, depth_t, goal_t)) for dx, dyaw in base.STRAIGHT
                ]).max()),
                "nose": float(surv(logits(groups["nose"], 0.0, base.YAW, 0.0, depth_t, goal_t)).min()),
                "nose_fwd": float(surv(logits(groups["nose"], 0.8, 0.0, 0.0, depth_t, goal_t)).max()),
                "pass": float(surv(logits(groups["pass"], 0.8, 0.0, 0.0, depth_t, goal_t)).min()),
                "hit": float(surv(logits(hit_i, 0.8, 0.0, 0.0, real_depth, real_goal)).max()),
            }
        model.train()
        print(
            f"step {step} shelf {stats['shelf']:.3f} into {stats['into']:.3f} block {stats['block']:.3f} "
            f"drive {stats['drive']:.3f} turn {stats['turn']:.3f} tfwd {stats['tfwd']:.3f} "
            f"nose {stats['nose']:.3f} pass {stats['pass']:.3f} hit {stats['hit']:.3f}",
            flush=True,
        )
        return stats

    def ok(stats: dict[str, float]) -> bool:
        return (
            stats["shelf"] >= 0.95 and stats["into"] <= 0.50 and stats["block"] <= 0.90
            and stats["drive"] >= 0.95 and stats["turn"] >= 0.95 and stats["tfwd"] <= 0.90
            and stats["nose"] >= 0.95 and stats["nose_fwd"] <= 0.35
            and stats["pass"] >= 0.95 and stats["hit"] <= 0.35
        )

    best_score, best_state = None, None
    report(0)
    for step in range(1, 41):
        model.train()
        loss = torch.zeros((), device=dev)
        for prev_dx in (0.0, 0.8):
            loss = loss + 4.0 * torch.relu(logits(groups["shelf"], 0.0, base.YAW, prev_dx, depth_t, goal_t) + 3.0).pow(2).mean()
            loss = loss + 4.0 * torch.relu(2.0 - logits(groups["shelf"], 0.0, -base.YAW, prev_dx, depth_t, goal_t)).pow(2).mean()
            for dx, dyaw in BLOCK:
                loss = loss + torch.relu(-2.2 - logits(groups["shelf"], dx, dyaw, prev_dx, depth_t, goal_t)).pow(2).mean()
            loss = loss + torch.relu(logits(groups["open"], 0.8, 0.0, prev_dx, depth_t, goal_t) + 3.0).pow(2).mean()
            loss = loss + torch.relu(logits(groups["turn"], 0.0, base.YAW, prev_dx, depth_t, goal_t) + 3.0).pow(2).mean()
            for dx, dyaw in base.STRAIGHT:
                loss = loss + torch.relu(-2.2 - logits(groups["turn"], dx, dyaw, prev_dx, depth_t, goal_t)).pow(2).mean()
            loss = loss + torch.relu(logits(groups["pass"], 0.8, 0.0, prev_dx, depth_t, goal_t) + 3.0).pow(2).mean()
            loss = loss + torch.relu(logits(groups["nose"], 0.0, base.YAW, prev_dx, depth_t, goal_t) + 3.0).pow(2).mean()
            loss = loss + 4.0 * torch.relu(2.0 - logits(groups["nose"], 0.8, 0.0, prev_dx, depth_t, goal_t)).pow(2).mean()
            loss = loss + 4.0 * torch.relu(2.0 - logits(hit_i, 0.8, 0.0, prev_dx, real_depth, real_goal)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 or step == 40:
            stats = report(step)
            score = stats["shelf"] + stats["drive"] + stats["turn"] + stats["pass"] - max(0.0, stats["block"] - 0.90) - max(0.0, stats["into"] - 0.50)
            if best_score is None or score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"SHELF_BEST {step} {score:.3f}", flush=True)
            if ok(stats):
                save_checkpoint(OUT, model.eval(), {"step": step})
                print("SHELF_SAVE", step, flush=True)
                print("SHELF_OUT", OUT, flush=True)
                return 0
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(OUT, model.eval(), {"step": 40, "fallback": True, "score": best_score})
    print("SHELF_SAVE_LAST", best_score, flush=True)
    print("SHELF_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
