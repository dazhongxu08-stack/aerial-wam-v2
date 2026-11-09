#!/usr/bin/env python3
"""Keep the learned open-side turn, and fit open depth maps back to driving.

Init is consequence_cam60_around.pt. Writes consequence_cam60_around_open.pt.
Does not overwrite the pass weights or the depth file.
"""
from __future__ import annotations

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
INIT = ART / "indoor_office_cam60_20261105/c_s218/consequence_cam60_around.pt"
OUT = ART / "indoor_office_cam60_20261105/c_s218/consequence_cam60_around_open.pt"
SIM_NPZ = base.SIM_NPZ
OPEN_NPZ = Path("/data/simworld/run/office_cam60_20261105/open_hold.npz")
POSES = [
    (-480.0, -170.0, 0.0, 220.0, -170.0),
    (-460.0, -170.0, 0.0, 220.0, -170.0),
    (125.0, -131.0, 36.0, 200.0, -200.0),
    (136.0, -123.0, 36.0, 200.0, -200.0),
    (147.0, -114.0, 36.0, 200.0, -200.0),
    (160.0, -108.0, 36.0, 200.0, -200.0),
]


def grab_open(ucv, cam) -> dict[str, np.ndarray]:
    if OPEN_NPZ.is_file():
        z = np.load(OPEN_NPZ)
        print(f"open_cache {len(z['rgb'])}", flush=True)
        return {k: z[k] for k in z.files}
    rgb, gt56, xy, goal = [], [], [], []
    for x, y, yaw, gx, gy in POSES:
        col.place(ucv, x, y, yaw)
        col.look(ucv, cam, x, y, yaw)
        packed = col.grab(ucv, cam)
        if packed is None:
            print(f"grab_fail {x:.0f}", flush=True)
            continue
        image, seen, warped = col.warp_pair(*packed)
        gt = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
        rgb.append(np.ascontiguousarray(image))
        gt56.append(gt)
        xy.append((x, y, yaw))
        goal.append((gx, gy))
    packed = {
        "rgb": np.stack(rgb),
        "gt56": np.stack(gt56).astype(np.float32),
        "xy": np.asarray(xy, np.float32),
        "goal": np.asarray(goal, np.float32),
    }
    np.savez(OPEN_NPZ, **packed)
    print(f"open_saved {len(rgb)}", flush=True)
    return packed


def main() -> int:
    if OUT.resolve() == INIT.resolve() or OUT.resolve() == base.INIT.resolve():
        raise RuntimeError("refusing to overwrite an existing consequence file")
    sim = np.load(SIM_NPZ)
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    loop.restore_boxes(ucv)
    try:
        extra = grab_open(ucv, cam)
    finally:
        loop.restore_boxes(ucv)
        print("RESTORED", flush=True)

    wrap = DepthMinPredictor.from_checkpoint(str(base.DEPTH), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    depth, da, goals, kind = [], [], [], []
    for i, rgb in enumerate(sim["rgb"]):
        x, y, yaw = (float(v) for v in sim["xy"][i])
        name = str(sim["kind"][i])
        gx, gy = base.GOAL_XY
        if name == "wall":
            gx, gy = 80.0, -250.0
        elif name == "side":
            gx, gy = 220.0, -170.0
        d, a, _g, goal = base.encode_rgb(wrap, rgb, sim["gt56"][i], x, y, yaw, gx, gy)
        depth.append(d)
        da.append(a)
        goals.append(goal)
        kind.append(name)
    for i, rgb in enumerate(extra["rgb"]):
        x, y, yaw = (float(v) for v in extra["xy"][i])
        gx, gy = (float(v) for v in extra["goal"][i])
        d, a, g, goal = base.encode_rgb(wrap, rgb, extra["gt56"][i], x, y, yaw, gx, gy)
        depth.append(d)
        da.append(a)
        goals.append(goal)
        kind.append("hold")
        print(
            f"hold x={x:.0f} y={y:.0f} yaw={yaw:.0f} da={a[0]:.2f},{a[1]:.2f},{a[2]:.2f} gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f}",
            flush=True,
        )
    del wrap
    torch.cuda.empty_cache()
    da_a = np.stack(da)
    kind_a = np.asarray(kind)
    turn = np.flatnonzero((kind_a == "gap") & (da_a[:, 1] < 0.85) & (da_a[:, 2] > 2.5) & (da_a[:, 0] > 1.5))
    opened = np.flatnonzero((da_a.min(axis=1) > 2.0) & (da_a[:, 0] > 2.5) & np.isin(kind_a, ["aisle", "side", "gap", "hold"]))
    nose = np.flatnonzero((kind_a == "front") & (da_a[:, 0] < 0.8) & (da_a[:, 2] > 2.5))
    passed = np.flatnonzero((kind_a == "front_yaw") & (da_a[:, 0] > 2.0))
    print(f"turn {len(turn)} open {len(opened)} nose {len(nose)} passed {len(passed)}", flush=True)
    if min(len(turn), len(opened), len(nose), len(passed)) == 0:
        print("OPEN_EMPTY", flush=True)
        return 1

    dev = torch.device("cuda")
    depth_t = torch.as_tensor(np.stack(depth), device=dev)
    goal_t = torch.as_tensor(np.stack(goals), device=dev)
    turn_i = torch.as_tensor(turn, device=dev, dtype=torch.long)
    open_i = torch.as_tensor(opened, device=dev, dtype=torch.long)
    nose_i = torch.as_tensor(nose, device=dev, dtype=torch.long)
    pass_i = torch.as_tensor(passed, device=dev, dtype=torch.long)
    real_rgb, real_gt = align.load_real()
    real_model = DepthMinPredictor.from_checkpoint(str(base.DEPTH), device="cuda")
    real_d, real_c, real_g = align.encode_split(real_model, real_rgb, real_gt)
    del real_model
    torch.cuda.empty_cache()
    real_depth = torch.as_tensor(real_d, device=dev)
    real_goal = torch.as_tensor(real_g, device=dev)
    hit_i = torch.as_tensor(np.flatnonzero(real_c[:, 0] < 1.15), device=dev, dtype=torch.long)

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
            stats = {
                "turn": float(surv(logits(turn_i, 0.0, base.YAW, 0.8, depth_t, goal_t)).min()),
                "fwd": float(torch.stack([
                    surv(logits(turn_i, dx, dyaw, 0.8, depth_t, goal_t)) for dx, dyaw in base.STRAIGHT
                ]).max()),
                "open": float(surv(logits(open_i, 0.8, 0.0, 0.0, depth_t, goal_t)).min()),
                "nose": float(surv(logits(nose_i, 0.0, base.YAW, 0.0, depth_t, goal_t)).min()),
                "nose_fwd": float(surv(logits(nose_i, 0.8, 0.0, 0.0, depth_t, goal_t)).max()),
                "pass": float(surv(logits(pass_i, 0.8, 0.0, 0.0, depth_t, goal_t)).min()),
                "hit": float(surv(logits(hit_i, 0.8, 0.0, 0.0, real_depth, real_goal)).max()),
            }
        model.train()
        print(
            f"step {step} turn {stats['turn']:.3f} fwd {stats['fwd']:.3f} open {stats['open']:.3f} "
            f"nose {stats['nose']:.3f} nose_fwd {stats['nose_fwd']:.3f} pass {stats['pass']:.3f} hit {stats['hit']:.3f}",
            flush=True,
        )
        return stats

    def ok(stats: dict[str, float]) -> bool:
        return (
            stats["turn"] >= 0.95 and stats["fwd"] <= 0.90 and stats["open"] >= 0.95
            and stats["nose"] >= 0.95 and stats["nose_fwd"] <= 0.35
            and stats["pass"] >= 0.95 and stats["hit"] <= 0.35
        )

    best_score, best_state = None, None
    report(0)
    for step in range(1, 41):
        model.train()
        loss = torch.zeros((), device=dev)
        for prev_dx in (0.0, 0.8):
            loss = loss + torch.relu(logits(turn_i, 0.0, base.YAW, prev_dx, depth_t, goal_t) + 3.0).pow(2).mean()
            loss = loss + torch.relu(2.0 - logits(turn_i, 0.0, -base.YAW, prev_dx, depth_t, goal_t)).pow(2).mean()
            for dx, dyaw in base.STRAIGHT:
                loss = loss + torch.relu(-2.2 - logits(turn_i, dx, dyaw, prev_dx, depth_t, goal_t)).pow(2).mean()
            loss = loss + torch.relu(logits(open_i, 0.8, 0.0, prev_dx, depth_t, goal_t) + 3.0).pow(2).mean()
            loss = loss + torch.relu(logits(pass_i, 0.8, 0.0, prev_dx, depth_t, goal_t) + 3.0).pow(2).mean()
            loss = loss + torch.relu(logits(nose_i, 0.0, base.YAW, prev_dx, depth_t, goal_t) + 3.0).pow(2).mean()
            loss = loss + 4.0 * torch.relu(2.0 - logits(nose_i, 0.8, 0.0, prev_dx, depth_t, goal_t)).pow(2).mean()
            loss = loss + 4.0 * torch.relu(2.0 - logits(hit_i, 0.8, 0.0, prev_dx, real_depth, real_goal)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 or step == 40:
            stats = report(step)
            score = stats["turn"] + stats["open"] + stats["pass"] + stats["nose"] - max(0.0, stats["fwd"] - 0.90) - max(0.0, stats["hit"] - 0.35)
            if best_score is None or score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"OPEN_BEST {step} {score:.3f}", flush=True)
            if ok(stats):
                save_checkpoint(OUT, model.eval(), {"step": step})
                print("OPEN_SAVE", step, flush=True)
                print("OPEN_OUT", OUT, flush=True)
                return 0
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(OUT, model.eval(), {"step": 40, "fallback": True, "score": best_score})
    print("OPEN_SAVE_LAST", best_score, flush=True)
    print("OPEN_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
