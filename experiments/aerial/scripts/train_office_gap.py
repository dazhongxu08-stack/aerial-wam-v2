#!/usr/bin/env python3
"""Keep the office head from spinning in the open gap.

The 3/4 run turned right in the middle of a clear corridor, then stopped on
the side face. On the straight gap, the next forward step stays safe and a
right spin does not. Wall face, wall approach, and the side crate stay as
they were. Far, wall-depth, and the car are not touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_office_cons_loop as loop  # noqa: E402
from experiments.aerial.rl.consequence_model import (  # noqa: E402
    depth_input,
    goal_features,
    load_checkpoint,
    save_checkpoint,
    survival,
)
from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402
from experiments.aerial.rl.env.obs import Observation  # noqa: E402
from office_bands import cones3_forward  # noqa: E402

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
DEPTH = ART / "da3_office_wall_20261104/depth_office_cone_wall.pt"
INIT = ART / "indoor_office_gap_20261104/c_s218/consequence_best.pt"
OUT = ART / "indoor_office_gap_20261104/c_s218"
GAP = Path("/data/simworld/run/office_gap_20261104/gap.npz")
HOLD = Path("/data/simworld/run/office_approach_20261104/anchors.npz")
YAW = 0.3141592653589793
STEPS = 20


def grab() -> None:
    xs = list(range(-168, 81, 8))
    xs += [-106, -62]
    poses = sorted(set(xs))
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(loop.col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    loop.restore_boxes(ucv)
    depth_model = DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    depth, cones, goals, xy = [], [], [], []
    for x in poses:
        loop.col.place(ucv, x, -200.0, 0.0)
        loop.col.look(ucv, cam, x, -200.0, 0.0)
        packed = loop.col.grab(ucv, cam)
        if packed is None:
            raise RuntimeError(f"grab failed {x}")
        image, seen, warped = loop.col.warp_pair(*packed)
        gt56 = loop.col.pool_min(loop.col.hide(warped, seen))
        depth_model.reset()
        pred = np.asarray(depth_model.predict_depth(Observation(
            rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False,
        )), np.float32)
        if pred.shape[:2] != warped.shape[:2]:
            import cv2
            pred = cv2.resize(pred, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
        da56 = loop.col.pool_min(loop.col.hide(pred, seen))
        g = cones3_forward(gt56)
        d = cones3_forward(da56)
        state = np.array([x / 100.0, -2.0, 0.52, 0, 0, 0, 0.0], np.float64)
        goal = np.array([2.0, -2.0, 0.52], np.float64)
        depth.append(depth_input(da56))
        cones.append(np.asarray(g, np.float32))
        goals.append(goal_features(state, goal))
        xy.append((x, -200.0))
        print(f"gap x={x:.0f} gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f} da={d[0]:.2f},{d[1]:.2f},{d[2]:.2f}", flush=True)
    GAP.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        GAP,
        depth=np.stack(depth).astype(np.float32),
        cones=np.stack(cones).astype(np.float32),
        goal=np.stack(goals).astype(np.float32),
        xy=np.asarray(xy, np.float32),
    )
    del depth_model
    torch.cuda.empty_cache()


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


def surv_of(model, depth, goal, prev, dx, dyaw):
    z = logits(model, depth, goal, prev, dx, dyaw)
    return (1.0 - torch.sigmoid(z)).detach()


def main() -> int:
    if not GAP.is_file():
        grab()
    gap = np.load(GAP)
    hold = np.load(HOLD)
    dev = torch.device("cuda")
    model = load_checkpoint(str(INIT), "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    gd, gg, gp = pack(gap["depth"], gap["goal"], dev)
    hd, hg, hp = pack(hold["depth"], hold["goal"], dev)
    gc = torch.as_tensor(gap["cones"], dtype=torch.float32, device=dev)
    hc = torch.as_tensor(hold["cones"], dtype=torch.float32, device=dev)
    # Straight corridor: do not spin, do go forward.
    gap_open = gc.min(dim=1).values > 2.5
    face = hc[:, 0] < 1.0
    side = (hc[:, 1] < 1.0) & (hc[:, 0] > 2.0) & (hc[:, 2] > 1.0)
    wall_open = (hc[:, 0] > 4.0) & (hc[:, 1] > 2.0) & (hc[:, 2] > 2.0)
    print(
        f"gap {len(gc)} open {int(gap_open.sum())} face {int(face.sum())} "
        f"side {int(side.sum())} wall_open {int(wall_open.sum())}",
        flush=True,
    )

    def report(step: int) -> tuple[float, float, float, float]:
        model.eval()
        with torch.no_grad():
            gf = surv_of(model, gd, gg, gp, 0.8, 0.0)
            gs = surv_of(model, gd, gg, gp, 0.0, YAW)
            wf = surv_of(model, hd, hg, hp, 0.8, 0.0)
        model.eval()
        gap_fwd = float(gf[gap_open].min()) if bool(gap_open.any()) else 0.0
        gap_med = float(gf[gap_open].median()) if bool(gap_open.any()) else 0.0
        gap_spin = float(gs[gap_open].max()) if bool(gap_open.any()) else 1.0
        face_fwd = float(wf[face].max()) if bool(face.any()) else 1.0
        side_fwd = float(wf[side].max()) if bool(side.any()) else 1.0
        xy = gap_np["xy"]
        named = []
        for want in (-160, -106, -62, 0, 40):
            hit = int(np.argmin(np.abs(xy[:, 0] - want)))
            named.append(f"x{int(xy[hit, 0])}={float(gf[hit]):.2f}")
        print(
            f"step {step} gap_fwd_min {gap_fwd:.3f} med {gap_med:.3f} gap_spin_max {gap_spin:.3f} "
            f"face_fwd_max {face_fwd:.3f} side_fwd_max {side_fwd:.3f} {' '.join(named)}",
            flush=True,
        )
        return gap_fwd, gap_spin, face_fwd, side_fwd

    global gap_np
    gap_np = gap
    OUT.mkdir(parents=True, exist_ok=True)
    report(0)
    best = None
    # The GRU backward refuses eval mode. Reports above stay in eval.
    model.train()
    for step in range(1, STEPS + 1):
        model.train()
        z_fwd = logits(model, gd, gg, gp, 0.8, 0.0)
        z_spin = logits(model, gd, gg, gp, 0.0, YAW)
        z_hold = logits(model, hd, hg, hp, 0.8, 0.0)
        z_yaw = logits(model, hd, hg, hp, 0.0, YAW)
        # Square the violations so one frame stuck near 0 is not averaged away.
        loss = torch.relu(z_fwd[gap_open] + 3.0).pow(2).mean()
        loss = loss + torch.relu(2.0 - z_spin[gap_open]).pow(2).mean()
        loss = loss + 4.0 * torch.relu(2.0 - z_hold[face]).pow(2).mean()
        loss = loss + torch.relu(z_hold[wall_open] + 2.0).pow(2).mean()
        if bool(side.any()):
            loss = loss + 4.0 * torch.relu(2.0 - z_hold[side]).pow(2).mean()
            loss = loss + 4.0 * torch.relu(z_yaw[side] + 2.0).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == STEPS:
            gap_fwd, gap_spin, face_fwd, side_fwd = report(step)
            ok = gap_fwd >= 0.95 and gap_spin <= 0.5 and face_fwd <= 0.2 and side_fwd <= 0.3
            save_checkpoint(OUT / "consequence_last.pt", model, {"step": step})
            if ok and (best is None or gap_fwd > best):
                best = gap_fwd
                save_checkpoint(OUT / "consequence_best.pt", model, {"step": step, "gap_fwd": gap_fwd})
                print("SAVE", step, flush=True)
    if best is None:
        save_checkpoint(OUT / "consequence_best.pt", model, {"step": STEPS, "fallback": True})
        print("SAVE_LAST", flush=True)
    print("BEST", best, "OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
