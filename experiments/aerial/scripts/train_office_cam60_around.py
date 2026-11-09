#!/usr/bin/env python3
"""Teach the survival head to turn toward the open side of its own depth.

The walker is not changed. On depth maps where the head's left cone is short
and its right cone is open, the right turn is fit above 0.95 and straight
steps are fit below it, so the existing picker turns. Views one turn later
stay drivable. A close nose, the barrel pass, and real hits stay as they are.

Writes consequence_cam60_around.pt. Does not overwrite the pass weights,
consequence_open.pt, or the depth file.
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
from experiments.aerial.rl.consequence_model import depth_input, goal_features, load_checkpoint, save_checkpoint
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation
from office_bands import cones3_forward

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
SIM_NPZ = Path("/data/simworld/run/office_cam60_20261105/sim.npz")
YAW_NPZ = Path("/data/simworld/run/office_cam60_20261105/gap_yaw.npz")
DEPTH = ART / "da3_office_cam60_20261105/depth_office_cone_cam60.pt"
INIT = ART / "indoor_office_cam60_20261105/c_s218/consequence_cam60_pass.pt"
OUT = ART / "indoor_office_cam60_20261105/c_s218/consequence_cam60_around.pt"
YAW = 0.3141592653589793
GOAL_XY = (200.0, -200.0)
STRAIGHT = ((0.35, 0.0), (0.60, 0.0), (0.80, 0.0), (1.00, 0.0), (0.80, 0.04), (0.80, -0.04), (0.80, 0.10), (0.80, -0.10))


def goal_at(x: float, y: float, yaw: float, gx: float, gy: float) -> np.ndarray:
    state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, math.radians(yaw)], np.float64)
    goal = np.array([gx / 100.0, gy / 100.0, 0.52], np.float64)
    return goal_features(state, goal)


def encode_rgb(wrap, rgb: np.ndarray, gt56: np.ndarray, x: float, y: float, yaw: float, gx: float, gy: float):
    wrap.reset()
    pred = np.asarray(wrap.predict_depth(Observation(
        rgb=np.ascontiguousarray(rgb), state=np.zeros(7, np.float32), collided=False,
    )), np.float32)
    if pred.shape[:2] != (224, 224):
        import cv2
        pred = cv2.resize(pred, (224, 224), interpolation=cv2.INTER_LINEAR)
    seen = rgb.sum(axis=-1) > 12
    da56 = np.asarray(col.pool_min(col.hide(pred, seen)), np.float32)
    return (
        depth_input(da56),
        cones3_forward(da56).astype(np.float32),
        cones3_forward(gt56).astype(np.float32),
        goal_at(x, y, yaw, gx, gy),
    )


def grab_yaw(ucv, cam) -> dict[str, np.ndarray]:
    if YAW_NPZ.is_file():
        z = np.load(YAW_NPZ)
        print(f"yaw_cache {len(z['rgb'])}", flush=True)
        return {k: z[k] for k in z.files}
    rgb, gt56, xy = [], [], []
    xs = [-160.0, -128.0, -112.0, -96.0, -80.0, -64.0, -59.0, -48.0, -32.0]
    for x in xs:
        for yaw in (18.0, 36.0):
            col.place(ucv, x, -200.0, yaw)
            col.look(ucv, cam, x, -200.0, yaw)
            packed = col.grab(ucv, cam)
            if packed is None:
                print(f"grab_fail {x:.0f} {yaw:.0f}", flush=True)
                continue
            image, seen, warped = col.warp_pair(*packed)
            gt = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
            g = cones3_forward(gt)
            print(f"yaw x={x:.0f} yaw={yaw:.0f} gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f}", flush=True)
            rgb.append(np.ascontiguousarray(image))
            gt56.append(gt)
            xy.append((x, -200.0, yaw))
    packed = {"rgb": np.stack(rgb), "gt56": np.stack(gt56).astype(np.float32), "xy": np.asarray(xy, np.float32)}
    np.savez(YAW_NPZ, **packed)
    print(f"yaw_saved {len(rgb)}", flush=True)
    return packed


def main() -> int:
    if OUT.resolve() == INIT.resolve():
        raise RuntimeError("refusing to overwrite the pass weights")
    sim = np.load(SIM_NPZ)
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    loop.boxes.save_restore(ucv)
    loop.restore_boxes(ucv)
    try:
        yaw_pack = grab_yaw(ucv, cam)
    finally:
        loop.restore_boxes(ucv)
        print("RESTORED", flush=True)

    wrap = DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    depth, da, gt, goals, kind = [], [], [], [], []
    for i, rgb in enumerate(sim["rgb"]):
        x, y, yaw = (float(v) for v in sim["xy"][i])
        name = str(sim["kind"][i])
        gx, gy = GOAL_XY
        if name == "wall":
            gx, gy = 80.0, -250.0
        elif name == "side":
            gx, gy = 220.0, -170.0
        d, a, g, goal = encode_rgb(wrap, rgb, sim["gt56"][i], x, y, yaw, gx, gy)
        depth.append(d)
        da.append(a)
        gt.append(g)
        goals.append(goal)
        kind.append(name)
    yaw0 = len(depth)
    for i, rgb in enumerate(yaw_pack["rgb"]):
        x, y, yaw = (float(v) for v in yaw_pack["xy"][i])
        d, a, g, goal = encode_rgb(wrap, rgb, yaw_pack["gt56"][i], x, y, yaw, *GOAL_XY)
        depth.append(d)
        da.append(a)
        gt.append(g)
        goals.append(goal)
        kind.append("gap_turned")
        print(
            f"turned x={x:.0f} yaw={yaw:.0f} da={a[0]:.2f},{a[1]:.2f},{a[2]:.2f} gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f}",
            flush=True,
        )
    del wrap
    torch.cuda.empty_cache()

    real_rgb, real_gt = align.load_real()
    real_d, real_c, real_g = align.encode_split(
        DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda"), real_rgb, real_gt,
    )
    torch.cuda.empty_cache()

    dev = torch.device("cuda")
    depth_t = torch.as_tensor(np.stack(depth), device=dev)
    goal_t = torch.as_tensor(np.stack(goals), device=dev)
    da_a = np.stack(da)
    kind_a = np.asarray(kind)
    gap = kind_a == "gap"
    turn = np.flatnonzero(gap & (da_a[:, 1] < 0.85) & (da_a[:, 2] > 2.5) & (da_a[:, 0] > 1.5))
    drive = np.flatnonzero((kind_a == "gap_turned") & (da_a[:, 0] > 2.0))
    nose = np.flatnonzero((kind_a == "front") & (da_a[:, 0] < 0.8) & (da_a[:, 2] > 2.5))
    passed = np.flatnonzero((kind_a == "front_yaw") & (da_a[:, 0] > 2.0))
    print(
        f"turn {turn.tolist()} drive {len(drive)} nose {len(nose)} passed {len(passed)}",
        flush=True,
    )
    if turn.size == 0 or drive.size == 0 or nose.size == 0 or passed.size == 0:
        print("AROUND_EMPTY", flush=True)
        return 1

    real_depth = torch.as_tensor(real_d, device=dev)
    real_goal = torch.as_tensor(real_g, device=dev)
    hit = np.flatnonzero(real_c[:, 0] < 1.15)
    print(f"real_hit {len(hit)}", flush=True)

    turn_i = torch.as_tensor(turn, device=dev, dtype=torch.long)
    drive_i = torch.as_tensor(drive, device=dev, dtype=torch.long)
    nose_i = torch.as_tensor(nose, device=dev, dtype=torch.long)
    pass_i = torch.as_tensor(passed, device=dev, dtype=torch.long)
    hit_i = torch.as_tensor(hit, device=dev, dtype=torch.long)

    model = load_checkpoint(str(INIT), "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def batch_logits(index, dx, dyaw, prev_dx, depth, goal):
        prev = torch.zeros(index.shape[0], 4, device=dev)
        prev[:, 0] = prev_dx
        return align.logits(model, depth[index], goal[index], prev, dx, dyaw)

    def surv_of(z: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.sigmoid(z)

    def report(step: int) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            right = surv_of(batch_logits(turn_i, 0.0, YAW, 0.8, depth_t, goal_t))
            fwd = torch.stack([
                surv_of(batch_logits(turn_i, dx, dyaw, 0.8, depth_t, goal_t)) for dx, dyaw in STRAIGHT
            ])
            gone = surv_of(batch_logits(drive_i, 0.8, 0.0, 0.0, depth_t, goal_t))
            nose_r = surv_of(batch_logits(nose_i, 0.0, YAW, 0.0, depth_t, goal_t))
            nose_f = surv_of(batch_logits(nose_i, 0.8, 0.0, 0.0, depth_t, goal_t))
            hold = surv_of(batch_logits(pass_i, 0.8, 0.0, 0.0, depth_t, goal_t))
            hits = surv_of(batch_logits(hit_i, 0.8, 0.0, 0.0, real_depth, real_goal)) if hit_i.numel() else torch.zeros(1, device=dev)
        stats = {
            "turn": float(right.min()),
            "fwd": float(fwd.max()),
            "drive": float(gone.min()),
            "nose": float(nose_r.min()),
            "nose_fwd": float(nose_f.max()),
            "pass": float(hold.min()),
            "hit": float(hits.max()),
        }
        model.train()
        print(
            f"step {step} turn {stats['turn']:.3f} fwd {stats['fwd']:.3f} drive {stats['drive']:.3f} "
            f"nose {stats['nose']:.3f} nose_fwd {stats['nose_fwd']:.3f} pass {stats['pass']:.3f} hit {stats['hit']:.3f}",
            flush=True,
        )
        return stats

    def ok(stats: dict[str, float]) -> bool:
        return (
            stats["turn"] >= 0.95
            and stats["fwd"] <= 0.90
            and stats["drive"] >= 0.95
            and stats["nose"] >= 0.95
            and stats["nose_fwd"] <= 0.35
            and stats["pass"] >= 0.95
            and stats["hit"] <= 0.35
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    best_score = None
    best_state = None
    report(0)
    for step in range(1, 81):
        model.train()
        loss = torch.zeros((), device=dev)
        for prev_dx in (0.0, 0.8):
            zr = batch_logits(turn_i, 0.0, YAW, prev_dx, depth_t, goal_t)
            zl = batch_logits(turn_i, 0.0, -YAW, prev_dx, depth_t, goal_t)
            loss = loss + torch.relu(zr + 3.0).pow(2).mean()
            loss = loss + torch.relu(2.0 - zl).pow(2).mean()
            for dx, dyaw in STRAIGHT:
                zf = batch_logits(turn_i, dx, dyaw, prev_dx, depth_t, goal_t)
                loss = loss + torch.relu(-2.2 - zf).pow(2).mean()
            zd = batch_logits(drive_i, 0.8, 0.0, prev_dx, depth_t, goal_t)
            loss = loss + torch.relu(zd + 3.0).pow(2).mean()
            zn = batch_logits(nose_i, 0.0, YAW, prev_dx, depth_t, goal_t)
            zfn = batch_logits(nose_i, 0.8, 0.0, prev_dx, depth_t, goal_t)
            loss = loss + torch.relu(zn + 3.0).pow(2).mean()
            loss = loss + 4.0 * torch.relu(2.0 - zfn).pow(2).mean()
            zp = batch_logits(pass_i, 0.8, 0.0, prev_dx, depth_t, goal_t)
            loss = loss + torch.relu(zp + 3.0).pow(2).mean()
            if hit_i.numel():
                zh = batch_logits(hit_i, 0.8, 0.0, prev_dx, real_depth, real_goal)
                loss = loss + 4.0 * torch.relu(2.0 - zh).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0 or step == 80:
            stats = report(step)
            score = (
                stats["turn"]
                + min(stats["drive"], 1.0)
                + min(stats["nose"], 1.0)
                + min(stats["pass"], 1.0)
                - max(0.0, stats["fwd"] - 0.90)
                - max(0.0, stats["nose_fwd"] - 0.35)
                - max(0.0, stats["hit"] - 0.35)
            )
            if best_score is None or score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"AROUND_BEST {step} {score:.3f}", flush=True)
            if ok(stats):
                save_checkpoint(OUT, model.eval(), {"step": step})
                print("AROUND_SAVE", step, flush=True)
                print("AROUND_OUT", OUT, flush=True)
                return 0
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(OUT, model.eval(), {"step": 80, "fallback": True, "score": best_score})
    print("AROUND_SAVE_LAST", best_score, flush=True)
    print("AROUND_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
