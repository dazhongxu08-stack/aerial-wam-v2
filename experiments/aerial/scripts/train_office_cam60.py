#!/usr/bin/env python3
"""Retrain office depth and consequence for the 60 cm, 30° camera.

The previous align depth and the open-side consequence stay where they are.
Sim frames are grabbed with the current camera, then the decoder is fit to
those frames plus the existing real frames. Consequence starts from the
open-side weights and is fit to the new depth maps.
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
import run_office_front_scene as front
import train_office_align as align
import train_office_da3_far as far
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
SIM_NPZ = Path("/data/simworld/run/office_cam60_20261105/sim.npz")
DEPTH_INIT = ART / "da3_office_align_20261104/depth_office_cone_align.pt"
DEPTH_OUT = ART / "da3_office_cam60_20261105/depth_office_cone_cam60.pt"
CONS_INIT = ART / "indoor_office_align_20261104/c_s218/consequence_open.pt"
CONS_OUT = ART / "indoor_office_cam60_20261105/c_s218/consequence_cam60.pt"
YAW = 0.3141592653589793
DEPTH_STEPS = 60
CONS_STEPS = 160


def poses():
    rows = []
    for x in np.arange(-480.0, -150.0, 16.0):
        rows.append((float(x), -200.0, 0.0, "aisle"))
    for x in np.arange(-520.0, -330.0, 16.0):
        rows.append((float(x), -250.0, 0.0, "wall"))
    for x in np.arange(-160.0, 90.0, 16.0):
        rows.append((float(x), -200.0, 0.0, "gap"))
    return rows


def grab_pose(cam, ucv, x, y, yaw, kind):
    col.place(ucv, x, y, yaw)
    col.look(ucv, cam, x, y, yaw)
    packed = col.grab(ucv, cam)
    if packed is None:
        print(f"grab_fail {kind} {x:.0f}", flush=True)
        return None
    image, seen, warped = col.warp_pair(*packed)
    from office_bands import cones3_forward
    gt = col.pool_min(col.hide(warped, seen)).astype(np.float32)
    g = cones3_forward(gt)
    print(
        f"{kind} x={x:.0f} y={y:.0f} yaw={yaw:.0f} gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f}",
        flush=True,
    )
    return np.ascontiguousarray(image), gt


def collect_sim() -> dict[str, np.ndarray]:
    if SIM_NPZ.is_file():
        z = np.load(SIM_NPZ)
        print(f"sim_cache {len(z['rgb'])}", flush=True)
        return {k: z[k] for k in z.files}
    if col.CAM_Z != 60.0 or col.PITCH != -30.0:
        raise RuntimeError(f"camera is {col.CAM_Z} {col.PITCH}, want 60 / -30")
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    loop.boxes.save_restore(ucv)
    loop.restore_boxes(ucv)
    rgb, gt56, kind, xy = [], [], [], []

    def take(x, y, yaw, name):
        got = grab_pose(cam, ucv, float(x), float(y), float(yaw), name)
        if got is None:
            return
        image, gt = got
        rgb.append(image)
        gt56.append(gt)
        kind.append(name)
        xy.append((float(x), float(y), float(yaw)))

    try:
        for x, y, yaw, name in poses():
            take(x, y, yaw, name)
        loop.boxes.place_boxes(ucv)
        for x in np.arange(-420.0, -250.0, 15.0):
            take(x, -170.0, 0.0, "side")
        loop.restore_boxes(ucv)
        front.place_front_barrel(ucv)
        for x in np.arange(-170.0, 20.0, 12.0):
            take(x, -200.0, 0.0, "front")
            take(x, -200.0, 18.0, "front_yaw")
    finally:
        loop.restore_boxes(ucv)
        print("cam60_restored", flush=True)
    if len(rgb) < 40:
        raise RuntimeError(f"sim sample too small: {len(rgb)}")
    packed = {
        "rgb": np.stack(rgb),
        "gt56": np.stack(gt56).astype(np.float32),
        "kind": np.asarray(kind),
        "xy": np.asarray(xy, np.float32),
    }
    SIM_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SIM_NPZ, **packed)
    print(f"sim_saved {len(rgb)} {SIM_NPZ}", flush=True)
    return packed


def train_depth(sim) -> None:
    if DEPTH_OUT.resolve() == DEPTH_INIT.resolve():
        raise RuntimeError("refusing to overwrite the align depth")
    real_rgb, real_gt = align.load_real()
    sim_rgb, sim_gt = sim["rgb"], sim["gt56"].astype(np.float32)
    print(f"real {len(real_rgb)} sim {len(sim_rgb)} hz56 {far.HZ56} cam {col.CAM_Z} {col.PITCH}", flush=True)
    wrap = DepthMinPredictor.from_checkpoint(str(DEPTH_INIT), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    model = wrap._model
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    model.encoder.eval()
    src = torch.load(str(DEPTH_INIT), map_location="cpu", weights_only=False)
    real0, _ = align.domain_score(model, real_rgb, real_gt, "real before")
    sim0, _ = align.domain_score(model, sim_rgb, sim_gt, "sim before")
    opt = torch.optim.AdamW((p for p in model.decoder.parameters() if p.requires_grad), lr=3e-6)
    rng = np.random.default_rng(4)
    best = real0 + sim0
    best_real = real0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for step in range(1, DEPTH_STEPS + 1):
        model.decoder.train()
        ri = rng.choice(len(real_rgb), size=4, replace=False)
        si = rng.choice(len(sim_rgb), size=4, replace=False)
        batch_rgb = np.concatenate([real_rgb[ri], sim_rgb[si]])
        batch_gt = np.concatenate([real_gt[ri], sim_gt[si]])
        pred = far.forward_maps(model, batch_rgb)
        gt_t = torch.as_tensor(batch_gt, dtype=torch.float32, device="cuda")
        loss = far.far_near_loss(pred[:4], gt_t[:4]) + far.far_near_loss(pred[4:], gt_t[4:])
        loss = loss + align.side_loss(pred[:4], gt_t[:4]) + align.side_loss(pred[4:], gt_t[4:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == DEPTH_STEPS:
            real_s, _ = align.domain_score(model, real_rgb, real_gt, f"real step {step}")
            sim_s, _ = align.domain_score(model, sim_rgb, sim_gt, f"sim step {step}")
            total = real_s + sim_s
            if total < best and real_s <= real0 + 0.05:
                best = total
                best_real = real_s
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                far.save_ckpt(model, src, DEPTH_OUT, {"step": step, "real": real_s, "sim": sim_s})
                print(f"DEPTH_SAVE {step} real {real_s:.3f} sim {sim_s:.3f}", flush=True)
    if not DEPTH_OUT.is_file():
        model.load_state_dict(best_state)
        far.save_ckpt(model, src, DEPTH_OUT, {"step": DEPTH_STEPS, "fallback": True, "real": best_real})
        print("DEPTH_SAVE_LAST", flush=True)
    print(f"DEPTH_BEST real0 {real0:.3f} sim0 {sim0:.3f} OUT {DEPTH_OUT}", flush=True)


def train_cons(sim) -> None:
    if CONS_OUT.resolve() == CONS_INIT.resolve():
        raise RuntimeError("refusing to overwrite the open consequence")
    wrap = DepthMinPredictor.from_checkpoint(str(DEPTH_OUT), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    real_rgb, real_gt = align.load_real()
    real_d, real_c, real_g = align.encode_split(wrap, real_rgb, real_gt)
    sim_d, sim_c, sim_g = align.encode_split(wrap, sim["rgb"], sim["gt56"].astype(np.float32))
    del wrap
    torch.cuda.empty_cache()
    dev = torch.device("cuda")
    splits = {
        "real": align.pack(real_d, real_g, dev) + (torch.as_tensor(real_c, device=dev),),
        "sim": align.pack(sim_d, sim_g, dev) + (torch.as_tensor(sim_c, device=dev),),
    }

    def masks_of(cones):
        fwd, left, right = cones[:, 0], cones[:, 1], cones[:, 2]
        side_l = (left < 1.0) & (fwd > 2.0) & (right > 1.2)
        side_r = (right < 1.0) & (fwd > 2.0) & (left > 1.2)
        return {
            "safe": (fwd > 2.0) & ~(side_l | side_r),
            "open": (fwd > 2.5) & (left > 1.5) & (right > 1.5),
            "hit": fwd < 1.15,
            "side_l": side_l,
            "side_r": side_r,
            "detour_l": (fwd < 1.8) & (fwd > 0.55) & (left > 2.2) & (right < left - 0.4),
            "detour_r": (fwd < 1.8) & (fwd > 0.55) & (right > 2.2) & (left < right - 0.4),
            "shut": (fwd < 1.2) & (left < 1.6) & (right < 1.6),
        }

    masks = {}
    for name, (_depth, _goal, _prev, cones) in splits.items():
        masks[name] = masks_of(cones)
        m = masks[name]
        print(
            f"{name} n {len(cones)} open {int(m['open'].sum())} hit {int(m['hit'].sum())} "
            f"side {int((m['side_l'] | m['side_r']).sum())} "
            f"detour {int((m['detour_l'] | m['detour_r']).sum())} shut {int(m['shut'].sum())}",
            flush=True,
        )
    from experiments.aerial.rl.consequence_model import load_checkpoint, save_checkpoint
    CONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(str(CONS_INIT), "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    def surv(depth, goal, prev, dx, dyaw):
        z = align.logits(model, depth, goal, prev, dx, dyaw)
        return 1.0 - torch.sigmoid(z)

    def report(step: int):
        model.eval()
        stats = {}
        with torch.no_grad():
            for name, (depth, goal, prev, _cones) in splits.items():
                sf = surv(depth, goal, prev, 0.8, 0.0)
                sl = surv(depth, goal, prev, 0.0, -YAW)
                sr = surv(depth, goal, prev, 0.0, YAW)
                m = masks[name]
                away = []
                if bool(m["side_l"].any()):
                    away.append(sr[m["side_l"]])
                if bool(m["side_r"].any()):
                    away.append(sl[m["side_r"]])
                if bool(m["detour_l"].any()):
                    away.append(sl[m["detour_l"]])
                if bool(m["detour_r"].any()):
                    away.append(sr[m["detour_r"]])
                side = m["side_l"] | m["side_r"]
                row = {
                    "open": float(sf[m["open"]].min()) if bool(m["open"].any()) else 1.0,
                    "hit": float(sf[m["hit"]].max()) if bool(m["hit"].any()) else 0.0,
                    "side": float(sf[side].max()) if bool(side.any()) else 0.0,
                    "away": float(torch.cat(away).min()) if away else 1.0,
                    "shut": float(torch.maximum(torch.maximum(sf, sl), sr)[m["shut"]].max()) if bool(m["shut"].any()) else 0.0,
                }
                stats[name] = row
                print(
                    f"step {step} {name} open {row['open']:.3f} hit {row['hit']:.3f} "
                    f"side {row['side']:.3f} away {row['away']:.3f} shut {row['shut']:.3f}",
                    flush=True,
                )
        return stats

    def ok(stats) -> bool:
        for row in stats.values():
            if row["open"] < 0.95 or row["hit"] > 0.25 or row["side"] > 0.35:
                return False
            if row["away"] < 0.95 or row["shut"] > 0.35:
                return False
        return True

    report(0)
    saved = False
    for step in range(1, CONS_STEPS + 1):
        model.train()
        loss = torch.zeros((), device=dev)
        for name, (depth, goal, prev, _cones) in splits.items():
            m = masks[name]
            z_fwd = align.logits(model, depth, goal, prev, 0.8, 0.0)
            z_left = align.logits(model, depth, goal, prev, 0.0, -YAW)
            z_right = align.logits(model, depth, goal, prev, 0.0, YAW)
            if bool(m["safe"].any()):
                loss = loss + torch.relu(z_fwd[m["safe"]] + 2.0).pow(2).mean()
            if bool(m["open"].any()):
                loss = loss + torch.relu(z_fwd[m["open"]] + 3.0).pow(2).mean()
                loss = loss + torch.relu(2.0 - z_left[m["open"]]).pow(2).mean()
                loss = loss + torch.relu(2.0 - z_right[m["open"]]).pow(2).mean()
            if bool(m["hit"].any()):
                loss = loss + 4.0 * torch.relu(2.0 - z_fwd[m["hit"]]).pow(2).mean()
            side = m["side_l"] | m["side_r"]
            if bool(side.any()):
                loss = loss + 4.0 * torch.relu(2.0 - z_fwd[side]).pow(2).mean()
            if bool(m["side_l"].any()):
                loss = loss + 4.0 * torch.relu(z_right[m["side_l"]] + 2.0).pow(2).mean()
            if bool(m["side_r"].any()):
                loss = loss + 4.0 * torch.relu(z_left[m["side_r"]] + 2.0).pow(2).mean()
            if bool(m["detour_l"].any()):
                loss = loss + 4.0 * torch.relu(z_left[m["detour_l"]] + 2.0).pow(2).mean()
                loss = loss + 4.0 * torch.relu(2.0 - z_fwd[m["detour_l"]]).pow(2).mean()
                loss = loss + torch.relu(2.0 - z_right[m["detour_l"]]).pow(2).mean()
            if bool(m["detour_r"].any()):
                loss = loss + 4.0 * torch.relu(z_right[m["detour_r"]] + 2.0).pow(2).mean()
                loss = loss + 4.0 * torch.relu(2.0 - z_fwd[m["detour_r"]]).pow(2).mean()
                loss = loss + torch.relu(2.0 - z_left[m["detour_r"]]).pow(2).mean()
            if bool(m["shut"].any()):
                loss = loss + 4.0 * torch.relu(2.0 - z_fwd[m["shut"]]).pow(2).mean()
                loss = loss + 4.0 * torch.relu(2.0 - z_left[m["shut"]]).pow(2).mean()
                loss = loss + 4.0 * torch.relu(2.0 - z_right[m["shut"]]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == CONS_STEPS:
            stats = report(step)
            save_checkpoint(CONS_OUT.with_name("consequence_cam60_last.pt"), model.eval(), {"step": step})
            model.train()
            if ok(stats):
                save_checkpoint(CONS_OUT, model.eval(), {"step": step})
                model.train()
                saved = True
                print("CONS_SAVE", step, flush=True)
                break
    if not saved:
        save_checkpoint(CONS_OUT, model.eval(), {"step": CONS_STEPS, "fallback": True})
        print("CONS_SAVE_LAST", flush=True)
    print("CONS_OUT", CONS_OUT, flush=True)


def main() -> int:
    print(f"camera {col.CAM_Z} {col.PITCH}", flush=True)
    sim = collect_sim()
    train_depth(sim)
    train_cons(sim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
