#!/usr/bin/env python3
"""Teach the depth decoder the dark live aisle, without moving the bright holdout.

The stored jpeg at the gap start already reads 2.47 m. The live frame is much
darker and reads about 1.5 m. Floor pixels (GT >= 2 m) are pushed back out.
Pixels whose GT is already under 1.6 m stay close.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
spec = importlib.util.spec_from_file_location("ft", "/data/simworld/run/finetune_office_da3_cone.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

import collect_office_train as col
import sim_verify_gap as gap
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

CKPT = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt")
CONS = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts/indoor_office_mix_angle_f_20261102/c_s218/consequence_best.pt")
OUT = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_cone_live_20261102/depth_office_cone_yaw.pt")
ANGLE = Path("/data/simworld/run/indoor_office_mix")
FAR = 30.0


def poses() -> list[tuple[str, float, float, float]]:
    rows = [("gap", x, -200.0, 0.0) for x in range(-180, 81, 20)]
    rows += [
        ("west", -520.0, -250.0, 0.0),
        ("east", -180.0, -450.0, 180.0),
        ("yaw", -170.0, -200.0, 18.0),
        ("south", -170.0, -250.0, 0.0),
        ("north", -170.0, -150.0, 0.0),
    ]
    return rows


def grab_live(ucv, cam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col.req(ucv, "vrun r.DefaultFeature.AutoExposure 0")
    col.req(ucv, "vrun r.ExposureOffset 8")
    rgb_l, gt_l, name_l = [], [], []
    for name, x, y, yaw in poses():
        col.place(ucv, x, y, yaw)
        col.look(ucv, cam, x, y, yaw)
        packed = col.grab(ucv, cam)
        if packed is None:
            print("grab fail", name, x, y, flush=True)
            continue
        rgb, depth = packed
        image, seen, warped = col.warp_pair(rgb, depth)
        gt = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
        rgb_l.append(image)
        gt_l.append(gt)
        name_l.append(f"{name}:{x:.0f},{y:.0f},{yaw:.0f}")
        c = ft.cones_np(gt)
        print(
            f"live {name_l[-1]} mean {float(image.mean()):.1f} "
            f"gt {float(c[0]):.2f},{float(c[1]):.2f},{float(c[2]):.2f}",
            flush=True,
        )
    return np.stack(rgb_l), np.stack(gt_l), np.asarray(name_l)


def live_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Push only the forward-cone floor outward. Getting farther than GT costs more."""
    h, w = pred.shape[-2:]
    r0, r1 = h // 4, h - h // 4
    c0, c1 = w // 4, w - w // 4
    p = pred[:, r0:r1].clamp(0.05, FAR - 1)
    g = gt[:, r0:r1]
    valid = torch.isfinite(g) & (g > 0.05) & (g < FAR - 1)
    forward = torch.zeros_like(valid)
    forward[:, :, c0:c1] = True
    floor = valid & forward & (g >= 2.0) & (p < 2.0)
    push = floor.float() * torch.relu(torch.log(p.new_tensor(2.0)) - torch.log(p))
    # A global scale-up makes side barrels and the thin face farther than GT.
    farther = valid.float() * torch.relu(torch.log(p) - torch.log(g.clamp(min=0.05)))
    return (3.0 * push.sum() + 8.0 * farther.sum()) / valid.float().sum().clamp(min=1)


def cones_batch(model, rgb: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(rgb), 8):
        with torch.no_grad():
            pred = ft.forward_maps(model, rgb[i : i + 8]).detach().float().cpu().numpy()
        out.append(np.stack([ft.cones_np(p) for p in pred]))
    return np.concatenate(out)


def straight_surv(cons, depth56: np.ndarray) -> float:
    state = np.array([-1.6, -2.0, 0.52, 0, 0, 0, 0.0])
    goal = np.array([0.2, -2.0, 0.52])
    action = np.zeros((1, 10, 4), np.float32)
    action[0, :, 0] = 0.8
    surv, _ = gap.score([cons], depth56, state, goal, action, np.zeros(4, np.float32))
    return float(surv[0])


def main() -> None:
    ft.start_rss_watch()
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    col.req(ucv, "vrun r.Fog 0")
    col.req(ucv, "vrun r.VolumetricFog 0")
    col.req(ucv, "vset /object/ExponentialHeightFog_1/hide")
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.ensure_dog(ucv)
    rgb, gt, names = grab_live(ucv, cam)
    gap_i = [i for i, n in enumerate(names) if n.startswith("gap:-160")]
    if not gap_i:
        gap_i = [0]
    print("captured", len(rgb), "rss", f"{ft.rss_mb():.0f}", flush=True)
    train_frames, hold_frames = ft.index_episodes()
    angle_train, _angle_hold = ft.index_angle(ANGLE)
    loader = ft.FrameLoader(None, ANGLE)
    open_i = [i for i, ref in enumerate(train_frames) if ref[3][0] >= 2.2]
    near_i = [i for i, ref in enumerate(train_frames) if ref[3][0] < 1.2]
    side_i = [
        i for i, ref in enumerate(angle_train)
        if len(ref[3]) >= 6 and min(ref[3][4], ref[3][5]) < 1.3
    ]
    print(f"stored open {len(open_i)} near {len(near_i)} side {len(side_i)}", flush=True)
    wrapper = DepthMinPredictor.from_checkpoint(str(CKPT), device="cuda")
    model = wrapper._model
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    model.eval()
    src_payload = torch.load(str(CKPT), map_location="cpu")
    opt = torch.optim.AdamW((p for p in model.decoder.parameters() if p.requires_grad), lr=2e-7)
    rng = np.random.default_rng(1)
    hold_open = [ref for ref in hold_frames if ref[3][0] >= 2.2]
    hold_near = [ref for ref in hold_frames if ref[3][0] < 1.2]
    rng_h = np.random.default_rng(2)
    hold_sample = [hold_open[int(i)] for i in rng_h.choice(len(hold_open), size=min(80, len(hold_open)), replace=False)]
    hold_sample += hold_near[:40]
    before = cones_batch(model, rgb)
    base_side = before[gap_i[0], 1:3].copy()
    _base_rlow, base_under = ft.show_turn(model, loader)
    print(
        "before live",
        names[gap_i[0]],
        np.round(before[gap_i[0], :3], 2),
        "gt",
        np.round(ft.cones_np(gt[gap_i[0]])[:3], 2),
        "under",
        f"{base_under:.3f}",
        flush=True,
    )
    best = None
    best_state = None
    for step in range(24):
        live_pick = rng.choice(len(rgb), size=4, replace=False)
        stored = []
        stored += [train_frames[int(i)] for i in rng.choice(open_i, size=2, replace=False)]
        if near_i:
            stored += [train_frames[int(i)] for i in rng.choice(near_i, size=2, replace=False)]
        if side_i:
            stored += [angle_train[int(i)] for i in rng.choice(side_i, size=2, replace=False)]
        srgb, _c, sgt = loader.batch(stored, with_depth=True)
        batch_rgb = np.concatenate([rgb[live_pick], srgb], axis=0)
        batch_gt = np.concatenate([gt[live_pick], sgt], axis=0)
        pred = ft.forward_maps(model, batch_rgb)
        gt_t = torch.tensor(batch_gt, dtype=torch.float32, device=pred.device)
        loss = live_loss(pred[:4], gt_t[:4]) + live_loss(pred[4:], gt_t[4:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        loss_v = float(loss.detach())
        del pred, gt_t, loss
        if step % 4 != 0 and step + 1 != 24:
            continue
        now = cones_batch(model, rgb)
        g = gap_i[0]
        da = now[g]
        truth = ft.cones_np(gt[g])
        hold_rows = []
        ordered = sorted(hold_sample, key=lambda ref: (ref[0], ref[1], ref[2]))
        for i in range(0, len(ordered), 8):
            chunk = ordered[i : i + 8]
            crgb, _cc = loader.batch(chunk)
            with torch.no_grad():
                maps = ft.forward_maps(model, crgb).detach().float().cpu().numpy()
            for j, ref in enumerate(chunk):
                hold_rows.append((float(ref[3][0]), float(ft.cones_np(maps[j])[0])))
        open_da = [a for g0, a in hold_rows if g0 >= 2.2]
        near_da = [a for g0, a in hold_rows if g0 < 1.2]
        open_med = float(np.median(open_da)) if open_da else 0.0
        near_med = float(np.median(near_da)) if near_da else 9.0
        _rlow, under = ft.show_turn(model, loader)
        side_ok = float(da[1]) <= float(base_side[0]) + 0.08 and float(da[2]) <= float(base_side[1]) + 0.08
        gates = (
            2.05 <= open_med <= 3.2
            and 0.75 <= near_med <= 1.3
            and float(da[0]) >= 2.0
            and side_ok
            and under + 1e-6 >= base_under - 0.002
        )
        print(
            f"step {step} loss {loss_v:.4f} "
            f"gap da {float(da[0]):.2f},{float(da[1]):.2f},{float(da[2]):.2f} "
            f"open {open_med:.2f} near {near_med:.2f} under {under:.3f} rss {ft.rss_mb():.0f}",
            flush=True,
        )
        if gates and (best is None or float(da[0]) > best):
            best = float(da[0])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ft.save_ckpt(model, src_payload, OUT, {"live_gap_da": float(da[0]), "open": open_med})
            print("  saved", flush=True)
        elif not gates:
            print("  gate hold", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("NO_SAVE", flush=True)
    # Fresh grab, same exposure, no thin-strip.
    image, seen, warped = None, None, None
    col.req(ucv, "vrun r.DefaultFeature.AutoExposure 0")
    col.req(ucv, "vrun r.ExposureOffset 8")
    col.place(ucv, -160.0, -200.0, 0.0)
    col.look(ucv, cam, -160.0, -200.0, 0.0)
    rgb2, depth2 = col.grab(ucv, cam)
    image, seen, warped = col.warp_pair(rgb2, depth2)
    gt2 = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
    with torch.no_grad():
        pred = ft.forward_maps(model, image[None]).detach().float().cpu().numpy()[0]
    print("regrab gt", np.round(ft.cones_np(gt2)[:3], 2), "da", np.round(ft.cones_np(pred)[:3], 2), flush=True)
    del wrapper
    torch.cuda.empty_cache()
    cons = gap.load_checkpoint(str(CONS), "cuda")
    print("straight surv", f"{straight_surv(cons, pred):.3f}", flush=True)
    print("OUT", OUT, "exists", OUT.is_file(), flush=True)


if __name__ == "__main__":
    main()
