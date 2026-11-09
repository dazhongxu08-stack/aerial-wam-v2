#!/usr/bin/env python3
"""Separate open corridor from the barrel face in the aligned depth head.

Decoder plus the last two encoder blocks. Open corridor cones are pulled up
to simulator depth. Barrel-face frames that already read under 1.3 m are not
allowed to grow, and side-crate frames are held short. Real frames stay in
every step. Writes a new file only when the corridor gets longer and the
face does not.
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
import train_office_da3_far as far
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from office_bands import cones3_forward

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "da3_office_align_20261104/depth_office_cone_align.pt"
OUT = ART / "da3_office_align_20261104/depth_office_cone_align_b.pt"
CACHE = Path("/data/simworld/run/office_align_depthb_20261104/poses_v2.npz")
STEPS = 20


def grab(ucv, cam, x: float, y: float, yaw: float):
    col.place(ucv, x, y, yaw)
    col.look(ucv, cam, x, y, yaw)
    packed = col.grab(ucv, cam)
    if packed is None:
        return None
    rgb, depth = packed
    image, seen, warped = col.warp_pair(rgb, depth)
    gt56 = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
    return np.ascontiguousarray(image), gt56


def sample() -> dict[str, np.ndarray]:
    if CACHE.is_file():
        z = np.load(CACHE)
        print(f"cache {len(z['rgb'])}", flush=True)
        return {k: z[k] for k in ("rgb", "gt56", "x", "y", "yaw", "kind")}
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    col.ensure_dog(ucv)
    loop.boxes.save_restore(ucv)
    loop.restore_boxes(ucv)
    rows = []
    for x in np.arange(-180.0, 161.0, 12.0):
        rows.append((float(x), -200.0, 0.0, "gap"))
    for x in np.arange(-390.0, -20.0, 12.0):
        y = -200.0 + (float(x) + 393.0) * math.tan(math.radians(18.0))
        rows.append((float(x), float(y), 18.0, "offset"))
    for x in np.arange(-332.0, -296.0, 8.0):
        rows.append((float(x), -250.0, 0.0, "face"))
    rgb, gt56, xs, ys, yaw, kind = [], [], [], [], [], []

    def take(x, y, heading, name):
        got = grab(ucv, cam, x, y, heading)
        if got is None:
            print(f"grab_fail {name} {x:.0f},{y:.0f}", flush=True)
            return
        image, gt = got
        cones = cones3_forward(gt)
        print(
            f"{name} x={x:.0f} y={y:.0f} gt={cones[0]:.2f},{cones[1]:.2f},{cones[2]:.2f}",
            flush=True,
        )
        rgb.append(image)
        gt56.append(gt)
        xs.append(x)
        ys.append(y)
        yaw.append(heading)
        kind.append(name)

    for x, y, heading, name in rows:
        take(x, y, heading, name)
    loop.boxes.place_boxes(ucv)
    for x in np.arange(-340.0, -250.0, 15.0):
        take(float(x), -170.0, 0.0, "side")
    loop.restore_boxes(ucv)
    packed = {
        "rgb": np.stack(rgb),
        "gt56": np.stack(gt56).astype(np.float32),
        "x": np.asarray(xs, np.float32),
        "y": np.asarray(ys, np.float32),
        "yaw": np.asarray(yaw, np.float32),
        "kind": np.asarray(kind),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **packed)
    print(f"sampled {len(rgb)}", flush=True)
    return packed


def p20(d56: torch.Tensor, c0: int, c1: int) -> torch.Tensor:
    h, w = d56.shape[-2:]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)
    band = d56[:, lo:hi, c0:c1]
    flat = band.reshape(band.shape[0], -1)
    k = max(1, int(round(0.20 * (flat.shape[1] - 1))))
    return flat.kthvalue(k + 1, dim=-1).values.clamp(0.05, far.FAR)


def cone_pull(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    w = pred.shape[-1]
    loss = pred.new_zeros(())
    n = 0
    for c0, c1 in ((w // 4, w - w // 4), (0, w // 3), (2 * w // 3, w)):
        pg = p20(gt, c0, c1)
        pp = p20(pred, c0, c1)
        far_enough = pg >= 3.0
        if bool(far_enough.any()):
            loss = loss + torch.relu(torch.log(pg[far_enough]) - torch.log(pp[far_enough])).mean()
            n += 1
    return loss / max(n, 1)


def side_hold(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    w = pred.shape[-1]
    loss = pred.new_zeros(())
    n = 0
    for c0, c1 in ((0, w // 3), (2 * w // 3, w)):
        pg = p20(gt, c0, c1)
        pp = p20(pred, c0, c1)
        close = pg < 1.2
        if bool(close.any()):
            loss = loss + torch.relu(torch.log(pp[close]) - torch.log(pg[close])).mean()
            n += 1
    return loss / max(n, 1)


def numpy_cones(model, rgb: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rgb), 4):
            pred = far.forward_maps(model, rgb[i:i + 4]).detach().float().cpu().numpy()
            out.append(np.stack([cones3_forward(p) for p in pred]))
    return np.concatenate(out)


def shortfall(da: np.ndarray, gt: np.ndarray) -> float:
    gaps = [max(0.0, np.log(g[i]) - np.log(max(d[i], 0.05)))
            for d, g in zip(da, gt) for i in range(3) if g[i] >= 3.5]
    return float(np.mean(gaps) if gaps else 0.0)


def main() -> int:
    sim = sample()
    gt_cones = np.stack([cones3_forward(g) for g in sim["gt56"]])
    open_i = np.flatnonzero(
        np.isin(sim["kind"], ["gap", "offset"])
        & (gt_cones[:, 0] >= 3.5)
        & (gt_cones[:, 1] >= 1.8)
        & (gt_cones[:, 2] >= 1.8)
    )
    face_i = np.flatnonzero((sim["kind"] == "face") & (gt_cones[:, 0] <= 1.2))
    side_i = np.flatnonzero((sim["kind"] == "side") & ((gt_cones[:, 1] < 1.0) | (gt_cones[:, 2] < 1.0)))
    print(f"open {len(open_i)} face {len(face_i)} side {len(side_i)}", flush=True)
    if len(open_i) < 4 or len(face_i) < 1:
        print("DEPTHB_TOO_FEW", flush=True)
        return 2
    real_rgb, real_gt = align.load_real()
    real_cones = np.stack([cones3_forward(g) for g in real_gt])
    wrap = DepthMinPredictor.from_checkpoint(str(INIT), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    model = wrap._model
    enc_params = []
    for name, param in model.encoder.named_parameters():
        train = name.startswith("pretrained.blocks.22") or name.startswith("pretrained.blocks.23") or name.startswith("pretrained.norm")
        param.requires_grad_(train)
        if train:
            enc_params.append(param)
    dec_params = []
    for param in model.decoder.parameters():
        param.requires_grad_(True)
        dec_params.append(param)
    print(f"trainable enc {len(enc_params)} dec {len(dec_params)}", flush=True)
    src = torch.load(str(INIT), map_location="cpu", weights_only=False)
    da0 = numpy_cones(model, sim["rgb"])
    real0 = numpy_cones(model, real_rgb)
    base_short = shortfall(da0[open_i], gt_cones[open_i])
    base_real = shortfall(real0, real_cones)
    face_base = {int(i): float(da0[i, 0]) for i in face_i}
    print(f"before short {base_short:.3f} real {base_real:.3f}", flush=True)
    for i in face_i:
        print(f"  face0 x={sim['x'][i]:.0f} gt={gt_cones[i, 0]:.2f} da={da0[i, 0]:.2f}", flush=True)
    opt = torch.optim.AdamW(
        [{"params": dec_params, "lr": 1e-6}, {"params": enc_params, "lr": 3e-7}],
    )
    rng = np.random.default_rng(11)
    best = base_short
    saved = False
    for step in range(1, STEPS + 1):
        model.train()
        model.encoder.eval()
        for param in enc_params:
            param.requires_grad_(True)
        model.decoder.train()
        oi = rng.choice(open_i, size=3, replace=False)
        fi = rng.choice(face_i, size=min(2, len(face_i)), replace=False)
        si = rng.choice(side_i, size=min(2, len(side_i)), replace=len(side_i) < 2) if len(side_i) else np.array([], int)
        ri = rng.choice(len(real_rgb), size=2, replace=False)
        parts = [sim["rgb"][oi], sim["rgb"][fi], real_rgb[ri]]
        gparts = [sim["gt56"][oi], sim["gt56"][fi], real_gt[ri]]
        if len(si):
            parts.insert(2, sim["rgb"][si])
            gparts.insert(2, sim["gt56"][si])
        pred = far.forward_maps(model, np.concatenate(parts))
        gt_t = torch.as_tensor(np.concatenate(gparts), dtype=torch.float32, device="cuda")
        n0, n1 = len(oi), len(fi)
        n2 = len(si)
        loss = cone_pull(pred[:n0], gt_t[:n0]) + 0.5 * far.far_near_loss(pred[:n0], gt_t[:n0])
        loss = loss + 8.0 * far.far_near_loss(pred[n0:n0 + n1], gt_t[n0:n0 + n1])
        cursor = n0 + n1
        if n2:
            loss = loss + 4.0 * side_hold(pred[cursor:cursor + n2], gt_t[cursor:cursor + n2])
            cursor += n2
        loss = loss + 2.0 * far.far_near_loss(pred[cursor:], gt_t[cursor:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.decoder.parameters()) + enc_params, 1.0)
        opt.step()
        if step % 5 != 0 and step != STEPS:
            continue
        da = numpy_cones(model, sim["rgb"])
        real = numpy_cones(model, real_rgb)
        short = shortfall(da[open_i], gt_cones[open_i])
        grown = [
            f"x={sim['x'][i]:.0f} {face_base[int(i)]:.2f}->{da[i, 0]:.2f}"
            for i in face_i if face_base[int(i)] < 1.3 and da[i, 0] > face_base[int(i)] + 0.25
        ]
        real_s = shortfall(real, real_cones)
        print(f"step {step} short {short:.3f} real {real_s:.3f} grown {grown or 'none'}", flush=True)
        for i in face_i:
            print(f"  face x={sim['x'][i]:.0f} gt={gt_cones[i, 0]:.2f} da={da[i, 0]:.2f}", flush=True)
        model.decoder.train()
        if short < best and not grown and real_s <= base_real + 0.05:
            best = short
            far.save_ckpt(model, src, OUT, {"step": step, "short": short, "real": real_s})
            saved = True
            print("DEPTHB_SAVE", step, flush=True)
    if not saved:
        print("DEPTHB_NO_IMPROVE", flush=True)
        return 2
    print("DEPTHB_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
