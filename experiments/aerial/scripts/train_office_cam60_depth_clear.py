#!/usr/bin/env python3
"""Train the 60 cm office depth from the align checkpoint.

Does not continue clear or stable. The decoder is fit to the office frames and
the real frames. An open cone below 0.80 of the truth moves up toward 0.90,
and one above 1.25 moves down toward 1.10, as one block. Close faces stay
close. Writes a new file and does not replace align, cam60, clear, or stable.
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
import train_office_da3_far as far
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation
from office_bands import cones3_forward, horizon_row_56

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "da3_office_align_20261104/depth_office_cone_align.pt"
OUT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_fresh.pt"
SIM_NPZ = Path("/data/simworld/run/office_cam60_20261105/sim.npz")
EXTRA_NPZ = Path("/data/simworld/run/office_cam60_20261106/smear.npz")
PROTECTED = {
    "depth_office_cone_cam60.pt",
    "depth_office_cone_align.pt",
    "depth_office_cone_far.pt",
    "depth_office_cone_lowbox.pt",
    "depth_office_cone_cam60_clear.pt",
    "depth_office_cone_cam60_stable.pt",
}
STEPS = 60
FAR = 30.0
PROBES = (
    ("west_end", 12.0, -237.0, 18.0),
    ("west_lane", 12.0, -250.0, 0.0),
    ("lane_late", -16.0, -200.0, 0.0),
    ("lane_m40", -40.0, -200.0, 0.0),
    ("gap_start", -160.0, -200.0, 0.0),
    ("barrel_start", -170.0, -200.0, 0.0),
)
REPEATS = 3


def runtime_p20(band: torch.Tensor) -> torch.Tensor:
    outs = []
    for row in band.reshape(band.shape[0], -1):
        kept = row[(row > 0.05) & (row < FAR - 0.5)]
        if kept.numel() < 8:
            outs.append(row.new_tensor(FAR))
            continue
        rank = max(1, int(round(0.20 * (kept.numel() - 1))) + 1)
        outs.append(kept.kthvalue(rank).values)
    return torch.stack(outs)


def cone_short_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    h, w = pred.shape[-2:]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)
    loss = pred.new_zeros(())
    n = 0
    for sl in (slice(w // 4, w - w // 4), slice(0, w // 3), slice(2 * w // 3, w)):
        pp = runtime_p20(pred[:, lo:hi, sl]).clamp(0.05, FAR)
        gg = runtime_p20(gt[:, lo:hi, sl]).clamp(0.05, FAR)
        use = (gg > 2.5) & (pp < 0.70 * gg)
        if bool(use.any()):
            loss = loss + torch.relu(torch.log(0.75 * gg[use]) - torch.log(pp[use])).mean()
            n += 1
    return loss / max(n, 1)


def cone_over_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Open cone read too far: pull it down into a band near the truth."""
    h, w = pred.shape[-2:]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)
    loss = pred.new_zeros(())
    n = 0
    for sl in (slice(w // 4, w - w // 4), slice(0, w // 3), slice(2 * w // 3, w)):
        pp = runtime_p20(pred[:, lo:hi, sl]).clamp(0.05, FAR)
        gg = runtime_p20(gt[:, lo:hi, sl]).clamp(0.05, FAR)
        use = (gg > 2.5) & (pp > 1.35 * gg)
        if bool(use.any()):
            loss = loss + torch.relu(torch.log(pp[use]) - torch.log(1.25 * gg[use])).mean()
            n += 1
    return loss / max(n, 1)


def hold_loss(pred: torch.Tensor, teacher: torch.Tensor, gt: torch.Tensor | None = None) -> torch.Tensor:
    """Keep pixels that are not an open, too-long reading on the teacher output."""
    p = pred.clamp(0.05, FAR)
    t = teacher.clamp(0.05, FAR)
    if gt is None:
        return (torch.log(p) - torch.log(t)).abs().mean()
    g = gt
    bad = torch.isfinite(g) & (g > 2.5) & (g < FAR - 0.5) & ((t < 0.80 * g) | (t > 1.25 * g))
    safe = torch.isfinite(t) & ~bad
    if not bool(safe.any()):
        return pred.new_zeros(())
    return (torch.log(p) - torch.log(t)).abs()[safe].mean()


def bracket_loss(pred: torch.Tensor, gt: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """On pixels the teacher read too far, stay inside 0.90–1.30 of the truth."""
    p = pred.clamp(0.05, FAR)
    g = gt
    t = teacher.clamp(0.05, FAR)
    use = torch.isfinite(g) & (g > 2.5) & (g < FAR - 0.5) & (t > 1.35 * g)
    if not bool(use.any()):
        return pred.new_zeros(())
    high = use & (p > 1.30 * g)
    low = use & (p < 0.90 * g)
    loss = pred.new_zeros(())
    n = 0
    if bool(high.any()):
        loss = loss + torch.relu(torch.log(p) - torch.log(1.20 * g.clamp(min=0.05)))[high].mean()
        n += 1
    if bool(low.any()):
        loss = loss + torch.relu(torch.log(0.95 * g.clamp(min=0.05)) - torch.log(p))[low].mean()
        n += 1
    return loss / max(n, 1)


def uniform_cone_loss(pred: torch.Tensor, gt: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Shift a whole open cone by one log amount.

    Cones the teacher already read too far move down toward 1.15 times the truth.
    If that update overshoots under 0.85, the same cone moves back up together.
    """
    h, w = pred.shape[-2:]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)
    loss = pred.new_zeros(())
    n = 0
    p = pred.clamp(0.05, FAR)
    t = teacher.clamp(0.05, FAR)
    for sl in (slice(w // 4, w - w // 4), slice(0, w // 3), slice(2 * w // 3, w)):
        band_p = p[:, lo:hi, sl]
        band_g = gt[:, lo:hi, sl]
        band_t = t[:, lo:hi, sl]
        pp = runtime_p20(band_p)
        gg = runtime_p20(band_g).clamp(0.05, FAR)
        tt = runtime_p20(band_t)
        down = (gg > 2.5) & (pp > 1.25 * gg)
        up = (gg > 2.5) & (pp < 0.80 * gg)
        if not bool(down.any() or up.any()):
            continue
        delta = torch.zeros_like(pp)
        if bool(down.any()):
            delta = torch.where(down, torch.log(pp) - torch.log(1.10 * gg), delta)
        if bool(up.any()):
            delta = torch.where(up, torch.log(pp) - torch.log(0.90 * gg), delta)
        delta = delta.detach()
        use = down | up
        mask = torch.isfinite(band_g) & (band_g > 2.5) & (band_g < FAR - 0.5) & use[:, None, None]
        if not bool(mask.any()):
            continue
        logp = torch.log(band_p)
        target = logp.detach() - delta[:, None, None]
        loss = loss + (logp - target)[mask].pow(2).mean()
        n += 1
    return loss / max(n, 1)


def band_over_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Pixels in the same upper band, not only the 20th percentile."""
    h = pred.shape[-2]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)
    p = pred[:, lo:hi].clamp(0.05, FAR)
    g = gt[:, lo:hi]
    mask = torch.isfinite(g) & (g > 2.5) & (g < FAR - 0.5) & (p > 1.40 * g)
    if not bool(mask.any()):
        return pred.new_zeros(())
    return torch.relu(torch.log(p) - torch.log(1.30 * g.clamp(min=0.05)))[mask].mean()


def cone_long_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """A cone that is actually close has to stay close."""
    h, w = pred.shape[-2:]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)
    loss = pred.new_zeros(())
    n = 0
    for sl in (slice(w // 4, w - w // 4), slice(0, w // 3), slice(2 * w // 3, w)):
        pp = runtime_p20(pred[:, lo:hi, sl]).clamp(0.05, FAR)
        gg = runtime_p20(gt[:, lo:hi, sl]).clamp(0.05, FAR)
        use = gg < 1.2
        if bool(use.any()):
            loss = loss + torch.relu(torch.log(pp[use]) - torch.log(gg[use])).mean()
            n += 1
    return loss / max(n, 1)


def near_pixel_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Pixels that are actually under 1 m have to stay near."""
    p = pred.clamp(0.05, FAR)
    g = gt
    mask = torch.isfinite(g) & (g > 0.05) & (g < 1.0)
    if not bool(mask.any()):
        return pred.new_zeros(())
    return torch.relu(torch.log(p) - torch.log(g.clamp(min=0.05)))[mask].mean()


def smear_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    h = pred.shape[-2]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)
    p = pred[:, lo:hi].clamp(0.05, FAR)
    g = gt[:, lo:hi]
    mask = torch.isfinite(g) & (g > 2.5) & (p < 1.2)
    if not bool(mask.any()):
        return pred.new_zeros(())
    return torch.relu(torch.log(g.clamp(min=0.05)) - torch.log(p))[mask].mean()


def short_open(da56: np.ndarray, gt56: np.ndarray) -> tuple[int, int, int]:
    """Open cones (ground truth above 2.5 m) read below 70 percent of that."""
    counts = [0, 0, 0]
    for dmap, gmap in zip(da56, gt56):
        d = cones3_forward(dmap)
        g = cones3_forward(gmap)
        for i in range(3):
            if float(g[i]) > 2.5 and float(d[i]) < 0.70 * float(g[i]):
                counts[i] += 1
    return counts[0], counts[1], counts[2]


def over_open(da56: np.ndarray, gt56: np.ndarray) -> tuple[tuple[int, int, int], float]:
    """Open cones read above 1.25 times ground truth, and the worst ratio."""
    counts = [0, 0, 0]
    worst = 1.0
    for dmap, gmap in zip(da56, gt56):
        d = cones3_forward(dmap)
        g = cones3_forward(gmap)
        for i in range(3):
            if float(g[i]) <= 2.5:
                continue
            ratio = float(d[i]) / float(g[i])
            worst = max(worst, ratio)
            if ratio > 1.25:
                counts[i] += 1
    return (counts[0], counts[1], counts[2]), worst


def false_close(da56: np.ndarray, gt56: np.ndarray) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    for dmap, gmap in zip(da56, gt56):
        d = cones3_forward(dmap)
        g = cones3_forward(gmap)
        for i in range(3):
            if float(d[i]) < 1.0 and float(g[i]) > 2.5:
                counts[i] += 1
    return counts[0], counts[1], counts[2]


def _band_p50(gmap: np.ndarray, sl: slice) -> float:
    h, w = gmap.shape
    hi = min(horizon_row_56(out=h), h - 1)
    lo = max(h // 8, 0)
    part = np.asarray(gmap[lo:hi, sl], np.float32)
    kept = part[np.isfinite(part) & (part > 0.05) & (part < FAR - 0.5)]
    if kept.size < 8:
        return FAR
    return float(np.percentile(kept, 50))


def missed_close(da56: np.ndarray, gt56: np.ndarray, xy=None) -> int:
    """Count cones whose body is close and the reading went long.

    A 20th-percentile of 0.13 m with a median near 3 m is a few pixels, not a
    close face, so it does not count.
    """
    n = 0
    names = ("fwd", "left", "right")
    slices = None
    for i, (dmap, gmap) in enumerate(zip(da56, gt56)):
        d = cones3_forward(dmap)
        g = cones3_forward(gmap)
        w = gmap.shape[1]
        slices = (slice(w // 4, w - w // 4), slice(0, w // 3), slice(2 * w // 3, w))
        for k in range(3):
            body = _band_p50(gmap, slices[k])
            if float(g[k]) < 1.2 and body < 1.5 and float(d[k]) > 2.0:
                n += 1
                where = ""
                if xy is not None and i < len(xy):
                    where = f" x={xy[i][0]:.0f},{xy[i][1]:.0f} yaw={xy[i][2]:.0f}"
                print(
                    f"  miss {names[k]} gt={float(g[k]):.2f} body={body:.2f} da={float(d[k]):.2f}{where}",
                    flush=True,
                )
    return n


def grab_extra() -> dict[str, np.ndarray]:
    if EXTRA_NPZ.is_file():
        z = np.load(EXTRA_NPZ)
        print(f"extra_cache {len(z['rgb'])}", flush=True)
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
    loop.restore_boxes(ucv)
    rgb, gt56, xy = [], [], []
    poses = []
    for x in np.arange(-160.0, 48.0, 14.0):
        poses.append((float(x), -200.0, 0.0))
        poses.append((float(x), -250.0, 0.0))
    for x in np.arange(-40.0, 28.0, 14.0):
        poses.append((float(x), -237.0, 18.0))
    for name, x, y, yaw in PROBES:
        poses.append((x, y, yaw))
    try:
        for x, y, yaw in poses:
            col.place(ucv, x, y, yaw)
            col.look(ucv, cam, x, y, yaw)
            packed = col.grab(ucv, cam)
            if packed is None:
                print(f"grab_fail {x:.0f}", flush=True)
                continue
            image, seen, warped = col.warp_pair(*packed)
            gt = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
            g = cones3_forward(gt)
            print(f"extra x={x:.0f} y={y:.0f} yaw={yaw:.0f} gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f}", flush=True)
            rgb.append(np.ascontiguousarray(image))
            gt56.append(gt)
            xy.append((x, y, yaw))
    finally:
        loop.restore_boxes(ucv)
        print("RESTORED", flush=True)
    if len(rgb) < 20:
        raise RuntimeError(f"extra sample too small: {len(rgb)}")
    packed = {"rgb": np.stack(rgb), "gt56": np.stack(gt56).astype(np.float32), "xy": np.asarray(xy, np.float32)}
    EXTRA_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EXTRA_NPZ, **packed)
    print(f"extra_saved {len(rgb)}", flush=True)
    return packed


def live_probe(path: Path) -> None:
    depth_model = DepthMinPredictor.from_checkpoint(str(path), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    loop.restore_boxes(ucv)
    grabbed: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name, *_ in PROBES}
    try:
        for repeat in range(REPEATS):
            for name, x, y, yaw in PROBES:
                col.place(ucv, x, y, yaw)
                col.look(ucv, cam, x, y, yaw)
                packed = col.grab(ucv, cam)
                rgb, _depth = packed
                image, seen, warped = col.warp_pair(rgb, _depth)
                gt56 = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
                depth_model.reset()
                pred = np.asarray(depth_model.predict_depth(
                    Observation(rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False)
                ), np.float32)
                if pred.shape[:2] != warped.shape[:2]:
                    import cv2
                    pred = cv2.resize(pred, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
                da56 = np.asarray(col.pool_min(col.hide(pred, seen)), np.float32)
                gt = cones3_forward(gt56)
                da = cones3_forward(da56)
                grabbed[name].append((gt, da))
                print(
                    f"PROBE {name} r={repeat} gt={gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f} "
                    f"da={da[0]:.2f},{da[1]:.2f},{da[2]:.2f}",
                    flush=True,
                )
        failed = 0
        for name, rows in grabbed.items():
            bits = []
            ok = True
            for k, label in enumerate(("fwd", "left", "right")):
                gt_k = [float(g[k]) for g, _ in rows]
                da_k = [float(d[k]) for _, d in rows]
                spread = max(da_k) - min(da_k)
                ratios = [d / g for g, d in zip(gt_k, da_k) if g > 2.5]
                dir_ok = spread <= 0.80 and (not ratios or (min(ratios) >= 0.75 and max(ratios) <= 1.35))
                ok = ok and dir_ok
                lo = min(ratios) if ratios else 1.0
                hi = max(ratios) if ratios else 1.0
                bits.append(f"{label} {min(da_k):.2f}-{max(da_k):.2f} r {lo:.2f}-{hi:.2f}")
            if not ok:
                failed += 1
            print(f"MATCH {name} {'PASS' if ok else 'FAIL'} " + " ".join(bits), flush=True)
        if failed:
            print(f"MATCH_UNMET {failed}", flush=True)
        else:
            print("MATCH_OK", flush=True)
    finally:
        loop.restore_boxes(ucv)
        print("RESTORED", flush=True)


def main() -> int:
    if OUT.name in PROTECTED or OUT.resolve() == INIT.resolve():
        raise RuntimeError("refusing to overwrite an existing depth file")
    if OUT.is_file():
        OUT.unlink()
        print(f"dropped {OUT.name}", flush=True)
    if col.CAM_Z != 60.0 or col.PITCH != -30.0:
        raise RuntimeError(f"camera is {col.CAM_Z} {col.PITCH}, want 60 / -30")
    sim = np.load(SIM_NPZ)
    extra = grab_extra()
    sim_rgb = np.concatenate([sim["rgb"], extra["rgb"]])
    sim_gt = np.concatenate([sim["gt56"].astype(np.float32), extra["gt56"].astype(np.float32)])
    real_rgb, real_gt = align.load_real()
    print(f"real {len(real_rgb)} sim {len(sim_rgb)} extra {len(extra['rgb'])}", flush=True)

    wrap = DepthMinPredictor.from_checkpoint(str(INIT), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    model = wrap._model
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    model.eval()
    src = torch.load(str(INIT), map_location="cpu", weights_only=False)

    def maps(rgb):
        model.eval()
        with torch.no_grad():
            return align.predict_56(model, rgb)

    before = maps(sim_rgb)
    real_teacher = maps(real_rgb)
    base_fc = false_close(before, sim_gt)
    base_short = short_open(before, sim_gt)
    base_over, base_worst = over_open(before, sim_gt)
    base_over_n = base_over[0] + base_over[1] + base_over[2]
    base_miss = missed_close(before, sim_gt)
    real0, _ = align.domain_score(model, real_rgb, real_gt, "real before")
    print(
        f"BEFORE false_close fwd {base_fc[0]} left {base_fc[1]} right {base_fc[2]} "
        f"short fwd {base_short[0]} left {base_short[1]} right {base_short[2]} "
        f"over fwd {base_over[0]} left {base_over[1]} right {base_over[2]} worst {base_worst:.2f} "
        f"missed {base_miss}",
        flush=True,
    )

    close_idx = [i for i, gmap in enumerate(sim_gt) if float(np.min(cones3_forward(gmap))) < 1.2]
    very_close = [i for i in close_idx if float(np.min(cones3_forward(sim_gt[i]))) < 0.4]
    close_idx = np.array(close_idx + very_close + very_close)
    short_idx = []
    long_idx = []
    long_w = []
    for i, (dmap, gmap) in enumerate(zip(before, sim_gt)):
        d = cones3_forward(dmap)
        g = cones3_forward(gmap)
        if np.any((g > 2.5) & (d < 0.80 * g)):
            short_idx.append(i)
        ratios = [float(d[k]) / float(g[k]) for k in range(3) if float(g[k]) > 2.5]
        excess = max(ratios) - 1.25 if ratios else 0.0
        if excess > 0.0:
            long_idx.append(i)
            long_w.append(excess)
    short_idx = np.array(short_idx if short_idx else close_idx)
    long_idx = np.array(long_idx)
    long_w = np.asarray(long_w, np.float64)
    if len(long_idx) >= 2:
        long_w = long_w / long_w.sum()
    print(
        f"close_frames {len(close_idx)} short_frames {len(short_idx)} long_frames {len(long_idx)}",
        flush=True,
    )
    opt = torch.optim.AdamW(
        (p for p in model.decoder.parameters() if p.requires_grad),
        lr=3e-6,
    )
    rng = np.random.default_rng(4)
    best = None
    best_state = None
    for step in range(1, STEPS + 1):
        model.encoder.eval()
        model.decoder.train()
        ri = rng.choice(len(real_rgb), size=min(4, len(real_rgb)), replace=False)
        picks = [rng.choice(short_idx, size=2, replace=len(short_idx) < 2)]
        if len(long_idx) >= 2:
            picks.append(rng.choice(long_idx, size=2, replace=False, p=long_w))
        picks.append(rng.choice(close_idx, size=2, replace=len(close_idx) < 2))
        si = np.concatenate(picks)
        batch_rgb = np.concatenate([real_rgb[ri], sim_rgb[si]])
        batch_gt = np.concatenate([real_gt[ri], sim_gt[si]])
        pred = far.forward_maps(model, batch_rgb)
        gt_t = torch.as_tensor(batch_gt, dtype=torch.float32, device="cuda")
        teach = torch.as_tensor(
            np.concatenate([real_teacher[ri], before[si]]), dtype=torch.float32, device="cuda"
        )
        n_real = len(ri)
        real_p, sim_p = pred[:n_real], pred[n_real:]
        real_g, sim_g = gt_t[:n_real], gt_t[n_real:]
        real_t, sim_t = teach[:n_real], teach[n_real:]
        loss = (
            far.far_near_loss(real_p, real_g)
            + far.far_near_loss(sim_p, sim_g)
            + align.side_loss(real_p, real_g)
            + align.side_loss(sim_p, sim_g)
            + 4.0 * near_pixel_loss(sim_p, sim_g)
            + 2.0 * near_pixel_loss(real_p, real_g)
            + 4.0 * cone_long_loss(sim_p, sim_g)
            + 2.0 * cone_long_loss(real_p, real_g)
            + 3.0 * uniform_cone_loss(sim_p, sim_g, sim_t)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.decoder.parameters() if p.requires_grad), 1.0)
        opt.step()
        if step % 10 != 0 and step != STEPS:
            continue
        da = maps(sim_rgb)
        fc = false_close(da, sim_gt)
        short = short_open(da, sim_gt)
        over, worst = over_open(da, sim_gt)
        xy = np.concatenate([sim["xy"], extra["xy"]]) if "xy" in sim.files else extra["xy"]
        miss = missed_close(da, sim_gt, xy if len(xy) == len(sim_gt) else None)
        real_s, _ = align.domain_score(model, real_rgb, real_gt, f"real step {step}")
        over_n = over[0] + over[1] + over[2]
        print(
            f"step {step} loss {float(loss):.3f} false_close fwd {fc[0]} left {fc[1]} right {fc[2]} "
            f"short fwd {short[0]} left {short[1]} right {short[2]} "
            f"over fwd {over[0]} left {over[1]} right {over[2]} worst {worst:.2f} "
            f"missed {miss} real {real_s:.3f}",
            flush=True,
        )
        ok = (
            short[1] <= max(8, int(0.6 * base_short[1]))
            and over_n <= max(12, int(0.6 * base_over_n))
            and worst <= max(1.45, base_worst * 0.85)
            and fc[1] <= base_fc[1]
            and real_s <= real0 + 0.05
            and miss <= base_miss
        )
        score = short[1] + over_n
        better = best is None or score < best[0] or (score == best[0] and worst < best[1])
        if real_s > real0 + 0.10 or miss > base_miss + 3:
            print(
                f"DEPTH_STOP step {step} real {real_s:.3f} short_left {short[1]} false_left {fc[1]}",
                flush=True,
            )
            break
        if ok and better:
            best = (score, worst, real_s, short[1], over_n)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"DEPTH_BEST step {step} short_left {short[1]} over {over_n} worst {worst:.2f} real {real_s:.3f}",
                flush=True,
            )
    if best_state is None:
        print("DEPTH_UNMET", flush=True)
        return 1
    model.load_state_dict(best_state)
    far.save_ckpt(model, src, OUT, {"short_left": best[3], "over": best[4], "worst": best[1], "real": best[2], "init": str(INIT)})
    print(
        f"DEPTH_SAVE short_left {best[3]} over {best[4]} worst {best[1]:.2f} real {best[2]:.3f} OUT {OUT}",
        flush=True,
    )
    del wrap
    torch.cuda.empty_cache()
    live_probe(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
