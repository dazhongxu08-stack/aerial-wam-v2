#!/usr/bin/env python3
"""Retrain the office depth decoder. The encoder stays frozen.

The current decoder stretches the right side of one image to about 6 m
while the left side of that same image sits on its own ground truth.
This run puts back the 1 November decoder, which had not done that, and
fits it to each cam60 image's own forward, left, and right meters.
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
import train_office_cam60_depth_clear as clear
import train_office_da3_far as far
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation
from office_bands import cones3_forward

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
STABLE = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_stable.pt"
DECODER_INIT = ART / "da3_office_cone_20261101/depth_office_cone.pt"
OUT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_decoder.pt"
SIM_NPZ = Path("/data/simworld/run/office_cam60_20261105/sim.npz")
EXTRA_NPZ = Path("/data/simworld/run/office_cam60_20261106/smear.npz")
PROTECTED = {
    "depth_office_cone.pt",
    "depth_office_cone_cam60.pt",
    "depth_office_cone_align.pt",
    "depth_office_cone_far.pt",
    "depth_office_cone_lowbox.pt",
    "depth_office_cone_cam60_clear.pt",
    "depth_office_cone_cam60_stable.pt",
    "depth_office_cone_cam60_scaled.pt",
    "depth_office_cone_cam60_right.pt",
    "depth_office_cone_cam60_right2.pt",
}
STEPS = 160
FAR = 30.0
PROBES = clear.PROBES
REPEATS = 3
OPEN_M = 2.5
CLOSE_M = 1.2
REAL_BAR = 0.319


def forward_meters(model, rgb: np.ndarray) -> torch.Tensor:
    window = torch.from_numpy(np.ascontiguousarray(rgb)).to("cuda").unsqueeze(1)
    pred, _ = model.predict_from_window(window)
    return far.hide_pool(pred, window[:, 0])


def maps_of(model, rgb: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rgb), 4):
            out.append(forward_meters(model, rgb[i : i + 4]).detach().float().cpu().numpy())
    model.encoder.eval()
    return np.concatenate(out)


def _stat(band: torch.Tensor, frac: float) -> torch.Tensor:
    outs = []
    for row in band.reshape(band.shape[0], -1):
        kept = row[(row > 0.05) & (row < FAR - 0.5)]
        if kept.numel() < 8:
            outs.append(row.new_tensor(FAR))
            continue
        rank = max(1, int(round(frac * (kept.numel() - 1))) + 1)
        outs.append(kept.kthvalue(min(rank, kept.numel())).values)
    return torch.stack(outs)


def readings(depth: torch.Tensor, frac: float) -> torch.Tensor:
    h, w = depth.shape[-2:]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)
    band = depth[:, lo:hi]
    cols = (slice(w // 4, w - w // 4), slice(0, w // 3), slice(2 * w // 3, w))
    return torch.stack([_stat(band[:, :, sl], frac) for sl in cols], dim=-1)


def _per_image(err: torch.Tensor, mask: torch.Tensor):
    use = mask.flatten(1).any(dim=1)
    num = (err * mask).flatten(1).sum(dim=1)
    den = mask.flatten(1).sum(dim=1).clamp(min=1)
    return num / den, use


def decoder_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Each direction's own meters. A direction already too long is weighted more.

    The right side is more sensitive than the left, so an even weight lengthens
    it past its own label before the left arrives. The long side is held back.
    """
    loss = pred.new_zeros(())
    n = 0
    h, w = pred.shape[-2:]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)
    cols = (slice(w // 4, w - w // 4), slice(0, w // 3), slice(2 * w // 3, w))
    with torch.no_grad():
        ref_g = readings(gt, 0.20).clamp(min=0.05)
        short_w = torch.ones_like(ref_g)
    for k, sl in enumerate(cols):
        p = pred[:, lo:hi, sl]
        g = gt[:, lo:hi, sl]
        valid = torch.isfinite(g) & (g > 0.05) & (g < FAR - 0.5)
        rel_g = g.clamp(min=0.05)
        pairs = (
            ((p - g) / rel_g, valid & (g > OPEN_M) & (p > g * 1.10), torch.full_like(short_w[:, k], 3.0)),
            ((g - p) / rel_g, valid & (g > OPEN_M) & (p < g * 0.90), short_w[:, k]),
        )
        for err, mask, scale in pairs:
            val, use = _per_image(err, mask)
            if bool(use.any()):
                loss = loss + (val[use] * scale[use]).mean()
                n += 1
    for frac in (0.20, 0.50):
        pred_r = readings(pred, frac).clamp(0.05, FAR)
        gt_r = readings(gt, frac).clamp(0.05, FAR)
        for k in range(3):
            p, g = pred_r[:, k], gt_r[:, k]
            too_long = (g > OPEN_M) & (p > g * 1.10)
            too_short = (g > OPEN_M) & (p < g * 0.90)
            closed = g < CLOSE_M
            if bool(too_long.any()):
                loss = loss + 3.0 * ((p[too_long] - g[too_long]) / g[too_long]).mean()
                n += 1
            if bool(too_short.any()):
                ww = short_w[:, k][too_short]
                loss = loss + (((g[too_short] - p[too_short]) / g[too_short]) * ww).mean()
                n += 1
            if bool(closed.any()):
                loss = loss + 2.0 * (torch.relu(p[closed] - g[closed]) / g[closed]).mean()
                n += 1
    return loss / n if n else loss


def open_index(xy: np.ndarray, gt: np.ndarray, x: float, y: float, yaw: float) -> int:
    dist = np.abs(xy[:, 0] - x) + np.abs(xy[:, 1] - y) + np.abs(xy[:, 2] - yaw)
    near = np.where(dist <= 30)[0]
    if len(near) == 0:
        return int(np.argmin(dist))
    fwd = np.array([float(cones3_forward(gt[int(j)])[0]) for j in near])
    pool = near[fwd > OPEN_M] if np.any(fwd > OPEN_M) else near
    return int(pool[np.argmin(dist[pool])])


def _np_stat(band: np.ndarray, pct: float) -> float:
    kept = band[np.isfinite(band) & (band > 0.05) & (band < FAR - 0.5)]
    if kept.size < 8:
        return FAR
    return float(np.percentile(kept, pct))


def direction_errors(da: np.ndarray, gt: np.ndarray, xy: np.ndarray) -> dict:
    out = {}
    for name, x, y, yaw in PROBES:
        i = open_index(xy, gt, x, y, yaw)
        d = cones3_forward(da[i])
        g = cones3_forward(gt[i])
        h, w = da[i].shape
        hi = min(int(far.HZ56), h - 1)
        lo = max(h // 8, 0)
        cols = (slice(w // 4, w - w // 4), slice(0, w // 3), slice(2 * w // 3, w))
        bits = []
        for k, label in enumerate(("fwd", "left", "right")):
            med_d = _np_stat(da[i, lo:hi, cols[k]], 50)
            med_g = _np_stat(gt[i, lo:hi, cols[k]], 50)
            if float(g[k]) > OPEN_M:
                out[(name, k)] = abs(float(d[k]) - float(g[k]))
                if med_g < FAR - 0.5:
                    out[(name, k, "med")] = abs(med_d - med_g)
                bits.append(f"{label} {float(d[k]):.2f}/{float(g[k]):.2f} med {med_d:.2f}/{med_g:.2f}")
            else:
                bits.append(f"{label} {float(d[k]):.2f} med {med_d:.2f}/{med_g:.2f}")
        print(f"PROBE {name} " + " ".join(bits), flush=True)
    return out


def close_lift(da: np.ndarray, base: np.ndarray, gt: np.ndarray) -> float:
    lift = 0.0
    for dmap, bmap, gmap in zip(da, base, gt):
        d, b, g = cones3_forward(dmap), cones3_forward(bmap), cones3_forward(gmap)
        for k in range(3):
            if float(g[k]) < CLOSE_M:
                lift = max(lift, float(d[k] - b[k]))
    return lift


def install_decoder(model) -> int:
    payload = torch.load(DECODER_INIT, map_location="cpu", weights_only=False)
    src = payload["model"]
    current = model.state_dict()
    copied = 0
    with torch.no_grad():
        for key, value in src.items():
            if not (torch.is_tensor(value) and key.startswith("decoder.")):
                continue
            if key not in current or current[key].shape != value.shape:
                raise RuntimeError(f"decoder tensor does not match: {key}")
            current[key].copy_(value.to(current[key].device))
            copied += 1
    if copied < 10:
        raise RuntimeError(f"only copied {copied} decoder tensors")
    return copied


def save_decoder(model_src: dict, model, report: dict) -> None:
    payload = dict(model_src)
    payload.pop("office_scale", None)
    payload.pop("office_meter", None)
    payload["forward_scale"] = 1.0
    payload["forward_exponent"] = 1.0
    payload["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload["office_decoder"] = report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(OUT))
    print(f"DECODER_SAVE {OUT}", flush=True)


def live_probe(path: Path) -> bool:
    depth_model = DepthMinPredictor.from_checkpoint(str(path), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    if depth_model._office_scale is not None or depth_model._office_meter is not None:
        raise RuntimeError("decoder checkpoint still carries a scale module")
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    loop.restore_boxes(ucv)
    grabbed = {name: [] for name, *_ in PROBES}
    try:
        for repeat in range(REPEATS):
            for name, x, y, yaw in PROBES:
                col.place(ucv, x, y, yaw)
                col.look(ucv, cam, x, y, yaw)
                rgb, depth = col.grab(ucv, cam)
                image, seen, warped = col.warp_pair(rgb, depth)
                gt56 = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
                depth_model.reset()
                pred = np.asarray(
                    depth_model.predict_depth(
                        Observation(
                            rgb=np.ascontiguousarray(image),
                            state=np.zeros(7, np.float32),
                            collided=False,
                        )
                    ),
                    np.float32,
                )
                if pred.shape[:2] != warped.shape[:2]:
                    import cv2

                    pred = cv2.resize(pred, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
                da56 = np.asarray(col.pool_min(col.hide(pred, seen)), np.float32)
                gt = cones3_forward(gt56)
                da = cones3_forward(da56)
                grabbed[name].append((gt, da))
                print(
                    f"LIVE {name} r={repeat} gt={gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f} "
                    f"da={da[0]:.2f},{da[1]:.2f},{da[2]:.2f}",
                    flush=True,
                )
        failed = 0
        for name, rows in grabbed.items():
            ok = True
            for k in range(3):
                gt_k = [float(g[k]) for g, _ in rows]
                da_k = [float(d[k]) for _, d in rows]
                spread = max(da_k) - min(da_k)
                ratios = [d / g for g, d in zip(gt_k, da_k) if g > OPEN_M]
                meter = [abs(d - g) for g, d in zip(gt_k, da_k) if g > OPEN_M]
                if spread > 0.80 or (ratios and (min(ratios) < 0.85 or max(ratios) > 1.20)):
                    ok = False
                if meter and max(meter) > 0.60:
                    ok = False
            if not ok:
                failed += 1
            print(f"MATCH {name} {'PASS' if ok else 'FAIL'}", flush=True)
        return failed == 0
    finally:
        loop.restore_boxes(ucv)
        print("RESTORED", flush=True)


def main() -> int:
    if OUT.name in PROTECTED or OUT.resolve() in {STABLE.resolve(), DECODER_INIT.resolve()}:
        raise RuntimeError("refusing to overwrite an existing depth file")
    if OUT.is_file():
        OUT.unlink()
    if col.CAM_Z != 60.0 or col.PITCH != -30.0:
        raise RuntimeError(f"camera is {col.CAM_Z} {col.PITCH}, want 60 / -30")
    sim = np.load(SIM_NPZ)
    extra = np.load(EXTRA_NPZ)
    sim_rgb = np.concatenate([sim["rgb"], extra["rgb"]])
    sim_gt = np.concatenate([sim["gt56"].astype(np.float32), extra["gt56"].astype(np.float32)])
    xy = np.concatenate([sim["xy"], extra["xy"]]).astype(np.float32)
    real_rgb, real_gt = align.load_real()
    real_gt = real_gt.astype(np.float32)
    print(f"real {len(real_rgb)} sim {len(sim_rgb)}", flush=True)

    wrap = DepthMinPredictor.from_checkpoint(str(STABLE), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    model = wrap._model
    before_copy = None
    for name, param in model.named_parameters():
        if name.startswith("decoder.") and param.ndim >= 1:
            before_copy = float(param.detach().float().mean())
            break
    copied = install_decoder(model)
    after_copy = None
    for name, param in model.named_parameters():
        if name.startswith("decoder.") and param.ndim >= 1:
            after_copy = float(param.detach().float().mean())
            break
    print(f"DECODER_INIT copied {copied} mean {before_copy:.6f} -> {after_copy:.6f}", flush=True)
    if before_copy is not None and abs(before_copy - after_copy) < 1e-8:
        raise RuntimeError("decoder weights did not change")
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    params = []
    for param in model.decoder.parameters():
        param.requires_grad_(True)
        params.append(param)
    model.encoder.eval()
    src = torch.load(str(STABLE), map_location="cpu", weights_only=False)
    before = maps_of(model, sim_rgb)
    print("BEFORE", flush=True)
    base_err = direction_errors(before, sim_gt, xy)
    base_miss = clear.missed_close(before, sim_gt, xy)
    close_idx = np.array(
        [i for i, gmap in enumerate(sim_gt) if float(np.min(cones3_forward(gmap))) < CLOSE_M],
        np.int64,
    )
    probe_idx = np.array([open_index(xy, sim_gt, *pose[1:]) for pose in PROBES], np.int64)
    watch = np.unique(np.concatenate([close_idx, probe_idx]))
    print(f"BEFORE missed {base_miss} close {len(close_idx)} watch {len(watch)}", flush=True)
    if len(close_idx) == 0:
        raise RuntimeError("missing close frames")
    real_parts0 = align.cone_rows(maps_of(model, real_rgb), real_gt)
    real0 = float(sum(real_parts0))
    print(
        f"real0 short {real_parts0[0]:.3f} over {real_parts0[1]:.3f} side {real_parts0[2]:.3f} sum {real0:.3f}",
        flush=True,
    )
    rng = np.random.default_rng(11)
    # beta2 is 0.9 so a large right-side step does not freeze the smaller left-side step.
    opt = torch.optim.AdamW(params, lr=1e-6, betas=(0.9, 0.9), weight_decay=0.0)
    base_watch = before[watch]
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_state = {k: v.clone() for k, v in init_state.items()}
    best_score = sum(value for key, value in base_err.items() if len(key) == 2)
    best_step = 0
    others = np.array([i for i in range(len(sim_rgb)) if i not in set(int(j) for j in probe_idx)], np.int64)

    def measure():
        watched = maps_of(model, sim_rgb[watch])
        after = before.copy()
        after[watch] = watched
        err = direction_errors(after, sim_gt, xy)
        lift = close_lift(watched, base_watch, sim_gt[watch])
        return err, lift, watched

    for step in range(1, STEPS + 1):
        si = np.unique(
            np.concatenate([probe_idx, rng.choice(others, size=4), rng.choice(close_idx, size=1)])
        )
        ri = rng.choice(len(real_rgb), size=8, replace=False)
        rgb_b = np.concatenate([sim_rgb[si], real_rgb[ri]])
        gt_b = np.concatenate([sim_gt[si], real_gt[ri]])
        model.encoder.eval()
        model.decoder.train()
        opt.zero_grad(set_to_none=True)
        loss_value = 0.0
        chunks = max(1, int(np.ceil(len(rgb_b) / 4)))
        for c in range(chunks):
            sl = slice(c * 4, (c + 1) * 4)
            pred = forward_meters(model, rgb_b[sl])
            gt = torch.as_tensor(gt_b[sl], dtype=torch.float32, device=pred.device)
            loss = decoder_loss(pred, gt) / chunks
            loss.backward()
            loss_value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 5 != 0 and step != STEPS:
            print(f"step {step} loss {loss_value:.4f}", flush=True)
            continue
        err, lift, watched = measure()
        wall = err[("west_lane", 2)]
        gap = max(err[("gap_start", k)] for k in range(3))
        score = sum(value for key, value in err.items() if len(key) == 2)
        miss_now = clear.missed_close(watched, sim_gt[watch])
        print(
            f"step {step} loss {loss_value:.4f} west_right {wall:.3f} gap {gap:.3f} "
            f"score {score:.3f} close_lift {lift:.3f} missed {miss_now}",
            flush=True,
        )
        if step == 1 and (lift > 0.50 or abs(wall - base_err[("west_lane", 2)]) > 1.5):
            model.load_state_dict(init_state)
            for group in opt.param_groups:
                group["lr"] *= 0.25
            print(f"LR_DOWN {opt.param_groups[0]['lr']:.3e}", flush=True)
            continue
        if step % 20 == 0:
            parts = align.cone_rows(maps_of(model, real_rgb), real_gt)
            print(
                f"real step {step} short {parts[0]:.3f} over {parts[1]:.3f} "
                f"side {parts[2]:.3f} sum {float(sum(parts)):.3f}",
                flush=True,
            )
        if miss_now == 0 and score < best_score - 0.05:
            best_score = score
            best_step = step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"DECODER_BEST step {step} score {score:.3f} west_right {wall:.3f} gap {gap:.3f}", flush=True)
    if best_step == 0:
        print("DECODER_UNMET", flush=True)
        return 1
    model.load_state_dict(best_state)
    final = maps_of(model, sim_rgb)
    miss = clear.missed_close(final, sim_gt, xy)
    print("BEST", flush=True)
    final_err = direction_errors(final, sim_gt, xy)
    real_parts = align.cone_rows(maps_of(model, real_rgb), real_gt)
    real1 = float(sum(real_parts))
    print(
        f"real1 short {real_parts[0]:.3f} over {real_parts[1]:.3f} side {real_parts[2]:.3f} sum {real1:.3f}",
        flush=True,
    )
    wall = final_err[("west_lane", 2)]
    wall_med = final_err.get(("west_lane", 2, "med"), 99.0)
    gap_p = max(final_err[("gap_start", k)] for k in range(3))
    gap_m = max(final_err.get(("gap_start", k, "med"), 99.0) for k in range(3))
    print(
        f"BEST step {best_step} west_right {wall:.3f} med {wall_med:.3f} "
        f"gap {gap_p:.3f} med {gap_m:.3f} missed {miss} real {real1:.3f}",
        flush=True,
    )
    probe_p = max(value for key, value in final_err.items() if len(key) == 2)
    probe_m = max(value for key, value in final_err.items() if len(key) == 3)
    if (
        miss > 0
        or wall > 0.70
        or wall_med > 0.80
        or gap_p > 0.50
        or gap_m > 0.60
        or probe_p > 0.80
        or probe_m > 0.90
        or real1 > REAL_BAR + 0.05
    ):
        print("DECODER_UNMET meters", flush=True)
        return 1
    save_decoder(src, model, {"step": best_step, "west_right": wall, "init": str(DECODER_INIT)})
    del wrap
    torch.cuda.empty_cache()
    if live_probe(OUT):
        print("DECODER_LIVE_OK", flush=True)
        return 0
    OUT.unlink()
    print("DECODER_UNMET live", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
