#!/usr/bin/env python3
"""Fit a local scale on the frozen office depth network.

The depth weights stay at the stable checkpoint. A small field reads this
picture only and scales pixels the network already places between 1.5 m
and 15 m. Closer pixels and the far plane stay as the frozen network
wrote them. Simulation and real photos are both pulled to their own
forward, left, and right readings.
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
from experiments.aerial.rl.office_meter_field import OfficeMeterField
from office_bands import cones3_forward

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_stable.pt"
OUT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_field.pt"
SIM_NPZ = Path("/data/simworld/run/office_cam60_20261105/sim.npz")
EXTRA_NPZ = Path("/data/simworld/run/office_cam60_20261106/smear.npz")
PROTECTED = {
    "depth_office_cone_cam60.pt",
    "depth_office_cone_align.pt",
    "depth_office_cone_far.pt",
    "depth_office_cone_lowbox.pt",
    "depth_office_cone_cam60_clear.pt",
    "depth_office_cone_cam60_stable.pt",
    "depth_office_cone_cam60_scaled.pt",
    "depth_office_cone_cam60_right.pt",
    "depth_office_cone_cam60_right2.pt",
    "depth_office_cone_cam60_body.pt",
    "depth_office_cone_cam60_fuse.pt",
    "depth_office_cone_cam60_enc.pt",
}
STEPS = 120
FAR = 30.0
PROBES = clear.PROBES
REPEATS = 3
OPEN_M = 2.5
CLOSE_M = 1.2


def frozen_depth(model, rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.from_numpy(np.ascontiguousarray(rgb)).to("cuda").unsqueeze(1)
    with torch.no_grad():
        pred, _ = model.predict_from_window(window)
    return pred.detach(), window[:, 0]


def cache_native(model, rgb: np.ndarray) -> torch.Tensor:
    chunks = []
    for i in range(0, len(rgb), 4):
        depth, _ = frozen_depth(model, rgb[i : i + 4])
        chunks.append(depth.cpu())
    return torch.cat(chunks, 0)


def pooled(field, rgb: np.ndarray, depth: torch.Tensor):
    image = torch.from_numpy(np.ascontiguousarray(rgb)).to(depth.device)
    corrected, scale = field(image, depth)
    maps = far.hide_pool(corrected, image)
    with torch.no_grad():
        base = far.hide_pool(depth, image)
    return maps, scale, depth, base


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


def _masked_mean(err: torch.Tensor, mask: torch.Tensor):
    use = mask.flatten(1).any(dim=1)
    if not bool(use.any()):
        return None
    num = (err * mask).flatten(1).sum(dim=1)
    den = mask.flatten(1).sum(dim=1).clamp(min=1)
    return (num[use] / den[use]).mean()


def field_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    scale: torch.Tensor,
    frozen: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    """Pull each direction's own surface. Pixels already within 12% stay put."""
    meter = pred.new_zeros(())
    holds = []
    n = 0
    h, w = pred.shape[-2:]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)
    cols = (slice(w // 4, w - w // 4), slice(0, w // 3), slice(2 * w // 3, w))
    for sl in cols:
        p = pred[:, lo:hi, sl]
        g = gt[:, lo:hi, sl]
        b = base[:, lo:hi, sl]
        valid = torch.isfinite(g) & (g > 0.05) & (g < FAR - 0.5)
        opened = valid & (g > OPEN_M)
        too_long = opened & (p > g * 1.10)
        too_short = opened & (p < g * 0.95)
        rel_g = g.clamp(min=0.05)
        rel0 = (b - g).abs() / rel_g
        rel1 = (p - g).abs() / rel_g
        held = valid & (g >= OfficeMeterField.NEAR_M) & (rel0 <= 0.12)
        long_term = _masked_mean((p - g) / rel_g, too_long)
        short_term = _masked_mean((g - p) / rel_g, too_short)
        hold_term = _masked_mean(torch.relu(rel1 - rel0), held)
        if long_term is not None:
            meter = meter + long_term
            n += 1
        if short_term is not None:
            meter = meter + short_term
            n += 1
        if hold_term is not None:
            holds.append(hold_term)
    for frac in (0.20, 0.50):
        pred_r = readings(pred, frac).clamp(0.05, FAR)
        gt_r = readings(gt, frac).clamp(0.05, FAR)
        base_r = readings(base, frac).clamp(0.05, FAR)
        for k in range(3):
            p, g = pred_r[:, k], gt_r[:, k]
            too_long = (g > OPEN_M) & (p > g * 1.10)
            too_short = (g > OPEN_M) & (p < g * 0.95)
            closed = g < CLOSE_M
            if bool(too_long.any()):
                meter = meter + ((p[too_long] - g[too_long]) / g[too_long]).mean()
                n += 1
            if bool(too_short.any()):
                meter = meter + ((g[too_short] - p[too_short]) / g[too_short]).mean()
                n += 1
            if bool(closed.any()):
                meter = meter + (torch.relu(p[closed] - g[closed]) / g[closed]).mean()
                n += 1
            rel0 = (base_r[:, k] - g).abs() / g
            rel1 = (p - g).abs() / g
            held_r = (g > OPEN_M) & (rel0 <= 0.12)
            if bool(held_r.any()):
                holds.append(torch.relu(rel1[held_r] - rel0[held_r]).mean())
    loss = meter / n if n else meter
    if holds:
        loss = loss + 2.0 * torch.stack(holds).mean()
    return loss


def maps_from(field, rgb: np.ndarray, depth: torch.Tensor) -> np.ndarray:
    field.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rgb), 4):
            pred, _, _, _ = pooled(field, rgb[i : i + 4], depth[i : i + 4].to("cuda"))
            out.append(pred.detach().float().cpu().numpy())
    field.train()
    return np.concatenate(out)


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


def direction_errors(da: np.ndarray, gt: np.ndarray, xy: np.ndarray) -> dict[tuple, float]:
    out = {}
    for name, x, y, yaw in PROBES:
        i = open_index(xy, gt, x, y, yaw)
        d = cones3_forward(da[i])
        g = cones3_forward(gt[i])
        h, w = da[i].shape
        hi = min(int(far.HZ56), h - 1)
        lo = max(h // 8, 0)
        band_d, band_g = da[i, lo:hi], gt[i, lo:hi]
        cols = (slice(w // 4, w - w // 4), slice(0, w // 3), slice(2 * w // 3, w))
        bits = []
        for k, label in enumerate(("fwd", "left", "right")):
            med_d = _np_stat(band_d[:, cols[k]], 50)
            med_g = _np_stat(band_g[:, cols[k]], 50)
            if float(g[k]) > OPEN_M:
                out[(name, k)] = abs(float(d[k]) - float(g[k]))
                if med_g < FAR - 0.5:
                    out[(name, k, "med")] = abs(med_d - med_g)
                bits.append(f"{label} {float(d[k]):.2f}/{float(g[k]):.2f} med {med_d:.2f}/{med_g:.2f}")
            else:
                bits.append(f"{label} {float(d[k]):.2f} med {med_d:.2f}/{med_g:.2f}")
        print(f"PROBE {name} " + " ".join(bits), flush=True)
    return out


def close_indices(gt: np.ndarray) -> np.ndarray:
    keep = [i for i, gmap in enumerate(gt) if float(np.min(cones3_forward(gmap))) < CLOSE_M]
    return np.asarray(keep, np.int64)


def close_lift(da: np.ndarray, base: np.ndarray, gt: np.ndarray) -> float:
    lift = 0.0
    for dmap, bmap, gmap in zip(da, base, gt):
        d, b, g = cones3_forward(dmap), cones3_forward(bmap), cones3_forward(gmap)
        for k in range(3):
            if float(g[k]) < CLOSE_M:
                lift = max(lift, float(d[k] - b[k]))
    return lift


def save_field(model_src: dict, field: OfficeMeterField, report: dict) -> None:
    payload = dict(model_src)
    payload.pop("office_scale", None)
    payload.pop("office_meter", None)
    payload["forward_scale"] = 1.0
    payload["forward_exponent"] = 1.0
    payload["office_meter"] = {
        "state": {k: v.detach().cpu() for k, v in field.state_dict().items()},
        "near_m": OfficeMeterField.NEAR_M,
        "far_m": OfficeMeterField.FAR_M,
        "report": report,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(OUT))
    print(f"FIELD_SAVE {OUT}", flush=True)


def live_probe(path: Path) -> bool:
    depth_model = DepthMinPredictor.from_checkpoint(str(path), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    if depth_model._office_meter is None:
        raise RuntimeError("field checkpoint has no local scale")
    if depth_model._office_scale is not None:
        raise RuntimeError("field checkpoint still carries a picture lookup")
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
    if OUT.name in PROTECTED or OUT.resolve() == INIT.resolve():
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

    wrap = DepthMinPredictor.from_checkpoint(str(INIT), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    if wrap._office_scale is not None or wrap._office_meter is not None:
        raise RuntimeError("stable checkpoint already carries a correction")
    model = wrap._model
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    src = torch.load(str(INIT), map_location="cpu", weights_only=False)
    print("CACHE frozen depth", flush=True)
    sim_depth = cache_native(model, sim_rgb)
    real_depth = cache_native(model, real_rgb)
    field = OfficeMeterField().to("cuda")
    if any(name == "slope" for name, _ in field.named_parameters()):
        raise RuntimeError("scale still has a depth slope")
    before = maps_from(field, sim_rgb, sim_depth)
    base_miss = clear.missed_close(before, sim_gt, xy)
    print("BEFORE", flush=True)
    base_err = direction_errors(before, sim_gt, xy)
    close_idx = close_indices(sim_gt)
    probe_idx = np.array([open_index(xy, sim_gt, *pose[1:]) for pose in PROBES], np.int64)
    watch = np.unique(np.concatenate([close_idx, probe_idx]))
    print(f"BEFORE missed {base_miss} close {len(close_idx)} watch {len(watch)}", flush=True)
    if len(close_idx) == 0:
        raise RuntimeError("missing close frames")
    real_base = maps_from(field, real_rgb, real_depth)
    real_parts0 = align.cone_rows(real_base, real_gt)
    real0 = float(real_parts0[0] + real_parts0[1] + real_parts0[2])
    print(
        f"real0 short {real_parts0[0]:.3f} over {real_parts0[1]:.3f} side {real_parts0[2]:.3f} sum {real0:.3f}",
        flush=True,
    )
    model.eval()
    rng = np.random.default_rng(11)
    opt = torch.optim.AdamW(
        [p for p in field.parameters() if p.requires_grad],
        lr=5e-3,
        weight_decay=0.0,
    )
    base_watch = before[watch]
    best_wall = base_err[("west_lane", 2)]
    best_state = {k: v.detach().cpu().clone() for k, v in field.state_dict().items()}
    best_step = 0
    others = np.array([i for i in range(len(sim_rgb)) if i not in set(probe_idx.tolist())], np.int64)

    def measure() -> tuple[dict, float]:
        watched = maps_from(field, sim_rgb[watch], sim_depth[watch])
        after = before.copy()
        after[watch] = watched
        err = direction_errors(after, sim_gt, xy)
        lift = close_lift(watched, base_watch, sim_gt[watch])
        return err, lift

    def real_sum() -> float:
        parts = align.cone_rows(maps_from(field, real_rgb, real_depth), real_gt)
        return float(parts[0] + parts[1] + parts[2])

    for step in range(1, STEPS + 1):
        si = np.unique(
            np.concatenate(
                [
                    probe_idx,
                    rng.choice(others, size=8),
                    rng.choice(close_idx, size=1),
                ]
            )
        )
        ri = rng.choice(len(real_rgb), size=8, replace=False)
        rgb_b = np.concatenate([sim_rgb[si], real_rgb[ri]])
        depth_b = torch.cat([sim_depth[si], real_depth[ri]], 0).to("cuda")
        gt_b = np.concatenate([sim_gt[si], real_gt[ri]])
        field.train()
        opt.zero_grad(set_to_none=True)
        pred, scale, depth, base = pooled(field, rgb_b, depth_b)
        gt = torch.as_tensor(gt_b, dtype=torch.float32, device=pred.device)
        loss = field_loss(pred, gt, scale, depth, base)
        probe_rgb = sim_rgb[probe_idx]
        probe_depth = sim_depth[probe_idx].to("cuda")
        probe_pred, probe_scale, probe_frozen, probe_base = pooled(field, probe_rgb, probe_depth)
        probe_gt = torch.as_tensor(sim_gt[probe_idx], dtype=torch.float32, device=probe_pred.device)
        loss = loss + 3.0 * field_loss(probe_pred, probe_gt, probe_scale, probe_frozen, probe_base)
        scale = torch.cat([scale, probe_scale], 0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
        opt.step()
        span = float((scale.detach().max() - scale.detach().min()))
        head = float(field.head.weight.detach().abs().max())
        if step % 5 != 0 and step != STEPS:
            print(f"step {step} loss {float(loss.detach()):.4f} scale_span {span:.3f} head {head:.4f}", flush=True)
            continue
        err, lift = measure()
        wall = err[("west_lane", 2)]
        gap = max(err[("gap_start", k)] for k in range(3))
        worse = [key for key, value in err.items() if value > base_err[key] + 0.25 and base_err[key] <= 0.55]
        gap_worse = any(err[("gap_start", k)] > base_err[("gap_start", k)] + 0.25 for k in range(3))
        real1 = real_sum()
        print(
            f"step {step} loss {float(loss.detach()):.4f} scale_span {span:.3f} "
            f"west_right {wall:.3f} gap {gap:.3f} close_lift {lift:.3f} "
            f"real {real1:.3f} worse {worse[:4]}",
            flush=True,
        )
        if (
            lift <= 0.15
            and not worse
            and not gap_worse
            and wall < best_wall - 0.05
            and real1 <= real0 + 0.05
        ):
            best_wall = wall
            best_step = step
            best_state = {k: v.detach().cpu().clone() for k, v in field.state_dict().items()}
            print(f"FIELD_BEST step {step} west_right {wall:.3f} gap {gap:.3f} real {real1:.3f}", flush=True)
    if best_step == 0:
        print("FIELD_UNMET", flush=True)
        return 1
    field.load_state_dict(best_state)
    final = maps_from(field, sim_rgb, sim_depth)
    miss = clear.missed_close(final, sim_gt, xy)
    print("BEST", flush=True)
    final_err = direction_errors(final, sim_gt, xy)
    real_da = maps_from(field, real_rgb, real_depth)
    real_parts = align.cone_rows(real_da, real_gt)
    real1 = float(real_parts[0] + real_parts[1] + real_parts[2])
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
    if (
        miss > base_miss
        or wall > 0.70
        or wall_med > 0.80
        or gap_p > 0.50
        or gap_m > 0.60
        or real1 > real0 + 0.05
    ):
        print("FIELD_UNMET meters", flush=True)
        return 1
    save_field(
        src,
        field,
        {"step": best_step, "west_right": wall, "west_right_med": wall_med, "init": str(INIT)},
    )
    del wrap
    torch.cuda.empty_cache()
    if live_probe(OUT):
        print("FIELD_LIVE_OK", flush=True)
        return 0
    OUT.unlink()
    print("FIELD_UNMET live", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
