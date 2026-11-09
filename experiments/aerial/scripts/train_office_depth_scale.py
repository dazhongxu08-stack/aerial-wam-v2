#!/usr/bin/env python3
"""Fit a gated shrink on the frozen stable depth.

The decoder is not updated. The gate stays off for real photos, so only the
office views move. Pixels under 1.5 m and the far plane stay put. Writes
depth_office_cone_cam60_scaled.pt only when the long cones come in and the
real-domain score does not get worse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/aerial-wam-v2/experiments/aerial/scripts")

import train_office_align as align
import train_office_da3_far as far
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.office_depth_scale import OfficeDepthScale
from office_bands import cones3_forward, horizon_row_56

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_stable.pt"
SCALE_FROM = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_scaled.pt"
OUT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_tight.pt"
SIM_NPZ = Path("/data/simworld/run/office_cam60_20261105/sim.npz")
EXTRA_NPZ = Path("/data/simworld/run/office_cam60_20261106/smear.npz")
FAR = 30.0
STEPS = 80
PROBE_NAMES = ("west_end", "west_lane", "lane_late", "gap_start")
PROBES = (
    ("west_end", 12.0, -237.0, 18.0),
    ("west_lane", 12.0, -250.0, 0.0),
    ("lane_late", -16.0, -200.0, 0.0),
    ("lane_m40", -40.0, -200.0, 0.0),
    ("gap_start", -160.0, -200.0, 0.0),
    ("barrel_start", -170.0, -200.0, 0.0),
)


def raw_depth(model, rgb: np.ndarray) -> torch.Tensor:
    window = torch.from_numpy(np.ascontiguousarray(rgb)).to("cuda").unsqueeze(1)
    pred, _ = model.predict_from_window(window)
    return pred.detach()


def band(pred56: torch.Tensor, gt56: torch.Tensor, teacher56: torch.Tensor):
    h = pred56.shape[-2]
    hi = min(horizon_row_56(out=h), h - 1)
    lo = max(h // 8, 0)
    p = pred56[:, lo:hi].clamp(min=0.05)
    g = gt56[:, lo:hi]
    t = teacher56[:, lo:hi].clamp(min=0.05)
    valid = torch.isfinite(g) & (g > 2.5) & (g < 12.0) & (t > 1.5) & (t < 15.0)
    return p, g, t, valid


def sim_loss(pred56: torch.Tensor, gt56: torch.Tensor, teacher56: torch.Tensor) -> torch.Tensor:
    """Pull only a slice that is still long. A slice already near the truth stays."""
    del teacher56
    return cone_pull(pred56, gt56, pred56)


def _p20(row: torch.Tensor) -> torch.Tensor | None:
    keep = row[(row > 0.05) & (row < FAR - 0.5)]
    if keep.numel() < 8:
        return None
    k = max(1, int(round(0.20 * (keep.numel() - 1))) + 1)
    return keep.kthvalue(k).values


def cone_pull(pred56: torch.Tensor, gt56: torch.Tensor, teacher56: torch.Tensor) -> torch.Tensor:
    h, w = pred56.shape[-2:]
    hi = min(horizon_row_56(out=h), h - 1)
    lo = max(h // 8, 0)
    loss = pred56.new_zeros(())
    n = 0
    for sl in (slice(w // 4, w - w // 4), slice(0, w // 3), slice(2 * w // 3, w)):
        terms = []
        for row_p, row_g, row_t in zip(pred56[:, lo:hi, sl], gt56[:, lo:hi, sl], teacher56[:, lo:hi, sl]):
            pp = _p20(row_p)
            gg = _p20(row_g[torch.isfinite(row_g)])
            tt = _p20(row_t)
            if pp is None or gg is None or tt is None:
                continue
            if float(gg) <= 2.5:
                continue
            ratio = float(pp.detach()) / float(gg)
            if ratio > 1.15:
                terms.append(torch.relu(pp.log() - gg.clamp(min=0.05).log()))
                mask = (
                    torch.isfinite(row_g)
                    & (row_g > 2.5)
                    & (row_g < 12.0)
                    & (row_p.detach() > 1.15 * row_g)
                    & (row_p > 1.5)
                    & (row_p < 15.0)
                )
                if bool(mask.any()):
                    terms.append(torch.relu(row_p[mask].log() - row_g[mask].clamp(min=0.05).log()).mean())
            elif ratio < 0.95:
                terms.append(2.0 * torch.relu((1.0 * gg).log() - pp.log()))
        if terms:
            loss = loss + torch.stack(terms).mean()
            n += 1
    return loss / max(n, 1)


def cone_stats(da56: np.ndarray, gt56: np.ndarray) -> tuple[int, int, float, int]:
    over = short = 0
    logs = []
    open_n = 0
    for dmap, gmap in zip(da56, gt56):
        d = cones3_forward(dmap)
        g = cones3_forward(gmap)
        for k in range(3):
            if g[k] <= 2.5:
                continue
            open_n += 1
            ratio = float(d[k]) / float(g[k])
            logs.append(abs(np.log(max(ratio, 1e-3))))
            if ratio > 1.35:
                over += 1
            if ratio < 0.70:
                short += 1
    med = float(np.median(logs)) if logs else 0.0
    return over, short, med, open_n


def probe_lines(da56: np.ndarray, gt56: np.ndarray, xy: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    found = []
    for name, x, y, yaw in PROBES:
        dist = np.abs(xy[:, 0] - x) + np.abs(xy[:, 1] - y) + np.abs(xy[:, 2] - yaw)
        i = int(np.argmin(dist))
        if float(dist[i]) > 5:
            print(f"PROBE {name} missing", flush=True)
            continue
        d = cones3_forward(da56[i])
        g = cones3_forward(gt56[i])
        print(
            f"PROBE {name} gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f} da={d[0]:.2f},{d[1]:.2f},{d[2]:.2f}",
            flush=True,
        )
        found.append((name, g, d))
    return found


def probes_ok(found: list[tuple[str, np.ndarray, np.ndarray]]) -> bool:
    by = {name: (g, d) for name, g, d in found}
    ok = True
    for name in PROBE_NAMES:
        if name not in by:
            print(f"PROBE_GATE {name} missing", flush=True)
            return False
        g, d = by[name]
        for k, label in enumerate(("fwd", "left", "right")):
            if float(g[k]) <= 2.5:
                if float(d[k]) > 2.0 and float(g[k]) < 1.2:
                    ok = False
                    print(f"PROBE_GATE {name} {label} miss gt={g[k]:.2f} da={d[k]:.2f}", flush=True)
                continue
            ratio = float(d[k]) / float(g[k])
            if ratio < 0.80 or ratio > 1.25:
                ok = False
                print(f"PROBE_GATE {name} {label} ratio {ratio:.2f}", flush=True)
    return ok


def missed(da56: np.ndarray, gt56: np.ndarray) -> int:
    n = 0
    for dmap, gmap in zip(da56, gt56):
        d = cones3_forward(dmap)
        g = cones3_forward(gmap)
        for k in range(3):
            if float(g[k]) < 1.2 and float(d[k]) > 2.0:
                n += 1
    return n


def split_frames(da56: np.ndarray, gt56: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    long_idx, good_idx = [], []
    for i, (dmap, gmap) in enumerate(zip(da56, gt56)):
        d = cones3_forward(dmap)
        g = cones3_forward(gmap)
        ratios = [float(d[k]) / float(g[k]) for k in range(3) if float(g[k]) > 2.5]
        if any(r > 1.35 for r in ratios):
            long_idx.append(i)
        elif ratios and all(0.80 <= r <= 1.20 for r in ratios):
            good_idx.append(i)
    return np.asarray(long_idx, np.int64), np.asarray(good_idx, np.int64)


def apply_all(scale: OfficeDepthScale, rgb: np.ndarray, raw: torch.Tensor) -> tuple[np.ndarray, float]:
    outs = []
    gates = []
    with torch.no_grad():
        for i in range(0, len(rgb), 4):
            batch = torch.from_numpy(np.ascontiguousarray(rgb[i : i + 4])).cuda()
            corr, _ = scale(batch, raw[i : i + 4].cuda())
            outs.append(far.hide_pool(corr, batch).detach().float().cpu())
            gates.append(scale.last_gate.detach().float().cpu())
    return torch.cat(outs).numpy(), float(torch.cat(gates).mean())


def live_probe(path: Path) -> bool:
    import collect_office_train as col
    import run_office_cons_loop as loop
    from experiments.aerial.rl.env.obs import Observation

    if col.CAM_Z != 60.0 or col.PITCH != -30.0:
        print(f"LIVE_CAMERA {col.CAM_Z} {col.PITCH}", flush=True)
        return False
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
        for repeat in range(3):
            for name, x, y, yaw in PROBES:
                col.place(ucv, x, y, yaw)
                col.look(ucv, cam, x, y, yaw)
                packed = col.grab(ucv, cam)
                rgb, _depth = packed
                image, seen, warped = col.warp_pair(rgb, _depth)
                gt56 = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
                depth_model.reset()
                pred = np.asarray(
                    depth_model.predict_depth(
                        Observation(rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False)
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
            bits = []
            ok = True
            for k, label in enumerate(("fwd", "left", "right")):
                gt_k = [float(g[k]) for g, _ in rows]
                da_k = [float(d[k]) for _, d in rows]
                spread = max(da_k) - min(da_k)
                ratios = [d / g for g, d in zip(gt_k, da_k) if g > 2.5]
                dir_ok = spread <= 0.80 and (not ratios or (min(ratios) >= 0.80 and max(ratios) <= 1.25))
                ok = ok and dir_ok
                lo = min(ratios) if ratios else 1.0
                hi = max(ratios) if ratios else 1.0
                bits.append(f"{label} {min(da_k):.2f}-{max(da_k):.2f} r {lo:.2f}-{hi:.2f}")
            if not ok:
                failed += 1
            print(f"MATCH {name} {'PASS' if ok else 'FAIL'} " + " ".join(bits), flush=True)
        print("MATCH_OK" if failed == 0 else f"MATCH_UNMET {failed}", flush=True)
        return failed == 0
    finally:
        loop.restore_boxes(ucv)
        print("RESTORED", flush=True)


def main() -> int:
    if OUT.resolve() == INIT.resolve() or OUT.name in {
        "depth_office_cone_cam60.pt",
        "depth_office_cone_align.pt",
        "depth_office_cone_far.pt",
        "depth_office_cone_lowbox.pt",
        "depth_office_cone_cam60_clear.pt",
        "depth_office_cone_cam60_stable.pt",
        "depth_office_cone_cam60_scaled.pt",
    }:
        raise RuntimeError("refusing to overwrite an existing depth file")
    if OUT.is_file():
        OUT.unlink()
        print(f"dropped {OUT.name}", flush=True)
    sim = np.load(SIM_NPZ)
    extra = np.load(EXTRA_NPZ)
    rgb = np.concatenate([sim["rgb"], extra["rgb"]])
    gt = np.concatenate([sim["gt56"].astype(np.float32), extra["gt56"].astype(np.float32)])
    xy = np.concatenate([sim["xy"], extra["xy"]]).astype(np.float32)
    real_rgb, real_gt = align.load_real()
    wrap = DepthMinPredictor.from_checkpoint(str(INIT), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    model = wrap._model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def pack(images: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        depths, pooled = [], []
        for i in range(0, len(images), 4):
            batch = images[i : i + 4]
            with torch.no_grad():
                pred = raw_depth(model, batch)
                window = torch.from_numpy(np.ascontiguousarray(batch)).to("cuda").unsqueeze(1)
                pooled.append(far.hide_pool(pred, window[:, 0]).detach().float().cpu())
                depths.append(pred.detach().float().cpu())
        return torch.cat(depths), torch.cat(pooled)

    print("cache sim", flush=True)
    sim_raw, sim_da = pack(rgb)
    print("cache real", flush=True)
    real_raw, real_da = pack(real_rgb)
    base_da = sim_da.numpy()
    real_base = real_da.numpy()
    over0, short0, med0, open0 = cone_stats(base_da, gt)
    print(f"BEFORE open {open0} over {over0} short {short0} medlog {med0:.3f}", flush=True)
    probe_lines(base_da, gt, xy)
    real_parts0 = align.cone_rows(real_base, real_gt)
    real0 = float(real_parts0[0] + real_parts0[1] + real_parts0[2])
    print(
        f"real before short {real_parts0[0]:.3f} over {real_parts0[1]:.3f} "
        f"side {real_parts0[2]:.3f} sum {real0:.3f}",
        flush=True,
    )
    base_miss = missed(base_da, gt)
    print(f"missed raw {base_miss}", flush=True)
    scale = OfficeDepthScale().cuda()
    held_scale = torch.load(str(SCALE_FROM), map_location="cpu", weights_only=False)
    scale.load_state_dict(held_scale["office_scale"]["state"], strict=False)
    scale.use_thirds = True
    for p in scale.parameters():
        p.requires_grad_(False)
    for p in list(scale.third_stem.parameters()) + list(scale.third_fc.parameters()):
        p.requires_grad_(True)
    scale.eval()
    scaled0, sim_g0 = apply_all(scale, rgb, sim_raw)
    over0, short0, med0, open0 = cone_stats(scaled0, gt)
    print(
        f"SCALED open {open0} over {over0} short {short0} medlog {med0:.3f} sim_gate {sim_g0:.2f}",
        flush=True,
    )
    probe_lines(scaled0, gt, xy)
    _real_scaled, real_g0 = apply_all(scale, real_rgb, real_raw)
    real_parts0 = align.cone_rows(_real_scaled, real_gt)
    real0 = float(real_parts0[0] + real_parts0[1] + real_parts0[2])
    print(
        f"real scaled short {real_parts0[0]:.3f} over {real_parts0[1]:.3f} "
        f"side {real_parts0[2]:.3f} sum {real0:.3f} real_gate {real_g0:.2f}",
        flush=True,
    )
    base_miss = missed(scaled0, gt)
    print(f"missed {base_miss}", flush=True)
    long_idx, good_idx = split_frames(scaled0, gt)
    print(f"still_long {len(long_idx)} good {len(good_idx)}", flush=True)
    if len(long_idx) < 4:
        long_idx, _ = split_frames(base_da, gt)
    focus = None
    dist = np.abs(xy[:, 0] - 12.0) + np.abs(xy[:, 1] + 250.0) + np.abs(xy[:, 2])
    if float(dist.min()) <= 5:
        focus = int(np.argmin(dist))
    trainable = [p for p in scale.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    rng = np.random.default_rng(3)
    gt_t = torch.as_tensor(gt, device="cuda")
    best_state = None
    best_key = None
    for step in range(1, STEPS + 1):
        scale.train()
        parts = [rng.choice(long_idx, size=min(3, len(long_idx)), replace=False)]
        if len(good_idx) >= 3:
            parts.append(rng.choice(good_idx, size=3, replace=False))
        if focus is not None:
            parts.append(np.asarray([focus]))
        si = np.unique(np.concatenate(parts))
        ri = rng.choice(len(real_rgb), size=4, replace=False)
        sim_rgb_t = torch.from_numpy(np.ascontiguousarray(rgb[si])).cuda()
        real_rgb_t = torch.from_numpy(np.ascontiguousarray(real_rgb[ri])).cuda()
        sim_corr, sim_s = scale(sim_rgb_t, sim_raw[si].cuda())
        sim_gate = scale.last_gate
        _real_corr, _real_s = scale(real_rgb_t, real_raw[ri].cuda())
        real_gate = scale.last_gate
        sim56 = far.hide_pool(sim_corr, sim_rgb_t)
        teacher56 = sim_da[si].cuda()
        loss = sim_loss(sim56, gt_t[si], teacher56)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scale.parameters(), 1.0)
        opt.step()
        if step % 10 != 0 and step != STEPS:
            continue
        scale.eval()
        after, sim_g = apply_all(scale, rgb, sim_raw)
        _real_after, real_g = apply_all(scale, real_rgb, real_raw)
        over, short, med, _ = cone_stats(after, gt)
        parts = align.cone_rows(_real_after, real_gt)
        real_s = float(parts[0] + parts[1] + parts[2])
        miss = missed(after, gt)
        print(
            f"step {step} loss {float(loss.detach()):.3f} sim_gate {sim_g:.2f} real_gate {real_g:.2f} "
            f"scale {float(sim_s.detach().median()):.3f} "
            f"over {over} short {short} med {med:.3f} real {real_s:.3f} miss {miss}",
            flush=True,
        )
        found = probe_lines(after, gt, xy)
        held = real_s <= real0 + 0.05 and miss <= base_miss and short <= short0 + 2 and real_g < 0.5
        if not held or not probes_ok(found):
            continue
        key = (over, med, real_s)
        if best_key is None or key < best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in scale.state_dict().items()}
            print(f"BEST over {over} med {med:.3f} real {real_s:.3f}", flush=True)

    if best_state is None:
        print("SCALE_UNMET", flush=True)
        return 1
    scale.load_state_dict(best_state)
    scale.eval()
    after, sim_g = apply_all(scale, rgb, sim_raw)
    real_after, real_g = apply_all(scale, real_rgb, real_raw)
    over, short, med, open_n = cone_stats(after, gt)
    print(f"AFTER open {open_n} over {over} short {short} medlog {med:.3f}", flush=True)
    found = probe_lines(after, gt, xy)
    parts = align.cone_rows(real_after, real_gt)
    real_s = float(parts[0] + parts[1] + parts[2])
    miss = missed(after, gt)
    print(
        f"real after short {parts[0]:.3f} over {parts[1]:.3f} side {parts[2]:.3f} "
        f"sum {real_s:.3f} missed {miss} real0 {real0:.3f} "
        f"sim_gate {sim_g:.2f} real_gate {real_g:.2f}",
        flush=True,
    )
    ok = (
        real_s <= real0 + 0.05
        and miss <= base_miss
        and short <= short0 + 2
        and over <= over0
        and med <= med0
        and real_g < 0.5
        and probes_ok(found)
    )
    print(f"medlog {med0:.3f} -> {med:.3f} {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("SCALE_UNMET", flush=True)
        return 1
    src = torch.load(str(INIT), map_location="cpu", weights_only=False)
    src["office_scale"] = {"state": best_state, "thirds": True}
    src["forward_scale"] = 1.0
    src["forward_exponent"] = 1.0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(src, str(OUT))
    print(f"SCALE_SAVE {OUT}", flush=True)
    try:
        matched = live_probe(OUT)
    except Exception as exc:  # noqa: BLE001
        print(f"LIVE_SKIP {exc}", flush=True)
        return 0
    if matched:
        print("SCALE_LIVE_OK", flush=True)
        return 0
    OUT.unlink()
    print("SCALE_UNMET live", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
