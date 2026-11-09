#!/usr/bin/env python3
"""Match each image's own forward, left, and right meter readings.

The reading is the same upper-band statistic the walker uses, and the
median of that same band. A direction that is already near its truth
only contributes when the network reads it longer. A direction that is
too long is pulled to that image's ground truth. Each direction has the
same weight. The encoder stays frozen. A step that lifts a close reading
is discarded, and the learning rate stays the one that moved the long
wall without that lift.
"""
from __future__ import annotations

import copy
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
INIT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_stable.pt"
OUT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_metric.pt"
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
STEPS = 40
FAR = 30.0
PROBES = clear.PROBES
REPEATS = 3
OPEN_M = 2.5
CLOSE_M = 1.2
LIFT_M = 0.30


def forward_meters(model, rgb: np.ndarray) -> torch.Tensor:
    window = torch.from_numpy(np.ascontiguousarray(rgb)).to("cuda").unsqueeze(1)
    pred, _ = model.predict_from_window(window)
    return far.hide_pool(pred, window[:, 0])


def maps_of(model, rgb: np.ndarray) -> np.ndarray:
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rgb), 4):
            pred = forward_meters(model, rgb[i : i + 4]).detach().float().cpu().numpy()
            out.append(pred)
    if was_training:
        model.decoder.train()
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


def split_loss(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull readings that are too long. Resist making a true reading shorter.

    Each direction of each image has the same weight. A reading already at or
    below its own ground truth is not asked to grow, and it pushes back if a
    shared update would shrink it.
    """
    fix = pred.new_zeros(())
    hold = pred.new_zeros(())
    n_fix, n_hold = 0, 0
    for frac in (0.20, 0.50):
        pred_r = readings(pred, frac).clamp(0.05, FAR)
        gt_r = readings(gt, frac).clamp(0.05, FAR)
        for k in range(3):
            p, g = pred_r[:, k], gt_r[:, k]
            opened = g > OPEN_M
            too_long = opened & (p > g * 1.10)
            keep = opened & (p <= g * 1.05)
            closed = g < CLOSE_M
            if bool(too_long.any()):
                fix = fix + ((p[too_long] - g[too_long]) / g[too_long]).mean()
                n_fix += 1
            if bool(keep.any()):
                hold = hold + (torch.relu(g[keep] - p[keep]) / g[keep]).mean()
                n_hold += 1
            if bool(closed.any()):
                hold = hold + (torch.relu(p[closed] - g[closed]) / g[closed]).mean()
                n_hold += 1
    if n_fix:
        fix = fix / n_fix
    if n_hold:
        hold = hold / n_hold
    return hold, fix


def project_hold(hold: torch.Tensor, fix: torch.Tensor, params: list[torch.nn.Parameter]) -> None:
    """Shorten long readings only in directions that do not shrink a true one."""
    if not fix.requires_grad:
        if hold.requires_grad:
            hold.backward()
        return
    if not hold.requires_grad:
        fix.backward()
        return
    hold.backward(retain_graph=True)
    saved = []
    for param in params:
        if param.grad is None:
            saved.append(None)
        else:
            saved.append(param.grad.detach().clone())
            param.grad = None
    fix.backward()
    dot = torch.zeros((), device=fix.device)
    denom = torch.zeros((), device=fix.device)
    for param, grad in zip(params, saved):
        if grad is None or param.grad is None:
            continue
        dot = dot + torch.sum(param.grad * grad)
        denom = denom + torch.sum(grad * grad)
    if float(dot) < 0.0 and float(denom) > 0.0:
        scale = dot / denom
        for param, grad in zip(params, saved):
            if grad is None or param.grad is None:
                continue
            param.grad.sub_(grad * scale)
    for param, grad in zip(params, saved):
        if grad is None:
            continue
        if param.grad is None:
            param.grad = grad.clone()
        else:
            param.grad.add_(grad)


def select_params(model, mode: str) -> list[torch.nn.Parameter]:
    chosen = []
    for name, param in model.named_parameters():
        if mode == "fuse":
            take = "refinenet" in name
        elif mode == "enc":
            take = "encoder.pretrained.blocks.22." in name or "encoder.pretrained.blocks.23." in name
        else:
            take = name.startswith("decoder.")
        param.requires_grad_(take)
        if take:
            chosen.append(param)
    if not chosen:
        raise RuntimeError(f"no parameters for {mode}")
    return chosen


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


def direction_errors(da: np.ndarray, gt: np.ndarray, xy: np.ndarray) -> dict[tuple[str, int], float]:
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
                bits.append(
                    f"{label} {float(d[k]):.2f}/{float(g[k]):.2f} med {med_d:.2f}/{med_g:.2f}"
                )
            else:
                bits.append(f"{label} {float(d[k]):.2f} med {med_d:.2f}/{med_g:.2f}")
        print(f"PROBE {name} " + " ".join(bits), flush=True)
    return out


def close_indices(gt: np.ndarray) -> np.ndarray:
    keep = [i for i, gmap in enumerate(gt) if float(np.min(cones3_forward(gmap))) < CLOSE_M]
    return np.asarray(keep, np.int64)


def long_indices(da: np.ndarray, gt: np.ndarray) -> np.ndarray:
    keep = []
    for i, (dmap, gmap) in enumerate(zip(da, gt)):
        d, g = cones3_forward(dmap), cones3_forward(gmap)
        if any(float(g[k]) > OPEN_M and float(d[k]) > float(g[k]) * 1.25 for k in range(3)):
            keep.append(i)
    return np.asarray(keep, np.int64)


def close_lift(da: np.ndarray, base: np.ndarray, gt: np.ndarray) -> float:
    lift = 0.0
    for dmap, bmap, gmap in zip(da, base, gt):
        d, b, g = cones3_forward(dmap), cones3_forward(bmap), cones3_forward(gmap)
        for k in range(3):
            if float(g[k]) < CLOSE_M:
                lift = max(lift, float(d[k] - b[k]))
    return lift


def snapshot(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_metric(model, src: dict, report: dict) -> None:
    payload = dict(src)
    payload["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload.pop("office_scale", None)
    payload["forward_scale"] = 1.0
    payload["forward_exponent"] = 1.0
    payload["office_metric"] = report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(OUT))
    print(f"METRIC_SAVE {OUT}", flush=True)


def live_probe(path: Path) -> bool:
    depth_model = DepthMinPredictor.from_checkpoint(str(path), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    if depth_model._office_scale is not None:
        raise RuntimeError("metric file still carries a depth lookup")
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
    if wrap._office_scale is not None:
        raise RuntimeError("stable checkpoint still carries a depth lookup")
    model = wrap._model
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    src = torch.load(str(INIT), map_location="cpu", weights_only=False)
    base_state = snapshot(model)
    before = maps_of(model, sim_rgb)
    base_miss = clear.missed_close(before, sim_gt, xy)
    print("BEFORE", flush=True)
    base_err = direction_errors(before, sim_gt, xy)
    base_worst = max(base_err.values())
    close_idx = close_indices(sim_gt)
    long_idx = long_indices(before, sim_gt)
    probe_idx = np.array([open_index(xy, sim_gt, *pose[1:]) for pose in PROBES], np.int64)
    watch = np.unique(np.concatenate([close_idx, probe_idx, long_idx[:8]]))
    print(
        f"BEFORE missed {base_miss} worst_m {base_worst:.3f} "
        f"close {len(close_idx)} long {len(long_idx)} watch {len(watch)}",
        flush=True,
    )
    if len(close_idx) == 0 or len(long_idx) == 0:
        raise RuntimeError("missing close or long frames")
    real0, _ = align.domain_score(model, real_rgb, real_gt, "real0")
    rng = np.random.default_rng(7)
    base_watch = before[watch]

    def step_batch(params, opt) -> float:
        # The reported frames are in every step, so a short reading is held
        # while the long wall in the same batch is shortened.
        si = np.unique(
            np.concatenate(
                [
                    probe_idx[:2],
                    rng.choice(probe_idx[2:], size=1),
                    rng.choice(close_idx, size=1),
                ]
            )
        )
        model.decoder.train()
        model.encoder.eval()
        opt.zero_grad(set_to_none=True)
        pred = forward_meters(model, sim_rgb[si])
        gt = torch.as_tensor(sim_gt[si], dtype=torch.float32, device=pred.device)
        hold, fix = split_loss(pred, gt)
        project_hold(hold, fix, params)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        return float((hold + fix).detach())

    def measure() -> tuple[dict[tuple[str, int], float], float]:
        watched = maps_of(model, sim_rgb[watch])
        after = before.copy()
        after[watch] = watched
        err = direction_errors(after, sim_gt, xy)
        lift = close_lift(watched, base_watch, sim_gt[watch])
        return err, lift

    def try_once(mode: str, lr: float) -> tuple[float, float, float]:
        model.load_state_dict(base_state)
        params = select_params(model, mode)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        loss = step_batch(params, opt)
        err, lift = measure()
        wall = err[("west_lane", 2)]
        collateral = max((value - base_err[key] for key, value in err.items()), default=0.0)
        print(
            f"TRIAL {mode} lr {lr:.1e} loss {loss:.4f} "
            f"west_right_err {wall:.3f} close_lift {lift:.3f} collateral {collateral:.3f}",
            flush=True,
        )
        model.load_state_dict(base_state)
        return wall, lift, collateral

    base_wall = base_err[("west_lane", 2)]
    best_trial = None
    for mode in ("fuse", "enc", "decoder"):
        for lr in (3e-6, 1e-6, 3e-7):
            wall, lift, collateral = try_once(mode, lr)
            gain = base_wall - wall
            safe = gain >= 0.03 and lift <= LIFT_M and collateral <= 0.10
            if safe and (best_trial is None or gain > best_trial[0]):
                best_trial = (gain, mode, lr, lift)
            if safe:
                break
    if best_trial is None:
        print("METRIC_UNMET no safe step", flush=True)
        return 1
    gain, mode, lr, lift = best_trial
    print(f"USE {mode} lr {lr:.1e} gain {gain:.3f} lift {lift:.3f}", flush=True)

    model.load_state_dict(base_state)
    params = select_params(model, mode)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    good_w = snapshot(model)
    good_opt = copy.deepcopy(opt.state_dict())
    best_worst = base_worst
    best_state = snapshot(model)
    best_step = 0
    reverts = 0
    for step in range(1, STEPS + 1):
        loss = step_batch(params, opt)
        err, lift = measure()
        worst = max(err.values())
        worse = [key for key, value in err.items() if value > base_err[key] + 0.10]
        print(
            f"step {step} loss {loss:.4f} worst_m {worst:.3f} close_lift {lift:.3f} "
            f"west_right {err[('west_lane', 2)]:.3f} worse {worse[:4]}",
            flush=True,
        )
        if lift > LIFT_M or worse:
            model.load_state_dict(good_w)
            opt.load_state_dict(good_opt)
            reverts += 1
            print(f"REVERT step {step} reverts {reverts}", flush=True)
            if reverts >= 5:
                print("METRIC_STOP reverts", flush=True)
                break
            continue
        reverts = 0
        good_w = snapshot(model)
        good_opt = copy.deepcopy(opt.state_dict())
        if worst < best_worst - 0.02:
            best_worst = worst
            best_step = step
            best_state = snapshot(model)
            print(f"METRIC_BEST step {step} worst_m {worst:.3f}", flush=True)
        if worst <= 0.60:
            best_step = step
            best_state = snapshot(model)
            best_worst = worst
            print(f"METRIC_REACHED step {step}", flush=True)
            break
    if best_step == 0:
        print("METRIC_UNMET", flush=True)
        return 1
    model.load_state_dict(best_state)
    final = maps_of(model, sim_rgb)
    miss = clear.missed_close(final, sim_gt, xy)
    print("BEST", flush=True)
    final_err = direction_errors(final, sim_gt, xy)
    real1, _ = align.domain_score(model, real_rgb, real_gt, "real1")
    worst = max(final_err.values())
    print(f"BEST step {best_step} worst_m {worst:.3f} missed {miss} real {real1:.3f}", flush=True)
    if miss > base_miss or worst > 0.60 or real1 > real0 + 0.05:
        print("METRIC_UNMET meters", flush=True)
        return 1
    save_metric(model, src, {"step": best_step, "worst_m": worst, "mode": mode, "init": str(INIT)})
    del wrap
    torch.cuda.empty_cache()
    if live_probe(OUT):
        print("METRIC_LIVE_OK", flush=True)
        return 0
    OUT.unlink()
    print("METRIC_UNMET live", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
