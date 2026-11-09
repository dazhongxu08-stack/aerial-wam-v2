#!/usr/bin/env python3
"""Train the office depth fusion layers on surfaces the head reads wrong.

Gradient comes only from open-surface cells whose current reading is more
than 20% away from the truth, in either direction. Close faces, cells already
near the truth, and real photos are not in the loss. The encoder and the
final 1x1 metric head stay frozen. A step that moves a close face, a correct
open cone, or the real-photo score is discarded and the fusion weights are
restored. Writes a new file only after a live check.
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
from office_bands import cones3_forward, horizon_row_56

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_stable.pt"
OUT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_enc.pt"
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
}
STEPS = 36
LR = 1e-4
LR_MIN = 1e-6
LR_MAX = 1e-3
TRAIN_MARKS = ("encoder.pretrained.blocks.22.", "encoder.pretrained.blocks.23.")
FAR = 30.0
OFF = 1.20
PROBES = clear.PROBES
REPEATS = 3


def wrong_surface_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Squared log error on the upper-band cells the current reading still has wrong.

    Squaring makes a surface that is twice the truth pull harder than a cell
    that is only just outside the band. A plain average gives every wrong
    cell the same pull, so the numerous small errors steer the step.
    """
    p = pred.clamp(0.05, FAR)
    g = gt
    h = pred.shape[-2]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)
    p = p[:, lo:hi]
    g = g[:, lo:hi]
    surface = torch.isfinite(g) & (g > 1.2) & (g < FAR - 0.5)
    wrong = surface & ((p.detach() > OFF * g) | (p.detach() < g / OFF))
    if not bool(wrong.any()):
        return pred.sum() * 0.0
    delta = torch.log(p) - torch.log(g.clamp(min=0.05))
    return delta[wrong].pow(2).mean()


def wrong_cone_error(ratios: np.ndarray, base: np.ndarray) -> tuple[float, int]:
    """Mean log error of cones that started out wrong, and how many long ones grew."""
    wrong = np.isfinite(base) & ((base > OFF) | (base < 1.0 / OFF))
    if not np.any(wrong):
        return 0.0, 0
    vals = np.where(np.isfinite(ratios), ratios, 10.0)
    score = float(np.mean(np.abs(np.log(np.clip(vals[wrong], 0.05, None)))))
    long = np.isfinite(base) & (base > OFF)
    grew = int(np.sum(long & np.isfinite(ratios) & (ratios > base + 0.02)))
    return score, grew


def band_wrong_error(da: np.ndarray, gt: np.ndarray, teacher: np.ndarray) -> float:
    """Mean log error on the cells that were wrong before any update."""
    h = da.shape[-2]
    hi = min(int(horizon_row_56(out=h)), h - 1)
    lo = max(h // 8, 0)
    d = da[:, lo:hi]
    g = gt[:, lo:hi]
    t = teacher[:, lo:hi]
    mask = np.isfinite(g) & (g > 1.2) & (g < FAR - 0.5) & ((t > OFF * g) | (t < g / OFF))
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(np.log(np.clip(d[mask], 0.05, None)) - np.log(g[mask]))))


def cone_ratios(da: np.ndarray, gt: np.ndarray) -> np.ndarray:
    ratios = np.full((len(da), 3), np.nan, np.float32)
    for i in range(len(da)):
        d = cones3_forward(da[i])
        g = cones3_forward(gt[i])
        for k in range(3):
            if float(g[k]) > 2.5:
                ratios[i, k] = float(d[k]) / float(g[k])
    return ratios


def open_index(xy: np.ndarray, gt: np.ndarray, x: float, y: float, yaw: float) -> int:
    dist = np.abs(xy[:, 0] - x) + np.abs(xy[:, 1] - y) + np.abs(xy[:, 2] - yaw)
    near = np.where(dist <= 30)[0]
    if len(near) == 0:
        return int(np.argmin(dist))
    fwd = np.array([float(cones3_forward(gt[int(j)])[0]) for j in near])
    pool = near[fwd > 2.5] if np.any(fwd > 2.5) else near
    return int(pool[np.argmin(dist[pool])])


def probe_line(ratios: np.ndarray, gt: np.ndarray, xy: np.ndarray) -> bool:
    ok = True
    for name, x, y, yaw in PROBES:
        i = open_index(xy, gt, x, y, yaw)
        r = ratios[i]
        text = ",".join("nan" if not np.isfinite(v) else f"{float(v):.2f}" for v in r)
        print(f"PROBE {name} {text}", flush=True)
        finite = r[np.isfinite(r)]
        if finite.size and (float(finite.min()) < 0.85 or float(finite.max()) > 1.18):
            ok = False
    return ok


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    kept = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            kept[name] = param.detach().cpu().clone()
    return kept


def restore(model: torch.nn.Module, kept: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in kept.items():
            params[name].copy_(value.to(params[name].device))


def save_fuse(model: torch.nn.Module, src: dict, report: dict) -> None:
    payload = dict(src)
    payload["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload.pop("office_scale", None)
    payload["forward_scale"] = 1.0
    payload["forward_exponent"] = 1.0
    payload["office_fuse"] = report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(OUT))
    print(f"ENC_SAVE {OUT}", flush=True)


def live_probe(path: Path) -> bool:
    depth_model = DepthMinPredictor.from_checkpoint(str(path), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    if depth_model._office_scale is not None:
        raise RuntimeError("fuse file still carries a depth lookup")
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
                ratios = [d / g for g, d in zip(gt_k, da_k) if g > 2.5]
                if spread > 0.80 or (ratios and (min(ratios) < 0.85 or max(ratios) > 1.20)):
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
    print(f"real {len(real_rgb)} sim {len(sim_rgb)}", flush=True)

    wrap = DepthMinPredictor.from_checkpoint(str(INIT), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    if wrap._office_scale is not None:
        raise RuntimeError("stable checkpoint still carries a depth lookup")
    model = wrap._model
    trainable = 0
    for name, param in model.named_parameters():
        use = any(mark in name for mark in TRAIN_MARKS)
        param.requires_grad_(use)
        if use:
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError("no encoder parameters")
    print(f"trainable {trainable}", flush=True)
    model.eval()
    src = torch.load(str(INIT), map_location="cpu", weights_only=False)

    def maps(rgb: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return align.predict_56(model, rgb)

    before = maps(sim_rgb)
    base_fc = clear.false_close(before, sim_gt)
    base_miss = clear.missed_close(before, sim_gt, xy)
    real0, _ = align.domain_score(model, real_rgb, real_gt, "real before")
    model.eval()
    base_ratios = cone_ratios(before, sim_gt)
    good = np.isfinite(base_ratios) & (base_ratios >= 0.85) & (base_ratios <= 1.15)
    base_err, _ = wrong_cone_error(base_ratios, base_ratios)
    print(
        f"BEFORE false_left {base_fc[1]} missed {base_miss} real {real0:.3f} "
        f"wrong_err {base_err:.4f} good_cones {int(good.sum())}",
        flush=True,
    )
    print("BEFORE", flush=True)
    probe_line(base_ratios, sim_gt, xy)

    bad_counts = []
    for tmap, gmap in zip(before, sim_gt):
        mask = np.isfinite(gmap) & (gmap > 1.2) & (gmap < FAR - 0.5)
        mask = mask & ((tmap > OFF * gmap) | (tmap < gmap / OFF))
        bad_counts.append(int(mask.sum()))
    bad_counts = np.asarray(bad_counts)
    bad_frames = np.where(bad_counts > 20)[0]
    if len(bad_frames) < 4:
        raise RuntimeError(f"need frames with a wrong surface, found {len(bad_frames)}")
    focus = [
        open_index(xy, sim_gt, x, y, yaw)
        for name, x, y, yaw in PROBES
        if name in {"west_lane", "lane_late", "gap_start"}
    ]
    print(
        f"bad_frames {len(bad_frames)} worst {int(bad_counts.max())} focus {focus}",
        flush=True,
    )

    opt = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR,
        momentum=0.0,
        weight_decay=0.0,
    )
    rng = np.random.default_rng(4)
    best_err = base_err
    best_snap = snapshot(model)
    best_step = 0
    accepted = 0
    damage_run = 0
    gain_run = 0
    for step in range(1, STEPS + 1):
        pick = rng.choice(bad_frames, size=3, replace=False)
        si = np.unique(np.concatenate([np.asarray(focus, np.int64), pick]))
        model.eval()
        kept = snapshot(model)
        pred = far.forward_maps(model, sim_rgb[si])
        gt_t = torch.as_tensor(sim_gt[si], dtype=torch.float32, device=pred.device)
        loss = wrong_surface_loss(pred, gt_t)
        if float(loss.detach()) == 0.0:
            print(f"step {step} EMPTY", flush=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 100.0)
        opt.step()
        da = maps(sim_rgb)
        ratios = cone_ratios(da, sim_gt)
        err, grew = wrong_cone_error(ratios, base_ratios)
        miss = clear.missed_close(da, sim_gt, xy)
        fc = clear.false_close(da, sim_gt)
        real_s, _ = align.domain_score(model, real_rgb, real_gt, f"real step {step}")
        model.eval()
        left_good = good & np.isfinite(ratios) & ((ratios < 0.80) | (ratios > 1.20))
        reason = ""
        if int(left_good.sum()) > 0:
            departed = ratios[left_good]
            reason = (
                f"good_cones {int(left_good.sum())} "
                f"span {float(np.min(departed)):.2f}-{float(np.max(departed)):.2f}"
            )
        elif miss > base_miss:
            reason = f"missed {miss}"
        elif fc[1] > base_fc[1]:
            reason = f"false_left {fc[1]}"
        elif real_s > real0 + 0.02:
            reason = f"real {real_s:.3f}"
        elif grew > 0:
            reason = f"long_grew {grew} err {err:.4f}"
        elif not err < best_err - 1e-4:
            reason = f"no_gain {err:.4f}"
        if reason:
            lr = float(opt.param_groups[0]["lr"])
            print(
                f"step {step} REJECT {reason} err {err:.4f} grad {float(grad):.3f} "
                f"lr {lr:.2e} loss {float(loss.detach()):.4f}",
                flush=True,
            )
            restore(model, kept)
            opt.zero_grad(set_to_none=True)
            kind = "damage" if not reason.startswith("no_gain") else "flat"
            if kind == "damage":
                damage_run += 1
                gain_run = 0
            else:
                gain_run += 1
                damage_run = 0
            if damage_run >= 5 and lr > LR_MIN * 1.01:
                opt.param_groups[0]["lr"] = max(LR_MIN, lr * 0.5)
                damage_run = 0
                print(f"LR {opt.param_groups[0]['lr']:.2e}", flush=True)
            elif gain_run >= 3 and lr < LR_MAX * 0.99:
                opt.param_groups[0]["lr"] = min(LR_MAX, lr * 2.0)
                gain_run = 0
                print(f"LR {opt.param_groups[0]['lr']:.2e}", flush=True)
            elif damage_run >= 5 and lr <= LR_MIN * 1.01:
                print("ENC_STOP damage at the smallest step", flush=True)
                break
            continue
        accepted += 1
        damage_run = 0
        gain_run = 0
        best_err = err
        best_snap = snapshot(model)
        best_step = step
        print(
            f"step {step} ACCEPT err {err:.4f} real {real_s:.3f} missed {miss} "
            f"grad {float(grad):.3f} accepted {accepted}",
            flush=True,
        )
        probe_line(ratios, sim_gt, xy)
    restore(model, best_snap)
    model.eval()
    final = maps(sim_rgb)
    final_ratios = cone_ratios(final, sim_gt)
    print(f"BEST step {best_step} err {best_err:.4f} accepted {accepted}", flush=True)
    print("BEST", flush=True)
    ready = probe_line(final_ratios, sim_gt, xy)
    if accepted == 0 or not ready:
        print("ENC_UNMET", flush=True)
        return 1
    save_fuse(
        model,
        src,
        {"step": best_step, "wrong_err": best_err, "accepted": accepted, "init": str(INIT)},
    )
    del wrap
    torch.cuda.empty_cache()
    if live_probe(OUT):
        print("ENC_LIVE_OK", flush=True)
        return 0
    OUT.unlink()
    print("ENC_UNMET live", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
