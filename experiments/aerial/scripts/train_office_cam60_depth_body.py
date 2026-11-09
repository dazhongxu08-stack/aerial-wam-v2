#!/usr/bin/env python3
"""Fit the office depth decoder to the depth map the walker reads.

The loss is the log error of each surface pixel against the ground-truth map,
in both directions. It does not use a picture lookup, a cone hinge, or a
per-column scale. The encoder stays frozen. Real photos are held on the
frozen output. Writes a new file and leaves the existing depth files in place.
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
INIT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_stable.pt"
OUT = ART / "da3_office_cam60_20261106/depth_office_cone_cam60_body.pt"
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
}
STEPS = 40
LR = 1e-6
FAR = 30.0
PROBES = clear.PROBES
REPEATS = 3


def anchored_loss(pred: torch.Tensor, gt: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Pull only surfaces the frozen head got wrong. Keep the rest where it was.

    Close faces are anchored to the frozen output. A two-sided pull on every
    pixel moves those faces through the shared decoder before the long surface
    has reached the truth.
    """
    p = pred.clamp(0.05, FAR)
    g = gt
    t = teacher.clamp(0.05, FAR)
    surface = torch.isfinite(g) & (g > 1.2) & (g < FAR - 0.5)
    bad = surface & ((t > 1.20 * g) | (t < g / 1.20))
    good = surface & ~bad
    close = torch.isfinite(g) & (g > 0.05) & (g < 1.2)
    loss = pred.new_zeros(())
    if bool(bad.any()):
        loss = loss + (torch.log(p) - torch.log(g.clamp(min=0.05))).abs()[bad].mean()
    if bool(good.any()):
        loss = loss + (torch.log(p) - torch.log(t)).abs()[good].mean()
    if bool(close.any()):
        loss = loss + 4.0 * (torch.log(p) - torch.log(t)).abs()[close].mean()
    return loss


def hold_log_l1(pred: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    p = pred.clamp(0.05, FAR)
    t = teacher.clamp(0.05, FAR)
    mask = torch.isfinite(t) & (t > 0.05) & (t < FAR - 0.5)
    if not bool(mask.any()):
        return pred.new_zeros(())
    return (torch.log(p) - torch.log(t)).abs()[mask].mean()


def open_index(xy: np.ndarray, gt: np.ndarray, x: float, y: float, yaw: float) -> int:
    dist = np.abs(xy[:, 0] - x) + np.abs(xy[:, 1] - y) + np.abs(xy[:, 2] - yaw)
    near = np.where(dist <= 30)[0]
    if len(near) == 0:
        return int(np.argmin(dist))
    fwd = np.array([float(cones3_forward(gt[int(j)])[0]) for j in near])
    pool = near[fwd > 2.5] if np.any(fwd > 2.5) else near
    return int(pool[np.argmin(dist[pool])])


def probe_table(da: np.ndarray, gt: np.ndarray, xy: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for name, x, y, yaw in PROBES:
        i = open_index(xy, gt, x, y, yaw)
        d = np.asarray(cones3_forward(da[i]), np.float32)
        g = np.asarray(cones3_forward(gt[i]), np.float32)
        ratio = np.ones(3, np.float32)
        for k in range(3):
            if float(g[k]) > 2.5:
                ratio[k] = float(d[k]) / float(g[k])
        out[name] = ratio
        print(
            f"PROBE {name} {float(ratio[0]):.2f},{float(ratio[1]):.2f},{float(ratio[2]):.2f}",
            flush=True,
        )
    return out


def probes_ok(table: dict[str, np.ndarray]) -> bool:
    for ratio in table.values():
        if float(ratio.min()) < 0.85 or float(ratio.max()) > 1.18:
            return False
    return True


def mean_abs_log(table: dict[str, np.ndarray]) -> float:
    vals = np.concatenate([np.log(np.clip(v, 0.05, None)) for v in table.values()])
    return float(np.mean(np.abs(vals)))


def save_body(model, src: dict, report: dict) -> None:
    payload = dict(src)
    payload["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload.pop("office_scale", None)
    payload["forward_scale"] = 1.0
    payload["forward_exponent"] = 1.0
    payload["office_body"] = report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(OUT))
    print(f"BODY_SAVE {OUT}", flush=True)


def live_probe(path: Path) -> bool:
    depth_model = DepthMinPredictor.from_checkpoint(str(path), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    if depth_model._office_scale is not None:
        raise RuntimeError("body file still carries a depth lookup")
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
                rgb, depth = packed
                image, seen, warped = col.warp_pair(rgb, depth)
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
    model = wrap._model
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    model.eval()
    src = torch.load(str(INIT), map_location="cpu", weights_only=False)

    def maps(rgb: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return align.predict_56(model, rgb)

    before = maps(sim_rgb)
    real_teacher = maps(real_rgb)
    base_fc = clear.false_close(before, sim_gt)
    base_miss = clear.missed_close(before, sim_gt, xy)
    real0, _ = align.domain_score(model, real_rgb, real_gt, "real before")
    print(f"BEFORE false_left {base_fc[1]} missed {base_miss} real {real0:.3f}", flush=True)
    print("BEFORE", flush=True)
    before_table = probe_table(before, sim_gt, xy)
    focus = [
        open_index(xy, sim_gt, x, y, yaw)
        for name, x, y, yaw in PROBES
        if name in {"west_end", "west_lane", "lane_m40"}
    ]
    close_idx = [i for i, gmap in enumerate(sim_gt) if float(np.min(cones3_forward(gmap))) < 1.2]
    if len(close_idx) < 2:
        raise RuntimeError(f"need close frames to anchor, found {len(close_idx)}")
    easy = []
    for i, (dmap, gmap) in enumerate(zip(before, sim_gt)):
        d = cones3_forward(dmap)
        g = cones3_forward(gmap)
        ratios = [float(d[k]) / float(g[k]) for k in range(3) if float(g[k]) > 2.5]
        if ratios and min(ratios) >= 0.90 and max(ratios) <= 1.15:
            easy.append(i)
    if not easy:
        easy = close_idx
    bad_n = 0
    surface_n = 0
    for i in focus:
        gmap = sim_gt[i]
        tmap = before[i]
        surface = np.isfinite(gmap) & (gmap > 1.2) & (gmap < FAR - 0.5)
        bad = surface & ((tmap > 1.20 * gmap) | (tmap < gmap / 1.20))
        bad_n += int(bad.sum())
        surface_n += int(surface.sum())
    print(
        f"focus {focus} close {len(close_idx)} easy {len(easy)} bad_px {bad_n}/{surface_n}",
        flush=True,
    )

    opt = torch.optim.AdamW(
        (p for p in model.decoder.parameters() if p.requires_grad),
        lr=LR,
        betas=(0.0, 0.999),
        weight_decay=0.0,
    )
    rng = np.random.default_rng(4)
    best_score = None
    best_state = None
    best_report = None
    for step in range(1, STEPS + 1):
        model.encoder.eval()
        model.decoder.train()
        si = np.concatenate([
            np.asarray(focus, np.int64),
            rng.choice(close_idx, size=2, replace=len(close_idx) < 2),
            rng.choice(easy, size=1, replace=False),
        ])
        ri = rng.choice(len(real_rgb), size=2, replace=False)
        pred = far.forward_maps(model, np.concatenate([sim_rgb[si], real_rgb[ri]]))
        sim_gt_t = torch.as_tensor(sim_gt[si], dtype=torch.float32, device="cuda")
        sim_teacher = torch.as_tensor(before[si], dtype=torch.float32, device="cuda")
        real_t = torch.as_tensor(real_teacher[ri], dtype=torch.float32, device="cuda")
        loss = anchored_loss(pred[: len(si)], sim_gt_t, sim_teacher) + hold_log_l1(pred[len(si) :], real_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.decoder.parameters() if p.requires_grad), 1.0)
        opt.step()
        if step % 10 != 0 and step != STEPS:
            print(f"step {step} loss {float(loss.detach()):.4f}", flush=True)
            continue
        da = maps(sim_rgb)
        fc = clear.false_close(da, sim_gt)
        miss = clear.missed_close(da, sim_gt, xy)
        real_s, _ = align.domain_score(model, real_rgb, real_gt, f"real step {step}")
        print(f"step {step} loss {float(loss.detach()):.4f} false_left {fc[1]} missed {miss} real {real_s:.3f}", flush=True)
        table = probe_table(da, sim_gt, xy)
        score = mean_abs_log(table)
        ok = probes_ok(table) and real_s <= real0 + 0.05 and miss <= base_miss and fc[1] <= base_fc[1]
        if ok and (best_score is None or score < best_score):
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_report = {"step": step, "score": score, "real": real_s, "missed": miss}
            print(f"BODY_BEST step {step} score {score:.4f} real {real_s:.3f}", flush=True)
        if real_s > real0 + 0.05 or miss > base_miss:
            print(f"BODY_STOP step {step} real {real_s:.3f} missed {miss}", flush=True)
            break
    if best_state is None:
        print("BODY_UNMET", flush=True)
        return 1
    model.load_state_dict(best_state)
    save_body(model, src, best_report)
    del wrap
    torch.cuda.empty_cache()
    if live_probe(OUT):
        print("BODY_LIVE_OK", flush=True)
        return 0
    OUT.unlink()
    print("BODY_UNMET live", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
