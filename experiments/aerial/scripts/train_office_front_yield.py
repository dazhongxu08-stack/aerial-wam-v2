#!/usr/bin/env python3
"""Move the side-crate turn one step earlier, and keep the barrel pass-by.

The closed loop still drives at the last pose the walker is allowed to turn
from (left cone 0.56 m). One step later it would spin, but that step is never
applied. The update follows only the part of that gradient that does not make
the barrel pass-by, the close spin, or a real hit less sure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/aerial-wam-v2/experiments/aerial/scripts")

import run_office_cons_loop as loop
import train_office_align as align
import train_office_align_detour as det
import train_office_front_detour as front
import train_office_front_pass as passed
from experiments.aerial.rl.consequence_model import load_checkpoint, save_checkpoint

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "indoor_office_align_20261104/c_s218/consequence_pass.pt"
OUT = ART / "indoor_office_align_20261104/c_s218/consequence_yield.pt"
CACHE = Path("/data/simworld/run/office_front_yield2_20261104/poses.npz")
STEPS = 40


def sample_side(depth_model) -> dict[str, np.ndarray]:
    if CACHE.is_file():
        z = np.load(CACHE, allow_pickle=False)
        print(f"yield_cache {len(z['depth'])}", flush=True)
        return {k: z[k] for k in z.files}
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(loop.col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    loop.restore_boxes(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    got = []
    try:
        loop.boxes.place_boxes(ucv)
        for x in np.arange(-318.0, -296.0, 2.0):
            row = det.pose_row(depth_model, ucv, cam, float(x), -170.0, 0.0, (220.0, -170.0), "yield")
            if row is not None:
                got.append(row)
    finally:
        loop.restore_boxes(ucv)
        print("yield_restored", flush=True)
    packed = {
        "depth": np.stack([r["depth"] for r in got]).astype(np.float32),
        "goal": np.stack([r["goal"] for r in got]).astype(np.float32),
        "gt": np.stack([r["gt"] for r in got]).astype(np.float32),
        "da": np.stack([r["da"] for r in got]).astype(np.float32),
        "x": np.asarray([r["x"] for r in got], np.float32),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **packed)
    print(f"yield_sampled {len(got)}", flush=True)
    return packed


def flat_grad(loss, params) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    parts = []
    for grad, param in zip(grads, params):
        parts.append(grad.reshape(-1) if grad is not None else torch.zeros_like(param).reshape(-1))
    return torch.cat(parts)


def project_out(task: torch.Tensor, hurt: torch.Tensor) -> torch.Tensor:
    denom = torch.dot(hurt, hurt).clamp_min(1e-12)
    coef = torch.dot(task, hurt) / denom
    if float(coef) > 0.0:
        return task - coef * hurt
    return task


def main() -> int:
    dev = torch.device("cuda")
    depth_model = loop.DepthMinPredictor.from_checkpoint(str(det.DEPTH), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    side = sample_side(depth_model)
    drive_np = passed.sample_pass(depth_model)
    nose = front.sample_front(depth_model)
    real_rgb, real_gt = align.load_real()
    real_d, real_c, real_g = align.encode_split(depth_model, real_rgb, real_gt)
    del depth_model
    torch.cuda.empty_cache()

    gt, da = side["gt"], side["da"]
    # Last poses the walker can still turn from: left side under 0.7 m but not
    # yet inside 0.55 m, and the depth map actually shows that side.
    give = (gt[:, 1] >= 0.55) & (gt[:, 1] < 0.70) & (da[:, 1] < 2.0) & (da[:, 0] > 2.0) & (da[:, 2] > 2.0)
    pda, pgt = drive_np["da"], drive_np["gt"]
    n = len(pda)
    yaw = np.concatenate([np.full(n // 2, 18.0), np.full(n - n // 2, 0.0)])
    drive = (yaw >= 12.0) & (pda[:, 0] > 2.2) & (pgt[:, 0] > 2.5) & (pgt[:, 1] > 0.55) & (pgt[:, 2] > 0.55)
    nda, ngt = nose["da"], nose["gt"]
    turn = (nda[:, 0] < 1.4) & (nda[:, 1] < 1.6) & (nda[:, 2] > 2.2) & (ngt[:, 2] > 2.0) & (ngt[:, 0] > 0.30)
    print(f"give {int(give.sum())} drive {int(drive.sum())} turn {int(turn.sum())}", flush=True)
    for i in np.flatnonzero(give):
        print(
            f"  give x={side['x'][i]:.0f} gt={gt[i, 0]:.2f},{gt[i, 1]:.2f},{gt[i, 2]:.2f} "
            f"da={da[i, 0]:.2f},{da[i, 1]:.2f},{da[i, 2]:.2f}",
            flush=True,
        )
    if int(give.sum()) < 1 or int(drive.sum()) < 8 or int(turn.sum()) < 4:
        print("YIELD_EMPTY", flush=True)
        return 2

    sd = torch.as_tensor(side["depth"], device=dev)
    sg = torch.as_tensor(side["goal"], device=dev)
    pd = torch.as_tensor(drive_np["depth"], device=dev)
    pg = torch.as_tensor(drive_np["goal"], device=dev)
    nd = torch.as_tensor(nose["depth"], device=dev)
    ng = torch.as_tensor(nose["goal"], device=dev)
    rd = torch.as_tensor(real_d, device=dev)
    rg = torch.as_tensor(real_g, device=dev)
    give_t = torch.as_tensor(give, device=dev)
    drive_t = torch.as_tensor(drive, device=dev)
    turn_t = torch.as_tensor(turn, device=dev)
    real_hit = torch.as_tensor(real_c[:, 0] < 1.15, device=dev)
    real_open = torch.as_tensor((real_c[:, 0] > 3.0) & (real_c[:, 1] > 1.5) & (real_c[:, 2] > 1.5), device=dev)

    model = load_checkpoint(str(INIT), "cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    for param in params:
        param.requires_grad_(True)
    enc, rest = [], []
    for name, param in model.named_parameters():
        (enc if name.startswith("enc.") else rest).append(param)
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": 5e-6}, {"params": rest, "lr": 2e-5}],
    )
    sz = torch.zeros(sd.shape[0], 4, device=dev)
    sc = torch.zeros_like(sz)
    sc[:, 0] = 0.8
    pz = torch.zeros(pd.shape[0], 4, device=dev)
    pc = torch.zeros_like(pz)
    pc[:, 0] = 0.8
    nz = torch.zeros(nd.shape[0], 4, device=dev)
    nc = torch.zeros_like(nz)
    nc[:, 0] = 0.8
    rz = torch.zeros(rd.shape[0], 4, device=dev)
    best = None

    def report(step: int) -> dict:
        model.eval()
        with torch.no_grad():
            away, fwds = [], []
            for prev in (sz, sc):
                away.append(det.surv_of(model, sd, sg, prev, 0.0, det.YAW)[give_t])
                for dx, dyaw in det.FWD:
                    fwds.append(det.surv_of(model, sd, sg, prev, dx, dyaw)[give_t])
            drive_vals = []
            for prev in (pz, pc):
                drive_vals.append(det.surv_of(model, pd, pg, prev, 0.8, 0.0)[drive_t])
                drive_vals.append(det.surv_of(model, pd, pg, prev, 0.35, 0.0)[drive_t])
            turn_away, turn_fwd = [], []
            for prev in (nz, nc):
                turn_away.append(det.surv_of(model, nd, ng, prev, 0.0, det.YAW)[turn_t])
                turn_fwd.append(det.surv_of(model, nd, ng, prev, 0.8, 0.0)[turn_t])
            ro = det.surv_of(model, rd, rg, rz, 0.8, 0.0)
        row = {
            "away": float(torch.cat(away).min()),
            "fwd": float(torch.cat(fwds).max()),
            "drive": float(torch.cat(drive_vals).min()),
            "turn": float(torch.cat(turn_away).min()),
            "turn_fwd": float(torch.cat(turn_fwd).max()),
            "real_hit": float(ro[real_hit].max()) if bool(real_hit.any()) else 0.0,
            "real_open": float(ro[real_open].min()) if bool(real_open.any()) else 1.0,
        }
        print(
            f"step {step} away {row['away']:.3f} fwd {row['fwd']:.3f} drive {row['drive']:.3f} "
            f"turn {row['turn']:.3f} tfwd {row['turn_fwd']:.3f} "
            f"real_hit {row['real_hit']:.3f} real_open {row['real_open']:.3f}",
            flush=True,
        )
        model.train()
        return row

    def ok(row: dict) -> bool:
        return (
            row["away"] >= 0.95 and row["fwd"] <= 0.90 and row["drive"] >= 0.95
            and row["turn"] >= 0.95 and row["turn_fwd"] <= 0.90
            and row["real_hit"] <= 0.25 and row["real_open"] >= 0.93
        )

    report(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, STEPS + 1):
        model.train()
        loss = sd.new_zeros(())
        for prev in (sz, sc):
            loss = loss + 2.0 * det.hinge_safe(det.logits(model, sd, sg, prev, 0.0, det.YAW)[give_t])
            loss = loss + 0.5 * det.hinge_unsafe(det.logits(model, sd, sg, prev, 0.0, -det.YAW)[give_t])
            for dx, dyaw in det.FWD:
                loss = loss + det.hinge_unsafe(det.logits(model, sd, sg, prev, dx, dyaw)[give_t])
        # Direction that would spoil the pass-by, the close spin, or a real hit.
        hurt = sd.new_zeros(())
        for prev in (pz, pc):
            hurt = hurt + det.logits(model, pd, pg, prev, 0.8, 0.0)[drive_t].sum()
        for prev in (nz, nc):
            hurt = hurt + det.logits(model, nd, ng, prev, 0.0, det.YAW)[turn_t].sum()
            hurt = hurt - det.logits(model, nd, ng, prev, 0.8, 0.0)[turn_t].sum()
        if bool(real_hit.any()):
            hurt = hurt - det.logits(model, rd, rg, rz, 0.8, 0.0)[real_hit].sum()
        if bool(real_open.any()):
            hurt = hurt + det.logits(model, rd, rg, rz, 0.8, 0.0)[real_open].sum()
        task_g = flat_grad(loss, params)
        hurt_g = flat_grad(hurt, params)
        step_g = project_out(task_g, hurt_g)
        scale = step_g.norm().clamp_min(1e-12)
        if float(scale) > 1.0:
            step_g = step_g * (1.0 / scale)
        offset = 0
        for param in params:
            numel = param.numel()
            param.grad = step_g[offset:offset + numel].view_as(param).clone()
            offset += numel
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 2 == 0 or step == STEPS:
            row = report(step)
            if row["drive"] < 0.93 or row["turn"] < 0.93 or row["real_hit"] > 0.30 or row["real_open"] < 0.90:
                print("YIELD_ABORT", flush=True)
                break
            if ok(row) and (best is None or row["away"] >= best):
                best = row["away"]
                save_checkpoint(OUT, model, {"step": step, "away": best, "fwd": row["fwd"], "drive": row["drive"]})
                print("YIELD_SAVE", step, flush=True)
    if best is None:
        print("YIELD_NO_SAVE", flush=True)
        return 2
    print("YIELD_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
