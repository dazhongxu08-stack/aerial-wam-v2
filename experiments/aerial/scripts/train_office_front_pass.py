#!/usr/bin/env python3
"""After the frontal spin, keep driving while the nose is open.

The close-nose spin is already in consequence_front.pt. On the closed loop
that spin fired, then a short left cone beside the barrel made every forward
step fall under 0.95, so it kept turning and left the lane. Frames whose
forward cone is still past 2 m have to stay a straight drive. The wall, the
gap, and the close spin stay put. Does not overwrite consequence_front.pt.
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
from experiments.aerial.rl.consequence_model import load_checkpoint, save_checkpoint

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "indoor_office_align_20261104/c_s218/consequence_front.pt"
OUT = ART / "indoor_office_align_20261104/c_s218/consequence_pass.pt"
PASS_CACHE = Path("/data/simworld/run/office_front_pass_20261104/poses.npz")
STEPS = 40


def sample_pass(depth_model) -> dict[str, np.ndarray]:
    if PASS_CACHE.is_file():
        z = np.load(PASS_CACHE, allow_pickle=False)
        print(f"pass_cache {len(z['depth'])}", flush=True)
        return {k: z[k] for k in z.files}
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(loop.col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    loop.restore_boxes(ucv)
    got = []
    try:
        front.place_barrel(ucv, 40.0, -225.0)
        for yaw in (18.0, 0.0):
            for y in (-200.0, -185.0, -170.0):
                for x in np.arange(-170.0, 120.0, 12.0):
                    row = det.pose_row(
                        depth_model, ucv, cam, float(x), y, yaw, (300.0, -200.0), "pass"
                    )
                    if row is not None:
                        got.append(row)
    finally:
        loop.restore_boxes(ucv)
        print("pass_restored", flush=True)
    if len(got) < 8:
        raise RuntimeError(f"pass sample too small: {len(got)}")
    packed = {
        "depth": np.stack([r["depth"] for r in got]).astype(np.float32),
        "goal": np.stack([r["goal"] for r in got]).astype(np.float32),
        "gt": np.stack([r["gt"] for r in got]).astype(np.float32),
        "da": np.stack([r["da"] for r in got]).astype(np.float32),
        "kind": np.asarray([r["kind"] for r in got]),
        "x": np.asarray([r["x"] for r in got], np.float32),
        "y": np.asarray([r["y"] for r in got], np.float32),
    }
    PASS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PASS_CACHE, **packed)
    print(f"pass_sampled {len(got)}", flush=True)
    return packed


def main() -> int:
    if OUT.resolve() in {INIT.resolve(), det.OUT.resolve(), det.INIT.resolve()}:
        print("PASS_REFUSES_OVERWRITE", flush=True)
        return 2
    dev = torch.device("cuda")
    depth_model = loop.DepthMinPredictor.from_checkpoint(str(front.DEPTH), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    passed = sample_pass(depth_model)
    nose = front.sample_front(depth_model)
    shelf = det.sample(depth_model)
    real_rgb, real_gt = align.load_real()
    live = np.load(align.SIM_LIVE)
    real_d, real_c, real_g = align.encode_split(depth_model, real_rgb, real_gt)
    live_d, live_c, live_g = align.encode_split(
        depth_model, live["rgb"], live["gt56"].astype(np.float32)
    )
    del depth_model
    torch.cuda.empty_cache()

    hold = np.load(det.HOLD, allow_pickle=False)
    hold_kind = hold["kind"]
    hold_gt = hold["cones"]
    gap_m = (hold_kind == "gap") & (hold_gt[:, 0] > 2.5) & (hold_gt[:, 1] > 2.0) & (hold_gt[:, 2] > 2.0)
    offset_m = (hold_kind == "offset") & (hold_gt[:, 0] > 2.5)
    face_m = (hold_kind == "face") & (hold_gt[:, 0] < 1.15)

    pda, pgt = passed["da"], passed["gt"]
    if "yaw" in passed:
        yaw = np.asarray(passed["yaw"], np.float32)
    else:
        # Sample order was yaw 18 then yaw 0, three lateral lines, x every 12 cm.
        yaw = np.concatenate([np.full(len(pda) // 2, 18.0), np.full(len(pda) - len(pda) // 2, 0.0)])
    # Already turned. Nose is open and neither side is inside the hit distance.
    # A straight approach (yaw 0) with a short nose stays a turn, not a drive.
    drive = (
        (yaw >= 12.0) & (pda[:, 0] > 2.2) & (pgt[:, 0] > 2.5)
        & (pgt[:, 1] > 0.55) & (pgt[:, 2] > 0.55)
    )
    nda, ngt = nose["da"], nose["gt"]
    turn_r = (nda[:, 0] < 1.4) & (nda[:, 1] < 1.6) & (nda[:, 2] > 2.2) & (ngt[:, 2] > 2.0) & (ngt[:, 0] > 0.30)
    shut = (nda[:, 0] < 1.2) & (nda[:, 1] < 1.5) & (nda[:, 2] < 1.5) & (ngt[:, 0] < 1.5)
    s_da, s_gt, s_kind = shelf["da"], shelf["gt"], shelf["kind"]
    roomy = np.isin(s_kind, ("lane", "side", "block"))
    shelf_r = (
        roomy & (s_da[:, 0] < 2.0) & (s_da[:, 1] < 2.2) & (s_da[:, 2] > 2.5)
        & (s_gt[:, 2] > 2.0) & (s_gt[:, 0] > 0.35)
    )
    wall_m = np.isin(s_kind, ("wall",)) & (s_gt[:, 0] > 4.0)
    face_wall = np.isin(s_kind, ("wall",)) & (s_gt[:, 0] < 1.2)
    print(
        f"drive {int(drive.sum())} turn_r {int(turn_r.sum())} shut {int(shut.sum())} "
        f"shelf_r {int(shelf_r.sum())} wall_face {int(face_wall.sum())} gap {int(gap_m.sum())}",
        flush=True,
    )
    if int(drive.sum()) < 8 or int(turn_r.sum()) < 4:
        print("PASS_EMPTY", flush=True)
        return 2

    pd = torch.as_tensor(passed["depth"], device=dev)
    pg = torch.as_tensor(passed["goal"], device=dev)
    nd = torch.as_tensor(nose["depth"], device=dev)
    ng = torch.as_tensor(nose["goal"], device=dev)
    sd = torch.as_tensor(shelf["depth"], device=dev)
    sg = torch.as_tensor(shelf["goal"], device=dev)
    hold_d = torch.as_tensor(hold["depth"], device=dev)
    hold_g = torch.as_tensor(hold["goal"], device=dev)
    drive_t = torch.as_tensor(drive, device=dev)
    turn_t = torch.as_tensor(turn_r, device=dev)
    shut_t = torch.as_tensor(shut, device=dev)
    shelf_t = torch.as_tensor(shelf_r, device=dev)
    wall_t = torch.as_tensor(wall_m, device=dev)
    face_wall_t = torch.as_tensor(face_wall, device=dev)
    gap_t = torch.as_tensor(gap_m, device=dev)
    offset_t = torch.as_tensor(offset_m, device=dev)
    face_t = torch.as_tensor(face_m, device=dev)
    real_depth = torch.as_tensor(real_d, device=dev)
    real_goal = torch.as_tensor(real_g, device=dev)
    live_depth = torch.as_tensor(live_d, device=dev)
    live_goal = torch.as_tensor(live_g, device=dev)
    real_open = torch.as_tensor((real_c[:, 0] > 3.0) & (real_c[:, 1] > 1.5) & (real_c[:, 2] > 1.5), device=dev)
    real_hit = torch.as_tensor(real_c[:, 0] < 1.15, device=dev)
    live_hit = torch.as_tensor(live_c[:, 0] < 1.15, device=dev)

    model = load_checkpoint(str(INIT), "cuda")
    # The close-spin features already exist. Moving the encoder made a real
    # hit look like these side-on frames. Only the head is allowed to move.
    rest_params = []
    for name, param in model.named_parameters():
        trainable = not name.startswith("enc.")
        param.requires_grad_(trainable)
        if trainable:
            rest_params.append(param)
    opt = torch.optim.AdamW(rest_params, lr=1e-4)
    pz = torch.zeros(pd.shape[0], 4, device=dev)
    pc = torch.zeros_like(pz)
    pc[:, 0] = 0.8
    nz = torch.zeros(nd.shape[0], 4, device=dev)
    nc = torch.zeros_like(nz)
    nc[:, 0] = 0.8
    sz = torch.zeros(sd.shape[0], 4, device=dev)
    scry = torch.zeros_like(sz)
    scry[:, 0] = 0.8
    hz = torch.zeros(hold_d.shape[0], 4, device=dev)
    hc = torch.zeros_like(hz)
    hc[:, 0] = 0.8
    best = None

    def report(step: int) -> dict:
        model.eval()
        with torch.no_grad():
            drive_vals = []
            for prev in (pz, pc):
                drive_vals.append(det.surv_of(model, pd, pg, prev, 0.8, 0.0)[drive_t])
                drive_vals.append(det.surv_of(model, pd, pg, prev, 0.35, 0.0)[drive_t])
                drive_vals.append(det.surv_of(model, pd, pg, prev, 0.8, 0.04)[drive_t])
                drive_vals.append(det.surv_of(model, pd, pg, prev, 0.8, -0.04)[drive_t])
            drive_min = float(torch.cat(drive_vals).min())
            turn_away, turn_fwd = [], []
            for prev in (nz, nc):
                turn_away.append(det.surv_of(model, nd, ng, prev, 0.0, det.YAW)[turn_t])
                for dx, dyaw in det.FWD:
                    turn_fwd.append(det.surv_of(model, nd, ng, prev, dx, dyaw)[turn_t])
            turn_min = float(torch.cat(turn_away).min())
            turn_fwd_max = float(torch.cat(turn_fwd).max())
            shut_vals = []
            for prev in (nz, nc):
                if bool(shut_t.any()):
                    shut_vals.append(det.surv_of(model, nd, ng, prev, 0.8, 0.0)[shut_t])
                    shut_vals.append(det.surv_of(model, nd, ng, prev, 0.0, det.YAW)[shut_t])
                    shut_vals.append(det.surv_of(model, nd, ng, prev, 0.0, -det.YAW)[shut_t])
            shut_max = float(torch.cat(shut_vals).max()) if shut_vals else 0.0
            shelf_vals, shelf_fwd = [], []
            for prev in (sz, scry):
                shelf_vals.append(det.surv_of(model, sd, sg, prev, 0.0, det.YAW)[shelf_t])
                shelf_fwd.append(det.surv_of(model, sd, sg, prev, 0.8, 0.0)[shelf_t])
            shelf_min = float(torch.cat(shelf_vals).min())
            shelf_fwd_max = float(torch.cat(shelf_fwd).max())
            gap_vals, offset_vals, face_vals, wall_vals = [], [], [], []
            for prev in (hz, hc):
                s = det.surv_of(model, hold_d, hold_g, prev, 0.8, 0.0)
                gap_vals.append(s[gap_t])
                offset_vals.append(s[offset_t])
                if bool(face_t.any()):
                    face_vals.append(s[face_t])
                    face_vals.append(det.surv_of(model, hold_d, hold_g, prev, 0.0, det.YAW)[face_t])
                    face_vals.append(det.surv_of(model, hold_d, hold_g, prev, 0.0, -det.YAW)[face_t])
            for prev in (sz, scry):
                wall_s = det.surv_of(model, sd, sg, prev, 0.8, 0.0)
                if bool(wall_t.any()):
                    wall_vals.append(wall_s[wall_t])
                if bool(face_wall_t.any()):
                    face_vals.append(wall_s[face_wall_t])
                    face_vals.append(det.surv_of(model, sd, sg, prev, 0.0, det.YAW)[face_wall_t])
                    face_vals.append(det.surv_of(model, sd, sg, prev, 0.0, -det.YAW)[face_wall_t])
            rz = torch.zeros(real_depth.shape[0], 4, device=dev)
            ro = det.surv_of(model, real_depth, real_goal, rz, 0.8, 0.0)
            lz = torch.zeros(live_depth.shape[0], 4, device=dev)
            lo = det.surv_of(model, live_depth, live_goal, lz, 0.8, 0.0)
        row = {
            "drive": drive_min,
            "turn": turn_min,
            "turn_fwd": turn_fwd_max,
            "shut": shut_max,
            "shelf": shelf_min,
            "shelf_fwd": shelf_fwd_max,
            "gap": float(torch.cat(gap_vals).min()),
            "wall": float(torch.cat(wall_vals).min()) if wall_vals else 1.0,
            "offset": float(torch.cat(offset_vals).min()),
            "face": float(torch.cat(face_vals).max()) if face_vals else 0.0,
            "real_open": float(ro[real_open].min()) if bool(real_open.any()) else 1.0,
            "real_hit": float(ro[real_hit].max()) if bool(real_hit.any()) else 0.0,
            "live_hit": float(lo[live_hit].max()) if bool(live_hit.any()) else 0.0,
        }
        print(
            f"step {step} drive {row['drive']:.3f} turn {row['turn']:.3f} tfwd {row['turn_fwd']:.3f} "
            f"shut {row['shut']:.3f} shelf {row['shelf']:.3f} sfwd {row['shelf_fwd']:.3f} "
            f"gap {row['gap']:.3f} wall {row['wall']:.3f} offset {row['offset']:.3f} face {row['face']:.3f} "
            f"real_open {row['real_open']:.3f} real_hit {row['real_hit']:.3f} live_hit {row['live_hit']:.3f}",
            flush=True,
        )
        return row

    def ok(row: dict) -> bool:
        return (
            row["drive"] >= 0.95
            and row["turn"] >= 0.95
            and row["turn_fwd"] <= 0.90
            and row["shut"] <= 0.25
            and row["shelf"] >= 0.95
            and row["shelf_fwd"] <= 0.90
            and row["gap"] >= 0.95
            and row["wall"] >= 0.89
            and row["offset"] >= 0.93
            and row["face"] <= 0.20
            and row["real_open"] >= 0.93
            and row["real_hit"] <= 0.25
            and row["live_hit"] <= 0.20
        )

    report(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, STEPS + 1):
        model.train()
        loss = pd.new_zeros(())
        for prev in (pz, pc):
            for dx, dyaw in ((0.8, 0.0), (0.35, 0.0), (0.6, 0.0), (1.0, 0.0), (0.8, 0.04), (0.8, -0.04)):
                loss = loss + 4.0 * det.hinge_safe(det.logits(model, pd, pg, prev, dx, dyaw)[drive_t])
        for prev in (nz, nc):
            z_r = det.logits(model, nd, ng, prev, 0.0, det.YAW)
            z_l = det.logits(model, nd, ng, prev, 0.0, -det.YAW)
            loss = loss + 2.0 * det.hinge_safe(z_r[turn_t]) + 0.5 * det.hinge_unsafe(z_l[turn_t])
            for dx, dyaw in det.FWD:
                loss = loss + 0.5 * det.hinge_unsafe(det.logits(model, nd, ng, prev, dx, dyaw)[turn_t])
            if bool(shut_t.any()):
                loss = loss + 4.0 * det.hinge_unsafe(det.logits(model, nd, ng, prev, 0.8, 0.0)[shut_t])
                loss = loss + 4.0 * det.hinge_unsafe(z_r[shut_t]) + 4.0 * det.hinge_unsafe(z_l[shut_t])
        for prev in (sz, scry):
            z_r = det.logits(model, sd, sg, prev, 0.0, det.YAW)
            z_l = det.logits(model, sd, sg, prev, 0.0, -det.YAW)
            z_wall = det.logits(model, sd, sg, prev, 0.8, 0.0)
            loss = loss + 2.0 * det.hinge_safe(z_r[shelf_t]) + 0.5 * det.hinge_unsafe(z_l[shelf_t])
            for dx, dyaw in det.FWD:
                loss = loss + 0.5 * det.hinge_unsafe(det.logits(model, sd, sg, prev, dx, dyaw)[shelf_t])
            if bool(wall_t.any()):
                loss = loss + det.hinge_safe(z_wall[wall_t])
            if bool(face_wall_t.any()):
                loss = loss + 4.0 * det.hinge_unsafe(z_wall[face_wall_t])
                loss = loss + 4.0 * det.hinge_unsafe(z_r[face_wall_t]) + 4.0 * det.hinge_unsafe(z_l[face_wall_t])
        for prev in (hz, hc):
            z = det.logits(model, hold_d, hold_g, prev, 0.8, 0.0)
            z_sr = det.logits(model, hold_d, hold_g, prev, 0.0, det.YAW)
            z_sl = det.logits(model, hold_d, hold_g, prev, 0.0, -det.YAW)
            loss = loss + 4.0 * det.hinge_safe(z[gap_t]) + 4.0 * det.hinge_safe(z[offset_t])
            loss = loss + det.hinge_unsafe(z_sr[gap_t]) + det.hinge_unsafe(z_sl[gap_t])
            if bool(face_t.any()):
                loss = loss + 4.0 * det.hinge_unsafe(z[face_t])
                loss = loss + 4.0 * det.hinge_unsafe(z_sr[face_t]) + 4.0 * det.hinge_unsafe(z_sl[face_t])
        rz = torch.zeros(real_depth.shape[0], 4, device=dev)
        z_real = det.logits(model, real_depth, real_goal, rz, 0.8, 0.0)
        if bool(real_open.any()):
            loss = loss + 4.0 * det.hinge_safe(z_real[real_open])
        if bool(real_hit.any()):
            loss = loss + 4.0 * det.hinge_unsafe(z_real[real_hit])
        lz = torch.zeros(live_depth.shape[0], 4, device=dev)
        z_live = det.logits(model, live_depth, live_goal, lz, 0.8, 0.0)
        if bool(live_hit.any()):
            loss = loss + 4.0 * det.hinge_unsafe(z_live[live_hit])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 8.0)
        opt.step()
        if step % 5 == 0 or step == STEPS:
            row = report(step)
            if (
                row["face"] > 0.30 or row["real_open"] < 0.90 or row["real_hit"] > 0.30
                or row["shut"] > 0.40 or row["turn_fwd"] > 0.90 or row["turn"] < 0.95
            ):
                print("PASS_ABORT", flush=True)
                break
            if step >= 20 and row["drive"] < 0.20:
                print("PASS_STUCK", flush=True)
                break
            if ok(row) and (best is None or row["drive"] >= best):
                best = row["drive"]
                save_checkpoint(OUT, model, {"step": step, "drive": best, "turn": row["turn"]})
                print("PASS_SAVE", step, flush=True)
    if best is None:
        print("PASS_NO_SAVE", flush=True)
        return 2
    print("PASS_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
