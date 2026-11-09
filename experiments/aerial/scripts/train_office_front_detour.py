#!/usr/bin/env python3
"""Teach the close frontal detour the car stopped on.

The saved shelf detour turns while the nose is still many meters out. On a
tall barrel the same weights still stop once the nose is about a meter and
one side is open: forward and the away spin both fall under 0.95. This run
lifts that away spin, keeps the forward steps under it, and leaves a true
wall (both sides short) as a stop.

Writes consequence_front.pt only. The nearmiss weights, the shelf detour
weights, and the align depth stay where they are.
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
from experiments.aerial.rl.consequence_model import load_checkpoint, save_checkpoint

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
DEPTH = ART / "da3_office_align_20261104/depth_office_cone_align.pt"
INIT = ART / "indoor_office_align_20261104/c_s218/consequence_detour.pt"
OUT = ART / "indoor_office_align_20261104/c_s218/consequence_front.pt"
FRONT_CACHE = Path("/data/simworld/run/office_front_detour_20261104/poses.npz")
BARREL = "SM_BarelPlastic_A_21"
STEPS = 80


def place_barrel(ucv, x: float, y: float) -> None:
    loop.col.req(ucv, f"vset /object/{BARREL}/object_mobility true")
    loop.col.req(ucv, f"vset /object/{BARREL}/physics false")
    loop.col.req(ucv, f"vset /object/{BARREL}/scale 1 1 3")
    loop.col.req(ucv, f"vset /object/{BARREL}/rotation 0 0 0")
    loop.col.req(ucv, f"vset /object/{BARREL}/location {x} {y} -48")
    print("barrel", loop.col.req(ucv, f"vget /object/{BARREL}/bounds"), flush=True)


def sample_front(depth_model) -> dict[str, np.ndarray]:
    if FRONT_CACHE.is_file():
        z = np.load(FRONT_CACHE, allow_pickle=False)
        print(f"front_cache {len(z['depth'])}", flush=True)
        return {k: z[k] for k in z.files}
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(loop.col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    loop.restore_boxes(ucv)
    got = []
    try:
        # Just left of the lane, in the open stretch east of the shelf.
        # The nose closes to about a meter while the right side stays open.
        place_barrel(ucv, 40.0, -225.0)
        for y in (-210.0, -200.0, -190.0):
            for x in np.arange(-170.0, -55.0, 8.0):
                row = det.pose_row(depth_model, ucv, cam, float(x), y, 0.0, (300.0, -200.0), "open")
                if row is not None:
                    got.append(row)
        for x in (-130.0, -110.0, -90.0, -70.0):
            for yaw in (12.0, 18.0, 28.0):
                row = det.pose_row(depth_model, ucv, cam, x, -200.0, yaw, (300.0, -200.0), "open")
                if row is not None:
                    got.append(row)
        # Same barrel on the lane center. Both sides close: this has to stay a stop.
        place_barrel(ucv, 40.0, -200.0)
        for x in np.arange(-160.0, -40.0, 10.0):
            row = det.pose_row(depth_model, ucv, cam, float(x), -200.0, 0.0, (300.0, -200.0), "shut")
            if row is not None:
                got.append(row)
    finally:
        loop.restore_boxes(ucv)
        print("front_restored", flush=True)
    if len(got) < 8:
        raise RuntimeError(f"front sample too small: {len(got)}")
    packed = {
        "depth": np.stack([r["depth"] for r in got]).astype(np.float32),
        "goal": np.stack([r["goal"] for r in got]).astype(np.float32),
        "gt": np.stack([r["gt"] for r in got]).astype(np.float32),
        "da": np.stack([r["da"] for r in got]).astype(np.float32),
        "kind": np.asarray([r["kind"] for r in got]),
        "x": np.asarray([r["x"] for r in got], np.float32),
        "y": np.asarray([r["y"] for r in got], np.float32),
    }
    FRONT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FRONT_CACHE, **packed)
    print(f"front_sampled {len(got)}", flush=True)
    return packed


def main() -> int:
    if OUT.resolve() == INIT.resolve():
        print("FRONT_REFUSES_OVERWRITE", flush=True)
        return 2
    dev = torch.device("cuda")
    depth_model = loop.DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    front = sample_front(depth_model)
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

    s_da, s_gt, s_kind = shelf["da"], shelf["gt"], shelf["kind"]
    roomy = np.isin(s_kind, ("lane", "side", "block"))
    shelf_r = (
        roomy & (s_da[:, 0] < 2.0) & (s_da[:, 1] < 2.2) & (s_da[:, 2] > 2.5)
        & (s_gt[:, 2] > 2.0) & (s_gt[:, 0] > 0.35)
    )
    shelf_l = (
        roomy & (s_da[:, 0] < 2.0) & (s_da[:, 2] < 2.2) & (s_da[:, 1] > 2.5)
        & (s_gt[:, 1] > 2.0) & (s_gt[:, 0] > 0.35)
    )
    wall_m = np.isin(s_kind, ("wall",)) & (s_gt[:, 0] > 4.0)
    face_wall = np.isin(s_kind, ("wall",)) & (s_gt[:, 0] < 1.2)

    da, gt = front["da"], front["gt"]
    # Nose about a meter, the other side still open. This is the stop the car made.
    turn_r = (da[:, 0] < 1.4) & (da[:, 1] < 1.6) & (da[:, 2] > 2.2) & (gt[:, 2] > 2.0) & (gt[:, 0] > 0.30)
    turn_l = (da[:, 0] < 1.4) & (da[:, 2] < 1.6) & (da[:, 1] > 2.2) & (gt[:, 1] > 2.0) & (gt[:, 0] > 0.30)
    shut = (da[:, 0] < 1.2) & (da[:, 1] < 1.5) & (da[:, 2] < 1.5) & (gt[:, 0] < 1.5)
    clear = (da[:, 0] > 2.5) & (gt[:, 0] > 2.5) & (da[:, 1] > 1.5) & (da[:, 2] > 1.5)
    print(
        f"turn_r {int(turn_r.sum())} turn_l {int(turn_l.sum())} shut {int(shut.sum())} "
        f"clear {int(clear.sum())} shelf_r {int(shelf_r.sum())} wall_open {int(wall_m.sum())} "
        f"wall_face {int(face_wall.sum())} gap {int(gap_m.sum())}",
        flush=True,
    )
    for i in np.flatnonzero(turn_r | turn_l):
        side = "R" if turn_r[i] else "L"
        print(
            f"  turn {side} {front['kind'][i]} x={front['x'][i]:.0f} y={front['y'][i]:.0f} "
            f"gt={gt[i, 0]:.2f},{gt[i, 1]:.2f},{gt[i, 2]:.2f} "
            f"da={da[i, 0]:.2f},{da[i, 1]:.2f},{da[i, 2]:.2f}",
            flush=True,
        )
    if int(turn_r.sum() + turn_l.sum()) < 4 or int(gap_m.sum()) < 5:
        print("FRONT_EMPTY", flush=True)
        return 2

    fd = torch.as_tensor(front["depth"], device=dev)
    fg = torch.as_tensor(front["goal"], device=dev)
    sd = torch.as_tensor(shelf["depth"], device=dev)
    sg = torch.as_tensor(shelf["goal"], device=dev)
    hold_d = torch.as_tensor(hold["depth"], device=dev)
    hold_g = torch.as_tensor(hold["goal"], device=dev)
    turn_r_t = torch.as_tensor(turn_r, device=dev)
    turn_l_t = torch.as_tensor(turn_l, device=dev)
    shut_t = torch.as_tensor(shut, device=dev)
    clear_t = torch.as_tensor(clear, device=dev)
    shelf_r_t = torch.as_tensor(shelf_r, device=dev)
    shelf_l_t = torch.as_tensor(shelf_l, device=dev)
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
    enc_params, rest_params = [], []
    for name, param in model.named_parameters():
        param.requires_grad_(True)
        (enc_params if name.startswith("enc.") else rest_params).append(param)
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": 2e-5}, {"params": rest_params, "lr": 1e-4}]
    )
    fz = torch.zeros(fd.shape[0], 4, device=dev)
    fc = torch.zeros_like(fz)
    fc[:, 0] = 0.8
    sz = torch.zeros(sd.shape[0], 4, device=dev)
    sc = torch.zeros_like(sz)
    sc[:, 0] = 0.8
    hz = torch.zeros(hold_d.shape[0], 4, device=dev)
    hc = torch.zeros_like(hz)
    hc[:, 0] = 0.8
    best = None

    def pack_away(depth, goal, prev, right_t, left_t):
        sr = det.surv_of(model, depth, goal, prev, 0.0, det.YAW)
        sl = det.surv_of(model, depth, goal, prev, 0.0, -det.YAW)
        away, into = [], []
        if bool(right_t.any()):
            away.append(sr[right_t])
            into.append(sl[right_t])
        if bool(left_t.any()):
            away.append(sl[left_t])
            into.append(sr[left_t])
        fwds = []
        detour = right_t | left_t
        for dx, dyaw in det.FWD:
            fwds.append(det.surv_of(model, depth, goal, prev, dx, dyaw)[detour])
        return away, into, fwds

    def report(step: int) -> dict:
        model.eval()
        with torch.no_grad():
            away, into, fwds = [], [], []
            for prev in (fz, fc):
                a, b, c = pack_away(fd, fg, prev, turn_r_t, turn_l_t)
                away += a
                into += b
                fwds += c
            front_away = float(torch.cat(away).min())
            front_into = float(torch.cat(into).max())
            front_fwd = float(torch.cat(fwds).max())
            saway, sinto, sfwds = [], [], []
            for prev in (sz, sc):
                a, b, c = pack_away(sd, sg, prev, shelf_r_t, shelf_l_t)
                saway += a
                sinto += b
                sfwds += c
            shelf_away = float(torch.cat(saway).min()) if saway else 1.0
            shelf_fwd = float(torch.cat(sfwds).max()) if sfwds else 0.0
            shut_vals = []
            for prev in (fz, fc):
                if bool(shut_t.any()):
                    shut_vals.append(det.surv_of(model, fd, fg, prev, 0.8, 0.0)[shut_t])
                    shut_vals.append(det.surv_of(model, fd, fg, prev, 0.0, det.YAW)[shut_t])
                    shut_vals.append(det.surv_of(model, fd, fg, prev, 0.0, -det.YAW)[shut_t])
            shut_max = float(torch.cat(shut_vals).max()) if shut_vals else 0.0
            clear_vals = []
            for prev in (fz, fc):
                if bool(clear_t.any()):
                    clear_vals.append(det.surv_of(model, fd, fg, prev, 0.8, 0.0)[clear_t])
            clear_min = float(torch.cat(clear_vals).min()) if clear_vals else 1.0
            gap_vals, offset_vals, face_vals, wall_vals = [], [], [], []
            for prev in (hz, hc):
                s = det.surv_of(model, hold_d, hold_g, prev, 0.8, 0.0)
                gap_vals.append(s[gap_t])
                offset_vals.append(s[offset_t])
                if bool(face_t.any()):
                    face_vals.append(s[face_t])
                    face_vals.append(det.surv_of(model, hold_d, hold_g, prev, 0.0, det.YAW)[face_t])
                    face_vals.append(det.surv_of(model, hold_d, hold_g, prev, 0.0, -det.YAW)[face_t])
            for prev in (sz, sc):
                wall_s = det.surv_of(model, sd, sg, prev, 0.8, 0.0)
                if bool(wall_t.any()):
                    wall_vals.append(wall_s[wall_t])
                if bool(face_wall_t.any()):
                    face_vals.append(wall_s[face_wall_t])
                    face_vals.append(det.surv_of(model, sd, sg, prev, 0.0, det.YAW)[face_wall_t])
                    face_vals.append(det.surv_of(model, sd, sg, prev, 0.0, -det.YAW)[face_wall_t])
            gap_min = float(torch.cat(gap_vals).min())
            offset_min = float(torch.cat(offset_vals).min())
            wall_min = float(torch.cat(wall_vals).min()) if wall_vals else 1.0
            face_max = float(torch.cat(face_vals).max()) if face_vals else 0.0
            rz = torch.zeros(real_depth.shape[0], 4, device=dev)
            ro = det.surv_of(model, real_depth, real_goal, rz, 0.8, 0.0)
            real_open_min = float(ro[real_open].min()) if bool(real_open.any()) else 1.0
            real_hit_max = float(ro[real_hit].max()) if bool(real_hit.any()) else 0.0
            lz = torch.zeros(live_depth.shape[0], 4, device=dev)
            lo = det.surv_of(model, live_depth, live_goal, lz, 0.8, 0.0)
            live_hit_max = float(lo[live_hit].max()) if bool(live_hit.any()) else 0.0
        print(
            f"step {step} front {front_away:.3f} fin {front_into:.3f} ffwd {front_fwd:.3f} "
            f"shut {shut_max:.3f} clear {clear_min:.3f} shelf {shelf_away:.3f} sfwd {shelf_fwd:.3f} "
            f"gap {gap_min:.3f} wall {wall_min:.3f} offset {offset_min:.3f} face {face_max:.3f} "
            f"real_open {real_open_min:.3f} real_hit {real_hit_max:.3f} live_hit {live_hit_max:.3f}",
            flush=True,
        )
        return {
            "front": front_away,
            "front_into": front_into,
            "front_fwd": front_fwd,
            "shut": shut_max,
            "clear": clear_min,
            "shelf": shelf_away,
            "shelf_fwd": shelf_fwd,
            "gap": gap_min,
            "wall": wall_min,
            "offset": offset_min,
            "face": face_max,
            "real_open": real_open_min,
            "real_hit": real_hit_max,
            "live_hit": live_hit_max,
        }

    def ok(row: dict) -> bool:
        return (
            row["front"] >= 0.95
            and row["front_fwd"] <= 0.90
            and row["front_into"] <= 0.90
            and row["shut"] <= 0.25
            and row["clear"] >= 0.93
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
        loss = fd.new_zeros(())
        for prev in (fz, fc):
            z_r = det.logits(model, fd, fg, prev, 0.0, det.YAW)
            z_l = det.logits(model, fd, fg, prev, 0.0, -det.YAW)
            if bool(turn_r_t.any()):
                loss = loss + 3.0 * det.hinge_safe(z_r[turn_r_t]) + 0.5 * det.hinge_unsafe(z_l[turn_r_t])
                for dx, dyaw in det.FWD:
                    loss = loss + 0.5 * det.hinge_unsafe(det.logits(model, fd, fg, prev, dx, dyaw)[turn_r_t])
            if bool(turn_l_t.any()):
                loss = loss + 3.0 * det.hinge_safe(z_l[turn_l_t]) + 0.5 * det.hinge_unsafe(z_r[turn_l_t])
                for dx, dyaw in det.FWD:
                    loss = loss + 0.5 * det.hinge_unsafe(det.logits(model, fd, fg, prev, dx, dyaw)[turn_l_t])
            if bool(shut_t.any()):
                loss = loss + 4.0 * det.hinge_unsafe(det.logits(model, fd, fg, prev, 0.8, 0.0)[shut_t])
                loss = loss + 4.0 * det.hinge_unsafe(z_r[shut_t]) + 4.0 * det.hinge_unsafe(z_l[shut_t])
            if bool(clear_t.any()):
                loss = loss + 4.0 * det.hinge_safe(det.logits(model, fd, fg, prev, 0.8, 0.0)[clear_t])
                loss = loss + det.hinge_unsafe(z_r[clear_t]) + det.hinge_unsafe(z_l[clear_t])
        for prev in (sz, sc):
            z_r = det.logits(model, sd, sg, prev, 0.0, det.YAW)
            z_l = det.logits(model, sd, sg, prev, 0.0, -det.YAW)
            z_wall = det.logits(model, sd, sg, prev, 0.8, 0.0)
            if bool(shelf_r_t.any()):
                loss = loss + 2.0 * det.hinge_safe(z_r[shelf_r_t]) + 0.5 * det.hinge_unsafe(z_l[shelf_r_t])
                for dx, dyaw in det.FWD:
                    loss = loss + 0.5 * det.hinge_unsafe(det.logits(model, sd, sg, prev, dx, dyaw)[shelf_r_t])
            if bool(shelf_l_t.any()):
                loss = loss + 2.0 * det.hinge_safe(z_l[shelf_l_t]) + 0.5 * det.hinge_unsafe(z_r[shelf_l_t])
                for dx, dyaw in det.FWD:
                    loss = loss + 0.5 * det.hinge_unsafe(det.logits(model, sd, sg, prev, dx, dyaw)[shelf_l_t])
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
            if row["face"] > 0.40 or row["real_open"] < 0.80 or row["real_hit"] > 0.40 or row["shut"] > 0.50:
                print("FRONT_ABORT", flush=True)
                break
            if step >= 40 and (
                row["gap"] < 0.90
                or row["offset"] < 0.90
                or row["wall"] < 0.70
                or row["face"] > 0.25
                or row["real_open"] < 0.90
                or row["shelf"] < 0.90
            ):
                print("FRONT_ABORT", flush=True)
                break
            if ok(row) and (best is None or row["front"] >= best):
                best = row["front"]
                save_checkpoint(OUT, model, {"step": step, "front": best, "front_fwd": row["front_fwd"]})
                print("FRONT_SAVE", step, flush=True)
    if best is None:
        print("FRONT_NO_SAVE", flush=True)
        return 2
    print("FRONT_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
