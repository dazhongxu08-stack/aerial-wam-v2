#!/usr/bin/env python3
"""Teach a turn away from a side that is closing, without touching the old weights.

Straight travel stays the choice while it is safe. On frames where the depth
map shows one side open and the other side or the nose closing, the away
spin has to clear 0.95 and the forward steps have to fall under it. The
nearmiss consequence, the align depth, and the gap consequence are not
overwritten.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/aerial-wam-v2/experiments/aerial/scripts")

import run_office_cons_loop as loop
import train_office_align as align
from experiments.aerial.rl.consequence_model import (
    depth_input,
    goal_features,
    load_checkpoint,
    save_checkpoint,
)

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
DEPTH = ART / "da3_office_align_20261104/depth_office_cone_align.pt"
INIT = ART / "indoor_office_align_20261104/c_s218/consequence_nearmiss.pt"
OUT = ART / "indoor_office_align_20261104/c_s218/consequence_detour.pt"
HOLD = Path("/data/simworld/run/office_align_nearmiss_20261104/poses.npz")
CACHE = Path("/data/simworld/run/office_align_detour_20261104/poses.npz")
YAW = 0.3141592653589793
STEPS = 80
FWD = ((0.8, 0.0), (0.35, 0.0), (0.6, 0.0), (1.0, 0.0), (0.8, 0.04), (0.8, -0.04))


def pose_row(depth_model, ucv, cam, x, y, yaw, goal_xy, kind):
    seen = loop.sense(ucv, cam, depth_model, float(x), float(y), float(yaw))
    if seen is None:
        print(f"grab_fail {kind} {x:.0f}", flush=True)
        return None
    state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, math.radians(yaw)], np.float64)
    goal = np.array([goal_xy[0] / 100.0, goal_xy[1] / 100.0, 0.52], np.float64)
    gt = np.asarray(seen["gt"], np.float32)
    da = np.asarray(seen["da"], np.float32)
    print(
        f"{kind} x={x:.0f} gt={gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f} "
        f"da={da[0]:.2f},{da[1]:.2f},{da[2]:.2f}",
        flush=True,
    )
    return {
        "depth": depth_input(np.asarray(seen["depth"], np.float32)),
        "goal": goal_features(state, goal),
        "gt": gt,
        "da": da,
        "kind": kind,
        "x": float(x),
        "y": float(y),
    }


def sample(depth_model) -> dict[str, np.ndarray]:
    if CACHE.is_file():
        z = np.load(CACHE, allow_pickle=False)
        print(f"cache {len(z['depth'])}", flush=True)
        return {k: z[k] for k in z.files}
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(loop.col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    loop.boxes.save_restore(ucv)
    loop.restore_boxes(ucv)
    rows = []
    for x in np.arange(-450.0, -255.0, 10.0):
        rows.append((float(x), -200.0, 0.0, (200.0, -200.0), "lane"))
    for x in (-500.0, -460.0, -420.0, -380.0, -350.0, -340.0, -335.0, -325.0):
        rows.append((x, -250.0, 0.0, (80.0, -250.0), "wall"))
    got = []
    for x, y, yaw, goal, kind in rows:
        row = pose_row(depth_model, ucv, cam, x, y, yaw, goal, kind)
        if row is not None:
            got.append(row)
    loop.boxes.place_boxes(ucv)
    for x in np.arange(-420.0, -245.0, 15.0):
        row = pose_row(depth_model, ucv, cam, float(x), -170.0, 0.0, (220.0, -170.0), "side")
        if row is not None:
            got.append(row)
    loop.restore_boxes(ucv)
    packed = {
        "depth": np.stack([r["depth"] for r in got]).astype(np.float32),
        "goal": np.stack([r["goal"] for r in got]).astype(np.float32),
        "gt": np.stack([r["gt"] for r in got]).astype(np.float32),
        "da": np.stack([r["da"] for r in got]).astype(np.float32),
        "kind": np.asarray([r["kind"] for r in got]),
        "x": np.asarray([r["x"] for r in got], np.float32),
        "y": np.asarray([r["y"] for r in got], np.float32),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **packed)
    print(f"sampled {len(got)}", flush=True)
    return packed


def logits(model, depth, goal, prev, dx, dyaw):
    actions = torch.zeros(depth.shape[0], 10, 4, device=depth.device)
    actions[:, 0, 0] = dx
    actions[:, 0, 3] = dyaw
    return model(depth, goal, prev, actions)["hazard_logit"][:, 0]


def surv_of(model, depth, goal, prev, dx, dyaw):
    z = logits(model, depth, goal, prev, dx, dyaw)
    return 1.0 - torch.sigmoid(z)


def hinge_safe(z):
    return torch.nn.functional.softplus(z + 3.0).mean()


def hinge_unsafe(z):
    return torch.nn.functional.softplus(2.0 - z).mean()


def anchor_high(z, base, slack=0.08):
    """Worst frame's logit must not rise. A safe action stays safe."""
    return torch.relu(z - base - slack).pow(2).amax()


def anchor_low(z, base, slack=0.08):
    """Worst frame's logit must not fall. An unsafe action stays unsafe."""
    return torch.relu(base - slack - z).pow(2).amax()


def main() -> int:
    dev = torch.device("cuda")
    depth_model = loop.DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    fresh = sample(depth_model)
    real_rgb, real_gt = align.load_real()
    live = np.load(align.SIM_LIVE)
    real_d, real_c, real_g = align.encode_split(depth_model, real_rgb, real_gt)
    live_d, live_c, live_g = align.encode_split(
        depth_model, live["rgb"], live["gt56"].astype(np.float32)
    )
    del depth_model
    torch.cuda.empty_cache()

    hold = np.load(HOLD, allow_pickle=False)
    hold_kind = hold["kind"]
    hold_gt = hold["cones"]
    gap_m = (hold_kind == "gap") & (hold_gt[:, 0] > 2.5) & (hold_gt[:, 1] > 2.0) & (hold_gt[:, 2] > 2.0)
    offset_m = (hold_kind == "offset") & (hold_gt[:, 0] > 2.5)
    face_m = (hold_kind == "face") & (hold_gt[:, 0] < 1.15)
    wall_m = np.isin(fresh["kind"], ("wall",)) & (fresh["gt"][:, 0] > 4.0)
    face_wall = np.isin(fresh["kind"], ("wall",)) & (fresh["gt"][:, 0] < 1.2)

    da = fresh["da"]
    gt = fresh["gt"]
    kind = fresh["kind"]
    # Forward has to look short as well. A gap frame can show a short left cone
    # while the nose is still past 2.5 m, and that must stay a straight drive.
    roomy = np.isin(kind, ("lane", "side", "block"))
    right = (
        roomy & (da[:, 0] < 2.0) & (da[:, 1] < 2.2) & (da[:, 2] > 2.5)
        & (gt[:, 2] > 2.0) & (gt[:, 0] > 0.35)
    )
    left = (
        roomy & (da[:, 0] < 2.0) & (da[:, 2] < 2.2) & (da[:, 1] > 2.5)
        & (gt[:, 1] > 2.0) & (gt[:, 0] > 0.35)
    )
    print(
        f"gap {int(gap_m.sum())} offset {int(offset_m.sum())} face {int(face_m.sum())} wall_open {int(wall_m.sum())} "
        f"wall_face {int(face_wall.sum())} detour_r {int(right.sum())} detour_l {int(left.sum())}",
        flush=True,
    )
    for i in np.flatnonzero(right | left):
        side = "R" if right[i] else "L"
        print(
            f"  detour {side} {kind[i]} x={fresh['x'][i]:.0f} "
            f"gt={gt[i, 0]:.2f},{gt[i, 1]:.2f},{gt[i, 2]:.2f} "
            f"da={da[i, 0]:.2f},{da[i, 1]:.2f},{da[i, 2]:.2f}",
            flush=True,
        )
    if int(right.sum() + left.sum()) < 3 or int(gap_m.sum()) < 5:
        print("DETOUR_EMPTY", flush=True)
        return 2

    depth = torch.as_tensor(fresh["depth"], device=dev)
    goal = torch.as_tensor(fresh["goal"], device=dev)
    hold_d = torch.as_tensor(hold["depth"], device=dev)
    hold_g = torch.as_tensor(hold["goal"], device=dev)
    right_t = torch.as_tensor(right, device=dev)
    left_t = torch.as_tensor(left, device=dev)
    gap_t = torch.as_tensor(gap_m, device=dev)
    offset_t = torch.as_tensor(offset_m, device=dev)
    face_t = torch.as_tensor(face_m, device=dev)
    wall_t = torch.as_tensor(wall_m, device=dev)
    face_wall_t = torch.as_tensor(face_wall, device=dev)
    real_depth = torch.as_tensor(real_d, device=dev)
    real_goal = torch.as_tensor(real_g, device=dev)
    live_depth = torch.as_tensor(live_d, device=dev)
    live_goal = torch.as_tensor(live_g, device=dev)
    real_open = torch.as_tensor((real_c[:, 0] > 3.0) & (real_c[:, 1] > 1.5) & (real_c[:, 2] > 1.5), device=dev)
    real_hit = torch.as_tensor(real_c[:, 0] < 1.15, device=dev)
    live_hit = torch.as_tensor(live_c[:, 0] < 1.15, device=dev)

    model = load_checkpoint(str(INIT), "cuda")
    # Offset frames sit on the same features as the turn. The conv stack has to
    # move a little, or holding that escape path cancels the turn entirely.
    enc_params, rest_params = [], []
    for name, param in model.named_parameters():
        param.requires_grad_(True)
        (enc_params if name.startswith("enc.") else rest_params).append(param)
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": 2e-5}, {"params": rest_params, "lr": 1e-4}]
    )
    zero = torch.zeros(depth.shape[0], 4, device=dev)
    carry = torch.zeros_like(zero)
    carry[:, 0] = 0.8
    hold_zero = torch.zeros(hold_d.shape[0], 4, device=dev)
    hold_carry = torch.zeros_like(hold_zero)
    hold_carry[:, 0] = 0.8
    best = None

    def report(step: int) -> dict:
        model.eval()
        with torch.no_grad():
            away = []
            into = []
            fwd_max = []
            for prev in (zero, carry):
                sr = surv_of(model, depth, goal, prev, 0.0, YAW)
                sl = surv_of(model, depth, goal, prev, 0.0, -YAW)
                if bool(right_t.any()):
                    away.append(sr[right_t])
                    into.append(sl[right_t])
                if bool(left_t.any()):
                    away.append(sl[left_t])
                    into.append(sr[left_t])
                fwd_here = []
                for dx, dyaw in FWD:
                    fwd_here.append(surv_of(model, depth, goal, prev, dx, dyaw))
                fwd_stack = torch.stack(fwd_here, 0)
                detour = right_t | left_t
                fwd_max.append(fwd_stack[:, detour].max(dim=0).values)
            away_min = float(torch.cat(away).min())
            into_max = float(torch.cat(into).max())
            fwd_detour = float(torch.cat(fwd_max).max())
            gap_vals = []
            offset_vals = []
            face_vals = []
            wall_vals = []
            for prev in (hold_zero, hold_carry):
                s = surv_of(model, hold_d, hold_g, prev, 0.8, 0.0)
                gap_vals.append(s[gap_t])
                offset_vals.append(s[offset_t])
                if bool(face_t.any()):
                    face_vals.append(s[face_t])
                    face_vals.append(surv_of(model, hold_d, hold_g, prev, 0.0, YAW)[face_t])
                    face_vals.append(surv_of(model, hold_d, hold_g, prev, 0.0, -YAW)[face_t])
            for prev in (zero, carry):
                wall_s = surv_of(model, depth, goal, prev, 0.8, 0.0)
                if bool(wall_t.any()):
                    wall_vals.append(wall_s[wall_t])
                if bool(face_wall_t.any()):
                    face_vals.append(wall_s[face_wall_t])
                    face_vals.append(surv_of(model, depth, goal, prev, 0.0, YAW)[face_wall_t])
                    face_vals.append(surv_of(model, depth, goal, prev, 0.0, -YAW)[face_wall_t])
            gap_min = float(torch.cat(gap_vals).min())
            offset_min = float(torch.cat(offset_vals).min())
            wall_min = float(torch.cat(wall_vals).min()) if wall_vals else 1.0
            face_max = float(torch.cat(face_vals).max()) if face_vals else 0.0
            rz = torch.zeros(real_depth.shape[0], 4, device=dev)
            ro = surv_of(model, real_depth, real_goal, rz, 0.8, 0.0)
            real_open_min = float(ro[real_open].min()) if bool(real_open.any()) else 1.0
            real_hit_max = float(ro[real_hit].max()) if bool(real_hit.any()) else 0.0
            lz = torch.zeros(live_depth.shape[0], 4, device=dev)
            lo = surv_of(model, live_depth, live_goal, lz, 0.8, 0.0)
            live_hit_max = float(lo[live_hit].max()) if bool(live_hit.any()) else 0.0
        print(
            f"step {step} away {away_min:.3f} into {into_max:.3f} fwd_detour {fwd_detour:.3f} "
            f"gap {gap_min:.3f} wall {wall_min:.3f} offset {offset_min:.3f} face {face_max:.3f} "
            f"real_open {real_open_min:.3f} real_hit {real_hit_max:.3f} live_hit {live_hit_max:.3f}",
            flush=True,
        )
        return {
            "away": away_min,
            "into": into_max,
            "fwd_detour": fwd_detour,
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
            row["away"] >= 0.95
            and row["fwd_detour"] <= 0.90
            and row["into"] <= 0.90
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
        loss = depth.new_zeros(())
        for prev in (zero, carry):
            z_r = logits(model, depth, goal, prev, 0.0, YAW)
            z_l = logits(model, depth, goal, prev, 0.0, -YAW)
            z_wall = logits(model, depth, goal, prev, 0.8, 0.0)
            if bool(right_t.any()):
                loss = loss + 2.0 * hinge_safe(z_r[right_t]) + 0.5 * hinge_unsafe(z_l[right_t])
                for dx, dyaw in FWD:
                    loss = loss + 0.5 * hinge_unsafe(logits(model, depth, goal, prev, dx, dyaw)[right_t])
            if bool(left_t.any()):
                loss = loss + 2.0 * hinge_safe(z_l[left_t]) + 0.5 * hinge_unsafe(z_r[left_t])
                for dx, dyaw in FWD:
                    loss = loss + 0.5 * hinge_unsafe(logits(model, depth, goal, prev, dx, dyaw)[left_t])
            if bool(wall_t.any()):
                loss = loss + hinge_safe(z_wall[wall_t])
            if bool(face_wall_t.any()):
                loss = loss + 4.0 * hinge_unsafe(z_wall[face_wall_t])
                loss = loss + 4.0 * hinge_unsafe(z_r[face_wall_t])
                loss = loss + 4.0 * hinge_unsafe(z_l[face_wall_t])
        for prev in (hold_zero, hold_carry):
            z = logits(model, hold_d, hold_g, prev, 0.8, 0.0)
            z_sr = logits(model, hold_d, hold_g, prev, 0.0, YAW)
            z_sl = logits(model, hold_d, hold_g, prev, 0.0, -YAW)
            loss = loss + 4.0 * hinge_safe(z[gap_t]) + 4.0 * hinge_safe(z[offset_t])
            loss = loss + hinge_unsafe(z_sr[gap_t]) + hinge_unsafe(z_sl[gap_t])
            if bool(face_t.any()):
                loss = loss + 4.0 * hinge_unsafe(z[face_t])
                loss = loss + 4.0 * hinge_unsafe(z_sr[face_t])
                loss = loss + 4.0 * hinge_unsafe(z_sl[face_t])
        rz = torch.zeros(real_depth.shape[0], 4, device=dev)
        z_real = logits(model, real_depth, real_goal, rz, 0.8, 0.0)
        if bool(real_open.any()):
            loss = loss + 4.0 * hinge_safe(z_real[real_open])
        if bool(real_hit.any()):
            loss = loss + 4.0 * hinge_unsafe(z_real[real_hit])
        lz = torch.zeros(live_depth.shape[0], 4, device=dev)
        z_live = logits(model, live_depth, live_goal, lz, 0.8, 0.0)
        if bool(live_hit.any()):
            loss = loss + 4.0 * hinge_unsafe(z_live[live_hit])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 8.0)
        opt.step()
        if step % 5 == 0 or step == STEPS:
            row = report(step)
            # The wall approach dips while the turn is forming, then comes back.
            if row["face"] > 0.40 or row["real_open"] < 0.80 or row["real_hit"] > 0.40:
                print("DETOUR_ABORT", flush=True)
                break
            if step >= 40 and (
                row["gap"] < 0.90
                or row["offset"] < 0.90
                or row["wall"] < 0.70
                or row["face"] > 0.25
                or row["real_open"] < 0.90
            ):
                print("DETOUR_ABORT", flush=True)
                break
            if ok(row) and (best is None or row["away"] >= best):
                best = row["away"]
                save_checkpoint(OUT, model, {"step": step, "away": best, "fwd_detour": row["fwd_detour"]})
                print("DETOUR_SAVE", step, flush=True)
    if best is None:
        print("DETOUR_NO_SAVE", flush=True)
        return 2
    print("DETOUR_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
