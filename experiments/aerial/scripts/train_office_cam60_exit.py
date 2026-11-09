#!/usr/bin/env python3
"""Turn around at the west dead end and drive back out.

The walker is unchanged. Facing the end, the left turn is fit above 0.95 and
a forward step stays below it. Once the heading faces the way in, driving is
safe. Init is consequence_cam60_shelf.pt. Writes consequence_cam60_exit.pt.
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

import collect_office_train as col
import run_office_cons_loop as loop
import train_office_align as align
import train_office_cam60_around as base
import train_office_cam60_shelf as shelf
from experiments.aerial.rl.consequence_model import load_checkpoint, save_checkpoint
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "indoor_office_cam60_20261105/c_s218/consequence_cam60_shelf.pt"
OUT = ART / "indoor_office_cam60_20261105/c_s218/consequence_cam60_exit.pt"
WALL_GOAL = (80.0, -250.0)


def back_step(x, y, yaw, dx=0.8):
    rad = math.radians(yaw)
    moved = max(0.0, dx) * 17.5
    return x + moved * math.cos(rad), y + moved * math.sin(rad), yaw


def main() -> int:
    protected = {INIT.resolve(), base.INIT.resolve(), base.OUT.resolve(), shelf.OUT.resolve()}
    if OUT.resolve() in protected:
        raise RuntimeError("refusing to overwrite an existing consequence file")
    poses = []
    for yaw in np.arange(0.0, -180.0, -18.0):
        poses.append((-260.0, -250.0, float(yaw), "exit_turn", WALL_GOAL))
    poses.append((-260.0, -250.0, 180.0, "exit_drive", WALL_GOAL))
    bx, by, byaw = back_step(-260.0, -250.0, 180.0)
    poses.append((bx, by, byaw, "exit_drive", WALL_GOAL))
    for x in (-280.0, -240.0):
        for yaw in (0.0, -90.0, 180.0):
            kind = "exit_drive" if abs(yaw) == 180.0 else "exit_turn"
            poses.append((x, -250.0, yaw, kind, WALL_GOAL))
    poses.append((205.0, -106.0, 12.3, "shelf", base.GOAL_XY))
    poses.append((205.0, -106.0, 30.3, "opened", base.GOAL_XY))

    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    loop.restore_boxes(ucv)
    grabbed = []
    try:
        for x, y, yaw, name, goal_xy in poses:
            col.place(ucv, x, y, yaw)
            col.look(ucv, cam, x, y, yaw)
            packed = col.grab(ucv, cam)
            if packed is None:
                print(f"grab_fail {name} {x:.0f} {yaw:.0f}", flush=True)
                continue
            image, seen, warped = col.warp_pair(*packed)
            gt = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
            grabbed.append((name, x, y, yaw, goal_xy, np.ascontiguousarray(image), gt))
    finally:
        loop.restore_boxes(ucv)
        print("RESTORED", flush=True)

    sim = np.load(base.SIM_NPZ)
    extra = np.load(Path("/data/simworld/run/office_cam60_20261105/open_hold.npz"))
    wrap = DepthMinPredictor.from_checkpoint(str(base.DEPTH), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    depth, da, goals, kind, yaws = [], [], [], [], []

    def add(rgb, gt56, x, y, yaw, gx, gy, name):
        d, a, g, goal = base.encode_rgb(wrap, rgb, gt56, x, y, yaw, gx, gy)
        depth.append(d)
        da.append(a)
        goals.append(goal)
        kind.append(name)
        yaws.append(float(yaw))
        if name.startswith("exit") or name in ("shelf", "opened"):
            print(
                f"{name} x={x:.0f} yaw={yaw:.0f} da={a[0]:.2f},{a[1]:.2f},{a[2]:.2f} gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f}",
                flush=True,
            )

    for i, rgb in enumerate(sim["rgb"]):
        x, y, yaw = (float(v) for v in sim["xy"][i])
        name = str(sim["kind"][i])
        gx, gy = (80.0, -250.0) if name == "wall" else (220.0, -170.0) if name == "side" else base.GOAL_XY
        add(rgb, sim["gt56"][i], x, y, yaw, gx, gy, name)
    for i, rgb in enumerate(extra["rgb"]):
        x, y, yaw = (float(v) for v in extra["xy"][i])
        gx, gy = (float(v) for v in extra["goal"][i])
        add(rgb, extra["gt56"][i], x, y, yaw, gx, gy, "hold")
    for name, x, y, yaw, goal_xy, rgb, gt56 in grabbed:
        add(rgb, gt56, x, y, yaw, goal_xy[0], goal_xy[1], name)
    del wrap
    torch.cuda.empty_cache()

    da_a = np.stack(da)
    kind_a = np.asarray(kind)
    turn = np.flatnonzero((kind_a == "gap") & (da_a[:, 1] < 0.85) & (da_a[:, 2] > 2.5) & (da_a[:, 0] > 1.5))
    opened = np.flatnonzero(
        ((da_a.min(axis=1) > 2.0) & (da_a[:, 0] > 2.5) & np.isin(kind_a, ["aisle", "side", "gap", "hold"]))
        | (kind_a == "opened")
        | (kind_a == "exit_drive")
    )
    nose = np.flatnonzero((kind_a == "front") & (da_a[:, 0] < 0.8) & (da_a[:, 2] > 2.5))
    passed = np.flatnonzero((kind_a == "front_yaw") & (da_a[:, 0] > 2.0))
    shelf_i = np.flatnonzero(kind_a == "shelf")
    exit_i = np.flatnonzero(kind_a == "exit_turn")
    print(
        f"turn {len(turn)} open {len(opened)} nose {len(nose)} pass {len(passed)} "
        f"shelf {len(shelf_i)} exit {len(exit_i)}",
        flush=True,
    )
    if min(len(turn), len(opened), len(nose), len(passed), len(shelf_i), len(exit_i)) == 0:
        print("EXIT_EMPTY", flush=True)
        return 1

    dev = torch.device("cuda")
    depth_t = torch.as_tensor(np.stack(depth), device=dev)
    goal_t = torch.as_tensor(np.stack(goals), device=dev)
    groups = {
        name: torch.as_tensor(index, device=dev)
        for name, index in {
            "turn": turn, "open": opened, "nose": nose, "pass": passed, "shelf": shelf_i, "exit": exit_i,
        }.items()
    }
    real_rgb, real_gt = align.load_real()
    real_model = DepthMinPredictor.from_checkpoint(str(base.DEPTH), device="cuda")
    real_d, real_c, real_g = align.encode_split(real_model, real_rgb, real_gt)
    del real_model
    torch.cuda.empty_cache()
    real_depth = torch.as_tensor(real_d, device=dev)
    real_goal = torch.as_tensor(real_g, device=dev)
    hit_i = torch.as_tensor(np.flatnonzero(real_c[:, 0] < 1.15), device=dev)

    model = load_checkpoint(str(INIT), "cuda")
    drive_out = torch.as_tensor(np.flatnonzero(kind_a == "exit_drive"), device=dev)
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()

    def logits(index, dx, dyaw, prev_dx, depth, goal):
        prev = torch.zeros(index.shape[0], 4, device=dev)
        prev[:, 0] = prev_dx
        return align.logits(model, depth[index], goal[index], prev, dx, dyaw)

    def surv(z):
        return 1.0 - torch.sigmoid(z)

    def report(step: int, quiet: bool = False) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            blocked = torch.stack([
                surv(logits(groups["exit"], dx, dyaw, 0.0, depth_t, goal_t)) for dx, dyaw in shelf.BLOCK
            ])
            stats = {
                "exit": float(surv(logits(groups["exit"], 0.0, -base.YAW, 0.0, depth_t, goal_t)).min()),
                "into": float(surv(logits(groups["exit"], 0.0, base.YAW, 0.0, depth_t, goal_t)).max()),
                "block": float(blocked.max()),
                "drive": float(surv(logits(groups["open"], 0.8, 0.0, 0.0, depth_t, goal_t)).min()),
                "turn": float(surv(logits(groups["turn"], 0.0, base.YAW, 0.8, depth_t, goal_t)).min()),
                "tfwd": float(torch.stack([
                    surv(logits(groups["turn"], dx, dyaw, 0.8, depth_t, goal_t)) for dx, dyaw in base.STRAIGHT
                ]).max()),
                "shelf": float(surv(logits(groups["shelf"], 0.0, base.YAW, 0.8, depth_t, goal_t)).min()),
                "nose": float(surv(logits(groups["nose"], 0.0, base.YAW, 0.0, depth_t, goal_t)).min()),
                "pass": float(surv(logits(groups["pass"], 0.8, 0.0, 0.0, depth_t, goal_t)).min()),
                "hit": float(surv(logits(hit_i, 0.8, 0.0, 0.0, real_depth, real_goal)).max()),
            }
        model.train()
        if not quiet:
            print(
                f"step {step} exit {stats['exit']:.3f} into {stats['into']:.3f} block {stats['block']:.3f} "
                f"drive {stats['drive']:.3f} turn {stats['turn']:.3f} shelf {stats['shelf']:.3f} "
                f"nose {stats['nose']:.3f} pass {stats['pass']:.3f} hit {stats['hit']:.3f}",
                flush=True,
            )
        return stats

    def ok(stats: dict[str, float]) -> bool:
        return (
            stats["exit"] >= 0.95 and stats["into"] <= 0.50 and stats["block"] <= 0.90
            and stats["drive"] >= 0.95 and stats["turn"] >= 0.95 and stats["tfwd"] <= 0.90
            and stats["shelf"] >= 0.95 and stats["nose"] >= 0.95
            and stats["pass"] >= 0.95 and stats["hit"] <= 0.35
        )

    def flat_grad(value):
        grads = torch.autograd.grad(value, params, retain_graph=False, allow_unused=True)
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).detach().reshape(-1)
            for g, p in zip(grads, params)
        ])

    def one_logit(index, dx, dyaw, prev_dx, depth, goal):
        return logits(index[:1] if index.ndim else index.view(1), dx, dyaw, prev_dx, depth, goal)[0]

    def apply_delta(delta):
        offset = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.add_(delta[offset:offset + n].view_as(p))
                offset += n

    def holds(stats: dict[str, float]) -> bool:
        return (
            stats["drive"] >= 0.90 and stats["turn"] >= 0.90 and stats["shelf"] >= 0.90
            and stats["nose"] >= 0.90 and stats["pass"] >= 0.90 and stats["hit"] <= 0.40
        )

    nonspin = (
        (0.80, 0.0), (0.35, 0.0), (1.00, 0.0), (0.20, 0.0), (0.50, 0.0),
        (0.80, 0.10), (0.80, -0.10), (0.80, base.YAW), (0.80, -base.YAW),
        (0.20, base.YAW), (0.20, -base.YAW),
    )
    yaw_a = np.asarray(yaws, np.float64)
    turn_frames = [int(i) for i in groups["exit"].tolist() if abs(yaw_a[i]) <= 36.0]
    turn_frames.sort(key=lambda i: (abs(yaw_a[i]), i))
    almost_out = [int(i) for i in groups["exit"].tolist() if abs(yaw_a[i]) >= 150.0]

    def frame_scores(ids: list[int]):
        idx = torch.as_tensor(ids, device=dev)
        with torch.no_grad():
            left = surv(logits(idx, 0.0, -base.YAW, 0.0, depth_t, goal_t))
            right = surv(logits(idx, 0.0, base.YAW, 0.0, depth_t, goal_t))
            ahead = torch.stack([
                surv(logits(idx, dx, dyaw, 0.0, depth_t, goal_t)) for dx, dyaw in nonspin
            ]).max(dim=0).values
        return left, right, ahead

    def frame_ready(left, right, ahead) -> bool:
        # The walker drives when any non-spin is at least 0.95, and otherwise
        # takes the higher spin. Ahead has to stay under that line, and the
        # left turn has to outscore the right turn.
        return float(left) >= 0.95 and float(ahead) <= 0.945 and float(left) >= float(right) + 0.01

    left0, right0, ahead0 = frame_scores(turn_frames)
    for row, i in enumerate(turn_frames):
        print(
            f"pose {i} yaw={yaw_a[i]:.0f} left {float(left0[row]):.3f} "
            f"right {float(right0[row]):.3f} ahead {float(ahead0[row]):.3f}",
            flush=True,
        )
    report(0)
    open_i = groups["open"][::4]

    def constraint_rows(locked: list[int], skip: tuple[str, ...] = ()):
        cons = []
        skipped = set(skip)
        if "open" not in skipped:
            for j in open_i.tolist():
                cons.append(flat_grad(one_logit(torch.tensor([j], device=dev), 0.8, 0.0, 0.0, depth_t, goal_t)))
        if "drive" not in skipped:
            for j in drive_out.tolist() + almost_out:
                cons.append(flat_grad(one_logit(torch.tensor([j], device=dev), 0.8, 0.0, 0.0, depth_t, goal_t)))
        if "turn" not in skipped:
            for j in groups["turn"].tolist():
                cons.append(flat_grad(one_logit(torch.tensor([j], device=dev), 0.0, base.YAW, 0.8, depth_t, goal_t)))
                cons.append(-flat_grad(one_logit(torch.tensor([j], device=dev), 0.8, 0.0, 0.8, depth_t, goal_t)))
        if "shelf" not in skipped:
            for j in groups["shelf"].tolist():
                cons.append(flat_grad(one_logit(torch.tensor([j], device=dev), 0.0, base.YAW, 0.8, depth_t, goal_t)))
                cons.append(-flat_grad(one_logit(torch.tensor([j], device=dev), 0.8, 0.0, 0.8, depth_t, goal_t)))
        if "nose" not in skipped:
            for j in groups["nose"][:2].tolist():
                cons.append(flat_grad(one_logit(torch.tensor([j], device=dev), 0.0, base.YAW, 0.0, depth_t, goal_t)))
                cons.append(-flat_grad(one_logit(torch.tensor([j], device=dev), 0.8, 0.0, 0.0, depth_t, goal_t)))
        if "pass" not in skipped:
            for j in groups["pass"][:4].tolist():
                cons.append(flat_grad(one_logit(torch.tensor([j], device=dev), 0.8, 0.0, 0.0, depth_t, goal_t)))
        if "hit" not in skipped:
            for j in hit_i[:4].tolist():
                cons.append(-flat_grad(one_logit(torch.tensor([j], device=dev), 0.8, 0.0, 0.0, real_depth, real_goal)))
        for j in locked:
            cons.append(flat_grad(one_logit(torch.tensor([j], device=dev), 0.0, -base.YAW, 0.0, depth_t, goal_t)))
            cons.append(-flat_grad(one_logit(torch.tensor([j], device=dev), 0.8, 0.0, 0.0, depth_t, goal_t)))
            cons.append(-flat_grad(one_logit(torch.tensor([j], device=dev), 0.0, base.YAW, 0.0, depth_t, goal_t)))
        rows = []
        for row in cons:
            if not torch.isfinite(row).all():
                continue
            norm = float(row.norm())
            if norm < 1e-8:
                continue
            rows.append(row / norm)
        return rows

    def projected(target, rows):
        if not rows or not torch.isfinite(target).all() or float(target.norm()) < 1e-8:
            return None, 0.0
        mat = torch.stack(rows).double()
        tgt = (target / target.norm()).double()
        gram = mat @ mat.T + torch.eye(mat.shape[0], dtype=torch.float64, device=dev) * 1e-3
        update = (tgt - mat.T @ torch.linalg.solve(gram, mat @ tgt)).float()
        align_g = float(torch.dot(update, tgt.float()))
        return update, align_g

    shelf_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    locked: list[int] = []
    best_n = -1
    best_state = None
    step = 0

    def locked_still_ready() -> bool:
        if not locked:
            return True
        lk, rk, ak = frame_scores(locked)
        return all(frame_ready(lk[n], rk[n], ak[n]) for n in range(len(locked)))

    def try_step(update, before, sl, sr, fw, kind):
        def improved(nsl, nsr, nfw, z0, z1) -> bool:
            if kind == "ahead":
                # Survival stays at 1 until the logit climbs out of the flat region.
                return z1 > z0 + 0.05 and nsl >= sl - 0.03
            if kind == "left":
                return nsl > sl + 0.002 and nfw <= max(0.945, fw) + 0.004
            return nsr < sr - 0.002 and nsl >= sl - 0.03

        def max_ahead_logit() -> float:
            with torch.no_grad():
                return max(
                    float(logits(torch.tensor([frame], device=dev), dx, dyaw, 0.0, depth_t, goal_t))
                    for dx, dyaw in nonspin
                )

        z0 = max_ahead_logit()
        for cap in (0.018, 0.007):
            model.load_state_dict(before)
            apply_delta(update * (cap / update.norm().clamp_min(1e-12)))
            nleft, nright, nahead = frame_scores([frame])
            nsl, nsr, nfw = float(nleft[0]), float(nright[0]), float(nahead[0])
            z1 = max_ahead_logit()
            stats = report(step, quiet=True)
            if improved(nsl, nsr, nfw, z0, z1) and holds(stats) and locked_still_ready():
                return True
            print(
                f"  reject {kind} cap {cap:.3f} left {sl:.3f}->{nsl:.3f} "
                f"right {sr:.3f}->{nsr:.3f} ahead {fw:.3f}->{nfw:.3f} z {z0:.2f}->{z1:.2f} "
                f"drive {stats['drive']:.3f} turn {stats['turn']:.3f} pass {stats['pass']:.3f}",
                flush=True,
            )
        model.load_state_dict(before)
        return False

    for frame in turn_frames:
        fit = False
        stall = 0
        idx = torch.tensor([frame], device=dev)
        for _attempt in range(80):
            step += 1
            left, right, ahead = frame_scores([frame])
            sl, sr, fw = float(left[0]), float(right[0]), float(ahead[0])
            if frame_ready(sl, sr, fw):
                fit = True
                break
            if sl < 0.95:
                kind = "left"
            elif sr + 0.01 > sl:
                kind = "right"
            else:
                kind = "ahead"
            rows = constraint_rows(locked, skip=("open",))
            if kind == "ahead":
                with torch.no_grad():
                    worst = max(
                        nonspin,
                        key=lambda a: float(surv(logits(idx, a[0], a[1], 0.0, depth_t, goal_t))),
                    )
                target = flat_grad(one_logit(idx, worst[0], worst[1], 0.0, depth_t, goal_t))
                guard = flat_grad(one_logit(idx, 0.0, -base.YAW, 0.0, depth_t, goal_t))
                if float(guard.norm()) > 1e-8:
                    rows.append(guard / guard.norm())
            elif kind == "left":
                target = -flat_grad(one_logit(idx, 0.0, -base.YAW, 0.0, depth_t, goal_t))
                with torch.no_grad():
                    worst = max(
                        nonspin,
                        key=lambda a: float(surv(logits(idx, a[0], a[1], 0.0, depth_t, goal_t))),
                    )
                for dx, dyaw in ((0.8, 0.0), worst):
                    guard = -flat_grad(one_logit(idx, dx, dyaw, 0.0, depth_t, goal_t))
                    if float(guard.norm()) > 1e-8:
                        rows.append(guard / guard.norm())
            else:
                target = flat_grad(one_logit(idx, 0.0, base.YAW, 0.0, depth_t, goal_t))
                guard = flat_grad(one_logit(idx, 0.0, -base.YAW, 0.0, depth_t, goal_t))
                if float(guard.norm()) > 1e-8:
                    rows.append(guard / guard.norm())
            update, align_g = projected(target, rows)
            if update is None or align_g < 0.004:
                stall += 1
                print(
                    f"step {step} frame {frame} yaw={yaw_a[frame]:.0f} {kind} orth {align_g:.4f} "
                    f"left {sl:.3f} right {sr:.3f} ahead {fw:.3f}",
                    flush=True,
                )
                if stall >= 4:
                    break
                continue
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if not try_step(update, before, sl, sr, fw, kind):
                stall += 1
                print(f"step {step} frame {frame} yaw={yaw_a[frame]:.0f} {kind} revert left {sl:.3f} ahead {fw:.3f}", flush=True)
                if stall >= 4:
                    break
                continue
            stall = 0
            if step % 8 == 0:
                nleft, nright, nahead = frame_scores([frame])
                print(
                    f"step {step} frame {frame} yaw={yaw_a[frame]:.0f} {kind} "
                    f"left {float(nleft[0]):.3f} right {float(nright[0]):.3f} ahead {float(nahead[0]):.3f}",
                    flush=True,
                )
        if fit:
            locked.append(frame)
            left, right, ahead = frame_scores([frame])
            print(
                f"LOCKED {frame} yaw={yaw_a[frame]:.0f} left {float(left[0]):.3f} "
                f"right {float(right[0]):.3f} ahead {float(ahead[0]):.3f}",
                flush=True,
            )
        else:
            left, right, ahead = frame_scores([frame])
            print(
                f"MISS {frame} yaw={yaw_a[frame]:.0f} left {float(left[0]):.3f} "
                f"right {float(right[0]):.3f} ahead {float(ahead[0]):.3f}",
                flush=True,
            )
            model.load_state_dict(best_state if best_state is not None else shelf_state)
        if len(locked) > best_n:
            best_n = len(locked)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    stats = report(step)
    start_id = turn_frames[0]
    start_ok = start_id in locked
    if best_state is not None and start_ok and holds(stats):
        model.load_state_dict(best_state)
        save_checkpoint(OUT, model.eval(), {"locked": best_n, "fallback": best_n < len(turn_frames)})
        print(f"EXIT_SAVE locked {best_n}/{len(turn_frames)}", flush=True)
        print("EXIT_OUT", OUT, flush=True)
        return 0
    print(f"EXIT_UNMET locked {best_n}/{len(turn_frames)} start {start_ok}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
