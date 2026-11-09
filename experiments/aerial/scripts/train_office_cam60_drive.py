#!/usr/bin/env python3
"""Open forward drives. The exit weights did the opposite and are not updated.

Facing an open aisle, a forward step stays at or above 0.95, so the walker
drives toward the goal. A real box face still turns and does not drive in.
Init is consequence_cam60_exit.pt. Writes consequence_cam60_drive.pt.
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
import sim_verify_gap as sim
import train_office_align as align
import train_office_cam60_around as base
import train_office_cam60_shelf as shelf
from experiments.aerial.rl.consequence_model import load_checkpoint, save_checkpoint
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts/indoor_office_cam60_20261105/c_s218")
INIT = ART / "consequence_cam60_exit.pt"
OUT = ART / "consequence_cam60_drive.pt"
WALL_GOAL = (80.0, -250.0)
PROTECTED = {
    "consequence_cam60.pt",
    "consequence_cam60_pass.pt",
    "consequence_cam60_around.pt",
    "consequence_cam60_around_open.pt",
    "consequence_cam60_shelf.pt",
    "consequence_cam60_exit.pt",
    "consequence_open.pt",
    "consequence_hold.pt",
    "consequence_yield.pt",
    "depth_office_cone_cam60.pt",
    "depth_office_cone_align.pt",
}
# Walker drives once any non-spin reaches 0.95. Stay above that line.
DRIVE_LINE = 0.96


def score_next(models, depth56, state, goal, cands, prev, da_f=0.0):
    curve, prog = sim._curves(models, depth56, state, goal, cands, prev)
    return curve[:, 0], prog


def pick_learned(da, low, names, cands, surv, allowed, bearing):
    surv = np.asarray(surv, np.float64)
    names = list(names)
    cands = np.asarray(cands, np.float64)
    ok = np.flatnonzero(surv >= loop.SURV_MIN)
    if ok.size:
        moving = [int(i) for i in ok if "spin" not in names[int(i)]]
        if moving:
            dx = cands[moving, 0, 0]
            chosen = int(moving[int(np.argmax(dx))])
        else:
            chosen = int(ok[int(np.argmax(surv[ok]))])
        return loop.first_cmd(cands[chosen]), str(names[chosen]), float(surv[chosen])
    return np.zeros(4, np.float64), "stop", float(np.max(surv))


def main() -> int:
    if OUT.name in PROTECTED or INIT.name in PROTECTED and OUT.resolve() == INIT.resolve():
        raise RuntimeError("refusing to overwrite an existing consequence file")
    if OUT.name in PROTECTED:
        raise RuntimeError("refusing to overwrite an existing consequence file")
    poses = []
    for x in (-520.0, -506.0, -492.0, -478.0, -464.0, -450.0, -436.0, -260.0):
        poses.append((x, -250.0, 0.0, "drive", WALL_GOAL))
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
                print(f"grab_fail {name} {x:.0f}", flush=True)
                continue
            image, seen, warped = col.warp_pair(*packed)
            gt = np.asarray(col.pool_min(col.hide(warped, seen)), np.float32)
            grabbed.append((name, x, y, yaw, goal_xy, np.ascontiguousarray(image), gt))
    except Exception:
        loop.restore_boxes(ucv)
        raise

    sim_npz = np.load(base.SIM_NPZ)
    extra = np.load(Path("/data/simworld/run/office_cam60_20261105/open_hold.npz"))
    wrap = DepthMinPredictor.from_checkpoint(str(base.DEPTH), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    depth, da, goals, kind, xy = [], [], [], [], []

    def add(rgb, gt56, x, y, yaw, gx, gy, name):
        d, a, g, goal = base.encode_rgb(wrap, rgb, gt56, x, y, yaw, gx, gy)
        depth.append(d)
        da.append(a)
        goals.append(goal)
        kind.append(name)
        xy.append((x, y, yaw))
        if name in ("drive", "shelf", "opened"):
            print(
                f"{name} x={x:.0f} yaw={yaw:.0f} da={a[0]:.2f},{a[1]:.2f},{a[2]:.2f} "
                f"gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f}",
                flush=True,
            )

    for i, rgb in enumerate(sim_npz["rgb"]):
        x, y, yaw = (float(v) for v in sim_npz["xy"][i])
        name = str(sim_npz["kind"][i])
        gx, gy = (80.0, -250.0) if name == "wall" else (220.0, -170.0) if name == "side" else base.GOAL_XY
        add(rgb, sim_npz["gt56"][i], x, y, yaw, gx, gy, name)
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
        | ((kind_a == "opened") & (da_a[:, 0] > 2.0))
    )
    nose = np.flatnonzero((kind_a == "front") & (da_a[:, 0] < 0.8) & (da_a[:, 2] > 2.5))
    passed = np.flatnonzero((kind_a == "front_yaw") & (da_a[:, 0] > 2.0))
    shelf_i = np.flatnonzero(kind_a == "shelf")
    drive_i = np.flatnonzero(kind_a == "drive")
    print(
        f"turn {len(turn)} open {len(opened)} nose {len(nose)} pass {len(passed)} "
        f"shelf {len(shelf_i)} drive {len(drive_i)}",
        flush=True,
    )
    if min(len(turn), len(opened), len(nose), len(passed), len(shelf_i), len(drive_i)) == 0:
        print("DRIVE_EMPTY", flush=True)
        loop.restore_boxes(ucv)
        return 1

    dev = torch.device("cuda")
    depth_t = torch.as_tensor(np.stack(depth), device=dev)
    goal_t = torch.as_tensor(np.stack(goals), device=dev)
    groups = {
        name: torch.as_tensor(index, device=dev)
        for name, index in {
            "turn": turn, "open": opened, "nose": nose, "pass": passed, "shelf": shelf_i, "drive": drive_i,
        }.items()
    }
    real_rgb, real_gt = align.load_real()
    real_model = DepthMinPredictor.from_checkpoint(str(base.DEPTH), device="cuda")
    real_d, _real_c, real_g = align.encode_split(real_model, real_rgb, real_gt)
    real_c = _real_c
    del real_model
    torch.cuda.empty_cache()
    real_depth = torch.as_tensor(real_d, device=dev)
    real_goal = torch.as_tensor(real_g, device=dev)
    hit_i = torch.as_tensor(np.flatnonzero(real_c[:, 0] < 1.15), device=dev)

    model = load_checkpoint(str(INIT), "cuda")
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    def logits(index, dx, dyaw, prev_dx, depth, goal):
        prev = torch.zeros(index.shape[0], 4, device=dev)
        prev[:, 0] = prev_dx
        return align.logits(model, depth[index], goal[index], prev, dx, dyaw)

    def surv(z):
        return 1.0 - torch.sigmoid(z)

    def one(index, dx, dyaw, depth, goal):
        idx = torch.as_tensor([index], device=dev)
        return logits(idx, dx, dyaw, 0.0, depth, goal)[0]

    def flat_grad(value):
        grads = torch.autograd.grad(value, params, retain_graph=False, allow_unused=True)
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).detach().reshape(-1)
            for g, p in zip(grads, params)
        ])

    def apply_delta(delta):
        offset = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.add_(delta[offset:offset + n].view_as(p))
                offset += n

    def fwd_surv(ids):
        idx = torch.as_tensor(list(ids), device=dev)
        model.eval()
        with torch.no_grad():
            out = surv(logits(idx, 0.8, 0.0, 0.0, depth_t, goal_t)).detach().cpu().numpy()
        model.train()
        return out

    def report(step: int) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            blocked = torch.stack([
                surv(logits(groups["shelf"], dx, dyaw, 0.8, depth_t, goal_t)) for dx, dyaw in shelf.BLOCK
            ])
            live = surv(logits(groups["drive"], 0.8, 0.0, 0.0, depth_t, goal_t))
            stats = {
                "live": float(live.min()),
                "drive": float(surv(logits(groups["open"], 0.8, 0.0, 0.0, depth_t, goal_t)).min()),
                "turn": float(surv(logits(groups["turn"], 0.0, base.YAW, 0.8, depth_t, goal_t)).min()),
                "tfwd": float(torch.stack([
                    surv(logits(groups["turn"], dx, dyaw, 0.8, depth_t, goal_t)) for dx, dyaw in base.STRAIGHT
                ]).max()),
                "shelf": float(surv(logits(groups["shelf"], 0.0, base.YAW, 0.8, depth_t, goal_t)).min()),
                "block": float(blocked.max()),
                "nose": float(surv(logits(groups["nose"], 0.0, base.YAW, 0.0, depth_t, goal_t)).min()),
                "nose_fwd": float(surv(logits(groups["nose"], 0.8, 0.0, 0.0, depth_t, goal_t)).max()),
                "pass": float(surv(logits(groups["pass"], 0.8, 0.0, 0.0, depth_t, goal_t)).min()),
                "hit": float(surv(logits(hit_i, 0.8, 0.0, 0.0, real_depth, real_goal)).max()),
            }
        model.train()
        xs = [xy[int(i)][0] for i in drive_i]
        bits = " ".join(f"{x:.0f}:{v:.3f}" for x, v in zip(xs, live.detach().cpu().numpy()))
        print(
            f"step {step} live {stats['live']:.3f} [{bits}] drive {stats['drive']:.3f} "
            f"turn {stats['turn']:.3f} tfwd {stats['tfwd']:.3f} shelf {stats['shelf']:.3f} "
            f"block {stats['block']:.3f} nose {stats['nose']:.3f} nfwd {stats['nose_fwd']:.3f} "
            f"pass {stats['pass']:.3f} hit {stats['hit']:.3f}",
            flush=True,
        )
        return stats

    def holds(stats: dict[str, float], floor: dict[str, float]) -> bool:
        return (
            stats["shelf"] >= min(0.95, floor["shelf"] - 0.02)
            and stats["block"] <= max(0.90, floor["block"] + 0.03)
            and stats["turn"] >= min(0.95, floor["turn"] - 0.02)
            and stats["tfwd"] <= max(0.90, floor["tfwd"] + 0.03)
            and stats["nose"] >= min(0.95, floor["nose"] - 0.02)
            and stats["nose_fwd"] <= max(0.35, floor["nose_fwd"] + 0.03)
            and stats["pass"] >= min(0.95, floor["pass"] - 0.02)
            and stats["hit"] <= max(0.35, floor["hit"] + 0.03)
            and stats["drive"] >= min(0.95, floor["drive"] - 0.02)
        )

    def ready(stats: dict[str, float]) -> bool:
        return (
            stats["live"] >= DRIVE_LINE
            and stats["drive"] >= 0.95
            and stats["shelf"] >= 0.95
            and stats["block"] <= 0.90
            and stats["turn"] >= 0.95
            and stats["tfwd"] <= 0.90
            and stats["nose"] >= 0.95
            and stats["nose_fwd"] <= 0.35
            and stats["pass"] >= 0.95
            and stats["hit"] <= 0.35
        )

    def projected(target, rows):
        if not rows or not torch.isfinite(target).all() or float(target.norm()) < 1e-8:
            return None
        mat = torch.stack(rows).double()
        tgt = (target / target.norm()).double()
        gram = mat @ mat.T + torch.eye(mat.shape[0], dtype=torch.float64, device=dev) * 1e-3
        update = (tgt - mat.T @ torch.linalg.solve(gram, mat @ tgt)).float()
        if float(torch.dot(update, tgt.float())) < 0.004:
            return None
        return update

    def guard_rows(skip: int):
        specs = []
        specs.append((int(shelf_i[0]), 0.8, 0.0, depth_t, goal_t))
        specs.append((int(shelf_i[0]), 0.0, base.YAW, depth_t, goal_t))
        for j in list(nose[:1]) + list(turn[:2]):
            specs.append((int(j), 0.8, 0.0, depth_t, goal_t))
        specs.append((int(nose[0]), 0.0, base.YAW, depth_t, goal_t))
        specs.append((int(turn[0]), 0.0, base.YAW, depth_t, goal_t))
        for j in hit_i[:2].tolist():
            specs.append((int(j), 0.8, 0.0, real_depth, real_goal))
        rows = []
        for j, dx, dyaw, depth, goal in specs:
            if j == skip:
                continue
            row = flat_grad(one(j, dx, dyaw, depth, goal))
            if not torch.isfinite(row).all():
                continue
            norm = float(row.norm())
            if norm < 1e-8:
                continue
            rows.append(row / norm)
        return rows

    floor = report(0)
    step = 0
    banned: set[int] = set()
    for _round in range(48):
        stats = report(step) if step else floor
        if ready(stats):
            break
        live_s = fwd_surv(drive_i)
        open_s = fwd_surv(opened)
        pass_s = fwd_surv(passed)
        under = [(float(live_s[n]), int(drive_i[n])) for n in range(len(drive_i)) if float(live_s[n]) < DRIVE_LINE]
        under += [(float(open_s[n]), int(opened[n])) for n in range(len(opened)) if float(open_s[n]) < 0.95]
        under += [(float(pass_s[n]), int(passed[n])) for n in range(len(passed)) if float(pass_s[n]) < 0.95]
        under = [item for item in under if item[1] not in banned]
        if not under:
            break
        under.sort()
        frame = under[0][1]
        before_s = under[0][0]
        target = -flat_grad(one(frame, 0.8, 0.0, depth_t, goal_t))
        update = projected(target, guard_rows(frame))
        step += 1
        if update is None:
            update = target
            print(f"step {step} frame {frame} unprojected surv {before_s:.3f}", flush=True)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        accepted = False
        for cap in (0.004, 0.0015, 0.0006):
            model.load_state_dict(before)
            apply_delta(update * (cap / update.norm().clamp_min(1e-12)))
            after = float(fwd_surv([frame])[0])
            stats = report(step)
            if after >= before_s + 0.004 and holds(stats, floor):
                accepted = True
                print(f"step {step} frame {frame} x={xy[frame][0]:.0f} {before_s:.3f}->{after:.3f} cap {cap}", flush=True)
                break
            print(f"  reject cap {cap:.4f} {before_s:.3f}->{after:.3f} live {stats['live']:.3f} block {stats['block']:.3f}", flush=True)
        if not accepted:
            model.load_state_dict(before)
            banned.add(frame)
            print(f"step {step} frame {frame} skip", flush=True)
            continue
        banned.clear()

    stats = report(step)
    if not ready(stats):
        print("DRIVE_UNMET", flush=True)
        loop.restore_boxes(ucv)
        print("RESTORED", flush=True)
        return 1
    save_checkpoint(OUT, model.eval(), {"step": step, "live": stats["live"]})
    print("DRIVE_SAVE", step, flush=True)
    print("DRIVE_OUT", OUT, flush=True)

    depth_model = DepthMinPredictor.from_checkpoint(str(base.DEPTH), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    model.eval()
    sim.score = score_next
    loop.sim.score = score_next
    loop.pick = pick_learned
    walk_entrance(ucv, cam, depth_model, model)
    loop.restore_boxes(ucv)
    print("RESTORED", flush=True)
    return 0


def walk_entrance(ucv, cam, depth_model, model) -> None:
    rng = np.random.default_rng(0)
    x, y, yaw = -520.0, -250.0, 0.0
    gx, gy = WALL_GOAL
    goal = np.array([gx / 100.0, gy / 100.0, 0.52], np.float64)
    prev = np.zeros(4, np.float32)
    stopped = None
    for step in range(18):
        seen = loop.sense(ucv, cam, depth_model, x, y, yaw)
        if seen is None:
            print(f"  s={step} grab_fail", flush=True)
            break
        state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, math.radians(yaw)], np.float64)
        bearing = sim.wrap(math.atan2(goal[1] - state[1], goal[0] - state[0]) - state[6])
        names, cands, allowed = sim.candidates(rng, bearing)
        surv, _prog = sim.score([model], seen["depth"], state, goal, cands, prev, seen["da"][0])
        action, family, s_pick = pick_learned(seen["da"], seen["low"], names, cands, surv, allowed, bearing)
        gt, da = seen["gt"], seen["da"]
        print(
            f"  s={step} x={x:.0f} yaw={yaw:.1f} gt={gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f} "
            f"da={da[0]:.2f},{da[1]:.2f},{da[2]:.2f} surv_max={float(np.max(surv)):.3f} "
            f"pick={s_pick:.3f} {family}",
            flush=True,
        )
        if family == "stop" or min(gt) < loop.HIT_M:
            stopped = x
            break
        x, y, yaw = loop.integrate(x, y, yaw, float(action[0]), float(action[3]))
        prev = np.asarray(action, np.float32)
    passed = x > -450.0 and stopped is None
    print(f"DRIVE_WALK x={x:.0f} passed={passed} stopped={stopped}", flush=True)
    score_pose(ucv, cam, depth_model, model, "shelf", 205.0, -106.0, 12.3, base.GOAL_XY)
    score_pose(ucv, cam, depth_model, model, "face", -260.0, -250.0, 0.0, WALL_GOAL)


def score_pose(ucv, cam, depth_model, model, name, x, y, yaw, goal_xy) -> None:
    seen = loop.sense(ucv, cam, depth_model, x, y, yaw)
    gx, gy = goal_xy
    goal = np.array([gx / 100.0, gy / 100.0, 0.52], np.float64)
    state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, math.radians(yaw)], np.float64)
    bearing = sim.wrap(math.atan2(goal[1] - state[1], goal[0] - state[0]) - state[6])
    names, cands, allowed = sim.candidates(np.random.default_rng(0), bearing)
    surv, _prog = sim.score([model], seen["depth"], state, goal, cands, np.zeros(4, np.float32), seen["da"][0])
    action, family, s_pick = pick_learned(seen["da"], seen["low"], names, cands, surv, allowed, bearing)
    print(
        f"DRIVE_POSE {name} gt={seen['gt'][0]:.2f} da={seen['da'][0]:.2f},{seen['da'][1]:.2f},{seen['da'][2]:.2f} "
        f"surv_max={float(np.max(surv)):.3f} pick={s_pick:.3f} {family}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
