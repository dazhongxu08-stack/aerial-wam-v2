#!/usr/bin/env python3
"""Stop on real frames whose nose is already inside 1.15 m.

consequence_yield.pt still scores a forward step as safe on a real frame
whose depth cone is 0.67 m. This projects that hazard up, one small step at
a time, and does not let the side yield, the barrel pass-by, or an open real
frame move the wrong way. Writes consequence_hold.pt. Does not overwrite
consequence_yield.pt.
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
import train_office_align as align
import train_office_align_detour as det
import run_office_cons_loop as loop
from experiments.aerial.rl.consequence_model import depth_input, goal_features, load_checkpoint, save_checkpoint
from experiments.aerial.rl.env.obs import Observation
from office_bands import cones3_forward

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "indoor_office_align_20261104/c_s218/consequence_yield.pt"
OUT = ART / "indoor_office_align_20261104/c_s218/consequence_hold.pt"
DEPTH = ART / "da3_office_align_20261104/depth_office_cone_align.pt"
REAL_CACHE = Path("/data/simworld/run/office_front_hold_20261104/real.npz")
ACTS = (
    (0.8, 0.0),
    (0.35, 0.0),
    (0.6, 0.0),
    (1.0, 0.0),
    (0.8, 0.04),
    (0.8, -0.04),
    (0.8, 0.10),
    (0.8, -0.10),
    (0.25, det.YAW),
    (0.25, -det.YAW),
    (0.0, det.YAW),
    (0.0, -det.YAW),
)


def _encode(depth_model, rgb, gt56) -> dict[str, np.ndarray]:
    state = np.array([0.0, 0.0, 0.34, 0, 0, 0, 0.0], np.float64)
    goal_v = goal_features(state, np.array([8.0, 0.0, 0.34], np.float64))
    depth, gt_c, da_c = [], [], []
    for i in range(len(rgb)):
        depth_model.reset()
        pred = depth_model.predict_depth(Observation(
            rgb=np.ascontiguousarray(rgb[i]), state=np.zeros(7, np.float32), collided=False,
        ))
        pred = np.asarray(pred, np.float32)
        seen = rgb[i].sum(axis=-1) > 12
        da56 = col.pool_min(col.hide(pred, seen))
        depth.append(depth_input(np.asarray(da56, np.float32)))
        gt_c.append(np.asarray(cones3_forward(gt56[i]), np.float32))
        da_c.append(np.asarray(cones3_forward(np.asarray(da56, np.float32)), np.float32))
    goal = np.repeat(goal_v[None], len(depth), 0).astype(np.float32)
    return {
        "depth": np.stack(depth).astype(np.float32),
        "goal": goal,
        "gt": np.stack(gt_c).astype(np.float32),
        "da": np.stack(da_c).astype(np.float32),
    }


def _load_real(depth_model) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if REAL_CACHE.is_file():
        z = np.load(REAL_CACHE, allow_pickle=False)
        real = {k: z[k] for k in ("depth", "goal", "gt", "da")}
        live = {k[5:]: z[k] for k in ("live_depth", "live_goal", "live_gt", "live_da")}
        print(f"hold_cache real {len(real['depth'])} live {len(live['depth'])}", flush=True)
        return real, live
    rgb, gt = align.load_real()
    real = _encode(depth_model, rgb, gt)
    live_z = np.load(align.SIM_LIVE)
    live = _encode(depth_model, live_z["rgb"], live_z["gt56"].astype(np.float32))
    REAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        REAL_CACHE,
        **real,
        live_depth=live["depth"],
        live_goal=live["goal"],
        live_gt=live["gt"],
        live_da=live["da"],
    )
    print(f"hold_encoded real {len(real['depth'])} live {len(live['depth'])}", flush=True)
    return real, live


def main() -> int:
    dev = torch.device("cuda")
    depth_model = loop.DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    real, live = _load_real(depth_model)
    del depth_model
    torch.cuda.empty_cache()

    side = np.load("/data/simworld/run/office_front_yield2_20261104/poses.npz", allow_pickle=False)
    passed = np.load("/data/simworld/run/office_front_pass_20261104/poses.npz", allow_pickle=False)
    nose = np.load("/data/simworld/run/office_front_detour_20261104/poses.npz", allow_pickle=False)
    gt, da = side["gt"], side["da"]
    give_i = np.flatnonzero((gt[:, 1] >= 0.55) & (gt[:, 1] < 0.70) & (da[:, 1] < 2.0) & (da[:, 0] > 2.0) & (da[:, 2] > 2.0))
    pda, pgt = passed["da"], passed["gt"]
    n = len(pda)
    yaw = np.concatenate([np.full(n // 2, 18.0), np.full(n - n // 2, 0.0)])
    drive_i = np.flatnonzero((yaw >= 12.0) & (pda[:, 0] > 2.2) & (pgt[:, 0] > 2.5) & (pgt[:, 1] > 0.55) & (pgt[:, 2] > 0.55))
    nda, ngt = nose["da"], nose["gt"]
    turn_i = np.flatnonzero((nda[:, 0] < 1.4) & (nda[:, 1] < 1.6) & (nda[:, 2] > 2.2) & (ngt[:, 2] > 2.0) & (ngt[:, 0] > 0.30))
    hit_i = np.flatnonzero(real["gt"][:, 0] < 1.15)
    open_i = np.flatnonzero((real["gt"][:, 0] > 3.0) & (real["gt"][:, 1] > 1.5) & (real["gt"][:, 2] > 1.5))
    live_i = np.flatnonzero(live["gt"][:, 0] < 1.15)
    print(
        f"give {len(give_i)} drive {len(drive_i)} turn {len(turn_i)} "
        f"hit {len(hit_i)} open {len(open_i)} live {len(live_i)}",
        flush=True,
    )
    if len(hit_i) < 1 or len(drive_i) < 8 or len(turn_i) < 4 or len(give_i) < 1:
        print("HOLD_EMPTY", flush=True)
        return 2

    packs = {
        "side": (torch.as_tensor(side["depth"], device=dev), torch.as_tensor(side["goal"], device=dev)),
        "pass": (torch.as_tensor(passed["depth"], device=dev), torch.as_tensor(passed["goal"], device=dev)),
        "nose": (torch.as_tensor(nose["depth"], device=dev), torch.as_tensor(nose["goal"], device=dev)),
        "real": (torch.as_tensor(real["depth"], device=dev), torch.as_tensor(real["goal"], device=dev)),
        "live": (torch.as_tensor(live["depth"], device=dev), torch.as_tensor(live["goal"], device=dev)),
    }
    model = load_checkpoint(str(INIT), "cuda").train()
    params = [p for p in model.parameters() if p.requires_grad]

    def prev_of(rows: int, dx: float) -> torch.Tensor:
        prev = torch.zeros(rows, 4, device=dev)
        prev[:, 0] = dx
        return prev

    def logit_at(pack: str, index: int, dx: float, dyaw: float, prev_dx: float) -> torch.Tensor:
        depth, goal = packs[pack]
        return det.logits(model, depth[index:index + 1], goal[index:index + 1], prev_of(1, prev_dx), dx, dyaw)[0]

    def logit_rows(pack: str, indexes, dx: float, dyaw: float, prev_dx: float) -> torch.Tensor:
        depth, goal = packs[pack]
        idx = torch.as_tensor(list(indexes), device=dev, dtype=torch.long)
        return det.logits(model, depth[idx], goal[idx], prev_of(len(idx), prev_dx), dx, dyaw)

    def flat_grad(loss: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)
        ]).detach()

    def constraint_matrix() -> tuple[torch.Tensor, torch.Tensor]:
        model.train()
        cons = []
        for prev_dx in (0.0, 0.8):
            for i in drive_i:
                cons.append(flat_grad(logit_at("pass", int(i), 0.8, 0.0, prev_dx)))
            for i in give_i:
                cons.append(flat_grad(logit_at("side", int(i), 0.0, det.YAW, prev_dx)))
                cons.append(-flat_grad(logit_at("side", int(i), 0.8, 0.0, prev_dx)))
            for i in turn_i:
                cons.append(flat_grad(logit_at("nose", int(i), 0.0, det.YAW, prev_dx)))
                cons.append(-flat_grad(logit_at("nose", int(i), 0.8, 0.0, prev_dx)))
            for i in open_i:
                cons.append(flat_grad(logit_at("real", int(i), 0.8, 0.0, prev_dx)))
        C = torch.stack(cons)
        gram = C @ C.T
        gram = gram + 1e-3 * torch.eye(gram.shape[0], device=dev, dtype=gram.dtype)
        return C, gram

    def open_terms() -> list[tuple[float, int, float, float, float, float]]:
        model.eval()
        found = []
        with torch.no_grad():
            for prev_dx in (0.0, 0.8):
                for dx, dyaw in ACTS:
                    z = logit_rows("real", hit_i, dx, dyaw, prev_dx)
                    surv = 1.0 - torch.sigmoid(z)
                    for n, i in enumerate(hit_i):
                        if float(surv[n]) > 0.25:
                            found.append((float(surv[n]), int(i), dx, dyaw, prev_dx, float(z[n])))
        model.train()
        found.sort(reverse=True)
        return found

    def surv_rows(pack: str, indexes, prev_dx: float) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            rows = [1.0 - torch.sigmoid(logit_rows(pack, indexes, dx, dyaw, prev_dx)) for dx, dyaw in ACTS]
        return torch.stack(rows, 0).detach().cpu().numpy()

    def report() -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            stats = []
            for prev_dx in (0.0, 0.8):
                give = surv_rows("side", give_i, prev_dx)
                drive = surv_rows("pass", drive_i, prev_dx)
                turn = surv_rows("nose", turn_i, prev_dx)
                hit = surv_rows("real", hit_i, prev_dx)
                opened = 1.0 - torch.sigmoid(logit_rows("real", open_i, 0.8, 0.0, prev_dx))
                fwd_ids = [k for k, (dx, dyaw) in enumerate(ACTS) if dx > 0.0 and abs(dyaw) < 0.05]
                spin_r = ACTS.index((0.0, det.YAW))
                stats.append({
                    "give_fwd": float(give[fwd_ids].max()),
                    "give_spin": float(give[spin_r].min()),
                    "drive": float(drive[ACTS.index((0.8, 0.0))].min()),
                    "turn": float(turn[spin_r].min()),
                    "turn_fwd": float(turn[ACTS.index((0.8, 0.0))].max()),
                    "hit": float(hit.max()),
                    "open": float(opened.min()),
                })
        model.train()
        row = stats[0]
        print(
            "prev0 "
            f"hit {row['hit']:.3f} open {row['open']:.3f} "
            f"give {row['give_fwd']:.3f}/{row['give_spin']:.3f} "
            f"drive {row['drive']:.3f} turn {row['turn']:.3f}/{row['turn_fwd']:.3f}",
            flush=True,
        )
        return row

    def dump_hits() -> None:
        table = surv_rows("real", hit_i, 0.0)
        spin_r = ACTS.index((0.0, det.YAW))
        fwd = ACTS.index((0.8, 0.0))
        for n, i in enumerate(hit_i):
            cones = real["da"][i]
            print(
                f"  hit i={int(i)} da={cones[0]:.2f}/{cones[1]:.2f}/{cones[2]:.2f} "
                f"fwd={table[fwd, n]:.3f} spinR={table[spin_r, n]:.3f} max={table[:, n].max():.3f}",
                flush=True,
            )

    def held_ok(row: dict[str, float]) -> bool:
        return (
            row["open"] >= 0.93
            and row["give_spin"] >= 0.95
            and row["drive"] >= 0.95
            and row["turn"] >= 0.95
            and row["turn_fwd"] <= 0.90
            and row["give_fwd"] <= 0.90
        )

    def apply_step(step: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for param in params:
                numel = param.numel()
                param.data += step[offset:offset + numel].view_as(param)
                offset += numel

    print("before", flush=True)
    report()
    dump_hits()
    saved = False
    gain = 1.5
    for it in range(1, 49):
        pending = open_terms()
        if not pending:
            row = report()
            if held_ok(row) and row["hit"] <= 0.25:
                save_checkpoint(OUT, model.eval(), {
                    "iters": it,
                    "hit": row["hit"],
                    "open": row["open"],
                    "give_fwd": row["give_fwd"],
                    "drive": row["drive"],
                    "turn_fwd": row["turn_fwd"],
                })
                model.train()
                print("HOLD_SAVE", it, flush=True)
                saved = True
                dump_hits()
            break
        C, gram = constraint_matrix()
        moved = False
        for surv, index, dx, dyaw, prev_dx, z0 in pending:
            grad = flat_grad(logit_at("real", index, dx, dyaw, prev_dx))
            resid = grad - C.T @ torch.linalg.solve(gram, C @ grad)
            orth = float(resid.norm() / grad.norm().clamp_min(1e-8))
            align = float(torch.dot(grad, resid))
            if orth < 0.08 or align <= 1.0:
                continue
            alpha = gain / align
            step = resid * alpha
            delta = float(step.norm())
            if delta > 0.15:
                step = step * (0.15 / delta)
                delta = 0.15
            backup = [p.detach().clone() for p in params]
            apply_step(step)
            row = report()
            print(
                f"it {it} i={index} act {dx:.2f},{dyaw:.2f} prev {prev_dx:.1f} "
                f"z {z0:.2f} orth {orth:.2f} delta {delta:.3f} left {len(pending)}",
                flush=True,
            )
            if not held_ok(row):
                with torch.no_grad():
                    for param, old in zip(params, backup):
                        param.copy_(old)
                gain *= 0.5
                print(f"HOLD_REVERT gain {gain:.3f}", flush=True)
                moved = True
                break
            moved = True
            if row["hit"] <= 0.25:
                save_checkpoint(OUT, model.eval(), {
                    "iters": it,
                    "hit": row["hit"],
                    "open": row["open"],
                    "give_fwd": row["give_fwd"],
                    "drive": row["drive"],
                    "turn_fwd": row["turn_fwd"],
                })
                model.train()
                print("HOLD_SAVE", it, flush=True)
                saved = True
                dump_hits()
            break
        del C, gram
        if saved:
            break
        if not moved:
            print("HOLD_LOCKED", flush=True)
            dump_hits()
            break
        if gain < 0.15:
            print("HOLD_CLIFF", flush=True)
            break
    if not saved:
        print("HOLD_NO_SAVE", flush=True)
        return 2
    print("HOLD_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
