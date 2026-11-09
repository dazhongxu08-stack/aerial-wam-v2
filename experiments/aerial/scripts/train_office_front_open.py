#!/usr/bin/env python3
"""Turn toward the open side on a close real nose.

consequence_hold.pt stops when the nose is close, including when its own
depth map shows one side several meters open. This lowers only that away
spin's hazard. Forward steps and the spin into the closed side stay unsafe,
and a nose with both sides closed still stops. Writes consequence_open.pt.
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
import train_office_align_detour as det
import run_office_cons_loop as loop
from experiments.aerial.rl.consequence_model import depth_input, goal_features, load_checkpoint, save_checkpoint
from experiments.aerial.rl.env.obs import Observation
from office_bands import cones3_forward

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "indoor_office_align_20261104/c_s218/consequence_hold.pt"
OUT = ART / "indoor_office_align_20261104/c_s218/consequence_open.pt"
DEPTH = ART / "da3_office_align_20261104/depth_office_cone_align.pt"
REAL_CACHE = Path("/data/simworld/run/office_front_hold_20261104/real.npz")
LIVE_RGB = Path("/data/simworld/run/office_front_open_20261105/live_rgb.npz")
# Cache index, away yaw. Positive yaw is image-right.
DETOUR = ((12, det.YAW), (53, det.YAW))
STOP = (7, 9, 13, 41, 52)
BLOCKED = (
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
)


def encode_live(depth_model) -> dict[str, np.ndarray]:
    z = np.load(LIVE_RGB)
    rgb = z["rgb"]
    depth_model.reset()
    pred = np.asarray(depth_model.predict_depth(Observation(
        rgb=np.ascontiguousarray(rgb), state=np.zeros(7, np.float32), collided=False,
    )), np.float32)
    seen = rgb.sum(axis=-1) > 12
    da56 = np.asarray(col.pool_min(col.hide(pred, seen)), np.float32)
    cones = cones3_forward(da56)
    state = np.array([0.0, 0.0, 0.34, 0, 0, 0, 0.0], np.float64)
    goal = goal_features(state, np.array([8.0, 0.0, 0.34], np.float64)).astype(np.float32)
    print(f"live da {cones[0]:.2f}/{cones[1]:.2f}/{cones[2]:.2f}", flush=True)
    return {
        "depth": depth_input(da56)[None].astype(np.float32),
        "goal": goal[None].astype(np.float32),
        "da": cones[None].astype(np.float32),
    }


def main() -> int:
    dev = torch.device("cuda")
    depth_model = loop.DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    depth_model.forward_scale = 1.0
    depth_model.forward_exponent = 1.0
    live = encode_live(depth_model)
    del depth_model
    torch.cuda.empty_cache()

    real = np.load(REAL_CACHE)
    side = np.load("/data/simworld/run/office_front_yield2_20261104/poses.npz")
    passed = np.load("/data/simworld/run/office_front_pass_20261104/poses.npz")
    nose = np.load("/data/simworld/run/office_front_detour_20261104/poses.npz")
    gt, da = side["gt"], side["da"]
    give_i = np.flatnonzero((gt[:, 1] >= 0.55) & (gt[:, 1] < 0.70) & (da[:, 1] < 2.0) & (da[:, 0] > 2.0) & (da[:, 2] > 2.0))
    pda, pgt = passed["da"], passed["gt"]
    n = len(pda)
    yaw = np.concatenate([np.full(n // 2, 18.0), np.full(n - n // 2, 0.0)])
    drive_i = np.flatnonzero((yaw >= 12.0) & (pda[:, 0] > 2.2) & (pgt[:, 0] > 2.5) & (pgt[:, 1] > 0.55) & (pgt[:, 2] > 0.55))
    nda, ngt = nose["da"], nose["gt"]
    turn_i = np.flatnonzero((nda[:, 0] < 1.4) & (nda[:, 1] < 1.6) & (nda[:, 2] > 2.2) & (ngt[:, 2] > 2.0) & (ngt[:, 0] > 0.30))
    open_i = np.flatnonzero((real["gt"][:, 0] > 3.0) & (real["gt"][:, 1] > 1.5) & (real["gt"][:, 2] > 1.5))
    print(
        f"give {len(give_i)} drive {len(drive_i)} turn {len(turn_i)} open {len(open_i)}",
        flush=True,
    )

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

    def flat_grad(loss: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)
        ]).detach()

    def surv_at(pack: str, index: int, dx: float, dyaw: float, prev_dx: float) -> float:
        model.eval()
        with torch.no_grad():
            z = logit_at(pack, index, dx, dyaw, prev_dx)
            return float(1.0 - torch.sigmoid(z))

    def constraint_matrix() -> tuple[torch.Tensor, torch.Tensor]:
        model.train()
        cons = []
        for prev_dx in (0.0, 0.8):
            for dx, dyaw in ((0.8, 0.0), (0.35, 0.0), (0.0, det.YAW)):
                cons.append(-flat_grad(logit_at("live", 0, dx, dyaw, prev_dx)))
            for i in (7, 13, 41):
                for dx, dyaw in ((0.8, 0.0), (0.35, 0.0), (0.0, det.YAW), (0.0, -det.YAW)):
                    cons.append(-flat_grad(logit_at("real", int(i), dx, dyaw, prev_dx)))
            for i in drive_i[::3]:
                cons.append(flat_grad(logit_at("pass", int(i), 0.8, 0.0, prev_dx)))
            for i in give_i:
                cons.append(flat_grad(logit_at("side", int(i), 0.0, det.YAW, prev_dx)))
            for i in turn_i[::2]:
                cons.append(flat_grad(logit_at("nose", int(i), 0.0, det.YAW, prev_dx)))
        C = torch.stack(cons)
        gram = C @ C.T
        gram = gram + 1e-3 * torch.eye(gram.shape[0], device=dev, dtype=gram.dtype)
        return C, gram

    def pending() -> list[tuple[str, int, float, float]]:
        model.eval()
        found = []
        with torch.no_grad():
            for prev_dx in (0.0, 0.8):
                if surv_at("live", 0, 0.0, -det.YAW, prev_dx) < 0.95:
                    found.append(("live", 0, -det.YAW, prev_dx))
                for i, away in DETOUR:
                    if surv_at("real", int(i), 0.0, away, prev_dx) < 0.95:
                        found.append(("real", int(i), away, prev_dx))
        model.train()
        return found

    def report() -> None:
        model.eval()
        bits = []
        for prev_dx in (0.0,):
            bits.append(
                f"live L {surv_at('live', 0, 0.0, -det.YAW, prev_dx):.3f} "
                f"R {surv_at('live', 0, 0.0, det.YAW, prev_dx):.3f} "
                f"fwd {surv_at('live', 0, 0.8, 0.0, prev_dx):.3f}"
            )
            for i, away in DETOUR:
                side = "R" if away > 0 else "L"
                bits.append(
                    f"i{i}{side} {surv_at('real', int(i), 0.0, away, prev_dx):.3f} "
                    f"in {surv_at('real', int(i), 0.0, -away, prev_dx):.3f} "
                    f"fwd {surv_at('real', int(i), 0.8, 0.0, prev_dx):.3f}"
                )
            stop_max = max(surv_at("real", int(i), dx, dyaw, prev_dx) for i in (7, 13) for dx, dyaw in BLOCKED + ((0.0, det.YAW), (0.0, -det.YAW)))
            bits.append(f"wall {stop_max:.3f}")
            bits.append(
                f"drive {surv_at('pass', int(drive_i[0]), 0.8, 0.0, prev_dx):.3f} "
                f"give {surv_at('side', int(give_i[0]), 0.0, det.YAW, prev_dx):.3f}"
            )
        model.train()
        print(" | ".join(bits), flush=True)

    def ok() -> bool:
        good = True
        for prev_dx in (0.0, 0.8):
            if surv_at("live", 0, 0.0, -det.YAW, prev_dx) < 0.95:
                good = False
            if surv_at("live", 0, 0.8, 0.0, prev_dx) > 0.25:
                good = False
            if surv_at("live", 0, 0.0, det.YAW, prev_dx) > 0.50:
                good = False
            for i, away in DETOUR:
                if surv_at("real", int(i), 0.0, away, prev_dx) < 0.95:
                    good = False
                if surv_at("real", int(i), 0.8, 0.0, prev_dx) > 0.25:
                    good = False
            for i in (7, 13):
                for dx, dyaw in ((0.8, 0.0), (0.0, det.YAW), (0.0, -det.YAW)):
                    if surv_at("real", int(i), dx, dyaw, prev_dx) > 0.25:
                        good = False
            if surv_at("pass", int(drive_i[0]), 0.8, 0.0, prev_dx) < 0.95:
                good = False
            if surv_at("side", int(give_i[0]), 0.0, det.YAW, prev_dx) < 0.95:
                good = False
        return good

    def apply_step(step: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for param in params:
                numel = param.numel()
                param.data += step[offset:offset + numel].view_as(param)
                offset += numel

    print("before", flush=True)
    report()
    gain = 1.5
    saved = False
    for it in range(1, 201):
        terms = pending()
        if not terms:
            break
        C, gram = constraint_matrix()
        moved = False
        for pack, index, away, prev_dx in terms:
            grad = flat_grad(logit_at(pack, index, 0.0, away, prev_dx))
            resid = -grad - C.T @ torch.linalg.solve(gram, C @ -grad)
            align = float(torch.dot(grad, resid))
            orth = float(resid.norm() / grad.norm().clamp_min(1e-8))
            if align >= -1.0 or orth < 0.03:
                continue
            alpha = -gain / align
            step = resid * alpha
            delta = float(step.norm())
            if delta > 0.005:
                step = step * (0.005 / delta)
                delta = 0.005
            backup = [p.detach().clone() for p in params]
            before = surv_at(pack, index, 0.0, away, prev_dx)
            apply_step(step)
            after = surv_at(pack, index, 0.0, away, prev_dx)
            report()
            print(
                f"it {it} {pack} i={index} prev {prev_dx:.1f} orth {orth:.2f} "
                f"delta {delta:.3f} spin {before:.3f}->{after:.3f} left {len(terms)}",
                flush=True,
            )
            broke = (
                surv_at("pass", int(drive_i[0]), 0.8, 0.0, 0.0) < 0.95
                or surv_at("side", int(give_i[0]), 0.0, det.YAW, 0.0) < 0.95
                or surv_at("nose", int(turn_i[0]), 0.0, det.YAW, 0.0) < 0.95
                or surv_at("live", 0, 0.8, 0.0, 0.0) > 0.25
                or surv_at("live", 0, 0.0, det.YAW, 0.0) > 0.50
                or surv_at("real", 7, 0.8, 0.0, 0.0) > 0.50
                or surv_at("real", 7, 0.0, -det.YAW, 0.0) > 0.30
                or surv_at("real", 13, 0.8, 0.0, 0.0) > 0.50
                or surv_at("real", 13, 0.0, det.YAW, 0.0) > 0.50
            )
            if broke or after + 1e-4 < before:
                with torch.no_grad():
                    for param, old in zip(params, backup):
                        param.copy_(old)
                gain *= 0.5
                print(f"OPEN_REVERT gain {gain:.3f}", flush=True)
                if gain < 0.15:
                    print("OPEN_CLIFF", flush=True)
                    moved = False
                    terms = []
                break
            moved = True
            if ok():
                save_checkpoint(OUT, model.eval(), {"iters": it, "live_spin": after})
                model.train()
                print("OPEN_SAVE", it, flush=True)
                saved = True
            break
        del C, gram
        if saved or not moved:
            if not moved:
                print("OPEN_LOCKED", flush=True)
            break
    if not saved:
        print("OPEN_NO_SAVE", flush=True)
        return 2
    print("OPEN_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
