"""Fit an image correction on the route-1 passage, held at zero on route 0."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.goal_features import goal_rel_from_obs
from experiments.aerial.rl.visual_residual import VisualResidual
from experiments.aerial.scripts.fly_gap_passage import _load_learner
from experiments.aerial.scripts.train_near_wall_turn import _predict


def _base_actions(policy, transitions) -> np.ndarray:
    out = []
    for tr in transitions:
        z = np.asarray(policy._dynamics.encode(tr.obs), dtype=np.float64).reshape(1, -1)
        g = goal_rel_from_obs(tr.obs).reshape(1, -1)
        with torch.no_grad():
            out.append(_predict(policy._ac, z, g).cpu().numpy()[0])
    return np.stack(out).astype(np.float32)


def _rgb(transitions) -> np.ndarray:
    return np.stack([np.asarray(tr.obs.rgb) for tr in transitions])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--route0", required=True)
    p.add_argument("--teacher", required=True, nargs="+")
    p.add_argument("--actor-ckpt", required=True)
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hold", type=float, default=8.0)
    p.add_argument("--turn-weight", type=float, default=30.0)
    args = p.parse_args()

    policy = _load_learner(args.config, args.wm_ckpt, args.actor_ckpt)
    r0 = load_episode(Path(args.route0))
    teacher = []
    for path in args.teacher:
        teacher.extend(load_episode(Path(path)))
    base_t = _base_actions(policy, teacher)
    act_t = np.stack([np.asarray(tr.action, np.float32) for tr in teacher])
    rgb_t = torch.as_tensor(_rgb(teacher), device="cuda")
    rgb_0 = torch.as_tensor(_rgb(r0), device="cuda")
    base = torch.as_tensor(base_t, device="cuda")
    target = torch.as_tensor(act_t, device="cuda")

    model = VisualResidual().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    limits = model.limits
    turn = (target[:, 0] < 0.3).float()
    weights = 1.0 + (float(args.turn_weight) - 1.0) * turn
    print(f"turn_frames={int(turn.sum())}/{len(teacher)}")

    def _mae(pred, tgt):
        return float((pred - tgt).abs().mean())

    for step in range(int(args.steps)):
        opt.zero_grad()
        res_t = model(rgb_t)
        res_0 = model(rgb_0)
        mixed = (base + res_t).clamp(-limits, limits)
        err = (mixed - target.clamp(-limits, limits)).pow(2).mean(dim=1)
        fit = (err * weights).sum() / weights.sum()
        hold = torch.nn.functional.mse_loss(res_0, torch.zeros_like(res_0))
        loss = fit + float(args.hold) * hold
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 50 == 0:
            with torch.no_grad():
                print(
                    f"step={step} fit={float(fit):.5f} hold={float(hold):.5f} "
                    f"teacher_mae={_mae(mixed, target):.4f} "
                    f"route0_res={_mae(res_0, torch.zeros_like(res_0)):.4f}"
                )

    with torch.no_grad():
        res_t = model(rgb_t)
        res_0 = model(rgb_0)
        mixed = (base + res_t).clamp(-limits, limits)
        dim = (res_0.abs().mean(0)).tolist()
        print(
            "done teacher_mae={:.4f} route0_res_dim dx,dy,dz,dyaw {}".format(
                _mae(mixed, target),
                " ".join(f"{v:.4f}" for v in dim),
            )
        )
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
