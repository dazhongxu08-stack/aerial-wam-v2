"""Fit the clearance correction on passage demos and hold it at zero on route 0."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.clearance_head import ClearanceHead, clearance_features
from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.goal_features import goal_rel_from_obs
from experiments.aerial.scripts.fly_gap_passage import _load_learner
from experiments.aerial.scripts.train_near_wall_turn import _predict


def _rows(policy, transitions):
    feats, bases, acts = [], [], []
    for tr in transitions:
        feats.append(clearance_features(tr.obs))
        z = np.asarray(policy._dynamics.encode(tr.obs), dtype=np.float64).reshape(1, -1)
        g = goal_rel_from_obs(tr.obs).reshape(1, -1)
        with torch.no_grad():
            bases.append(_predict(policy._ac, z, g).cpu().numpy()[0])
        acts.append(np.asarray(tr.action, np.float32))
    return (
        torch.as_tensor(np.stack(feats), device="cuda"),
        torch.as_tensor(np.stack(bases), device="cuda"),
        torch.as_tensor(np.stack(acts), device="cuda"),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--route0", required=True)
    p.add_argument("--teacher", required=True, nargs="+")
    p.add_argument("--actor-ckpt", required=True)
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--hold", type=float, default=12.0)
    p.add_argument("--turn-weight", type=float, default=40.0)
    args = p.parse_args()

    policy = _load_learner(args.config, args.wm_ckpt, args.actor_ckpt)
    r0 = load_episode(Path(args.route0))
    teacher = []
    for path in args.teacher:
        teacher.extend(load_episode(Path(path)))
    f0, _b0, _a0 = _rows(policy, r0)
    ft, base, target = _rows(policy, teacher)
    print("feat0", np.round(f0[0].detach().cpu().numpy(), 3).tolist())
    print("featt", np.round(ft[0].detach().cpu().numpy(), 3).tolist())
    model = ClearanceHead().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    limits = model.limits
    weights = 1.0 + (float(args.turn_weight) - 1.0) * (target[:, 0] < 0.3).float()
    for step in range(int(args.steps)):
        opt.zero_grad()
        mixed = (base + model(ft)).clamp(-limits, limits)
        err = (mixed - target.clamp(-limits, limits)).pow(2).mean(dim=1)
        fit = (err * weights).sum() / weights.sum()
        res0 = model(f0)
        hold = torch.nn.functional.mse_loss(res0, torch.zeros_like(res0))
        loss = fit + float(args.hold) * hold
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == int(args.steps):
            with torch.no_grad():
                mae = float((mixed - target).abs().mean())
                r0m = float(res0.abs().mean())
            print(f"step={step} fit={float(fit):.5f} teacher_mae={mae:.4f} route0_res={r0m:.4f}")
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
