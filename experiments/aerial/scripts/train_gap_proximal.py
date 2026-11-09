"""Fit the route-1 teacher prefix without walking away from the route-0 actor.

Loss is the teacher action error on the latched route-1 steps, plus the
original actor's own actions on the route-0 flight, plus a penalty on the
parameter move. The route-0 targets are computed once from the frozen weights,
so they keep a gradient after the teacher steps start to move the network.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.goal_features import goal_rel_from_obs
from experiments.aerial.scripts.fly_gap_passage import _load_learner
from experiments.aerial.scripts.train_gap_correction import _latched_mask


def _encode(policy, transitions):
    zs = []
    goals = []
    actions = []
    for tr in transitions:
        zs.append(np.asarray(policy._dynamics.encode(tr.obs), dtype=np.float64).reshape(-1))
        goals.append(goal_rel_from_obs(tr.obs))
        actions.append(np.asarray(tr.action, dtype=np.float64).reshape(4))
    return np.stack(zs), np.stack(goals), np.stack(actions)


def _predict(ac, z, goal):
    feat = ac._feat_tensor(z, goal)
    mean, _std = ac._pre_dist(feat)
    return ac._limits * torch.tanh(mean)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--route0", required=True)
    p.add_argument("--route1", required=True)
    p.add_argument("--actor-ckpt", required=True)
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--anchor", type=float, default=4.0)
    p.add_argument("--prox", type=float, default=1.0)
    args = p.parse_args()

    policy = _load_learner(args.config, args.wm_ckpt, args.actor_ckpt)
    ac = policy._ac
    r0 = load_episode(Path(args.route0))
    r1 = load_episode(Path(args.route1))
    latched = _latched_mask(r1)
    r1 = [tr for tr, keep in zip(r1, latched) if keep]
    if len(r1) < 8:
        raise SystemExit(f"only {len(r1)} teacher steps")

    z0, g0, _a0 = _encode(policy, r0)
    z1, g1, a1 = _encode(policy, r1)
    with torch.no_grad():
        anchor = _predict(ac, z0, g0).detach()
        before_r0 = (anchor - _predict(ac, z0, g0)).abs().mean().item()
        before_r1 = (
            _predict(ac, z1, g1) - torch.as_tensor(a1, dtype=torch.float32, device=ac._device)
        ).abs().mean().item()

    frozen = [p.detach().clone() for p in list(ac._actor.parameters()) + [ac._log_std]]
    params = list(ac._actor.parameters()) + [ac._log_std]
    opt = torch.optim.Adam(params, lr=float(args.lr))
    a1_t = torch.as_tensor(a1, dtype=torch.float32, device=ac._device).clamp(-ac._limits, ac._limits)

    for step in range(int(args.steps)):
        opt.zero_grad()
        pred1 = _predict(ac, z1, g1)
        pred0 = _predict(ac, z0, g0)
        bc = torch.nn.functional.mse_loss(pred1, a1_t)
        hold = torch.nn.functional.mse_loss(pred0, anchor)
        prox = torch.zeros((), device=ac._device)
        n = 0
        for p, p0 in zip(params, frozen):
            prox = prox + (p - p0).pow(2).sum()
            n += p.numel()
        prox = prox / max(n, 1)
        loss = bc + float(args.anchor) * hold + float(args.prox) * prox
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 10 == 0:
            print(
                f"step={step} bc={float(bc.detach()):.5f} "
                f"hold={float(hold.detach()):.5f} prox={float(prox.detach()):.6f}"
            )

    with torch.no_grad():
        after_r0 = (anchor - _predict(ac, z0, g0)).abs().mean().item()
        after_r1 = (_predict(ac, z1, g1) - a1_t).abs().mean().item()
    print(
        f"route0_mae {before_r0:.4f} -> {after_r0:.4f}  "
        f"teacher_mae {before_r1:.4f} -> {after_r1:.4f}  "
        f"n_teacher={len(r1)} n_route0={len(r0)}"
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor": ac._actor.state_dict(),
        "critic": ac._critic.state_dict(),
        "log_std": ac._log_std.detach().cpu(),
        "config": ac.config.__dict__,
        "iter_idx": 0,
    }
    torch.save(payload, out / "v4_ac_latest.pt")
    print(f"saved {out / 'v4_ac_latest.pt'}")


if __name__ == "__main__":
    main()
