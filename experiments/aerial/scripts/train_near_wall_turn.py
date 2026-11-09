"""Teach the west turn only on near-wall frames east of the gap.

Route 0 frames keep the frozen actor's own actions. Route 1 frames outside
the near wall do too. Nothing here writes the full gap passage into the net.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.goal_features import goal_rel_from_obs
from experiments.aerial.rl.near_wall_turn import near_wall_east, west_turn_action
from experiments.aerial.scripts.fly_gap_passage import _load_learner


def _forward_m(obs) -> float:
    depth = getattr(obs, "depth", None)
    if depth is None:
        return float("nan")
    from experiments.aerial.rl.depth_geometry import approach_clearances

    arr = np.asarray(depth, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    fwd, _, _ = approach_clearances(arr, center_frac=0.12)
    return float(fwd)


def _encode(policy, transitions):
    zs, goals = [], []
    for tr in transitions:
        zs.append(np.asarray(policy._dynamics.encode(tr.obs), dtype=np.float64).reshape(-1))
        goals.append(goal_rel_from_obs(tr.obs))
    return np.stack(zs), np.stack(goals)


def _predict(ac, z, goal):
    feat = ac._feat_tensor(z, goal)
    mean, _std = ac._pre_dist(feat)
    return ac._limits * torch.tanh(mean)


def _split(transitions):
    keep = []
    labels = []
    for tr in transitions:
        pos = np.asarray(tr.obs.position, dtype=np.float64)
        fwd = _forward_m(tr.obs)
        if near_wall_east(float(pos[0]), float(pos[1]), fwd):
            labels.append(west_turn_action(pos, float(tr.obs.yaw)))
            keep.append(True)
        else:
            labels.append(np.asarray(tr.action, dtype=np.float64))
            keep.append(False)
    return np.asarray(keep, dtype=bool), np.stack(labels)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--route0", required=True)
    p.add_argument("--route1", required=True)
    p.add_argument("--actor-ckpt", required=True)
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--anchor", type=float, default=8.0)
    p.add_argument("--prox", type=float, default=1.0)
    args = p.parse_args()

    policy = _load_learner(args.config, args.wm_ckpt, args.actor_ckpt)
    ac = policy._ac
    r0 = load_episode(Path(args.route0))
    r1 = load_episode(Path(args.route1))
    keep0, _ = _split(r0)
    keep1, labels1 = _split(r1)
    if int(keep0.sum()) > 0:
        raise SystemExit(f"refuse: route 0 has {int(keep0.sum())} near-wall frames")
    if int(keep1.sum()) < 4:
        raise SystemExit(f"refuse: route 1 has only {int(keep1.sum())} near-wall frames")

    print(f"near_wall route1={int(keep1.sum())}/{len(r1)} route0={int(keep0.sum())}/{len(r0)}")
    for i, tr in enumerate(r1):
        if not keep1[i]:
            continue
        pos = np.asarray(tr.obs.position, dtype=np.float64)
        print(
            f"  i={i} xy=({pos[0]:.1f},{pos[1]:.1f}) z={pos[2]:.1f} "
            f"yaw={float(tr.obs.yaw):.2f} label={np.round(labels1[i], 3).tolist()}"
        )

    z0, g0 = _encode(policy, r0)
    z1, g1 = _encode(policy, r1)
    with torch.no_grad():
        anchor0 = _predict(ac, z0, g0).detach()
        anchor1 = _predict(ac, z1, g1).detach()

    frozen = [p.detach().clone() for p in list(ac._actor.parameters()) + [ac._log_std]]
    params = list(ac._actor.parameters()) + [ac._log_std]
    opt = torch.optim.Adam(params, lr=float(args.lr))
    target = torch.as_tensor(labels1, dtype=torch.float32, device=ac._device)
    target = target.clamp(-ac._limits, ac._limits)
    mask = torch.as_tensor(keep1, dtype=torch.bool, device=ac._device)
    hold_mask = ~mask

    def _mae(pred, tgt, sel):
        if int(sel.sum()) == 0:
            return float("nan")
        return float((pred[sel] - tgt[sel]).abs().mean())

    with torch.no_grad():
        before = _predict(ac, z1, g1)
        before_turn = _mae(before, target, mask)
        before_hold = _mae(before, anchor1, hold_mask)
        before_r0 = _mae(_predict(ac, z0, g0), anchor0, torch.ones(len(r0), dtype=torch.bool, device=ac._device))
    print(f"before turn_mae={before_turn:.4f} hold_mae={before_hold:.4f} route0_mae={before_r0:.4f}")

    for step in range(int(args.steps)):
        opt.zero_grad()
        pred1 = _predict(ac, z1, g1)
        pred0 = _predict(ac, z0, g0)
        turn = torch.nn.functional.mse_loss(pred1[mask], target[mask])
        hold1 = torch.nn.functional.mse_loss(pred1[hold_mask], anchor1[hold_mask])
        hold0 = torch.nn.functional.mse_loss(pred0, anchor0)
        prox = torch.zeros((), device=ac._device)
        n = 0
        for p, p0 in zip(params, frozen):
            prox = prox + (p - p0).pow(2).sum()
            n += p.numel()
        prox = prox / max(n, 1)
        loss = turn + float(args.anchor) * hold0 + float(args.anchor) * hold1 + float(args.prox) * prox
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 10 == 0:
            print(
                f"step={step} turn={float(turn.detach()):.5f} "
                f"hold0={float(hold0.detach()):.5f} hold1={float(hold1.detach()):.5f} "
                f"prox={float(prox.detach()):.6f}"
            )

    with torch.no_grad():
        after = _predict(ac, z1, g1)
        after0 = _predict(ac, z0, g0)
        ones = torch.ones(len(r0), dtype=torch.bool, device=ac._device)
        after_turn = _mae(after, target, mask)
        after_hold = _mae(after, anchor1, hold_mask)
        after_r0 = _mae(after0, anchor0, ones)
        yaw = float((after[mask][:, 3] - target[mask][:, 3]).abs().mean())
        r0_dim = (after0 - anchor0).abs().mean(dim=0)
        turn_dim = (after[mask] - target[mask]).abs().mean(dim=0)
    print(
        f"after turn_mae={after_turn:.4f} yaw_mae={yaw:.4f} "
        f"hold_mae={after_hold:.4f} route0_mae={after_r0:.4f}"
    )
    print(
        "route0_dim dx,dy,dz,dyaw "
        + " ".join(f"{float(v):.4f}" for v in r0_dim)
    )
    print(
        "turn_dim dx,dy,dz,dyaw "
        + " ".join(f"{float(v):.4f}" for v in turn_dim)
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
