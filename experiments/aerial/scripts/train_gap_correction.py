"""Fit a gated residual so the frozen route-0 actor matches the route-1 teacher
inside y<80 and x>-960. The actor weights are not updated.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.gap_correction import GapCorrection, _net, correction_features, correction_gate
from experiments.aerial.rl.gap_passage import LIMITS
from experiments.aerial.scripts.fly_gap_passage import _load_learner


def _latched_mask(transitions) -> np.ndarray:
    """Same latch as the route-1 handover flight. True on teacher steps."""
    latched = False
    prev_y = None
    mask = np.zeros(len(transitions), dtype=bool)
    for i, tr in enumerate(transitions):
        pos = np.asarray(tr.obs.position, dtype=np.float64)
        x, y = float(pos[0]), float(pos[1])
        yaw = float(tr.obs.yaw)
        yaw_err = (yaw + math.pi / 2.0 + math.pi) % (2.0 * math.pi) - math.pi
        facing_south = abs(yaw_err) < 0.45
        if latched and x >= -948.0 and y <= 26.0 and facing_south:
            latched = False
        elif (
            not latched
            and y > 60.0
            and x > -960.0
            and prev_y is not None
            and y < prev_y - 0.3
        ):
            latched = True
        prev_y = y
        mask[i] = latched
    return mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True)
    p.add_argument("--actor-ckpt", required=True)
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()

    transitions = load_episode(Path(args.episode))
    teacher = _latched_mask(transitions)
    policy = _load_learner(args.config, args.wm_ckpt, args.actor_ckpt)
    feats = []
    targets = []
    for i, tr in enumerate(transitions):
        pos = np.asarray(tr.obs.position, dtype=np.float64)
        if not teacher[i] or correction_gate(float(pos[0]), float(pos[1])) == 0.0:
            continue
        goal = tr.obs.info.get("goal")
        base = np.asarray(policy.act(tr.obs), dtype=np.float64).reshape(4)
        label = np.asarray(tr.action, dtype=np.float64).reshape(4)
        feats.append(correction_features(pos, float(tr.obs.yaw), goal))
        targets.append(label - base)
    if len(feats) < 4:
        raise SystemExit(f"only {len(feats)} gated teacher steps")
    x = np.stack(feats).astype(np.float32)
    y = np.stack(targets).astype(np.float32)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-3)
    xn = (x - mean) / std
    net = _net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    xt = torch.as_tensor(xn)
    yt = torch.as_tensor(y)
    loss = None
    for _ in range(int(args.steps)):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        opt.step()
    final = float(loss.detach())
    print(f"samples={len(feats)} mse={final:.6f}")
    GapCorrection(net.eval(), mean, std, LIMITS).save(Path(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
