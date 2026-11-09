"""Map passage pictures to sticks. The street gate is a separate, frozen code."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.gap_passage import gap_action
from experiments.aerial.rl.visual_residual import VisualResidual
from experiments.aerial.scripts.train_passage_code import _release_index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--passage", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()
    passage = load_episode(Path(args.passage))
    goal = np.asarray(passage[0].obs.info["goal"], dtype=np.float64).reshape(3)
    release = _release_index(passage, goal)
    frames = passage[:release]
    rgb = torch.as_tensor(np.stack([tr.obs.rgb for tr in frames]), device="cuda")
    acts = []
    for tr in frames:
        action, _phase = gap_action(
            tr.obs.position, float(tr.obs.yaw), goal, z_hold=29.0
        )
        acts.append(np.asarray(action, dtype=np.float32))
    target = torch.as_tensor(np.stack(acts), device="cuda")
    print(f"frames={len(frames)} release={release}")
    model = VisualResidual().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    turn = (target[:, 0] < 0.3).float()
    weights = 1.0 + 30.0 * turn
    for step in range(int(args.steps)):
        opt.zero_grad()
        pred = model(rgb)
        err = (pred - target).pow(2).mean(dim=1)
        loss = (err * weights).sum() / weights.sum()
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 100 == 0:
            with torch.no_grad():
                mae = float((pred - target).abs().mean())
                turn_mae = float((pred[turn > 0] - target[turn > 0]).abs().mean()) if int(turn.sum()) else float("nan")
            print(f"step={step} mae={mae:.4f} turn_mae={turn_mae:.4f}")
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
