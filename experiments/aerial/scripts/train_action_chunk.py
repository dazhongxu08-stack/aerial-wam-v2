"""Fit an 8-step stick sequence from the current picture.

Passage frames learn the teacher's next sticks. Route 0 frames learn to repeat
the stick the frozen actor already flew there.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.action_chunk import ActionChunkNet
from experiments.aerial.rl.dataset import load_episode


def _windows(transitions, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    rgb, acts = [], []
    actions = [np.asarray(tr.action, np.float32) for tr in transitions]
    for i, tr in enumerate(transitions):
        chunk = []
        for k in range(horizon):
            chunk.append(actions[min(i + k, len(actions) - 1)])
        rgb.append(np.asarray(tr.obs.rgb))
        acts.append(np.stack(chunk))
    return np.stack(rgb), np.stack(acts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--route0", required=True)
    p.add_argument("--teacher", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--hold", type=float, default=4.0)
    args = p.parse_args()
    h = int(args.horizon)
    r0 = load_episode(Path(args.route0))
    teacher = []
    for path in args.teacher:
        teacher.extend(load_episode(Path(path)))
    rgb0, act0 = _windows(r0, h)
    rgbt, actt = _windows(teacher, h)
    rgb0_t = torch.as_tensor(rgb0, device="cuda")
    rgbt_t = torch.as_tensor(rgbt, device="cuda")
    act0_t = torch.as_tensor(act0, device="cuda")
    actt_t = torch.as_tensor(actt, device="cuda")
    model = ActionChunkNet(horizon=h).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    turn = (actt_t[:, 0, 0] < 0.3).float()
    weights = 1.0 + 30.0 * turn
    for step in range(int(args.steps)):
        opt.zero_grad()
        pred_t = model(rgbt_t)
        pred_0 = model(rgb0_t)
        err = (pred_t - actt_t).pow(2).mean(dim=(1, 2))
        fit = (err * weights).sum() / weights.sum()
        hold = torch.nn.functional.mse_loss(pred_0, act0_t)
        loss = fit + float(args.hold) * hold
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == int(args.steps):
            with torch.no_grad():
                print(
                    f"step={step} teacher_mae={float((pred_t-actt_t).abs().mean()):.4f} "
                    f"route0_mae={float((pred_0-act0_t).abs().mean()):.4f}"
                )
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
