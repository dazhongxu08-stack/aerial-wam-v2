"""Learn the stick from the current picture.

Open-street frames keep the sticks that flew route 0. Passage frames get the
side-passage sticks. The two pictures must produce different sticks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.visual_residual import VisualResidual


def _batch(transitions):
    rgb = torch.as_tensor(np.stack([tr.obs.rgb for tr in transitions]), device="cuda")
    act = torch.as_tensor(
        np.stack([np.asarray(tr.action, np.float32) for tr in transitions]),
        device="cuda",
    )
    return rgb, act


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--street", required=True)
    p.add_argument("--passage", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()
    street = load_episode(Path(args.street))
    passage = load_episode(Path(args.passage))
    rgb_s, act_s = _batch(street)
    rgb_p, act_p = _batch(passage)
    model = VisualResidual().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    turn = (act_p[:, 0] < 0.3).float()
    w_p = 1.0 + 20.0 * turn
    for step in range(int(args.steps)):
        opt.zero_grad()
        pred_s = model(rgb_s)
        pred_p = model(rgb_p)
        loss_s = torch.nn.functional.mse_loss(pred_s, act_s)
        err_p = (pred_p - act_p).pow(2).mean(dim=1)
        loss_p = (err_p * w_p).sum() / w_p.sum()
        loss = loss_s + loss_p
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 100 == 0:
            with torch.no_grad():
                print(
                    f"step={step} street_mae={float((pred_s-act_s).abs().mean()):.4f} "
                    f"passage_mae={float((pred_p-act_p).abs().mean()):.4f}"
                )
    with torch.no_grad():
        pred_all = model(rgb_s).detach().cpu().numpy()
        a0 = pred_all[0]
        b0 = model(rgb_p[:1]).detach().cpu().numpy()[0]
        print(
            "street_dx_std", float(pred_all[:, 0].std()),
            "street_dyaw_std", float(pred_all[:, 3].std()),
        )
    print("street0", np.round(a0, 3).tolist(), "target", np.round(act_s[0].detach().cpu().numpy(), 3).tolist())
    print("wall0", np.round(b0, 3).tolist(), "target", np.round(act_p[0].detach().cpu().numpy(), 3).tolist())
    if abs(float(a0[0] - b0[0])) < 0.2 and abs(float(a0[3] - b0[3])) < 0.15:
        raise SystemExit("refuse: street and wall pictures still produce the same stick")
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
