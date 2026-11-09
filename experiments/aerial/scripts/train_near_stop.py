"""Train the blocked-front stop. The route 0 canyon stays out of the loss."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.aerial.rl.clearance_stick import _opening
from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.near_stop import NearStop
from experiments.aerial.scripts.train_clearance_view import _center_stats, _depth
from experiments.aerial.scripts.train_posterior_gate import _batch, _clear


def _blocked(tr) -> bool:
    mean_m, _p10 = _center_stats(_depth(tr.obs))
    _fwd, left, right = _clear(tr.obs)
    return float(mean_m) < 12.0 and not _opening(mean_m, left, right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--street", required=True)
    parser.add_argument("--passage", nargs="+", required=True)
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init", default="")
    args = parser.parse_args()

    street = load_episode(Path(args.street))
    train = [tr for tr in street if float(np.asarray(tr.obs.position)[1]) > 50.0]
    canyon = [tr for tr in street if float(np.asarray(tr.obs.position)[1]) <= 50.0]
    for path in list(args.passage) + list(args.extra):
        item = Path(path)
        files = sorted(item.glob("episode_*.npz")) if item.is_dir() else [item]
        for ep_path in files:
            train.extend(load_episode(ep_path))
    labels = np.asarray([1.0 if _blocked(tr) else 0.0 for tr in train], dtype=np.float32)
    print(f"train={len(train)} blocked={int(labels.sum())} canyon={len(canyon)}")
    if int(labels.sum()) < 10 or len(canyon) < 10:
        raise SystemExit("refuse: not enough frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NearStop().to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
        print(f"init={args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    rgb, _prop = _batch(train, None)
    target = torch.as_tensor(labels)
    pos = torch.nonzero(target > 0.5, as_tuple=False).flatten()
    batch = min(int(args.batch), len(train))
    model.train()
    for step in range(int(args.steps)):
        opt.zero_grad()
        n_pos = batch // 2
        pick_pos = pos[torch.randint(0, int(pos.numel()), (n_pos,))]
        pick_rest = torch.randint(0, len(train), (batch - n_pos,))
        pick = torch.cat([pick_pos, pick_rest])
        logit = model.logit(rgb[pick].to(device))
        loss = F.binary_cross_entropy_with_logits(logit, target[pick].to(device))
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 200 == 0:
            print(f"step={step} loss={float(loss.detach()):.4f}")

    model.eval()

    def rate(frames) -> tuple[int, int]:
        hits = 0
        rgb_all, _ = _batch(frames, None)
        with torch.no_grad():
            logit = model.logit(rgb_all.to(device)).cpu().numpy()
        for flag in logit > 0.0:
            hits += int(bool(flag))
        return hits, len(frames)

    canyon_hits, canyon_n = rate(canyon)
    blocked_frames = [tr for tr, flag in zip(train, labels) if flag > 0.5]
    blocked_hits, blocked_n = rate(blocked_frames[:240])
    print(f"canyon_blocked={canyon_hits}/{canyon_n} train_blocked={blocked_hits}/{blocked_n}")
    ok = canyon_hits == 0 and blocked_hits >= int(0.8 * blocked_n) and blocked_n >= 10
    out = Path(args.out)
    if ok:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, str(out))
        print(f"PASS saved {out}")
    else:
        print("FAIL near stop")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
