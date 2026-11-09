"""Train a continuous stick network on the depth map. The canyon and the crash stay out."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.clearance_stick import _opening
from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.depth_policy import DepthPolicy, resize_depth
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.goal_features import goal_rel_from_obs
from experiments.aerial.scripts.train_clearance_view import _center_stats, _depth
from experiments.aerial.scripts.train_posterior_gate import _clear


def _close(tr) -> bool:
    mean_m, _p10 = _center_stats(_depth(tr.obs))
    _fwd, left, right = _clear(tr.obs)
    return float(mean_m) < 12.0 and not _opening(mean_m, left, right)


def _target(tr, close: bool) -> np.ndarray:
    goal = goal_rel_from_obs(tr.obs)
    bearing = float(np.arctan2(goal[1], max(float(goal[0]), 1e-3)))
    dyaw = float(np.clip(0.12 * bearing, -0.08, 0.08))
    if close:
        return np.array([0.0, 0.0, 0.0, dyaw], dtype=np.float32)
    flown = np.asarray(tr.action, dtype=np.float32).reshape(4)
    return np.array(
        [
            float(np.clip(flown[0], 0.0, 1.0)),
            0.0,
            0.0,
            float(np.clip(flown[3], -0.08, 0.08)),
        ],
        dtype=np.float32,
    )


def _maps(predictor, frames) -> np.ndarray:
    rows = []
    for tr in frames:
        predictor.reset()
        depth = predictor.predict_depth(tr.obs)
        rows.append(resize_depth(depth))
    return np.stack(rows).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", required=True)
    parser.add_argument("--street", required=True)
    parser.add_argument("--passage", nargs="+", required=True)
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument("--crash", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    street = load_episode(Path(args.street))
    train = [tr for tr in street if float(np.asarray(tr.obs.position)[1]) > 50.0]
    canyon = [tr for tr in street if float(np.asarray(tr.obs.position)[1]) <= 50.0]
    for path in args.passage:
        train.extend(load_episode(Path(path)))
    for item in args.extra:
        root = Path(item)
        files = sorted(root.glob("episode_*.npz")) if root.is_dir() else [root]
        for ep_path in files:
            for tr in load_episode(ep_path):
                if _close(tr):
                    train.append(tr)
    crash = [tr for tr in load_episode(Path(args.crash)) if _close(tr)]
    print(f"train={len(train)} canyon={len(canyon)} crash_close={len(crash)}")
    if len(train) < 20 or len(canyon) < 10 or len(crash) < 5:
        raise SystemExit("refuse: not enough frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = DepthMinPredictor.from_checkpoint(args.depth, device=str(device))
    print("depth head ready", flush=True)
    train_depth = _maps(predictor, train)
    canyon_depth = _maps(predictor, canyon)
    crash_depth = _maps(predictor, crash)
    train_goal = np.stack([goal_rel_from_obs(tr.obs) for tr in train])
    canyon_goal = np.stack([goal_rel_from_obs(tr.obs) for tr in canyon])
    crash_goal = np.stack([goal_rel_from_obs(tr.obs) for tr in crash])
    close_flags = np.asarray([_close(tr) for tr in train], dtype=bool)
    train_target = np.stack([_target(tr, bool(flag)) for tr, flag in zip(train, close_flags)])
    print(f"train_close={int(close_flags.sum())}", flush=True)

    policy = DepthPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=float(args.lr))
    depth_t = torch.as_tensor(train_depth)
    goal_t = torch.as_tensor(train_goal)
    target_t = torch.as_tensor(train_target)
    close_idx = np.flatnonzero(close_flags)
    batch = min(int(args.batch), len(train))
    policy.train()
    for step in range(int(args.steps)):
        opt.zero_grad()
        if close_idx.size and batch > 1:
            n_close = batch // 2
            a = close_idx[np.random.randint(0, close_idx.size, size=n_close)]
            b = np.random.randint(0, len(train), size=batch - n_close)
            pick = torch.as_tensor(np.concatenate([a, b]))
        else:
            pick = torch.randint(0, len(train), (batch,))
        pred = policy(depth_t[pick].to(device), goal_t[pick].to(device))
        loss = torch.mean((pred - target_t[pick].to(device)) ** 2)
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 200 == 0:
            print(f"step={step} loss={float(loss.detach()):.4f}", flush=True)

    policy.eval()

    def run(depth, goal) -> np.ndarray:
        with torch.no_grad():
            return policy(torch.as_tensor(depth).to(device), torch.as_tensor(goal).to(device)).cpu().numpy()

    canyon_act = run(canyon_depth, canyon_goal)
    canyon_ok = (canyon_act[:, 0] > 0.5) & (np.abs(canyon_act[:, 3]) < 0.1)
    crash_act = run(crash_depth, crash_goal)
    crash_ok = crash_act[:, 0] < 0.2
    print(
        f"canyon forward={int(canyon_ok.sum())}/{len(canyon)} "
        f"dx={float(canyon_act[:, 0].mean()):.2f} |dyaw|max={float(np.abs(canyon_act[:, 3]).max()):.3f}"
    )
    print(
        f"crash stop={int(crash_ok.sum())}/{len(crash)} "
        f"dx_med={float(np.median(crash_act[:, 0])):.2f}"
    )
    ok = bool(canyon_ok.all()) and bool(crash_ok.all())
    out = Path(args.out)
    if ok:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": policy.state_dict()}, str(out))
        print(f"PASS saved {out}")
    else:
        print("FAIL depth policy")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
