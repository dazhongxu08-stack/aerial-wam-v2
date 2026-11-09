"""Step 2: sticks from the frozen clearance code and each frame's goal.

The route 0 canyon stays out of the loss. Nothing is flown.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.clearance_stick import ClearanceStick, stick_target
from experiments.aerial.rl.clearance_view import ClearanceView
from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.goal_features import goal_rel_from_obs
from experiments.aerial.scripts.train_clearance_view import _center_stats, _depth
from experiments.aerial.scripts.train_posterior_gate import _batch, _clear


def _rows(frames):
    goals, targets, yaw = [], [], []
    for tr in frames:
        mean_m, _p10 = _center_stats(_depth(tr.obs))
        _fwd, left, right = _clear(tr.obs)
        goal = goal_rel_from_obs(tr.obs)
        target = stick_target(mean_m, left, right, goal)
        goals.append(goal)
        targets.append(target)
        yaw.append(abs(float(target[3])) >= 0.15)
    return (
        np.asarray(goals, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(yaw, dtype=bool),
    )


def _predict(view, frames) -> np.ndarray:
    preds = []
    for start in range(0, len(frames), 64):
        rgb, _prop = _batch(frames[start : start + 64], None)
        with torch.no_grad():
            pred = view(rgb.to(next(view.parameters()).device))
        preds.append(torch.expm1(pred).cpu().numpy())
    return np.concatenate(preds)


def _act(stick, clearance, goals) -> np.ndarray:
    outs = []
    for start in range(0, len(clearance), 64):
        code = torch.as_tensor(clearance[start : start + 64])
        goal = torch.as_tensor(goals[start : start + 64])
        device = next(stick.parameters()).device
        with torch.no_grad():
            outs.append(stick(code.to(device), goal.to(device)).cpu().numpy())
    return np.concatenate(outs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", required=True)
    parser.add_argument("--street", required=True)
    parser.add_argument("--passage", nargs="+", required=True)
    parser.add_argument("--facade", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    street = load_episode(Path(args.street))
    train = [tr for tr in street if float(np.asarray(tr.obs.position)[1]) > 50.0]
    canyon = [tr for tr in street if float(np.asarray(tr.obs.position)[1]) <= 50.0]
    for path in args.passage:
        train.extend(load_episode(Path(path)))
    facade = []
    for path in args.facade:
        facade.extend(load_episode(Path(path)))
    print(f"train={len(train)} canyon={len(canyon)} facade_pool={len(facade)}")
    if len(train) < 20 or len(canyon) < 10:
        raise SystemExit("refuse: not enough frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    view = ClearanceView().to(device)
    view.load_state_dict(torch.load(args.view, map_location=device)["model"])
    view.eval()
    for param in view.parameters():
        param.requires_grad_(False)

    code = _predict(view, train)
    goals, targets, _yaw = _rows(train)
    n_yaw = int(np.sum(np.abs(targets[:, 3]) >= 0.15))
    print(f"train_yaw_frames={n_yaw}")
    yaw_idx = np.flatnonzero(np.abs(targets[:, 3]) >= 0.15)
    stick = ClearanceStick().to(device)
    opt = torch.optim.Adam(stick.parameters(), lr=float(args.lr))
    target_t = torch.as_tensor(targets)
    goal_t = torch.as_tensor(goals)
    code_t = torch.as_tensor(code)
    batch = min(int(args.batch), len(train))
    stick.train()
    for step in range(int(args.steps)):
        opt.zero_grad()
        if yaw_idx.size and batch > 1:
            n_turn = batch // 2
            turn = yaw_idx[np.random.randint(0, yaw_idx.size, size=n_turn)]
            rest = np.random.randint(0, len(train), size=batch - n_turn)
            pick = torch.as_tensor(np.concatenate([turn, rest]))
        else:
            pick = torch.randint(0, len(train), (batch,))
        pred = stick(code_t[pick].to(device), goal_t[pick].to(device))
        loss = torch.mean((pred - target_t[pick].to(device)) ** 2)
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 200 == 0:
            print(f"step={step} loss={float(loss.detach()):.4f}")

    stick.eval()
    canyon_goal, _canyon_target, _ = _rows(canyon)
    canyon_act = _act(stick, _predict(view, canyon), canyon_goal)
    canyon_ok = (canyon_act[:, 0] > 0.5) & (np.abs(canyon_act[:, 3]) < 0.1)
    print(
        f"canyon forward={int(canyon_ok.sum())}/{len(canyon)} "
        f"dx={float(canyon_act[:, 0].mean()):.2f} "
        f"|dyaw|max={float(np.abs(canyon_act[:, 3]).max()):.3f}"
    )

    facade_goal, facade_target, facade_yaw = _rows(facade)
    exam = np.flatnonzero(facade_yaw)
    if exam.size:
        facade_act = _act(stick, _predict(view, [facade[int(i)] for i in exam]), facade_goal[exam])
        true_sign = np.sign(facade_target[exam, 3])
        yaw_ok = (np.abs(facade_act[:, 3]) >= 0.15) & (facade_act[:, 3] * true_sign > 0)
        print(
            f"facade yaw={int(yaw_ok.sum())}/{int(exam.size)} "
            f"|dyaw|min={float(np.abs(facade_act[:, 3]).min()):.3f}"
        )
    else:
        yaw_ok = np.zeros(0, dtype=bool)
        print("facade yaw=0/0")

    passed = bool(canyon_ok.all()) and int(exam.size) >= 4 and bool(yaw_ok.all())
    out = Path(args.out)
    if passed:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": stick.state_dict()}, str(out))
        print(f"PASS saved {out}")
    else:
        print("FAIL clearance stick")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
