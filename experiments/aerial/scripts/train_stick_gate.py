"""Step 2: sticks from the frozen encoder. Nothing is flown.

Street targets are the sticks route 0 flew. A facade frame yaws toward the
side with more room and holds forward until that side is the forward view.
The acceptance goal is fixed at 86.6 m ahead and 21 m to the left.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.aerial.rl.actor_critic import LatentActorCritic
from experiments.aerial.rl.corrector import _save_actor_ckpt
from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.train_rl import load_torch_dynamics
from experiments.aerial.rl.gap_passage import passage_phase
from experiments.aerial.rl.goal_features import goal_rel_body
from experiments.aerial.scripts.train_posterior_gate import _batch, _clear, _logits

_YAW_LIMIT = 0.3141592653589793
_GOAL = np.array([86.6, 21.0, 0.0, float(np.hypot(86.6, 21.0))], dtype=np.float32)


def _latent(model, transitions) -> np.ndarray:
    rgb, prop = _batch(transitions, None)
    with torch.no_grad():
        logits = _logits(model, rgb, prop, 1.0)
        eye = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(model.torch_dtype)
        z = eye.reshape(eye.shape[0], -1)
        h = model.rssm.initial_h(z.shape[0], model.device, model.torch_dtype)
        return torch.cat([h, z], dim=-1).float().cpu().numpy()


def _wall_target(tr) -> np.ndarray:
    fwd, left, right = _clear(tr.obs)
    sign = 1.0 if float(left) >= float(right) else -1.0
    # The opening is already the forward view: fly. Otherwise yaw in place.
    aligned = float(fwd) >= max(float(left), float(right))
    dx = 1.0 if aligned else 0.0
    dyaw = 0.0 if aligned else sign * _YAW_LIMIT
    return np.array([dx, 0.0, 0.0, dyaw], dtype=np.float64)


def _act(ac: LatentActorCritic, z: np.ndarray) -> np.ndarray:
    goal = _GOAL
    return np.stack(
        [ac.act_latent(z[i], goal_rel=goal, deterministic=True) for i in range(len(z))]
    )


def _gate(street_a, wall_a, wall_frames, canyon_a) -> bool:
    s_dx, s_yaw = street_a[:, 0], street_a[:, 3]
    street_ok = int(np.sum((s_dx >= 0.75) & (s_dx <= 1.0) & (np.abs(s_yaw) < 0.08)))
    wall_ok = 0
    for act, tr in zip(wall_a, wall_frames):
        _fwd, left, right = _clear(tr.obs)
        sign = 1.0 if float(left) >= float(right) else -1.0
        if abs(float(act[3])) >= 0.15 and float(act[3]) * sign > 0.0:
            wall_ok += 1
    c_dx, c_yaw = canyon_a[:, 0], canyon_a[:, 3]
    canyon_good = (c_dx > 0.5) & (np.abs(c_yaw) < 0.1)
    canyon_ok = int(np.sum(canyon_good))
    if canyon_ok != len(canyon_a):
        print("canyon miss", np.nonzero(~canyon_good)[0].tolist(), np.round(c_yaw[~canyon_good], 3).tolist())
    print(
        f"street {street_ok}/{len(street_a)} dx={s_dx.mean():.3f} |dyaw|={np.abs(s_yaw).mean():.3f}"
    )
    print(
        f"wall {wall_ok}/{len(wall_a)} dyaw_min={wall_a[:, 3].min():.3f} "
        f"dyaw_max={wall_a[:, 3].max():.3f}"
    )
    print(
        f"canyon {canyon_ok}/{len(canyon_a)} dx={c_dx.mean():.3f} "
        f"|dyaw|={np.abs(c_yaw).mean():.3f}"
    )
    return street_ok >= 10 and wall_ok >= 10 and canyon_ok == len(canyon_a)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--actor", required=True)
    p.add_argument("--street", required=True)
    p.add_argument("--facade", nargs="+", required=True)
    p.add_argument("--break-flight", nargs="*", default=[])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    model, _ = load_torch_dynamics(
        cfg.get("world_model") or {}, args.wm, device="cuda", freeze=True
    )
    model.eval()
    street_ep = load_episode(Path(args.street))
    street, canyon, wall = [], [], []
    for tr in street_ep:
        y = float(np.asarray(tr.obs.position)[1])
        fwd = float(_clear(tr.obs)[0])
        if y <= 50.0:
            canyon.append(tr)
        elif fwd > 20.0:
            street.append(tr)
    for path in args.facade:
        fac_ep = load_episode(Path(path))
        info = getattr(fac_ep[0].obs, "info", None) or {}
        goal = np.asarray(info["goal"], dtype=np.float64).reshape(3)
        for tr in fac_ep:
            pos = np.asarray(tr.obs.position, dtype=np.float64)
            _yaw, phase = passage_phase(pos, goal)
            if phase == "west" and float(_clear(tr.obs)[0]) < 20.0:
                wall.append(tr)
    print(f"street={len(street)} wall={len(wall)} canyon={len(canyon)}")
    if len(street) < 10 or len(wall) < 10 or len(canyon) < 5:
        raise SystemExit("refuse: not enough frames")

    z_s = _latent(model, street)
    z_w = _latent(model, wall)
    z_c = _latent(model, canyon)
    rec = int(model.rssm.recurrent_dim)
    wall_codes = z_w[:, rec:].reshape(len(z_w), 32, 32).argmax(-1)
    street_codes = z_s[:, rec:].reshape(len(z_s), 32, 32).argmax(-1)
    # Frames already on the street code cannot also be told to yaw.
    ham_street = (wall_codes != street_codes[0]).sum(axis=1)
    far = ham_street >= 5
    print(
        f"wall_unique_modes={len({tuple(r) for r in wall_codes})} "
        f"wall_far={int(far.sum())}"
    )
    if int(far.sum()) < 10:
        raise SystemExit("refuse: fewer than 10 wall frames off the street code")
    a_s = np.stack([np.asarray(tr.action, dtype=np.float64)[:4] for tr in street])
    a_c = np.stack([np.asarray(tr.action, dtype=np.float64)[:4] for tr in canyon])
    a_c[:, 3] = np.clip(a_c[:, 3], -0.05, 0.05)
    a_w = np.stack([_wall_target(tr) for tr in wall])
    print(
        f"target street dx={a_s[:, 0].mean():.3f} dyaw={a_s[:, 3].mean():.3f} "
        f"wall_far dyaw={a_w[far, 3].mean():.3f}"
    )
    goals = np.repeat(_GOAL.reshape(1, 4), len(z_s), axis=0)
    goalc = np.repeat(_GOAL.reshape(1, 4), len(z_c), axis=0)
    goalw = np.repeat(_GOAL.reshape(1, 4), int(far.sum()), axis=0)
    break_street, break_wall = [], []
    for broke_path in args.break_flight:
        broke = load_episode(Path(broke_path))
        for tr in broke:
            pos = np.asarray(tr.obs.position, dtype=np.float64)
            yaw = float(tr.obs.yaw)
            fwd, left, right = _clear(tr.obs)
            longer = max(float(left), float(right))
            if fwd < 20.0 and longer >= fwd + 12.0 and longer >= 20.0:
                break_wall.append(tr)
            elif float(pos[1]) >= 78.0 and fwd > 20.0 and not (0.4 < yaw < 2.6):
                break_street.append(tr)
    z_bs = _latent(model, break_street) if break_street else None
    z_bw = _latent(model, break_wall) if break_wall else None
    if break_street:
        g_bs = np.stack([
            goal_rel_body(tr.obs.position, float(tr.obs.yaw), np.asarray(tr.obs.info["goal"]))
            for tr in break_street
        ])
        a_bs = np.tile(np.array([float(a_s[:, 0].mean()), 0.0, 0.0, 0.0]), (len(break_street), 1))
    if break_wall:
        g_bw = np.stack([
            goal_rel_body(tr.obs.position, float(tr.obs.yaw), np.asarray(tr.obs.info["goal"]))
            for tr in break_wall
        ])
        a_bw = np.stack([_wall_target(tr) for tr in break_wall])
    print(f"break_street={len(break_street)} break_wall={len(break_wall)}")
    ac = LatentActorCritic.load_from_checkpoint(args.actor, device="cuda")
    for step in range(int(args.steps)):
        street_stat = ac.update_bc(z_s, a_s, goals)
        ac.update_bc(z_c, a_c, goalc, loss_scale=6.0)
        wall_stat = ac.update_bc(z_w[far], a_w[far], goalw)
        if z_bs is not None:
            ac.update_bc(z_bs, a_bs, g_bs)
        if z_bw is not None:
            ac.update_bc(z_bw, a_bw, g_bw)
            ac.update_bc(z_bw, a_bw, np.repeat(_GOAL.reshape(1, 4), len(z_bw), axis=0))
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 50 == 0:
            print(
                f"step={step} street_mae={street_stat['bc_mae']:.4f} "
                f"wall_mae={wall_stat['bc_mae']:.4f}"
            )
    passed = _gate(_act(ac, z_s), _act(ac, z_w), wall, _act(ac, z_c))
    if passed:
        _save_actor_ckpt(args.out_dir, ac, int(args.steps))
        print(f"PASS saved {args.out_dir}")
    else:
        print("FAIL stick gate")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
