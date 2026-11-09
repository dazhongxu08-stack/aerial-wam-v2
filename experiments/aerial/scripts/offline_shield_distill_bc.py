"""Offline BC of route-1 shield-executed actions into a frozen-WM actor.

The npz ``actions`` are the post-shield commands from
``dataset_urban_hard134_depth_scene_20260918_hard134`` (shield=three_zone_fwd).
Route 1 is the around-building arrival. The world model is not updated.
A later no-shield actor eval is a separate process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.actor_critic import LatentActorCritic
from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.goal_features import (
    attach_goal,
    goal_rel_from_obs,
    resolve_episode_goal,
)
from experiments.aerial.rl.train_rl import load_torch_dynamics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--wm-ckpt", type=Path, required=True)
    p.add_argument("--init-actor", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    episodes = []
    for name in args.episodes:
        path = args.dataset_dir / name
        ep = load_episode(path)
        goal = resolve_episode_goal(ep, allow_end_proxy=True)
        if goal is None:
            raise SystemExit(f"no goal for {name}")
        attach_goal(ep, goal)
        episodes.append(ep)
        print(
            f"loaded {name} steps={len(ep)} goal={np.round(goal, 1).tolist()}",
            flush=True,
        )
    flat = [t for ep in episodes for t in ep]
    actions = np.stack([np.asarray(t.action, dtype=np.float64).reshape(4) for t in flat])
    dx = actions[:, 0]
    print(
        f"n={len(flat)} dx<0={(dx < -1e-3).mean():.3f} "
        f"mean|dyaw|={np.abs(actions[:, 3]).mean():.3f} "
        f"mean_dx={dx.mean():.3f}",
        flush=True,
    )

    dynamics, payload = load_torch_dynamics(
        {"device": args.device},
        args.wm_ckpt,
        device=args.device,
        freeze=True,
    )
    print(
        f"wm latent={dynamics.latent_dim} obstacle_cost_trained="
        f"{payload.get('obstacle_cost_trained')}",
        flush=True,
    )
    actor = LatentActorCritic.load_from_checkpoint(args.init_actor, device=args.device)
    if int(actor.config.latent_dim) != int(dynamics.latent_dim):
        raise SystemExit(
            f"actor latent {actor.config.latent_dim} != wm {dynamics.latent_dim}"
        )
    if not actor.bounded:
        raise SystemExit(f"refusing unbounded policy {actor.config.policy_class}")

    zs = []
    gs = []
    for i, t in enumerate(flat):
        zs.append(dynamics.encode(t.obs))
        gs.append(goal_rel_from_obs(t.obs))
        if i % 200 == 0:
            print(f"encode {i}/{len(flat)}", flush=True)
    z = np.stack(zs).astype(np.float32)
    g = np.stack(gs).astype(np.float32)
    if float(np.linalg.norm(g[:, :3], axis=1).mean()) < 1.0:
        raise SystemExit("goal_rel is ~0; refusing to BC without a goal")
    print(
        f"encoded {len(flat)} mean_goal_rel={float(np.linalg.norm(g[:, :3], axis=1).mean()):.1f}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    history = []
    n = len(flat)
    batch = min(int(args.batch), n)
    for step in range(int(args.updates)):
        idx = rng.integers(0, n, size=batch)
        out = actor.update_bc(z[idx], actions[idx], goal_rel=g[idx], loss_scale=1.0)
        if step % int(args.log_every) == 0 or step + 1 == int(args.updates):
            row = {"step": step, "bc_loss": out["bc_loss"], "bc_mae": out["bc_mae"]}
            history.append(row)
            print(f"bc step={step} loss={out['bc_loss']:.4f} mae={out['bc_mae']:.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor._actor.state_dict(),
            "critic": actor._critic.state_dict(),
            "log_std": actor._log_std.detach().cpu(),
            "config": actor.config.__dict__,
        },
        args.out / "v4_ac_latest.pt",
    )
    meta = {
        "kind": "shield_executed_bc",
        "routes": [1],
        "episodes": list(args.episodes),
        "n_steps": int(n),
        "updates": int(args.updates),
        "batch": int(batch),
        "dx_neg_frac": float((dx < -1e-3).mean()),
        "init_actor": str(args.init_actor),
        "wm_ckpt": str(args.wm_ckpt),
        "history": history,
        "note": "post-shield route-1 actions; WM frozen; eval must be no-shield",
    }
    (args.out / "bc_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out / 'v4_ac_latest.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
