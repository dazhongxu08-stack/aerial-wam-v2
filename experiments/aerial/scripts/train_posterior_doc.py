"""Step 1 of the main-channel plan. The actor is not updated. Nothing is flown.

Street frames: the arrived route 0 flight, forward clearance above 20 m, and
north of the held-out canyon. Wall frames: the exam-camera passage, west phase,
forward clearance under 20 m. The route 0 canyon and the early part of another
wall approach are held out.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.gap_passage import passage_phase
from experiments.aerial.rl.train_rl import load_torch_dynamics
from experiments.aerial.scripts.train_posterior_gate import (
    _bake_pose_scale,
    _batch,
    _clear,
    _logits,
)


def _goal(ep):
    info = getattr(ep[0].obs, "info", None) or {}
    return np.asarray(info["goal"], dtype=np.float64).reshape(3)


def _west_near(ep, *, fwd_lo: float, fwd_hi: float, heading_max: Optional[float] = None):
    goal = _goal(ep)
    kept = []
    for tr in ep:
        pos = np.asarray(tr.obs.position, dtype=np.float64)
        yaw, phase = passage_phase(pos, goal)
        if phase != "west":
            continue
        fwd = float(_clear(tr.obs)[0])
        if not (fwd_lo <= fwd < fwd_hi):
            continue
        if heading_max is not None:
            err = abs((float(tr.obs.yaw) + 2.094395 + np.pi) % (2.0 * np.pi) - np.pi)
            if err > float(heading_max):
                continue
        kept.append(tr)
    return kept


def _modes(model, frames):
    rgb, prop = _batch(frames, None)
    with torch.no_grad():
        logits = _logits(model, rgb, prop, 1.0)
        flat = logits.reshape(logits.shape[0], -1)
        probs = model.rssm._logits_to_probs(flat)
    modes = probs.argmax(-1).detach().cpu().numpy().astype(np.int16)
    maxp = float(probs.max(-1).values.mean().detach())
    return modes, maxp


def _as_set(modes: np.ndarray) -> set:
    return {tuple(int(v) for v in row) for row in modes}


def _nearer(query: np.ndarray, prefer: np.ndarray, other: np.ndarray) -> int:
    """How many query modes sit closer to ``prefer`` than to ``other``."""
    prefer_d = np.stack([(query != row).sum(axis=1) for row in prefer], axis=1).min(axis=1)
    other_d = np.stack([(query != row).sum(axis=1) for row in other], axis=1).min(axis=1)
    return int(np.sum(prefer_d < other_d))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--street", required=True)
    p.add_argument("--passage", required=True)
    p.add_argument("--held-wall", required=True)
    p.add_argument("--break-flight", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    model, _ = load_torch_dynamics(
        cfg.get("world_model") or {}, args.wm, device="cuda", freeze=False
    )
    street_ep = load_episode(Path(args.street))
    street, canyon = [], []
    for tr in street_ep:
        y = float(np.asarray(tr.obs.position)[1])
        fwd = float(_clear(tr.obs)[0])
        if y <= 50.0:
            canyon.append(tr)
        elif fwd > 20.0:
            street.append(tr)
    wall = _west_near(load_episode(Path(args.passage)), fwd_lo=0.0, fwd_hi=20.0)
    extra_street = extra_wall = 0
    for broke_path in args.break_flight:
        broke = load_episode(Path(broke_path))
        for tr in broke:
            pos = np.asarray(tr.obs.position, dtype=np.float64)
            yaw = float(tr.obs.yaw)
            fwd, left, right = _clear(tr.obs)
            longer = max(float(left), float(right))
            # Facade in front: forward is short and one side is clearly open.
            if fwd < 20.0 and longer >= fwd + 12.0 and longer >= 20.0:
                wall.append(tr)
                extra_wall += 1
                continue
            # Still on the outbound corridor, before the nose swings north.
            north = 0.4 < yaw < 2.6
            if float(pos[1]) >= 78.0 and fwd > 20.0 and not north:
                street.append(tr)
                extra_street += 1
        print(f"break_street={extra_street} break_wall={extra_wall}")
    # Early part of the other approach: still in front of the facade, not the contact.
    held_wall = _west_near(
        load_episode(Path(args.held_wall)),
        fwd_lo=0.0,
        fwd_hi=20.0,
        heading_max=0.30,
    )
    print(
        f"street={len(street)} wall={len(wall)} "
        f"canyon={len(canyon)} held_wall={len(held_wall)}"
    )
    if min(len(street), len(wall), len(canyon)) < 5 or len(held_wall) < 4:
        raise SystemExit("refuse: not enough frames")

    for param in model.parameters():
        param.requires_grad_(False)
    for module in (model.encoder, model.rssm.post_net):
        for param in module.parameters():
            param.requires_grad_(True)
    opt = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=float(args.lr)
    )
    rgb_s, prop_s = _batch(street, None)
    rgb_w, prop_w = _batch(wall, None)
    with torch.no_grad():
        pose_ref = model.proprio_mlp(prop_s.to(model.device, model.torch_dtype)).float()
        scale = float(pose_ref.std().clamp_min(1.0))
    print(f"pose_scale={scale:.1f}")

    def _ce(rgb, prop, class_id: int):
        logits = _logits(model, rgb, prop, scale)
        target = torch.full(
            (logits.shape[0], logits.shape[1]),
            int(class_id),
            device=logits.device,
            dtype=torch.long,
        )
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

    model.train()
    for step in range(int(args.steps)):
        opt.zero_grad()
        loss = _ce(rgb_s, prop_s, 0) + _ce(rgb_w, prop_w, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad], 5.0
        )
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 50 == 0:
            print(f"step={step} loss={float(loss.detach()):.4f}")

    _bake_pose_scale(model, scale)
    model.eval()
    sm, sp = _modes(model, street)
    wm, wp = _modes(model, wall)
    cm, cp = _modes(model, canyon)
    hm, hp = _modes(model, held_wall)
    street_set, wall_set = _as_set(sm), _as_set(wm)
    overlap = len(street_set & wall_set)
    # Training modes must not be the same tuple. Held-out frames must belong
    # to the nearer group, which is what 同一群 asks for.
    canyon_near = _nearer(cm, sm, wm)
    held_near = _nearer(hm, wm, sm)
    mean_max = float(np.mean([sp, wp, cp, hp]))
    print(
        f"overlap={overlap} canyon_near_street={canyon_near}/{len(cm)} "
        f"held_wall_near_wall={held_near}/{len(hm)} maxp={mean_max:.3f}"
    )
    ok = (
        overlap == 0
        and canyon_near == len(cm)
        and held_near == len(hm)
        and mean_max >= 0.15
    )
    if ok:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(str(out), step=int(args.steps))
        print(f"PASS saved {out}")
    else:
        print("FAIL posterior gate")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
