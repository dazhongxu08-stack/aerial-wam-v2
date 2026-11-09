"""Separate street and wall codes. The passage tape is only played for the wall code."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.gap_passage import gap_action, passage_phase
from experiments.aerial.rl.passage_code import PassageEncoder, PassagePlayer


def _facing_south(yaw: float) -> bool:
    err = (float(yaw) + math.pi / 2.0 + math.pi) % (2.0 * math.pi) - math.pi
    return abs(err) < 0.45


def _release_index(transitions, goal) -> int:
    for i, tr in enumerate(transitions):
        pos = np.asarray(tr.obs.position, dtype=np.float64)
        if float(pos[0]) >= -948.0 and float(pos[1]) <= 26.0 and _facing_south(tr.obs.yaw):
            return i
    return len(transitions)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--street", required=True, help="route 0 episode, open street")
    p.add_argument("--passage", required=True, help="route 1 passage recorded in the eval camera")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()

    street = load_episode(Path(args.street))
    passage = load_episode(Path(args.passage))
    goal = None
    for tr in passage:
        info = getattr(tr.obs, "info", None) or {}
        if isinstance(info, dict) and info.get("goal") is not None:
            goal = np.asarray(info["goal"], dtype=np.float64).reshape(3)
            break
    if goal is None:
        raise SystemExit("passage episode has no goal")
    release = _release_index(passage, goal)
    if release < 8:
        raise SystemExit(f"release index {release} is too early")
    # Trigger code uses the west approach, before the canyon shared with route 0.
    trigger = []
    tape = []
    for i, tr in enumerate(passage[:release]):
        pos = np.asarray(tr.obs.position, dtype=np.float64)
        _yaw, phase = passage_phase(pos, goal)
        action, _ph = gap_action(pos, float(tr.obs.yaw), goal, z_hold=29.0)
        tape.append(np.asarray(action, dtype=np.float32))
        if phase == "west":
            trigger.append(np.asarray(tr.obs.rgb))
    if len(trigger) < 4:
        raise SystemExit(f"only {len(trigger)} west frames")
    print(f"street={len(street)} west={len(trigger)} release_idx={release}")

    rgb_s = torch.as_tensor(np.stack([tr.obs.rgb for tr in street]), device="cuda")
    rgb_w = torch.as_tensor(np.stack(trigger), device="cuda")
    model = PassageEncoder().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(int(args.steps)):
        opt.zero_grad()
        ss, sw = model.scores(rgb_s)
        ws, ww = model.scores(rgb_w)
        street_loss = F.cross_entropy(
            torch.stack([ss, sw], dim=1) * 10.0,
            torch.zeros(len(street), dtype=torch.long, device="cuda"),
        )
        wall_loss = F.cross_entropy(
            torch.stack([ws, ww], dim=1) * 10.0,
            torch.ones(len(trigger), dtype=torch.long, device="cuda"),
        )
        loss = street_loss + wall_loss
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 100 == 0:
            with torch.no_grad():
                print(
                    f"step={step} street_wall_gap={(ss-sw).mean().item():.3f} "
                    f"wall_gap={(ww-ws).mean().item():.3f}"
                )

    with torch.no_grad():
        ss, sw = model.scores(rgb_s)
        ws, ww = model.scores(rgb_w)
        street_margin = (ss - sw).detach().cpu().numpy()
        wall_margin = (ww - ws).detach().cpu().numpy()
    # Route 0 must stay on the street code. The first west frame must be the wall code.
    need = float(np.max(-street_margin))
    margin = max(0.05, need + 0.02)
    n_street_fire = int(np.sum(-street_margin >= margin))
    n_wall_hit = int(np.sum(wall_margin > margin))
    print(
        f"margin={margin:.3f} street_fires={n_street_fire}/{len(street)} "
        f"west_hits={n_wall_hit}/{len(trigger)} first_west_gap={float(wall_margin[0]):.3f}"
    )
    if n_street_fire > 0 or float(wall_margin[0]) <= margin:
        raise SystemExit("refuse: codes do not separate the first wall frame from route 0")

    z_w = model.embed(rgb_w).detach().cpu().numpy()
    player = PassagePlayer(
        model.cpu().eval(),
        z_w,
        np.stack(tape),
        release,
        margin,
    )
    player.save(args.out)
    print(f"saved {args.out} tape={len(tape)} release={release}")


if __name__ == "__main__":
    main()
