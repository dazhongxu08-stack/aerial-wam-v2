"""Learn sticks on the wall code and, where separable, on a third gap code.

Street frames keep the frozen actor. The head is not applied there.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.gap_passage import gap_action, passage_phase
from experiments.aerial.rl.passage_code import LearnedPassage
from experiments.aerial.scripts.train_passage_code import _release_index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--street", required=True)
    p.add_argument("--passage", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()

    street = load_episode(Path(args.street))
    episodes = [load_episode(Path(path)) for path in args.passage]
    goal = np.asarray(episodes[0][0].obs.info["goal"], dtype=np.float64).reshape(3)
    groups = {"street": [], "wall": [], "gap": []}
    targets = {"wall": [], "gap": []}
    for tr in street:
        groups["street"].append(np.asarray(tr.obs.rgb))
    for passage in episodes:
        release = _release_index(passage, goal)
        for tr in passage[:release]:
            pos = np.asarray(tr.obs.position, dtype=np.float64)
            _yaw, phase = passage_phase(pos, goal)
            if phase == "west":
                key = "wall"
            elif phase in ("south", "east"):
                key = "gap"
            else:
                continue
            groups[key].append(np.asarray(tr.obs.rgb))
            action, _ph = gap_action(pos, float(tr.obs.yaw), goal, z_hold=29.0)
            targets[key].append(np.asarray(action, dtype=np.float32))
    print(
        f"street={len(groups['street'])} wall={len(groups['wall'])} "
        f"gap={len(groups['gap'])} episodes={len(episodes)}"
    )
    if len(groups["wall"]) < 4 or len(groups["gap"]) < 4:
        raise SystemExit("not enough labeled frames")

    rgb = {k: torch.as_tensor(np.stack(v), device="cuda") for k, v in groups.items()}
    tgt = {k: torch.as_tensor(np.stack(v), device="cuda") for k, v in targets.items()}
    model = LearnedPassage().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    names = ("street", "wall", "gap")
    for step in range(int(args.steps)):
        opt.zero_grad()
        loss_ce = 0.0
        loss_act = 0.0
        packed = []
        for lab, name in enumerate(names):
            z, ss, sw, sg = model.scores(rgb[name])
            logits = torch.stack([ss, sw, sg], dim=1) * 10.0
            y = torch.full((z.shape[0],), lab, dtype=torch.long, device=z.device)
            loss_ce = loss_ce + F.cross_entropy(logits, y)
            packed.append((name, z))
        for name, z in packed:
            if name == "street":
                continue
            pred = model.stick(z)
            turn = (tgt[name][:, 0] < 0.3).float()
            weights = 1.0 + 25.0 * turn
            err = (pred - tgt[name]).pow(2).mean(dim=1)
            loss_act = loss_act + (err * weights).sum() / weights.sum()
        loss = loss_ce + loss_act
        loss.backward()
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 100 == 0:
            print(f"step={step} ce={float(loss_ce):.4f} act={float(loss_act):.4f}")

    def _decide(name: str) -> np.ndarray:
        with torch.no_grad():
            _z, ss, sw, sg = model.scores(rgb[name])
        return torch.stack([ss, sw, sg], dim=1).detach().cpu().numpy()

    street_s = _decide("street")
    wall_s = _decide("wall")
    gap_s = _decide("gap")
    # positive means a non-street code beats street
    street_fire = np.maximum(street_s[:, 1], street_s[:, 2]) - street_s[:, 0]
    margin = max(0.05, float(street_fire.max()) + 0.05)
    model.margin = margin

    def _hits(scores: np.ndarray, code: int) -> int:
        best = scores.argmax(1)
        beat = np.maximum(scores[:, 1], scores[:, 2]) - scores[:, 0]
        return int(np.sum((best == code) & (beat > margin)))

    print(
        f"margin={margin:.3f} street_fires={int(np.sum(street_fire > margin))}/{len(street_s)} "
        f"wall_hits={_hits(wall_s, 1)}/{len(wall_s)} gap_hits={_hits(gap_s, 2)}/{len(gap_s)}"
    )
    with torch.no_grad():
        pred_w = model.stick(model.scores(rgb["wall"])[0])
        pred_g = model.stick(model.scores(rgb["gap"])[0])
        turn = tgt["wall"][:, 0] < 0.3
        if int(turn.sum()) > 0:
            print(
                "west_turn_mae",
                float((pred_w[turn] - tgt["wall"][turn]).abs().mean()),
                "pred",
                np.round(pred_w[turn][0].detach().cpu().numpy(), 3).tolist(),
                "tgt",
                np.round(tgt["wall"][turn][0].detach().cpu().numpy(), 3).tolist(),
            )
        print(
            "gap_mae",
            float((pred_g - tgt["gap"]).abs().mean()),
        )
    if int(np.sum(street_fire > margin)) > 0 or _hits(wall_s, 1) < len(wall_s):
        raise SystemExit("refuse: street code is not clean or the west frames are not all wall")
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
