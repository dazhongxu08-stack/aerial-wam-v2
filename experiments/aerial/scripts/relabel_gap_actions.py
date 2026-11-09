"""Replace episode actions with the side-passage teacher at the same poses."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.aerial.rl.dataset import load_episode, write_episode
from experiments.aerial.rl.gap_passage import gap_action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--z-hold", type=float, default=29.0)
    args = p.parse_args()
    transitions = load_episode(Path(args.src))
    goal = None
    for tr in transitions:
        info = getattr(tr.obs, "info", None) or {}
        if isinstance(info, dict) and info.get("goal") is not None:
            goal = np.asarray(info["goal"], dtype=np.float64).reshape(3)
            break
    if goal is None:
        raise SystemExit("episode has no goal")
    for tr in transitions:
        action, _phase = gap_action(
            tr.obs.position, float(tr.obs.yaw), goal, z_hold=float(args.z_hold)
        )
        tr.action = np.asarray(action, dtype=np.float32)
    out = write_episode(Path(args.out_dir), 0, transitions)
    print(f"relabeled {len(transitions)} steps -> {out}")


if __name__ == "__main__":
    main()
