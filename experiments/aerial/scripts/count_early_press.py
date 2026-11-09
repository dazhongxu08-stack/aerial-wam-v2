"""Count stable frontal approaches, not every frame inside 8–18 m.

A frame counts only inside a straight-press run where the 10th-percentile
forward range stays in (8, 18] m, neither side is closer than that, the run
lasts at least 3 frames, forward distance falls by at least 0.5 m, and no
step jumps by more than 2 m. Writes AUTO_REVIEW.json next to the dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.aerial.rl.depth_geometry import (
    approach_clearances,
    frontal_obstacle_frame,
    stable_approach,
)


def _runs(flags, values):
    t = 0
    n = len(flags)
    kept = 0
    n_runs = 0
    while t < n:
        if not flags[t]:
            t += 1
            continue
        start = t
        seq = []
        while t < n and flags[t]:
            seq.append(values[t])
            t += 1
        if stable_approach(seq):
            n_runs += 1
            kept += len(seq)
    return n_runs, kept


def count_dataset(root: Path) -> dict:
    files = sorted(root.glob("episode_*.npz"))
    n_frames = n_runs = n_stable = 0
    per_ep = []
    for path in files:
        with np.load(path) as raw:
            if "depth" not in raw.files or "actions" not in raw.files:
                per_ep.append({"file": path.name, "drop": "missing depth or actions"})
                continue
            depth = raw["depth"]
            actions = raw["actions"]
        flags = []
        values = []
        for t in range(int(depth.shape[0])):
            d_fwd, d_left, d_right = approach_clearances(depth[t])
            action = np.asarray(actions[t], dtype=np.float64).reshape(-1)
            press = bool(action[0] > 0.25 and abs(action[3]) < 0.08)
            frontal = frontal_obstacle_frame(d_fwd, d_left, d_right)
            flags.append(bool(press and frontal))
            values.append(float(d_fwd) if np.isfinite(d_fwd) else 1e9)
            n_frames += 1
        ep_runs, ep_kept = _runs(flags, values)
        n_runs += ep_runs
        n_stable += ep_kept
        per_ep.append(
            {
                "file": path.name,
                "steps": int(depth.shape[0]),
                "stable_press": ep_kept,
                "stable_runs": ep_runs,
                "keep": ep_kept > 0,
            }
        )
    return {
        "n_episodes": len(files),
        "n_frames": n_frames,
        "stable_runs": n_runs,
        "early_and_press": n_stable,
        "episodes": per_ep,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gate", type=int, default=40)
    args = parser.parse_args()
    root = Path(args.dataset)
    report = count_dataset(root)
    report["gate"] = int(args.gate)
    report["gate_pass"] = bool(report["early_and_press"] >= int(args.gate))
    report["scene"] = "urban_interior"
    report["depth"] = "airsim_depth_planar_p10"
    report["mask"] = (
        "p10 forward in (8,18], one side median >= forward+2, "
        "straight press run >=3 frames, drop>=0.5m, jump<=2m"
    )
    out = root / "AUTO_REVIEW.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"episodes={report['n_episodes']} frames={report['n_frames']} "
        f"stable_runs={report['stable_runs']} stable_press={report['early_and_press']} "
        f"gate={report['gate']} pass={report['gate_pass']}"
    )
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
