#!/usr/bin/env python3
"""Ask the flown step whether it meets a remembered 3D cell. No survival.

Past depth stays at the range where it was seen. The current pose puts those
points into 0.5 m cells indexed by forward, left, and up. Points already
inside the body are not written. The step meets a cell when the cell center
is within 0.4 m of the body path. Open frames ask the same of a forward 0.8 m
step.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/yao/aerial-wam-v2")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = ROOT / "experiments/aerial/rl/artifacts"
OUT = ART / "scenevoxels_20261108"
PAST = 16
BODY_M = 0.4


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "check.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def empty():
    return {
        "contact": 0,
        "memory": 0,
        "seen": 0,
        "gaps": [],
        "ups": [],
        "open": 0,
        "open_memory": 0,
        "open_gaps": [],
        "open_ups": [],
    }


def take(depth, state, t):
    s0 = max(0, t - PAST)
    if t - s0 < 1:
        return None
    return depth[s0:t], state[s0:t], state[t: t + 1]


def main() -> None:
    from experiments.aerial.rl.consequence_model import load_episode
    from experiments.aerial.rl.scene_voxels import nearest_voxel
    from experiments.aerial.scripts.consequence.check_scene_points import contact_at, episodes, is_open

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"start device {dev}")
    groups = {"val": empty(), "train": empty()}
    rows = episodes()
    log(f"episodes {len(rows)}")
    fwd = np.array([0.8, 0.0, 0.0, 0.0], np.float64)
    for n_ep, (is_val, path) in enumerate(rows):
        ep = load_episode(path)
        if "da3_depth56" not in ep or "collided" not in ep:
            continue
        depth_np = np.asarray(ep["da3_depth56"], np.float32)
        state_np = np.asarray(ep["state"], np.float32)
        action = np.asarray(ep["action"], np.float32)
        collided = np.asarray(ep["collided"]).astype(bool)
        n = min(len(depth_np), len(state_np), len(action), len(collided))
        if n < 4:
            continue
        depth = torch.as_tensor(depth_np[:n], device=dev)
        state = torch.as_tensor(state_np[:n], device=dev)
        key = "val" if is_val else "train"
        with torch.no_grad():
            for t in range(1, n - 1):
                hit = contact_at(collided, t)
                opened = is_val and is_open(depth_np[t])
                if not hit and not opened:
                    continue
                got = take(depth, state, t)
                if got is None:
                    if hit:
                        groups[key]["contact"] += 1
                    if opened:
                        groups[key]["open"] += 1
                    continue
                past_d, past_s, here = got
                if hit:
                    groups[key]["contact"] += 1
                    gone, stay = nearest_voxel(past_d, past_s, here, action[t])
                    if gone is not None:
                        groups[key]["gaps"].append(gone[0])
                        groups[key]["ups"].append(gone[1])
                        if gone[0] <= BODY_M:
                            groups[key]["memory"] += 1
                    if stay is not None and stay[0] <= BODY_M:
                        groups[key]["seen"] += 1
                if opened:
                    groups[key]["open"] += 1
                    gone, _stay = nearest_voxel(past_d, past_s, here, fwd)
                    if gone is not None:
                        groups[key]["open_gaps"].append(gone[0])
                        if gone[0] <= BODY_M:
                            groups[key]["open_memory"] += 1
                            groups[key]["open_ups"].append(gone[1])
        if n_ep % 40 == 0:
            log(f"ep {n_ep} {key} contact {groups[key]['contact']} memory {groups[key]['memory']} "
                f"open {groups[key]['open']}")

    def med(xs):
        arr = np.asarray(xs, np.float64)
        return None if arr.size == 0 else round(float(np.median(arr)), 3)

    def finish(cnt):
        return {
            "contact": cnt["contact"],
            "memory": cnt["memory"],
            "seen": cnt["seen"],
            "gap_n": len(cnt["gaps"]),
            "gap_med": med(cnt["gaps"]),
            "up_med": med(cnt["ups"]),
            "open": cnt["open"],
            "open_memory": cnt["open_memory"],
            "open_gap_n": len(cnt["open_gaps"]),
            "open_gap_med": med(cnt["open_gaps"]),
            "open_hit_up_med": med(cnt["open_ups"]),
        }

    result = {"val": finish(groups["val"]), "train": finish(groups["train"])}
    (OUT / "CHECK.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log("val " + json.dumps(result["val"]))
    log("train " + json.dumps(result["train"]))
    log("done")


if __name__ == "__main__":
    main()
