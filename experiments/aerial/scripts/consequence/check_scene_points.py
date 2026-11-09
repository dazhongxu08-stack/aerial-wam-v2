#!/usr/bin/env python3
"""Ask the flown step whether it meets a remembered point. No grid, no survival.

Past depth stays at the range where it was seen. The current pose puts those
points in the body. Points already inside the body are dropped. The step meets
a remaining point when the horizontal distance to the body path is within
0.4 m. Open frames ask the same of a forward 0.8 m step.
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
DATA = ART / "consequence_data_20261004"
OUT = ART / "scenepoints_20261104_body"
VAL = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
PAST = 16
BODY_M = 0.4


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "check.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def episodes():
    review = json.loads((DATA / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    rows = []
    for e in review["episodes"]:
        if not e.get("keep") or not e.get("file"):
            continue
        rid = str(e["route_id"])
        if not rid.startswith("od_train_"):
            continue
        path = DATA / "episodes" / e["file"]
        if path.is_file():
            rows.append((rid in VAL, path))
    return rows


def is_open(depth: np.ndarray) -> bool:
    h, w = depth.shape
    band = depth[h // 4: h - h // 4]

    def near(block: np.ndarray) -> float:
        vals = block[np.isfinite(block) & (block > 0.05)]
        return float(vals.min()) if vals.size else 100.0

    return min(
        near(band[:, : w // 3]),
        near(band[:, w // 3: w - w // 3]),
        near(band[:, w - w // 3:]),
    ) >= 8.0


def contact_at(collided: np.ndarray, t: int) -> bool:
    return bool(t + 1 < len(collided) and collided[t + 1])


def empty():
    return {"contact": 0, "memory": 0, "seen": 0, "gaps": [], "open": 0, "open_memory": 0, "open_gaps": []}


def take(depth, state, t, dev):
    s0 = max(0, t - PAST)
    if t - s0 < 1:
        return None
    return depth[s0:t], state[s0:t], state[t: t + 1]


def main() -> None:
    from experiments.aerial.rl.consequence_model import load_episode
    from experiments.aerial.rl.scene_points import nearest
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
                got = take(depth, state, t, dev)
                if got is None:
                    if hit:
                        groups[key]["contact"] += 1
                    if opened:
                        groups[key]["open"] += 1
                    continue
                past_d, past_s, here = got
                if hit:
                    groups[key]["contact"] += 1
                    gone, stay = nearest(past_d, past_s, here, action[t])
                    if gone is not None:
                        groups[key]["gaps"].append(gone)
                        if gone <= BODY_M:
                            groups[key]["memory"] += 1
                    if stay is not None and stay <= BODY_M:
                        groups[key]["seen"] += 1
                if opened:
                    groups[key]["open"] += 1
                    gone, _stay = nearest(past_d, past_s, here, fwd)
                    if gone is not None:
                        groups[key]["open_gaps"].append(gone)
                        if gone <= BODY_M:
                            groups[key]["open_memory"] += 1
        if n_ep % 40 == 0:
            log(f"ep {n_ep} {key} contact {groups[key]['contact']} memory {groups[key]['memory']} "
                f"open {groups[key]['open']}")

    def finish(cnt):
        gaps = np.asarray(cnt["gaps"], np.float64)
        opens = np.asarray(cnt["open_gaps"], np.float64)
        return {
            "contact": cnt["contact"],
            "memory": cnt["memory"],
            "seen": cnt["seen"],
            "gap_n": int(gaps.size),
            "gap_med": None if gaps.size == 0 else round(float(np.median(gaps)), 3),
            "open": cnt["open"],
            "open_memory": cnt["open_memory"],
            "open_gap_n": int(opens.size),
            "open_gap_med": None if opens.size == 0 else round(float(np.median(opens)), 3),
        }

    result = {"val": finish(groups["val"]), "train": finish(groups["train"])}
    (OUT / "CHECK.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log("val " + json.dumps(result["val"]))
    log("train " + json.dumps(result["train"]))
    log("done")


if __name__ == "__main__":
    main()
