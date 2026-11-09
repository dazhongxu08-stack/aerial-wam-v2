#!/usr/bin/env python3
"""Offline read of the trained scene. Does not fly and does not score survival.

Two counts, on flights already collected:

- A step that hits, where a past surface has left the camera and lies in the
  body sweep of the flown action. Does that cell still hold the surface, and
  does the depth read back within 1 m?
- An open frame, all three depth thirds at least 8 m. Does the forward 0.8 m
  step sweep a cell the scene calls nearer than 8 m?
"""
from __future__ import annotations

import json
import os
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
OUT = Path(os.environ.get("SCENE_OUT", str(ART / "scenemap_20261104_cont")))
CKPT = Path(os.environ.get("SCENE_CKPT", str(OUT / "map_best.pt")))
VAL = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
PAST = 16
DEPTH_OK_M = 1.0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
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

    thirds = (
        near(band[:, : w // 3]),
        near(band[:, w // 3: w - w // 3]),
        near(band[:, w - w // 3:]),
    )
    return min(thirds) >= 8.0


def contact_at(collided: np.ndarray, t: int) -> bool:
    """``action[t]`` hits on arrival, flagged at ``collided[t + 1]``."""
    return bool(t + 1 < len(collided) and collided[t + 1])


def sweep_ids(action: np.ndarray) -> set:
    from experiments.aerial.rl.body_surround import N, step_cells
    return {i * N + j for i, j in step_cells(np.zeros(7), action)}


def reach_m(action: np.ndarray) -> float:
    return float(np.hypot(action[0], action[1]) + 0.4)


def read_map(enc, feat, depth, state, t, dev):
    """Past frames placed at pose ``t``. Current frame is not written."""
    from experiments.aerial.rl.latent_map import block_body, fuse_nearest, outside_view
    s0 = max(0, t - PAST)
    if t - s0 < 1:
        return None
    past_d = depth[s0:t]
    past_s = state[s0:t]
    here = state[t: t + 1].expand(past_d.shape[0], -1)
    cell, rng, valid, dist, fwd, left, _dz = block_body(past_d, past_s, here)
    gone_pt = outside_view(fwd, left)
    p = past_d.shape[0]
    c = feat.shape[1]
    past_f = feat[s0:t].permute(1, 0, 2, 3).reshape(1, c, p * 14, 14)
    pc = cell.reshape(1, p * 14, 14)
    pr = rng.reshape(1, p * 14, 14)
    pv = valid.reshape(1, p * 14, 14)
    target = dist.clamp(min=0.05).log().reshape(1, 1, p * 14, 14)
    flag = gone_pt.reshape(1, 1, p * 14, 14).to(past_f.dtype)
    grid, horiz = fuse_nearest(past_f, pc, pr, pv)
    true_log, _ = fuse_nearest(target, pc, pr, pv)
    gone, _ = fuse_nearest(flag, pc, pr, pv)
    only = pv & gone_pt.reshape(1, p * 14, 14)
    oracle_log, oracle_dist = fuse_nearest(target, pc, pr, only)
    pred = enc.depth_of(grid)
    return {
        "pred": pred[0].clamp(-3.0, 3.0).exp(),
        "true": true_log[0, 0].clamp(min=-3.0).exp(),
        "oracle": oracle_log[0, 0].clamp(min=-3.0).exp(),
        "gone": gone[0, 0] > 0.5,
        "written": horiz[0] < 1e8,
        "horiz": horiz[0],
        "oracle_h": oracle_dist[0],
    }


def empty_count():
    return {
        "contact": 0,
        "entered": 0,
        "held": 0,
        "depth_ok": 0,
        "covers": 0,
        "err_sum": 0.0,
        "err_n": 0,
        "open": 0,
        "phantom8": 0,
        "phantom_step": 0,
    }


def add_hit(cnt, scene, sweep, reach):
    cnt["contact"] += 1
    if scene is None:
        return
    ids = [i for i in sweep if i < scene["written"].numel()]
    if not ids:
        return
    idx = torch.tensor(ids, device=scene["written"].device)
    gone = scene["gone"].reshape(-1)[idx]
    horiz = scene["oracle_h"].reshape(-1)[idx]
    pred = scene["pred"].reshape(-1)[idx]
    true = scene["oracle"].reshape(-1)[idx]
    entered = (horiz < 1e8) & (horiz <= reach + 0.5)
    if not bool(entered.any()):
        return
    cnt["entered"] += 1
    held = entered & gone
    err = (pred - true).abs()
    if bool(held.any()):
        cnt["held"] += 1
        cnt["err_sum"] += float(err[held].sum())
        cnt["err_n"] += int(held.sum())
        if bool((held & (err <= DEPTH_OK_M)).any()):
            cnt["depth_ok"] += 1
        if bool((held & (pred <= reach + 0.25)).any()):
            cnt["covers"] += 1


def add_open(cnt, scene, sweep):
    cnt["open"] += 1
    if scene is None:
        return
    ids = [i for i in sweep if i < scene["written"].numel()]
    if not ids:
        return
    idx = torch.tensor(ids, device=scene["written"].device)
    written = scene["written"].reshape(-1)[idx]
    pred = scene["pred"].reshape(-1)[idx]
    if bool((written & (pred < 8.0)).any()):
        cnt["phantom8"] += 1
    if bool((written & (pred < 1.2)).any()):
        cnt["phantom_step"] += 1


def finish(cnt):
    out = dict(cnt)
    out["err_m"] = None if cnt["err_n"] == 0 else round(cnt["err_sum"] / cnt["err_n"], 3)
    for k in ("err_sum",):
        out.pop(k, None)
    return out


def main() -> None:
    from experiments.aerial.rl.consequence_model import load_episode
    from experiments.aerial.rl.latent_map import MapEncoder, _depth_units
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"start device {dev} ckpt {CKPT.name}")
    raw = torch.load(CKPT, map_location=dev)
    enc = MapEncoder(int(raw.get("dim", 16))).to(dev)
    enc.load_state_dict(raw["model"])
    enc.eval()
    fwd = sweep_ids(np.array([0.8, 0.0, 0.0, 0.0], np.float64))
    groups = {"val": empty_count(), "train": empty_count()}
    rows = episodes()
    log(f"episodes {len(rows)}")
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
        with torch.no_grad():
            feat = enc.features(_depth_units(depth))
        key = "val" if is_val else "train"
        queries = []
        for t in range(1, n - 1):
            hit = contact_at(collided, t)
            opened = is_val and is_open(depth_np[t])
            if hit or opened:
                queries.append((t, hit, opened))
        with torch.no_grad():
            for t, hit, opened in queries:
                scene = read_map(enc, feat, depth, state, t, dev)
                if hit and t < len(action):
                    action_t = np.asarray(action[t], np.float64)
                    add_hit(groups[key], scene, sweep_ids(action_t), reach_m(action_t))
                if opened:
                    add_open(groups[key], scene, fwd)
        if n_ep % 40 == 0:
            log(f"ep {n_ep} {key} contact {groups[key]['contact']} entered {groups[key]['entered']} "
                f"open {groups[key]['open']}")
    result = {"ckpt": CKPT.name, "val": finish(groups["val"]), "train": finish(groups["train"])}
    (OUT / "CHECK.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log("val " + json.dumps(result["val"]))
    log("train " + json.dumps(result["train"]))
    log("done")


if __name__ == "__main__":
    main()
