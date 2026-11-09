#!/usr/bin/env python3
"""Teach the office consequence head to avoid, instead of handing that to rules.

The chassis camera was fitted from a RealSense floor plane: 34.2 cm up and
14.5° down. The simulator camera is already that pose (34 cm, -14.3°, 40 cm
ahead). These episodes were collected there. A forward step into a short
obstacle is labeled a hit even when the logged trajectory yawed away, and the
yaw is labeled safe. The car weights are not touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from office_bands import short_clearance  # noqa: E402
from experiments.aerial.rl.consequence_model import (  # noqa: E402
    consequence_loss,
    episode_windows,
    load_checkpoint,
    load_episode,
    mirror_batch,
    save_checkpoint,
    survival,
)

MIX = Path("/data/simworld/run/indoor_office_mix")
ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "indoor_office_mix_far_20261103/c_s218/consequence_best.pt"
OUT = ART / "indoor_office_avoid_20261103/c_s218"
YAW = 0.3141592653589793
STRIP_M = 1.50
STEPS = 120


def strip_center(depth56: np.ndarray) -> float:
    low = short_clearance(np.asarray(depth56, np.float32))
    center = low[1] if low is not None else None
    return 30.0 if center is None else float(center)


def stack(paths, horizon=10):
    parts = {}
    strips = []
    for path in paths:
        ep = load_episode(path)
        if "da3_depth56" not in ep or len(ep["state"]) < 2:
            continue
        ep.pop("env", None)
        ep.pop("env_flip", None)
        windows = episode_windows(ep, horizon)
        if not len(windows["depth"]):
            continue
        gt = ep["depth56"]
        for t in range(len(ep["state"]) - 1):
            strips.append(strip_center(gt[t]))
        for key, value in windows.items():
            parts.setdefault(key, []).append(value)
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data["strip"] = np.asarray(strips, np.float32)
    if len(data["strip"]) != len(data["depth"]):
        raise RuntimeError(f"strip {len(data['strip'])} != windows {len(data['depth'])}")
    return data


def hazard0(model, batch):
    out = model(batch["depth"], batch["goal"], batch["prev_action"], batch["actions"])
    alive = batch["alive"][:, 0] > 0
    prob = 1.0 - survival(out["hazard_logit"])[:, 0]
    return prob, alive


def counterfactual(batch, dx: float, dyaw: float, hit: float):
    alt = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    alt["actions"] = batch["actions"].clone()
    alt["actions"][:, 0, :] = 0
    alt["actions"][:, 0, 0] = dx
    alt["actions"][:, 0, 3] = dyaw
    alt["hit"] = batch["hit"].clone()
    alt["hit"][:, 0] = hit
    return alt


def main() -> int:
    review = json.loads((MIX / "AUTO_REVIEW.json").read_text())
    route_of = {e["file"]: e["route_id"] for e in review["episodes"] if e.get("keep") and e.get("file")}
    eps = sorted(
        p for p in (MIX / "episodes").glob("episode_*.npz")
        if p.name.count(".") == 1 and p.name in route_of
    )
    tr_paths = [p for p in eps if route_of[p.name] not in ("office_val", "office_yaw_val")]
    va_paths = [p for p in eps if route_of[p.name] == "office_val"]
    print(f"paths tr {len(tr_paths)} va {len(va_paths)}", flush=True)
    tr = stack(tr_paths)
    va = stack(va_paths)
    close = tr["strip"] < STRIP_M
    print(
        f"windows tr {len(tr['depth'])} strip<{STRIP_M} {int(close.sum())} va {len(va['depth'])}",
        flush=True,
    )
    dev = torch.device("cuda")
    model = load_checkpoint(str(INIT), "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal")
    rng = np.random.default_rng(218)

    def take(data, idx):
        picked = {k: torch.as_tensor(data[k][idx], dtype=torch.float32, device=dev) for k in keys}
        picked["strip"] = torch.as_tensor(data["strip"][idx], dtype=torch.float32, device=dev)
        return picked

    @torch.no_grad()
    def rank(data):
        """On close-strip frames, forward hazard should exceed yaw hazard."""
        model.eval()
        sel = np.flatnonzero(data["strip"] < STRIP_M)
        if sel.size == 0:
            model.train()
            return 0.0, 0.0
        if sel.size > 512:
            sel = sel[:512]
        batch = take(data, sel)
        fwd = hazard0(model, counterfactual(batch, 0.8, 0.0, 1.0))[0]
        yaw = hazard0(model, counterfactual(batch, 0.0, YAW, 0.0))[0]
        model.train()
        return float(fwd.mean()), float(yaw.mean())

    OUT.mkdir(parents=True, exist_ok=True)
    best = None
    fwd_p, yaw_p = rank(va)
    print(f"step 0 fwd_hazard {fwd_p:.3f} yaw_hazard {yaw_p:.3f}", flush=True)
    model.train()
    n = len(tr["depth"])
    for step in range(1, STEPS + 1):
        idx = rng.choice(n, size=128)
        batch = take(tr, idx)
        if rng.random() < 0.5:
            flipped = mirror_batch(batch)
            flipped["strip"] = batch["strip"]
            batch = flipped
        out = model(batch["depth"], batch["goal"], batch["prev_action"], batch["actions"])
        loss, _parts = consequence_loss(out, batch)
        near = batch["strip"] < STRIP_M
        if bool(near.any()):
            fwd_h, _ = hazard0(model, counterfactual(batch, 0.8, 0.0, 1.0))
            yaw_h, _ = hazard0(model, counterfactual(batch, 0.0, YAW, 0.0))
            loss = loss + 4.0 * (1.0 - fwd_h[near]).mean() + 4.0 * yaw_h[near].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == STEPS:
            fwd_p, yaw_p = rank(va)
            gap = fwd_p - yaw_p
            print(f"step {step} fwd_hazard {fwd_p:.3f} yaw_hazard {yaw_p:.3f} gap {gap:.3f}", flush=True)
            save_checkpoint(OUT / "consequence_last.pt", model, {"step": step, "fwd": fwd_p, "yaw": yaw_p})
            if best is None or gap > best[0]:
                best = (gap, step, fwd_p, yaw_p)
                save_checkpoint(OUT / "consequence_best.pt", model, {"step": step, "fwd": fwd_p, "yaw": yaw_p})
                print("SAVE", step, flush=True)
    print("BEST", best, "OUT", OUT, flush=True)
    return 0 if best is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
