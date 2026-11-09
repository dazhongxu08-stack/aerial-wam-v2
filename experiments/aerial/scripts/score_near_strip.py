#!/usr/bin/env python3
"""Score the saved thin-strip correction: safe tail, gap frames, candidate actions."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("fit", "/data/simworld/run/fit_near_strip.py")
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)

from experiments.aerial.rl.consequence_model import (  # noqa: E402
    episode_windows,
    load_checkpoint,
    survival,
)
from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
cfg = json.loads((ART / "near_strip_20261102/near_strip.json").read_text())["best"]


def apply_ep(raw: np.ndarray) -> np.ndarray:
    return fit.correct_thin(raw, cfg["alpha"], cfg["lo"], cfg["hi"], cfg["max_run"])


def window_p(cons, ep, depth):
    ep = dict(ep)
    ep["da3_depth56"] = depth
    w = episode_windows(ep, 10)
    if len(w["depth"]) == 0:
        return w, np.zeros(0)
    with torch.no_grad():
        b = {k: torch.tensor(w[k], dtype=torch.float32) for k in ("depth", "goal", "prev_action", "actions", "alive")}
        out = cons(b["depth"], b["goal"], b["prev_action"], b["actions"])
        last = (b["alive"].sum(-1) - 1).clamp(min=0).long()
        p = (1.0 - survival(out["hazard_logit"]).gather(1, last[:, None]).squeeze(1)).numpy()
    return w, p


def cand_p(cons, depth_map, goal_f, prev, dx, dyaw):
    a = np.zeros((10, 4), np.float32)
    a[:, 0] = dx
    a[:, 3] = dyaw
    return fit._full_p(cons, fit.depth_input(depth_map), goal_f, prev, a)


def main() -> None:
    pred = DepthMinPredictor.from_checkpoint(
        str(ART / "da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt"), device="cuda")
    model = pred._model
    model.eval()
    review = json.loads((fit.MIX / "AUTO_REVIEW.json").read_text())
    packed = []
    for row in review["episodes"]:
        if row.get("route_id") not in ("office_val", "office_yaw_val"):
            continue
        fname = Path(row["file"]).name
        path = fit.MIX / "episodes" / fname
        raw = fit.predict_episode(model, path)
        ep = fit.load_episode(path)
        raw = raw[: len(ep["state"])]
        packed.append((fname, row["route_id"], ep, apply_ep(raw)))
        print("cached", fname, "rss", f"{fit.ft.rss_mb():.0f}", flush=True)
    del pred, model
    torch.cuda.empty_cache()
    cons = load_checkpoint(fit.CONS, "cpu")
    safe_p = []
    worst = []
    for fname, route, ep, corr in packed:
        w, p = window_p(cons, ep, corr)
        y = w["hit"].sum(-1) > 0 if len(p) else np.zeros(0, dtype=bool)
        if route == "office_val":
            for i, (pi, yi) in enumerate(zip(p, y)):
                if not yi:
                    safe_p.append(float(pi))
                    worst.append((float(pi), fname, i))
        if fname in ("episode_20030.npz", "episode_20005.npz"):
            print(fname, " ".join(f"{float(x):.3f}" for x in p), flush=True)
            if fname == "episode_20030.npz":
                d4 = corr[4]
                print(
                    "cands",
                    "straight", f"{cand_p(cons, d4, w['goal'][4], w['prev_action'][4], 0.8, 0.0):.3f}",
                    "right", f"{cand_p(cons, d4, w['goal'][4], w['prev_action'][4], 0.8, 0.088):.3f}",
                    "left", f"{cand_p(cons, d4, w['goal'][4], w['prev_action'][4], 0.8, -0.088):.3f}",
                    "creep", f"{cand_p(cons, d4, w['goal'][4], w['prev_action'][4], 0.4, 0.0):.3f}",
                    "yaw", f"{cand_p(cons, d4, w['goal'][4], w['prev_action'][4], 0.0, 0.314):.3f}",
                    flush=True,
                )
        print("scored", fname, "rss", f"{fit.ft.rss_mb():.0f}", flush=True)
    s = np.asarray(safe_p)
    worst.sort(reverse=True)
    print(
        f"safe n={len(s)} mean={s.mean():.3f} p50={np.median(s):.3f} "
        f"p90={np.quantile(s, 0.9):.3f} p99={np.quantile(s, 0.99):.3f} max={s.max():.3f} "
        f"gt05={(s > 0.05).mean():.3f} gt15={(s > 0.15).mean():.3f}",
        flush=True,
    )
    print("worst", worst[:8], flush=True)


if __name__ == "__main__":
    main()
