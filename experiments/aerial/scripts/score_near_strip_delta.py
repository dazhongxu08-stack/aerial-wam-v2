#!/usr/bin/env python3
"""Compare the safe-window tail with and without the thin-strip correction."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("fit", "/data/simworld/run/fit_near_strip.py")
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)
spec2 = importlib.util.spec_from_file_location("sc", "/data/simworld/run/score_near_strip.py")
sc = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(sc)

from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
NAMES = (
    "episode_00204.npz",
    "episode_00390.npz",
    "episode_00450.npz",
    "episode_00402.npz",
    "episode_00102.npz",
    "episode_00096.npz",
    "episode_00162.npz",
    "episode_00006.npz",
    "episode_20030.npz",
)


def main() -> None:
    pred = DepthMinPredictor.from_checkpoint(
        str(ART / "da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt"), device="cuda")
    model = pred._model
    model.eval()
    packed = []
    for fname in NAMES:
        path = fit.MIX / "episodes" / fname
        raw = fit.predict_episode(model, path)
        ep = fit.load_episode(path)
        raw = raw[: len(ep["state"])]
        packed.append((fname, ep, raw))
    del pred, model
    torch.cuda.empty_cache()
    cons = fit.load_checkpoint(fit.CONS, "cpu")
    cfg = sc.cfg
    for fname, ep, raw in packed:
        corr = sc.apply_ep(raw)
        _, p0 = sc.window_p(cons, ep, raw)
        _, p1 = sc.window_p(cons, ep, corr)
        d = p1 - p0
        print(
            fname,
            "max0", f"{p0.max():.3f}",
            "max1", f"{p1.max():.3f}",
            "mean_d", f"{d.mean():+.3f}",
            "max_d", f"{d.max():+.3f}",
            flush=True,
        )
        if fname == "episode_20030.npz":
            print(" p0", " ".join(f"{x:.3f}" for x in p0), flush=True)
            print(" p1", " ".join(f"{x:.3f}" for x in p1), flush=True)
            h, w = raw[4].shape
            band = raw[4][h // 4 : h - h // 4]
            cols = fit.thin_columns(raw[4], cfg["lo"], cfg["hi"], cfg["max_run"])
            print(" thin_cols", int(cols.sum()), "band_min", f"{band.min():.2f}", flush=True)


if __name__ == "__main__":
    main()
