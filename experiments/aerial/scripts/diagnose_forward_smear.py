#!/usr/bin/env python3
"""Count open-floor frames whose forward cone is still read near 1.5 m."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("ft", "/data/simworld/run/finetune_office_da3_cone.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402

CKPT = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt")
ANGLE = Path("/data/simworld/run/indoor_office_mix")


def main() -> None:
    train, hold = ft.index_episodes()
    angle_train, angle_hold = ft.index_angle(ANGLE)
    print(
        f"gap train {len(train)} hold {len(hold)} angle train {len(angle_train)} hold {len(angle_hold)}",
        flush=True,
    )
    pred = DepthMinPredictor.from_checkpoint(str(CKPT), device="cuda")
    model = pred._model
    model.eval()
    loader = ft.FrameLoader(None, ANGLE)
    groups = {
        "gap_hold": [r for r in hold if r[3][0] >= 2.2],
        "gap_train": [r for r in train if r[3][0] >= 2.2 and min(r[3][1], r[3][2]) < 1.8],
        "yaw_hold": [r for r in angle_hold if r[3][0] >= 2.2] if False else angle_hold,
    }
    # office_yaw_val hold is the close angled set, not the open floor.
    # Score open frames from the yaw train split's siblings via gap hold, and
    # a sample of gap-like train frames.
    rng = np.random.default_rng(0)
    if len(groups["gap_train"]) > 240:
        pick = rng.choice(len(groups["gap_train"]), size=240, replace=False)
        groups["gap_train"] = [groups["gap_train"][int(i)] for i in pick]
    for name, frames in groups.items():
        da_f = []
        worst = []
        for i in range(0, len(frames), 8):
            chunk = frames[i : i + 8]
            rgb, _c = loader.batch(chunk)
            with torch.no_grad():
                maps = ft.forward_maps(model, rgb).detach().float().cpu().numpy()
            for j, ref in enumerate(chunk):
                da = ft.cones_np(maps[j])
                gt = ref[3]
                da_f.append(float(da[0]))
                if float(da[0]) < 1.8:
                    worst.append((float(da[0]), float(gt[0]), float(da[1]), float(da[2]), ref[1], int(ref[2])))
            print(f"{name} {i + len(chunk)}/{len(frames)} rss {ft.rss_mb():.0f}", flush=True)
        arr = np.asarray(da_f, np.float32)
        if arr.size == 0:
            print(name, "empty", flush=True)
            continue
        print(
            f"{name} n={arr.size} med={np.median(arr):.2f} "
            f"lt1.8={(arr < 1.8).mean():.3f} lt1.6={(arr < 1.6).mean():.3f} "
            f"lt1.3={(arr < 1.3).mean():.3f}",
            flush=True,
        )
        worst.sort()
        for row in worst[:8]:
            print("  worst", row, flush=True)


if __name__ == "__main__":
    main()
