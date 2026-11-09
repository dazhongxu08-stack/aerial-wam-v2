#!/usr/bin/env python3
"""Score stored frames nearest the live gap and west starts."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("ft", "/data/simworld/run/finetune_office_da3_cone.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402

CKPT = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt"
ROOTS = (
    Path("/data/simworld/run/indoor_office_gap"),
    Path("/data/simworld/run/indoor_office_mix"),
)
TARGETS = (("gap", -160.0, -200.0, 0.0), ("west", -520.0, -250.0, 0.0), ("yaw", -170.0, -200.0, 18.0))


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def main() -> None:
    hits = []
    seen = set()
    for root in ROOTS:
        review = json.loads((root / "AUTO_REVIEW.json").read_text())
        for row in review["episodes"]:
            if not row.get("file"):
                continue
            path = root / "episodes" / Path(row["file"]).name
            if not path.is_file() or path.name in seen:
                continue
            seen.add(path.name)
            with np.load(path) as z:
                state = np.asarray(z["state"], np.float64)
                depth = np.asarray(z["depth56"], np.float32)
            for i in range(len(state)):
                x = float(state[i, 0]) * 100.0
                y = float(state[i, 1]) * 100.0
                yaw = math.degrees(float(state[i, 6]))
                gt = ft.cones_np(depth[i])
                for name, tx, ty, tyaw in TARGETS:
                    err = abs(x - tx) + abs(y - ty) + 4.0 * abs(wrap(yaw - tyaw))
                    if err < 80.0:
                        hits.append((err, name, path, i, x, y, yaw, float(gt[0]), float(gt[1]), float(gt[2])))
    hits.sort()
    picked = []
    used = set()
    for name, *_rest in TARGETS:
        n = 0
        for row in hits:
            if row[1] != name or (row[2], row[3]) in used:
                continue
            picked.append(row)
            used.add((row[2], row[3]))
            n += 1
            if n >= 6:
                break
    print(f"near poses {len(picked)}", flush=True)
    pred = DepthMinPredictor.from_checkpoint(CKPT, device="cuda")
    model = pred._model
    model.eval()
    loader_cache = {}
    for err, name, path, i, x, y, yaw, gf, gl, gr in picked:
        root = path.parent.parent
        loader = loader_cache.get(str(root))
        if loader is None:
            loader = ft.FrameLoader(None, root if root.name != "indoor_office_gap" else None)
            loader_cache[str(root)] = loader
        kind = "e" if root.name == "indoor_office_gap" else "a"
        # FrameLoader uses DATA for kind e and angle dir for kind a.
        if kind == "e":
            loader._angle = None
        rgb, _c = loader.batch([(kind, path.name, i, (0, 0, 0, 0, 0, 0))])
        with torch.no_grad():
            da = ft.cones_np(ft.forward_maps(model, rgb).detach().float().cpu().numpy()[0])
        print(
            f"{name} err {err:.0f} {path.parent.parent.name}/{path.name} i={i} "
            f"xy {x:.0f},{y:.0f} yaw {yaw:.0f} gt {gf:.2f},{gl:.2f},{gr:.2f} "
            f"da {float(da[0]):.2f},{float(da[1]):.2f},{float(da[2]):.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
