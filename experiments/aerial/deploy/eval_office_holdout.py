#!/usr/bin/env python3
"""Score the frozen office decoder on pairs it did not train on.

Does not train, does not command the chassis, and does not write weights.
"""
from __future__ import annotations

import json
import os

import torch

os.environ.setdefault("AERIAL_DA3_FP16", "1")

import finetune_realsense_pairs as ft
from experiments.aerial.rl.depth_predictor import DepthMinPredictor


def main() -> None:
    names = [f"pair_{i:02d}.npz" for i in range(33, 58)]
    frames = [ft._load(name) for name in names]
    pred = DepthMinPredictor.from_checkpoint(ft.WEIGHTS, device="cuda")
    pred.forward_scale = 1.0
    pred.forward_exponent = 1.0
    model = pred._model
    blob = torch.load("/dev/shm/rtw/m_d_office_dec.pt", map_location="cpu", weights_only=False)
    model.decoder.load_state_dict(blob["decoder"])
    model.decoder.to("cuda").eval()
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    feats = ft._encode_all(model, frames)
    model.encoder.cpu()
    rows, pixels = ft._eval(model, frames, feats)
    worst = None
    compact = []
    for row in rows:
        deltas = []
        model_min = min(v for v in row["cones_model"] if v is not None)
        real_min = min(v for v in row["cones_real"] if v is not None)
        gap = round(model_min - real_min, 2)
        for pred_m, real_m in zip(row["cones_model"], row["cones_real"]):
            deltas.append(round(pred_m - real_m, 2))
            if worst is None or pred_m - real_m > worst[0]:
                worst = (pred_m - real_m, row["file"], pred_m, real_m)
        compact.append({
            "file": row["file"],
            "pose": [round(v, 2) for v in row["pose"]],
            "model": row["cones_model"],
            "real": row["cones_real"],
            "min_gap_m": gap,
        })
    optimistic = [row for row in compact if row["min_gap_m"] > 0.2]
    print(json.dumps({
        "checkpoint_step": blob.get("step"),
        "holdout_frames": len(rows),
        "cone_mae_m": round(ft._cone_score(rows), 3),
        "pixel_mae_m": round(pixels, 3),
        "worst_optimistic_m": None if worst is None else round(worst[0], 3),
        "worst_file": None if worst is None else worst[1],
        "frames_min_long_by_over_0_2": [row["file"] for row in optimistic],
    }, ensure_ascii=False))
    for row in compact:
        if row["min_gap_m"] > 0.15 or row["file"] in {"pair_33.npz", "pair_36.npz", "pair_57.npz"}:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
