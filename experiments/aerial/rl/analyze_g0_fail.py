#!/usr/bin/env python3
"""Break down the G0 gate on flights already on disk.

Counts why windows miss each denominator, and how the step-8 prior rays
miss the action-integrated label. Does not fly and does not write weights.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments.aerial.rl.eval_occ_topo_g0 import HISTORY, load_model, mode_code
from experiments.aerial.rl.occ_grid import (
    LEFT_ANGLES,
    NX,
    NY,
    NZ,
    RIGHT_ANGLES,
    counted_same_sign,
    integrate_label,
    labels_for_episode,
    occupied_median,
    ray_length,
)
from experiments.aerial.rl.occ_topo_loss import HORIZON


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    ordered = sorted(xs)
    return float(ordered[len(ordered) // 2])


def open_reason(start, label8, pred8, net_dyaw: float) -> str:
    left0 = occupied_median(start, LEFT_ANGLES)
    right0 = occupied_median(start, RIGHT_ANGLES)
    if left0 is None or right0 is None:
        return "start_side_missing"
    if abs(float(left0) - float(right0)) < 1.0:
        return "sides_similar"
    turn_left = float(net_dyaw) > 0.0
    open_is_left = float(left0) > float(right0)
    if open_is_left != turn_left:
        return "yaw_away"
    left8 = occupied_median(label8, LEFT_ANGLES)
    right8 = occupied_median(label8, RIGHT_ANGLES)
    if left8 is None or right8 is None:
        return "label8_side_missing"
    pred_l = occupied_median(pred8, LEFT_ANGLES)
    pred_r = occupied_median(pred8, RIGHT_ANGLES)
    if pred_l is None or pred_r is None:
        return "pred_side_missing"
    open_s, wall_s = (left0, right0) if open_is_left else (right0, left0)
    open_l, wall_l = (left8, right8) if open_is_left else (right8, left8)
    open_p, wall_p = (pred_l, pred_r) if open_is_left else (pred_r, pred_l)
    open_ok = counted_same_sign(open_p, open_l, open_s, label_stop="occupied")
    wall_ok = counted_same_sign(wall_p, wall_l, wall_s, label_stop="occupied")
    if open_ok is None or wall_ok is None:
        return "change_under_1m"
    return "pass" if bool(open_ok and wall_ok) else "fail"


def forward_case(start, label8, pred8) -> Tuple[str, Optional[Dict[str, Any]]]:
    start_m, start_stop = ray_length(start, 0.0)
    label_m, label_stop = ray_length(label8, 0.0)
    pred_m, pred_stop = ray_length(pred8, 0.0)
    if start_stop != "occupied":
        return "start_" + start_stop, None
    if label_stop != "occupied":
        return "label8_" + label_stop, None
    if float(label_m) > float(start_m) - 1.0:
        return "label_not_shorter", None
    ok = counted_same_sign(pred_m, label_m, start_m, label_stop="occupied")
    detail = {
        "pass": bool(ok),
        "start_m": float(start_m),
        "label_m": float(label_m),
        "pred_m": float(pred_m),
        "pred_stop": pred_stop,
        "signed_m": float(pred_m) - float(label_m),
        "label_delta_m": float(label_m) - float(start_m),
        "pred_delta_m": float(pred_m) - float(start_m),
    }
    return ("pass" if ok else "fail"), detail


def ray_pairs(label, pred) -> List[Dict[str, Any]]:
    rows = []
    for theta in (0.0,) + LEFT_ANGLES + RIGHT_ANGLES:
        label_m, label_stop = ray_length(label, float(theta))
        if label_stop != "occupied":
            continue
        pred_m, pred_stop = ray_length(pred, float(theta))
        rows.append(
            {
                "signed_m": float(pred_m) - float(label_m),
                "abs_m": abs(float(pred_m) - float(label_m)),
                "pred_stop": pred_stop,
                "label_m": float(label_m),
                "pred_m": float(pred_m),
            }
        )
    return rows


def analyze_episode(path: Path, model, decoder, stride: int) -> Dict[str, Any]:
    from experiments.aerial.rl.train_occ_topo import decode_rgb

    with np.load(path) as z:
        blob = np.asarray(z["jpeg"])
        offsets = np.asarray(z["jpeg_offsets"])
        proprio = np.asarray(z["proprio"], dtype=np.float32)
        action = np.asarray(z["action"], dtype=np.float64)
        depth = np.asarray(z["depth"])
        pose = np.asarray(z["label_state"])
    grids = labels_for_episode(depth, pose, stride=stride)
    length = int(proprio.shape[0])
    h = model.rssm.initial_h(1, model.device, model.torch_dtype)
    last = length - HORIZON - 1
    open_reasons: List[str] = []
    forward_reasons: List[str] = []
    forward_rows: List[Dict[str, Any]] = []
    rays: List[Dict[str, Any]] = []
    anchors = 0
    with torch.no_grad():
        for t in range(length):
            rgb = torch.from_numpy(decode_rgb(blob, offsets, t)).unsqueeze(0).to(model.device)
            prop = torch.from_numpy(proprio[t : t + 1]).to(model.device, model.torch_dtype)
            embed = model._embed(rgb, prop)
            z = mode_code(model.rssm.post_probs(h, embed))
            if HISTORY - 1 <= t <= last and (t - (HISTORY - 1)) % HORIZON == 0:
                anchors += 1
                acts = action[t : t + HORIZON]
                label8 = integrate_label(grids[t], acts)
                h_k, z_k = h, z
                for k in range(HORIZON):
                    a = torch.from_numpy(acts[k].astype(np.float32)).reshape(1, 4).to(
                        model.device, model.torch_dtype
                    )
                    h_k = model.rssm.advance_h(h_k, z_k, a)
                    z_k = mode_code(model.rssm.prior_probs(h_k))
                logits = decoder(torch.cat([h_k, z_k], dim=-1))
                pred = logits.argmax(dim=-1).reshape(NZ, NY, NX).cpu().numpy().astype(np.uint8)
                net = float(acts[:, 3].sum())
                open_reasons.append(open_reason(grids[t], label8, pred, net))
                name, detail = forward_case(grids[t], label8, pred)
                forward_reasons.append(name)
                if detail is not None:
                    forward_rows.append(detail)
                    rays.extend(ray_pairs(label8, pred))
            if t + 1 < length:
                a = torch.from_numpy(action[t].astype(np.float32)).reshape(1, 4).to(
                    model.device, model.torch_dtype
                )
                h = model.rssm.advance_h(h, z, a)
    return {
        "frames": length,
        "anchors": anchors,
        "open": open_reasons,
        "forward": forward_reasons,
        "forward_rows": forward_rows,
        "rays": rays,
    }


def _pack(counter: Counter) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> int:
    holdout = Path("experiments/aerial/rl/artifacts/occ_topo_g0_20261010/holdout_gate")
    ckpt = Path("experiments/aerial/rl/artifacts/occ_topo_g0_20261012/occ_topo_wm.pt")
    model, decoder = load_model(ckpt, "cuda")
    open_c: Counter = Counter()
    fwd_c: Counter = Counter()
    frames = []
    anchors = 0
    rows: List[Dict[str, Any]] = []
    rays: List[Dict[str, Any]] = []
    for path in sorted(holdout.glob("route*.npz")):
        part = analyze_episode(path, model, decoder, 2)
        frames.append(part["frames"])
        anchors += int(part["anchors"])
        open_c.update(part["open"])
        fwd_c.update(part["forward"])
        rows.extend(part["forward_rows"])
        rays.extend(part["rays"])
        print("done", path.name, "frames", part["frames"], "anchors", part["anchors"], flush=True)
    fail = [r for r in rows if not r["pass"]]
    ok = [r for r in rows if r["pass"]]
    stop_c: Counter = Counter(r["pred_stop"] for r in rays)
    signed = [r["signed_m"] for r in rays]
    report = {
        "episodes": len(frames),
        "frames_under_24": sum(1 for n in frames if n < 24),
        "anchors": anchors,
        "open_reasons": _pack(open_c),
        "forward_reasons": _pack(fwd_c),
        "forward_counted": len(rows),
        "forward_pass": len(ok),
        "forward_fail_pred_stop": _pack(Counter(r["pred_stop"] for r in fail)),
        "forward_pass_pred_stop": _pack(Counter(r["pred_stop"] for r in ok)),
        "forward_fail_signed_median_m": _median([r["signed_m"] for r in fail]),
        "forward_pass_signed_median_m": _median([r["signed_m"] for r in ok]),
        "forward_label_delta_median_m": _median([r["label_delta_m"] for r in rows]),
        "forward_pred_delta_median_m": _median([r["pred_delta_m"] for r in rows]),
        "ray_count": len(rays),
        "ray_abs_median_m": _median([r["abs_m"] for r in rays]),
        "ray_signed_median_m": _median(signed),
        "ray_pred_longer": sum(1 for s in signed if s > 0),
        "ray_pred_shorter": sum(1 for s in signed if s < 0),
        "ray_pred_stop": _pack(stop_c),
        "ray_abs_median_by_stop": {
            name: _median([r["abs_m"] for r in rays if r["pred_stop"] == name])
            for name in sorted(stop_c)
        },
    }
    out = Path("experiments/aerial/rl/artifacts/occ_topo_g0_20261012/G0_FAIL.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
