#!/usr/bin/env python3
"""Directional obstacle_cost vs GT cone clearances (geometry falsification).

For near-field frames: score fwd / left / right / hover / climb costs from the
WM obstacle head and compare ranks to GT ``cone_clearances``. Pass when the
head prefers the emptier side on left-near / right-near / fwd-near slices.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


_ACTIONS = {
    "fwd": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    "left": np.array([0.0, 0.9, 0.0, 0.0], dtype=np.float64),
    "right": np.array([0.0, -0.9, 0.0, 0.0], dtype=np.float64),
    "climb": np.array([0.0, 0.0, 0.9, 0.0], dtype=np.float64),
    "hover": np.zeros(4, dtype=np.float64),
}


def _score_row(dyn: Any, feat: np.ndarray) -> Dict[str, float]:
    return {k: float(dyn.predict_obstacle_cost(feat, a)) for k, a in _ACTIONS.items()}


def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(description="Obstacle cost vs GT cone rank")
    p.add_argument(
        "--dataset",
        default="experiments/aerial/rl/artifacts/dataset_v0_three_zone_near_20260823fg",
    )
    p.add_argument(
        "--wm-ckpt",
        default=(
            "experiments/aerial/rl/artifacts/"
            "wm_ckpt_obstacle_cost_gatefix_20260922_161641/wm_obs.pt"
        ),
    )
    p.add_argument("--config", default="configs/aerial_rl.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--max-forward-depth-m", type=float, default=12.0)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--center-frac", type=float, default=0.5)
    p.add_argument("--margin-m", type=float, default=1.0)
    p.add_argument("--out", default="artifacts/wam_obstacle_cost_cone_rank.json")
    args = p.parse_args()

    from experiments.aerial.rl import dataset as ds
    from experiments.aerial.rl.depth_geometry import cone_clearances, forward_min_depth
    from experiments.aerial.rl.latent_encode_probe import encode_window_packed
    from experiments.aerial.rl.reward import RewardConfig
    from experiments.aerial.rl.train_rl import load_torch_dynamics

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with cfg_path.open() as f:
        config: Dict[str, Any] = yaml.safe_load(f) or {}

    dataset = Path(args.dataset).expanduser()
    if not dataset.is_absolute():
        dataset = root / dataset
    wm_ckpt = Path(args.wm_ckpt).expanduser()
    if not wm_ckpt.is_absolute():
        wm_ckpt = root / wm_ckpt
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reward_cfg = (
        RewardConfig(**(config.get("reward", {}) or {}))
        if config.get("reward")
        else RewardConfig()
    )
    wm_cfg = config.get("world_model", {}) or {}
    dynamics, _ = load_torch_dynamics(
        wm_cfg,
        wm_ckpt,
        device=str(args.device),
        success_dist_m=float(reward_cfg.success_dist_m),
    )
    if not bool(getattr(dynamics, "obstacle_cost_trained", False)):
        payload = {
            "error": "obstacle_cost_not_trained",
            "wm_ckpt": str(wm_ckpt),
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return 2

    episodes = ds.load_dataset(dataset, skip_quarantined=True)
    margin = float(args.margin_m)
    rows: List[Dict[str, Any]] = []
    checks = {
        "fwd_empty_fwd_cheaper_than_hover": [],
        "fwd_near_side_or_climb_cheaper_than_fwd": [],
        "left_near_fwd_cheaper_than_left": [],
        "right_near_fwd_cheaper_than_right": [],
    }

    for ep_i, ep in enumerate(episodes):
        if len(rows) >= int(args.max_samples):
            break
        if not ep:
            continue
        for t_i in range(0, len(ep), max(1, int(args.stride))):
            if len(rows) >= int(args.max_samples):
                break
            obs = ep[t_i].obs
            depth = getattr(obs, "depth", None)
            if depth is None:
                continue
            d = np.asarray(depth, dtype=np.float64)
            if d.ndim != 2:
                continue
            fwd = float(forward_min_depth(d, center_frac=float(args.center_frac)))
            if not np.isfinite(fwd) or fwd > float(args.max_forward_depth_m):
                continue
            cones = cone_clearances(d, center_frac=float(args.center_frac))
            feat = encode_window_packed(
                dynamics, ep, t_i, window=int(args.window)
            )
            costs = _score_row(dynamics, feat)

            left_c = float(cones["left"])
            right_c = float(cones["right"])
            fwd_c = float(cones["forward"])
            # Classify slice by GT cones
            slice_name = "fwd_empty"
            if fwd_c <= 4.0 and min(left_c, right_c) >= fwd_c + margin:
                slice_name = "fwd_near"
            elif left_c + margin < fwd_c and left_c < right_c:
                slice_name = "left_near"
            elif right_c + margin < fwd_c and right_c < left_c:
                slice_name = "right_near"
            elif fwd_c > 8.0:
                slice_name = "fwd_empty"

            ok: Optional[bool] = None
            if slice_name == "fwd_empty":
                ok = costs["fwd"] < costs["hover"]
                checks["fwd_empty_fwd_cheaper_than_hover"].append(bool(ok))
            elif slice_name == "fwd_near":
                side_best = min(costs["left"], costs["right"], costs["climb"])
                ok = side_best < costs["fwd"]
                checks["fwd_near_side_or_climb_cheaper_than_fwd"].append(bool(ok))
            elif slice_name == "left_near":
                ok = costs["fwd"] < costs["left"]
                checks["left_near_fwd_cheaper_than_left"].append(bool(ok))
            elif slice_name == "right_near":
                ok = costs["fwd"] < costs["right"]
                checks["right_near_fwd_cheaper_than_right"].append(bool(ok))

            rows.append(
                {
                    "episode_idx": ep_i,
                    "t_idx": t_i,
                    "slice": slice_name,
                    "cones": {k: round(float(v), 3) for k, v in cones.items()},
                    "costs": {k: round(float(v), 4) for k, v in costs.items()},
                    "ok": ok,
                }
            )

    rates: Dict[str, Any] = {}
    for k, vals in checks.items():
        if vals:
            rates[k] = {
                "n": len(vals),
                "rate": float(np.mean(vals)),
                "pass": bool(np.mean(vals) >= 0.6),
            }
        else:
            rates[k] = {"n": 0, "rate": None, "pass": None}

    core = [
        "fwd_near_side_or_climb_cheaper_than_fwd",
        "left_near_fwd_cheaper_than_left",
    ]
    core_pass = all(rates[k]["pass"] for k in core if rates[k]["n"] >= 20)
    core_enough = all(rates[k]["n"] >= 20 for k in core)

    payload = {
        "step": "geom_obstacle_cost_cone_rank",
        "dataset": str(dataset),
        "wm_ckpt": str(wm_ckpt),
        "n_samples": len(rows),
        "max_forward_depth_m": float(args.max_forward_depth_m),
        "margin_m": margin,
        "rates": rates,
        "verdict": (
            "PASS"
            if core_enough and core_pass
            else ("INSUFFICIENT" if not core_enough else "FAIL")
        ),
        "samples_head": rows[:20],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "verdict": payload["verdict"],
                "rates": rates,
                "n": len(rows),
            },
            indent=2,
        )
    )
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
