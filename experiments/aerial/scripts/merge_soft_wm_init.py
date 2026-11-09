#!/usr/bin/env python3
"""Merge antisat/geomfix WM (imagine-PASS coll) + gatefix obstacle_cost head.

Soft needs both: action-conditioned coll from READY_SOFT imagine pass, and
obstacle_cost_trained=True for directional OA reward. Pure gatefix loses
imagine gap; pure antisat refuses / zeros learned cost.

  $AERIAL_PY experiments/aerial/scripts/merge_soft_wm_init.py \\
    --antisat .../wm_step_2500.pt \\
    --cost .../wm_obs.pt \\
    --out .../wm_soft_init_antisat_cost.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--antisat", required=True, type=Path)
    p.add_argument("--cost", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    anti = torch.load(args.antisat, map_location="cpu")
    cost = torch.load(args.cost, map_location="cpu")
    if not isinstance(anti, dict) or not isinstance(cost, dict):
        raise SystemExit("ckpts must be dict payloads")

    am = anti.get("model") or anti
    cm = cost.get("model") or cost
    if not isinstance(am, dict) or not isinstance(cm, dict):
        raise SystemExit("missing model state_dict")

    n_copied = 0
    for k, v in cm.items():
        if "obstacle_cost" in k:
            am[k] = v
            n_copied += 1
    if n_copied == 0:
        raise SystemExit("no obstacle_cost_* keys in cost ckpt")

    anti["model"] = am
    anti["obstacle_cost_trained"] = True
    # Keep antisat step / depth_decoder_trained / coll_head.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(anti, args.out)
    print(
        f"wrote {args.out} obstacle_cost_keys={n_copied} "
        f"obstacle_cost_trained=True step={anti.get('step')} "
        f"from_antisat={args.antisat.name} cost={args.cost.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
