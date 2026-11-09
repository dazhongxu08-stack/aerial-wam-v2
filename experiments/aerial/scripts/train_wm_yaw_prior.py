"""Train the yaw-consequence prior. Does not overwrite wm_soft_init."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aerial.rl.wm_yaw_prior import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default="configs/_tmp_mainchannel_soft_h84.yaml")
    parser.add_argument(
        "--wm-ckpt",
        default="experiments/aerial/rl/artifacts/wm_ckpt_mainchannel_s0_20261008/wm_soft_init.pt",
    )
    parser.add_argument(
        "--depth-ckpt",
        default="experiments/aerial/rl/artifacts/da3_cone_yaw_20261009/depth_cone_yaw40.pt",
    )
    parser.add_argument(
        "--actor-ckpt",
        default="experiments/aerial/rl/artifacts/v4_ac_ckpt_mainchannel_s0_20261008/v4_ac_latest.pt",
    )
    parser.add_argument(
        "--annotation",
        default="experiments/aerial/phase3_unified/annotations/outdoor_complex_inland_patched.json",
    )
    parser.add_argument("--route", type=int, default=0)
    parser.add_argument("--camera", default="front_custom")
    parser.add_argument("--spacing", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--stick-steps", type=int, default=400)
    parser.add_argument("--retrain-prior", action="store_true")
    parser.add_argument("--actor-steps", type=int, default=200)
    parser.add_argument(
        "--out-dir",
        default="experiments/aerial/rl/artifacts/wm_ckpt_yaw_prior_20261009",
    )
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    return int(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
