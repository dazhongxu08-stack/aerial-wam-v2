#!/usr/bin/env python3
"""Dry-run the Tello consequence policy on the dog's camera.

Reads a RealSense color node, runs the same DA3 depth head and reactive
cones as the drone, and logs the walk command it would send. It does not
open LCM and does not publish ``lcm_robot_cmd``, so an H12 bridge can own
the dog at the same time.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import cv2
import numpy as np

from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.tello_consequence_policy import (
    MAX_V_MPS,
    MAX_YAW_RAD_S,
    SLOW_M,
    TelloConsequencePolicy,
)

# SDK example's tested fore-aft command is vx=-0.2. Map a positive forward
# stick onto that sign. Yaw uses the example's positive vyaw.
DOG_VX_MAX = 0.5
DOG_VYAW_MAX = 0.5


def pool_min(depth: np.ndarray, out: int = 56) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    h, w = d.shape
    fh, fw = h // out, w // out
    if fh < 1 or fw < 1:
        raise ValueError(f"depth {h}x{w} cannot pool to {out}")
    d = d[: fh * out, : fw * out].reshape(out, fh, out, fw)
    return d.min(axis=(1, 3)).astype(np.float32)


def open_color(prefer: str) -> Tuple[cv2.VideoCapture, str]:
    candidates = [prefer] if prefer else []
    candidates += ["/dev/video4", "/dev/video2", "/dev/video0"]
    seen = set()
    for dev in candidates:
        if not dev or dev in seen:
            continue
        seen.add(dev)
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue
        return cap, dev
    raise RuntimeError("no color frame from RealSense video nodes")


def dog_cmd(rc: Tuple[int, int, int, int]) -> Tuple[float, float]:
    _a, b, _c, d = rc
    vx = -float(b) / 100.0 * MAX_V_MPS
    vyaw = float(d) / 100.0 * MAX_YAW_RAD_S
    vx = max(-DOG_VX_MAX, min(DOG_VX_MAX, vx))
    vyaw = max(-DOG_VYAW_MAX, min(DOG_VYAW_MAX, vyaw))
    return vx, vyaw


def main() -> int:
    p = argparse.ArgumentParser(description="Log dog commands from the drone policy. Does not publish.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--camera", default="")
    p.add_argument("--near-m", type=float, default=1.2)
    p.add_argument("--hz", type=float, default=2.0)
    p.add_argument("--log", default="/tmp/lud_policy_dryrun.log")
    args = p.parse_args()
    if str(args.device).startswith("cuda"):
        os.environ.setdefault("AERIAL_DA3_FP16", "1")

    pred = DepthMinPredictor.from_checkpoint(args.ckpt, device=args.device)
    policy = TelloConsequencePolicy(device=args.device, near_m=args.near_m, slow_m=SLOW_M)
    cap, dev = open_color(args.camera)
    print(
        f"dry-run camera={dev} near_m={args.near_m} device={args.device} "
        f"publish=off families=hover/yaw/fwd",
        flush=True,
    )
    period = 1.0 / max(0.2, float(args.hz))
    log = open(args.log, "a", buffering=1)
    try:
        while True:
            t0 = time.time()
            ok, bgr = cap.read()
            if not ok or bgr is None:
                print("camera read failed", flush=True)
                time.sleep(0.5)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            obs = Observation(rgb=rgb, state=np.zeros(7, np.float32), t=t0)
            depth = pred.predict_depth(obs)
            depth56 = pool_min(depth) if depth is not None else None
            dec = policy.decide(
                rgb=rgb,
                height_m=None,
                yaw_rad=0.0,
                takeoff_yaw_rad=0.0,
                goal_bearing0_rad=0.0,
                depth56=depth56,
            )
            vx, vyaw = dog_cmd(dec.rc)
            line = (
                f"{time.strftime('%H:%M:%S')} family={dec.family} "
                f"fwd={dec.fwd_m:.2f} left={dec.left_m:.2f} right={dec.right_m:.2f} "
                f"src={dec.depth_source} would_vx={vx:.3f} would_vyaw={vyaw:.3f} "
                f"note={dec.note}"
            )
            print(line, flush=True)
            log.write(line + "\n")
            delay = period - (time.time() - t0)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("stop", flush=True)
    finally:
        cap.release()
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
