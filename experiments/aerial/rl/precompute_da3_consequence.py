#!/usr/bin/env python3
"""Write ``episode_*.da3.npz`` sidecars: the deployed depth head on each frame.

Frames are fed in order per episode so the head's frame history matches what
it sees in flight. Output is the predicted depth block-min pooled to 56x56
(float16), the same layout as the GT ``depth56`` in the episode file.
Skips episodes that already have a sidecar, so it can run while collection
is still appending.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("precompute_da3")

DEFAULT_CKPT = "experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"


def decode_frames(z: np.lib.npyio.NpzFile) -> list:
    import cv2

    buf, off = z["jpeg"], z["jpeg_offsets"]
    return [cv2.imdecode(buf[off[i]:off[i + 1]], cv2.IMREAD_COLOR)[:, :, ::-1] for i in range(len(off) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--depth-ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--watch", action="store_true", help="keep polling for new episodes")
    ap.add_argument("--poll-s", type=float, default=120.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from experiments.aerial.rl.collect_consequence_dataset import pool_min
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation

    pred = DepthMinPredictor.from_checkpoint(root / args.depth_ckpt, device=args.device)
    ep_dir = root / args.data_dir / "episodes"
    while True:
        todo = sorted(p for p in ep_dir.glob("episode_*.npz")
                      if re.fullmatch(r"episode_\d+\.npz", p.name) and not p.with_suffix(".da3.npz").is_file())
        for p in todo:
            t0 = time.time()
            try:
                z = np.load(p)
                frames = decode_frames(z)
            except Exception as exc:  # noqa: BLE001 - file still being written
                logger.warning("skip %s: %s", p.name, exc)
                continue
            pred.reset()
            out = []
            for rgb in frames:
                obs = Observation(rgb=np.ascontiguousarray(rgb), state=np.zeros(7, np.float32), collided=False)
                d = pred.predict_depth(obs)
                out.append(pool_min(d))
            tmp = p.with_name(p.stem + ".da3.tmp.npz")
            np.savez(tmp, depth56=np.stack(out))
            tmp.rename(p.with_suffix(".da3.npz"))
            logger.info("%s frames=%d %.1fs", p.name, len(frames), time.time() - t0)
        if not args.watch:
            break
        time.sleep(float(args.poll_s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
