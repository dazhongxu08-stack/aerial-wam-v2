#!/usr/bin/env python3
"""Write ``episode_*.env.npz`` sidecars: the depth head's DA3 encoder scene vector per frame.

``env`` is the vector of each frame and ``env_flip`` that of the left-right
mirrored frame, so ``mirror_batch`` can swap them instead of pretending the
vector is mirror-symmetric. Frames are resized exactly as ``DepthMinPredictor``
does before the encoder. Skips episodes that already have a sidecar.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger("precompute_env")

DEFAULT_CKPT = "experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"


def load_da3_head(path: Path, device: str):
    import torch

    from experiments.aerial.rl.dynamics_torch import build_depth_head

    payload = torch.load(str(path), map_location="cpu")
    head = build_depth_head(payload)
    head.load_state_dict(payload["model"], strict=True)
    if not hasattr(head, "encode_env"):
        raise SystemExit(f"{path} is not a DA3 depth head")
    return head.float().to(device).eval()


def env_pair(head, frames: Sequence[np.ndarray], batch: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """RGB frames -> (env, env_flip) float16 ``[N, ENV_DIM]``."""
    import cv2
    import torch

    size = int(head.image_size)
    dev = next(head.parameters()).device
    env: List[np.ndarray] = []
    env_flip: List[np.ndarray] = []
    for s in range(0, len(frames), batch):
        imgs = []
        for rgb in frames[s:s + batch]:
            rgb = np.asarray(rgb, np.uint8)
            if rgb.shape[:2] != (size, size):
                rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            imgs.append(rgb)
        x = torch.from_numpy(np.stack(imgs)).unsqueeze(1).to(dev)  # [B,1,H,W,3]
        with torch.no_grad():
            env.append(head.encode_env(x).cpu().numpy().astype(np.float16))
            env_flip.append(head.encode_env(torch.flip(x, dims=[3])).cpu().numpy().astype(np.float16))
    return np.concatenate(env), np.concatenate(env_flip)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--depth-ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from experiments.aerial.rl.precompute_da3_consequence import decode_frames

    head = load_da3_head(root / args.depth_ckpt, args.device)
    ep_dir = root / args.data_dir / "episodes"
    todo = sorted(p for p in ep_dir.glob("episode_*.npz")
                  if re.fullmatch(r"episode_\d+\.npz", p.name) and not p.with_suffix(".env.npz").is_file())
    for p in todo:
        t0 = time.time()
        frames = decode_frames(np.load(p))
        env, env_flip = env_pair(head, frames)
        tmp = p.with_name(p.stem + ".env.tmp.npz")
        np.savez(tmp, env=env, env_flip=env_flip)
        tmp.rename(p.with_suffix(".env.npz"))
        logger.info("%s frames=%d %.1fs", p.name, len(frames), time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
