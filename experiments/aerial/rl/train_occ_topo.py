#!/usr/bin/env python3
"""Train the occ_topo occupancy prior (G0) on collected episodes.

The world model reads RGB and the stored 10-D proprio (commanded body delta
plus IMU). Simulator pose and depth are used only to build the occupancy
label. Weights are written under an ``occ_topo_*`` directory.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from experiments.aerial.rl.occ_grid import integrate_label, labels_for_episode
from experiments.aerial.rl.occ_topo_loss import HORIZON, OccDecoder, occ_topo_loss

logger = logging.getLogger("train_occ_topo")

HISTORY = 16
KL_STOP = 10.0


def decode_rgb(blob: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    import cv2

    raw = np.asarray(blob[int(offsets[index]) : int(offsets[index + 1])])
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("jpeg decode failed at frame %s" % index)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def label_cache(episode: Path) -> Path:
    return episode.with_suffix(".occ.npz")


def ensure_labels(episode: Path, stride: int) -> np.ndarray:
    cache = label_cache(episode)
    if cache.is_file():
        with np.load(cache) as z:
            return np.asarray(z["grid"])
    with np.load(episode) as z:
        depth = np.asarray(z["depth"])
        pose = np.asarray(z["label_state"])
    grid = labels_for_episode(depth, pose, stride=stride)
    np.savez(cache, grid=grid)
    return grid


def episode_paths(data_dir: Path) -> List[Path]:
    paths = sorted(p for p in data_dir.glob("ep_*.npz") if not p.name.endswith(".occ.npz"))
    # ``ep_0001.occ.npz`` does not match ``ep_*.npz`` via this suffix check;
    # glob ``ep_*.npz`` also matches ``ep_0001.occ.npz`` because of the extra dot.
    return [p for p in paths if ".occ." not in p.name and p.suffix == ".npz"]


def anchors(length: int, history: int, horizon: int) -> List[int]:
    start = int(history) - 1
    last = int(length) - int(horizon) - 1
    if last < start:
        return []
    return list(range(start, last + 1))


def load_batch(
    items: Sequence[Tuple[Path, int, np.ndarray]],
    *,
    history: int,
    horizon: int,
) -> Dict[str, torch.Tensor]:
    rgbs = []
    proprios = []
    actions = []
    occs = []
    for path, t, grid in items:
        s = int(t) - int(history) + 1
        with np.load(path) as z:
            blob = np.asarray(z["jpeg"])
            offsets = np.asarray(z["jpeg_offsets"])
            proprio = np.asarray(z["proprio"], dtype=np.float32)
            action = np.asarray(z["action"], dtype=np.float32)
        if proprio.shape[-1] != 10:
            raise RuntimeError("%s proprio width %s" % (path, proprio.shape[-1]))
        frames = [decode_rgb(blob, offsets, i) for i in range(s, int(t) + 1)]
        rgbs.append(np.stack(frames, axis=0))
        proprios.append(proprio[s : int(t) + 1])
        actions.append(action[s : int(t) + horizon])
        labels = [grid[int(t)]]
        for k in range(1, horizon + 1):
            labels.append(integrate_label(grid[int(t)], action[int(t) : int(t) + k]))
        occs.append(np.stack(labels, axis=0))
    return {
        "rgb": torch.from_numpy(np.stack(rgbs, axis=0)),
        "proprio": torch.from_numpy(np.stack(proprios, axis=0)),
        "action": torch.from_numpy(np.stack(actions, axis=0)),
        "occ": torch.from_numpy(np.stack(occs, axis=0)),
    }


def main() -> int:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="experiments/aerial/rl/artifacts/occ_topo_collect",
    )
    parser.add_argument("--init", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.out_dir)
    if "occ_topo" not in out.name:
        raise SystemExit("refusing to write outside an occ_topo_* directory: %s" % out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "occ_topo_wm.pt"
    data_dir = Path(args.data_dir)
    paths = episode_paths(data_dir)
    if not paths:
        raise SystemExit("no episodes in %s" % data_dir)
    logger.info("episodes=%s", len(paths))

    index: List[Tuple[Path, int]] = []
    grids: Dict[Path, np.ndarray] = {}
    workers = max(1, int(args.workers))
    t_label = time.time()
    if workers == 1:
        built = [(path, ensure_labels(path, int(args.stride))) for path in paths]
    else:
        built = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(ensure_labels, path, int(args.stride)): path for path in paths}
            done = 0
            for fut in as_completed(futs):
                path = futs[fut]
                grid = fut.result()
                built.append((path, grid))
                done += 1
                if done == 1 or done % 10 == 0 or done == len(paths):
                    logger.info("labels %s/%s %s frames=%s", done, len(paths), path.name, grid.shape[0])
    for path, grid in built:
        grids[path] = grid
        for t in anchors(grid.shape[0], int(args.history), HORIZON):
            index.append((path, t))
    logger.info("labels ready anchors=%s sec=%.0f", len(index), time.time() - t_label)
    if not index:
        raise SystemExit("no anchors")

    from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics

    model = TorchRSSMDynamics(proprio_dim=10, device=args.device)
    payload = model.load_checkpoint(str(args.init))
    skipped = payload.get("load_skipped") or []
    logger.info("init %s skipped %s", args.init, skipped)
    decoder = OccDecoder(in_dim=int(model.latent_dim)).to(model.device, model.torch_dtype)
    params = list(model.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr), betas=(0.9, 0.95))

    rng = np.random.default_rng(0)
    order = rng.permutation(len(index))

    model.train()
    decoder.train()
    cursor = 0
    for step in range(1, int(args.steps) + 1):
        batch_items = []
        for _ in range(int(args.batch)):
            path, t = index[int(order[cursor % len(order)])]
            cursor += 1
            batch_items.append((path, t, grids[path]))
        batch = load_batch(batch_items, history=int(args.history), horizon=HORIZON)
        rgb = batch["rgb"].to(model.device)
        proprio = batch["proprio"].to(model.device, model.torch_dtype)
        action = batch["action"].to(model.device, model.torch_dtype)
        occ = batch["occ"].to(model.device)
        opt.zero_grad(set_to_none=True)
        loss, stats = occ_topo_loss(model, decoder, rgb, proprio, action, occ)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(params, model.grad_clip))
        opt.step()
        if step == 1 or step % 20 == 0:
            logger.info("step=%s grad=%.3f %s", step, grad, " ".join(
                "%s=%.4f" % (k, stats[k]) for k in (
                    "loss", "loss_occ", "loss_prior", "loss_post", "loss_dyn", "loss_rep"
                )
            ))
        if stats["loss_dyn"] >= KL_STOP * max(stats["loss_occ"], 1e-8):
            _save(ckpt_path, model, decoder, step, stats)
            logger.error(
                "stop: loss_dyn %.4f >= %.0f x loss_occ %.4f",
                stats["loss_dyn"],
                KL_STOP,
                stats["loss_occ"],
            )
            return 2
        if step % 500 == 0 or step == int(args.steps):
            _save(ckpt_path, model, decoder, step, stats)
    logger.info("done %s", ckpt_path)
    return 0


def _save(
    path: Path,
    model: torch.nn.Module,
    decoder: OccDecoder,
    step: int,
    stats: Dict[str, float],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "occ_decoder": decoder.state_dict(),
            "step": int(step),
            "proprio_dim": 10,
            "stats": stats,
        },
        path,
    )
    logger.info("saved %s step=%s", path, step)


if __name__ == "__main__":
    raise SystemExit(main())
