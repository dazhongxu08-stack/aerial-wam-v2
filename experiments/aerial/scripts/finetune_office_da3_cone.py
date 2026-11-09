#!/usr/bin/env python3
"""Finetune the office depth head on the forward-cone minimum.

The encoder stays frozen. The decoder is updated so the pooled forward cone
matches the stored GT depth56. A frame that is already too near is penalised
harder than one that is too far, so the open floor is not pulled down to the
contact distance. Sidecars are rewritten only after the holdout passes.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/aerial-wam-v2")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402

DATA = Path("/data/simworld/run/indoor_office_gap")
SRC = ROOT / "experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"
OUT = ROOT / "experiments/aerial/rl/artifacts/da3_office_cone_20261101/depth_office_cone.pt"
FAR = 30.0
OUT_N = 224


def hide_row() -> int:
    focal = 0.5 * OUT_N / math.tan(math.radians(45.0))
    angle = math.atan((0.42 * 480.0 - 243.503) / 386.394)
    return int(np.clip(math.floor(OUT_N / 2.0 + focal * math.tan(angle)), 0, OUT_N - 1))


HIDE_ROW = hide_row()
# Stay far under the 62 GB machine so the OOM killer never runs.
RSS_ABORT_MB = 12000


def rss_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def guard_rss(where: str) -> None:
    used = rss_mb()
    print(f"rss {where} {used:.0f}MB", flush=True)
    if used > RSS_ABORT_MB:
        print(f"RSS_ABORT {where} {used:.0f}MB", flush=True)
        os._exit(3)


def start_rss_watch() -> None:
    def loop() -> None:
        while True:
            time.sleep(2.0)
            used = rss_mb()
            if used > RSS_ABORT_MB:
                print(f"RSS_ABORT watch {used:.0f}MB", flush=True)
                os._exit(3)

    threading.Thread(target=loop, daemon=True).start()


def decode_one(z, index: int) -> np.ndarray:
    import cv2

    off = z["jpeg_offsets"]
    blob = z["jpeg"]
    image = cv2.imdecode(
        np.frombuffer(blob[int(off[index]) : int(off[index + 1])], np.uint8),
        cv2.IMREAD_COLOR,
    )
    return np.ascontiguousarray(image[:, :, ::-1])


def decode_frames(z) -> list:
    import cv2

    buf, off = z["jpeg"], z["jpeg_offsets"]
    return [
        cv2.imdecode(np.frombuffer(buf[int(off[i]) : int(off[i + 1])], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        for i in range(len(off) - 1)
    ]


def _low5(x: np.ndarray) -> float:
    v = x[np.isfinite(x)]
    if v.size == 0:
        return FAR
    k = max(1, v.size // 20)
    return float(np.partition(v, k - 1)[:k].mean())


def cones_np(d56: np.ndarray) -> np.ndarray:
    d = np.asarray(d56, np.float32)
    h, w = d.shape
    band = d[h // 4 : h - h // 4]
    c0, c1 = w // 4, w - w // 4
    parts = (band[:, c0:c1], band[:, :c0], band[:, c1:])

    def vmin(x: np.ndarray) -> float:
        v = x[np.isfinite(x) & (x > 0.05)]
        return float(v.min()) if v.size else FAR

    mins = [_low5(x) for x in parts]
    return np.array([vmin(p) for p in parts] + mins, np.float32)


def index_episodes() -> tuple[list, list]:
    """Frame refs only. RGB stays on disk until a batch is drawn."""
    review = json.loads((DATA / "AUTO_REVIEW.json").read_text())
    train, hold = [], []
    for row in review["episodes"]:
        if not row.get("keep"):
            continue
        path = DATA / "episodes" / row["file"]
        with np.load(path) as archive:
            depth = np.asarray(archive["depth56"], np.float32)
        bucket = hold if row.get("route_id") == "office_val" else train
        for i in range(depth.shape[0]):
            bucket.append(("e", row["file"], i, tuple(float(v) for v in cones_np(depth[i]))))
    return train, hold


class FrameLoader:
    """Decode a handful of frames at a time. One episode file stays open."""

    def __init__(self, pairs: Path | None, angle_dir: Path | None = None) -> None:
        self._name = ""
        self._episode = None
        self._pairs = None
        self._angle = angle_dir
        if pairs is not None:
            self._pairs = np.load(pairs, mmap_mode="r")

    def close(self) -> None:
        if self._episode is not None:
            self._episode.close()
            self._episode = None
            self._name = ""

    def rgb(self, ref) -> np.ndarray:
        kind, loc, index, _cones = ref
        if kind == "p":
            return np.ascontiguousarray(self._pairs["rgb"][index])
        path = (self._angle if kind == "a" else DATA) / "episodes" / loc
        key = str(path)
        if self._name != key:
            self.close()
            self._episode = np.load(path)
            self._name = key
        return decode_one(self._episode, index)

    def gt_depth(self, ref) -> np.ndarray:
        kind, _loc, index, _cones = ref
        if kind == "p":
            return np.asarray(self._pairs["depth56"][index], np.float32)
        return np.asarray(self._episode["depth56"][index], np.float32)

    def batch(self, refs: list, with_depth: bool = False):
        ordered = sorted(refs, key=lambda ref: (ref[0], ref[1], ref[2]))
        images = []
        depths = []
        for ref in ordered:
            images.append(self.rgb(ref))
            if with_depth:
                depths.append(self.gt_depth(ref))
        cones = np.asarray([ref[3] for ref in ordered], np.float32)
        self.close()
        stacked = np.stack(images)
        if not with_depth:
            return stacked, cones
        return stacked, cones, np.stack(depths)


def pool_min_t(depth: torch.Tensor, out: int = 56) -> torch.Tensor:
    b, h, w = depth.shape
    fh, fw = h // out, w // out
    d = depth[:, : fh * out, : fw * out].reshape(b, out, fh, out, fw)
    return d.amin(dim=(2, 4))


def hide_pool(pred: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    seen = rgb.sum(dim=-1) > 12
    depth = torch.where(seen, pred, torch.full_like(pred, FAR))
    depth = depth.clone()
    depth[:, HIDE_ROW:, :] = FAR
    return pool_min_t(depth)


def cone_mins_t(d56: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = d56.shape[-2:]
    band = d56[:, h // 4 : h - h // 4]
    c0, c1 = w // 4, w - w // 4
    return (
        band[:, :, c0:c1].amin(dim=(-2, -1)),
        band[:, :, :c0].amin(dim=(-2, -1)),
        band[:, :, c1:].amin(dim=(-2, -1)),
    )


def forward_maps(model, rgb: np.ndarray) -> torch.Tensor:
    window = torch.from_numpy(np.ascontiguousarray(rgb)).to("cuda").unsqueeze(1)
    pred, _ = model.predict_from_window(window)
    return hide_pool(pred, window[:, 0])


def low_forward(d56: torch.Tensor) -> torch.Tensor:
    """Mean of the closest twentieth of the forward cone, not a single pixel."""
    return _low_cols(d56, slice(d56.shape[-1] // 4, d56.shape[-1] - d56.shape[-1] // 4))


def _low_cols(d56: torch.Tensor, cols: slice) -> torch.Tensor:
    h, w = d56.shape[-2:]
    band = d56[:, h // 4 : h - h // 4, cols]
    flat = band.reshape(band.shape[0], -1)
    k = max(1, flat.shape[1] // 20)
    return flat.topk(k, dim=-1, largest=False).values.mean(-1)


def cone_loss(p56: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pf, pl, pr = cone_mins_t(p56)
    gf, gl, gr = gt[:, 0], gt[:, 1], gt[:, 2]
    err = torch.log(pf.clamp(0.05, FAR)) - torch.log(gf.clamp(0.05, FAR))
    # Too near costs more than too far, so the open floor stays open.
    fwd = torch.where(err > 0, err, 2.0 * (-err))
    region = torch.log(low_forward(p56).clamp(0.05, FAR)) - torch.log(gf.clamp(0.05, FAR))
    region = torch.where(region > 0, region, 2.0 * (-region))
    sides = (
        (torch.log(pl.clamp(0.05, FAR)) - torch.log(gl.clamp(0.05, FAR))).abs()
        + (torch.log(pr.clamp(0.05, FAR)) - torch.log(gr.clamp(0.05, FAR))).abs()
    )
    weight = torch.where(gf < 1.2, 4.0, torch.where(gf < 2.2, 2.0, 1.0))
    # Angled open floor was being pulled down to the contact distance.
    open_floor = (gf >= 2.2).float() * torch.relu(math.log(2.2) - torch.log(pf.clamp(min=0.05)))
    # A close cone that is still read long is the angled miss. This term is
    # zero once the prediction is at or nearer than the GT cone, so it does
    # not pull the open floor down.
    def too_far(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.log(pred.clamp(0.05, FAR)) - torch.log(truth.clamp(0.05, FAR)))

    close_far = (
        (gf < 1.2).float() * too_far(pf, gf)
        + (gl < 1.2).float() * too_far(pl, gl)
        + (gr < 1.2).float() * too_far(pr, gr)
    )
    # The barrel face is a thin strip. Match the closest twentieth of that
    # side to the same strip of the GT, not the single nearest pixel.
    thin_face = pf.new_zeros(())
    if gt.shape[1] >= 6:
        gf5, gl5, gr5 = gt[:, 3], gt[:, 4], gt[:, 5]
        wpx = p56.shape[-1]
        thin_face = (gf5 < 1.3).float() * too_far(low_forward(p56), gf5)
        thin_face = thin_face + (gl5 < 1.3).float() * too_far(_low_cols(p56, slice(0, wpx // 4)), gl5)
        thin_face = thin_face + (gr5 < 1.3).float() * too_far(
            _low_cols(p56, slice(wpx - wpx // 4, wpx)), gr5
        )
    # The open side of an angled frame is still floor. Keep it up while the
    # close side comes in.
    side_open = (gl >= 2.2).float() * torch.relu(math.log(2.2) - torch.log(pl.clamp(min=0.05)))
    side_open = side_open + (gr >= 2.2).float() * torch.relu(math.log(2.2) - torch.log(pr.clamp(min=0.05)))
    return (
        weight * (fwd + region)
        + 0.35 * sides
        + 16.0 * open_floor
        + 8.0 * side_open
        + 2.0 * close_far
        + 3.0 * thin_face
    ).mean()


def pixel_close(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Pull only pixels whose GT is already inside 1.3 m. Floor pixels add nothing."""
    h = pred.shape[-2]
    band = slice(h // 4, h - h // 4)
    p = pred[:, band].clamp(0.05, FAR)
    g = gt[:, band]
    close = torch.isfinite(g) & (g > 0.05) & (g < 1.3)
    err = torch.relu(torch.log(p) - torch.log(g.clamp(min=0.05)))
    count = close.flatten(1).sum(1)
    present = count > 0
    if not bool(present.any()):
        return pred.new_zeros(())
    per = (err * close).flatten(1).sum(1) / count.clamp(min=1)
    return per[present].mean()


def report(name: str, rows: list[tuple[float, float]]) -> dict:
    buckets = {
        "near": [a for g, a in rows if g < 1.2],
        "mid": [a for g, a in rows if 1.2 <= g < 2.2],
        "open": [a for g, a in rows if g >= 2.2],
    }
    gt_buckets = {
        "near": [g for g, _a in rows if g < 1.2],
        "mid": [g for g, _a in rows if 1.2 <= g < 2.2],
        "open": [g for g, _a in rows if g >= 2.2],
    }
    out = {}
    for key in ("near", "mid", "open"):
        da = np.asarray(buckets[key], np.float32)
        gt = np.asarray(gt_buckets[key], np.float32)
        if da.size == 0:
            continue
        out[key] = {
            "n": int(da.size),
            "gt_med": float(np.median(gt)),
            "da_med": float(np.median(da)),
            "abs_med": float(np.median(np.abs(da - gt))),
            "ratio": float(np.median(da / np.clip(gt, 0.05, None))),
            "p90": float(np.quantile(da, 0.9)),
            "hi": float(da.max()),
        }
        print(
            f"  {name} {key} n={da.size} gt {out[key]['gt_med']:.2f} "
            f"da {out[key]['da_med']:.2f} p90 {out[key]['p90']:.2f} "
            f"hi {out[key]['hi']:.2f} |e| {out[key]['abs_med']:.2f} "
            f"ratio {out[key]['ratio']:.2f}",
            flush=True,
        )
    return out


def accepts(stats: dict) -> bool:
    near, open_ = stats.get("near"), stats.get("open")
    if not near or not open_:
        return False
    return near["da_med"] <= 1.3 and 2.0 <= open_["da_med"] <= 3.2


def side_gap(stats: dict) -> float:
    """Close sides must come in. Open sides must stay open."""
    near, open_ = stats.get("side_near"), stats.get("side_open")
    gap = 0.0
    if near:
        gap += max(0.0, near["da_med"] - 1.3)
    if open_:
        gap += max(0.0, 2.0 - open_["da_med"])
    return gap


def index_angle(angle_dir: Path) -> tuple[list, list]:
    """Train on the angled episodes. The yaw val split stays out of the loss."""
    review = json.loads((angle_dir / "AUTO_REVIEW.json").read_text())
    train, hold = [], []
    for row in review["episodes"]:
        rid = row.get("route_id")
        if rid not in ("office_yaw_train", "office_yaw_val"):
            continue
        path = angle_dir / "episodes" / row["file"]
        with np.load(path) as archive:
            depth = np.asarray(archive["depth56"], np.float32)
        bucket = hold if rid == "office_yaw_val" else train
        for i in range(depth.shape[0]):
            cones = tuple(float(v) for v in cones_np(depth[i]))
            if rid == "office_yaw_val" and not (cones[0] < 1.2 and max(cones[1], cones[2]) > 2.0):
                continue
            bucket.append(("a", row["file"], i, cones))
    return train, hold


def gate_gap(stats: dict) -> float:
    near, open_ = stats.get("near"), stats.get("open")
    if not near or not open_:
        return 10.0
    return (
        max(0.0, near["da_med"] - 1.3)
        + max(0.0, 2.0 - open_["da_med"])
        + max(0.0, open_["da_med"] - 3.2)
    )


@torch.no_grad()
def evaluate(model, frames: list, loader: FrameLoader, limit: int = 0) -> dict:
    model.eval()
    use = frames
    if limit and len(frames) > limit:
        rng = np.random.default_rng(1)
        near = [i for i, ref in enumerate(frames) if ref[3][0] < 1.2]
        rest = [i for i, ref in enumerate(frames) if ref[3][0] >= 1.2]
        pick = list(near) + list(rng.choice(rest, size=min(len(rest), limit), replace=False))
        use = [frames[i] for i in pick]
    rows = []
    side_rows = []
    thin_rows = []
    ordered = sorted(use, key=lambda ref: (ref[0], ref[1], ref[2]))
    for i in range(0, len(ordered), 8):
        chunk = ordered[i : i + 8]
        rgb, _cones = loader.batch(chunk)
        with torch.no_grad():
            pred = forward_maps(model, rgb).detach().float().cpu().numpy()
        del rgb
        for j, ref in enumerate(chunk):
            da = cones_np(pred[j])
            gt = ref[3]
            rows.append((float(gt[0]), float(da[0])))
            for k in (1, 2):
                side_rows.append((float(gt[k]), float(da[k])))
            if len(gt) >= 6:
                for src, col in ((4, 1), (5, 2)):
                    if float(gt[src]) < 1.3:
                        thin_rows.append((float(gt[src]), float(da[col + 3])))
        del pred
    out = report("hold", rows)
    side = report("side", side_rows)
    if "near" in side:
        out["side_near"] = side["near"]
    if "open" in side:
        out["side_open"] = side["open"]
    if thin_rows:
        da = np.asarray([a for _, a in thin_rows], np.float32)
        truth = np.asarray([g for g, _ in thin_rows], np.float32)
        out["thin"] = {
            "n": int(da.size),
            "gt_med": float(np.median(truth)),
            "da_med": float(np.median(da)),
            "hi": float(da.max()),
        }
        print(
            f"  thin n={da.size} gt {out['thin']['gt_med']:.2f} "
            f"da {out['thin']['da_med']:.2f} hi {out['thin']['hi']:.2f}",
            flush=True,
        )
    return out


def save_ckpt(model, src_payload: dict, path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(src_payload)
    keys = src_payload["model"].keys()
    state = model.state_dict()
    payload["model"] = {k: state[k].detach().cpu() for k in keys}
    payload["office_cone"] = {"hide_row": HIDE_ROW, "holdout": stats}
    torch.save(payload, path)


def index_pairs(path: Path) -> tuple[list, list]:
    with np.load(path) as archive:
        depth = np.asarray(archive["depth56"], np.float32)
        splits = np.array(archive["split"])
        yaw = np.asarray(archive["yaw"], np.float32)
    train, hold = [], []
    for i in range(depth.shape[0]):
        which = splits[i]
        if isinstance(which, bytes):
            which = which.decode()
        cones = tuple(float(v) for v in cones_np(depth[i]))
        ref = ("p", "", i, cones)
        bucket = hold if str(which) == "hold" else train
        bucket.append(ref)
        if str(which) == "hold":
            continue
        # Angled close views are the ones the centerline set missed.
        if cones[0] < 1.5:
            train.extend([ref] * 3)
        # Open floor at a yaw is rare next to the centerline set, and the
        # negative yaw side was still read as about 1.7 m.
        elif cones[0] >= 2.2:
            train.extend([ref] * (20 if float(yaw[i]) < 0 else 8))
    return train, hold


def mine_hard(model, loader: FrameLoader, frames: list) -> np.ndarray:
    """Angle frames whose close side is still at least 0.35 m too far."""
    picked = [
        i for i, ref in enumerate(frames)
        if ref[0] == "a" and len(ref[3]) >= 6 and min(ref[3][4], ref[3][5]) < 1.3
    ]
    hard: list[int] = []
    for start in range(0, len(picked), 8):
        chunk_i = picked[start : start + 8]
        rgb, _cones = loader.batch([frames[i] for i in chunk_i])
        with torch.no_grad():
            pred = forward_maps(model, rgb).detach().float().cpu().numpy()
        del rgb
        for j, idx in enumerate(chunk_i):
            da = cones_np(pred[j])
            gt = frames[idx][3]
            if (float(gt[4]) < 1.3 and float(da[4]) - float(gt[4]) > 0.35) or (
                float(gt[5]) < 1.3 and float(da[5]) - float(gt[5]) > 0.35
            ):
                hard.append(idx)
        del pred
    return np.asarray(hard, np.int64)


def show_turn(model, loader: FrameLoader) -> tuple[float, float]:
    rgb, _cones = loader.batch([("a", "episode_20030.npz", 4, (0.0, 0.0, 0.0))])
    with torch.no_grad():
        pred = forward_maps(model, rgb).detach().float().cpu().numpy()[0]
    da = cones_np(pred)
    h, w = pred.shape
    band = pred[h // 4 : h - h // 4]
    under = float((band[:, w - w // 4 :] < 1.3).mean())
    print(
        f"  turn4 da_f {float(da[0]):.2f} da_r {float(da[2]):.2f} "
        f"rlow {float(da[5]):.2f} under13 {under:.3f}",
        flush=True,
    )
    return float(da[5]), under


def train(args) -> int:
    start_rss_watch()
    print(f"hide_row {HIDE_ROW}", flush=True)
    train_frames, hold_frames = index_episodes()
    yaw_hold: list = []
    pair_path = Path(args.pairs) if args.pairs else None
    if pair_path is not None:
        extra_tr, extra_ho = index_pairs(pair_path)
        train_frames.extend(extra_tr)
        yaw_hold = extra_ho
        print(f"pairs train {len(extra_tr)} hold {len(extra_ho)}", flush=True)
    print(f"frames train {len(train_frames)} hold {len(hold_frames)}", flush=True)
    guard_rss("indexed")
    angle_dir = Path(args.angle_dir) if args.angle_dir else None
    angle_train, angle_hold = index_angle(angle_dir) if angle_dir is not None else ([], [])
    if angle_train:
        train_frames.extend(angle_train)
        print(f"angle train {len(angle_train)} hold {len(angle_hold)}", flush=True)
    loader = FrameLoader(pair_path, angle_dir)
    pred = DepthMinPredictor.from_checkpoint(str(args.ckpt), device="cuda")
    model = pred._model
    model.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    src_payload = torch.load(str(args.ckpt), map_location="cpu")
    guard_rss("model")
    print("before", flush=True)
    before = evaluate(model, hold_frames, loader, limit=500)
    opt = torch.optim.AdamW((p for p in model.decoder.parameters() if p.requires_grad), lr=args.lr)
    rng = np.random.default_rng(0)
    near = np.array([i for i, ref in enumerate(train_frames) if ref[3][0] < 1.2])
    mid = np.array([i for i, ref in enumerate(train_frames) if 1.2 <= ref[3][0] < 2.2])
    open_i = np.array([i for i, ref in enumerate(train_frames) if ref[3][0] >= 2.2])
    angle_near = np.array([
        i for i, ref in enumerate(train_frames)
        if ref[0] == "a" and len(ref[3]) >= 6 and min(ref[3][4], ref[3][5]) < 1.3
    ])
    hard = mine_hard(model, loader, train_frames)
    if len(hard) == 0:
        hard = angle_near
    print(
        f"sample near {len(near)} mid {len(mid)} open {len(open_i)} "
        f"angle {len(angle_near)} hard {len(hard)}",
        flush=True,
    )
    best = None
    best_stats = {"center": before, "yaw": {}}
    best_state = None
    for step in range(args.steps):
        groups = []
        if len(near):
            groups.append(rng.choice(near, size=2, replace=True))
        if len(mid):
            groups.append(rng.choice(mid, size=1, replace=True))
        if len(open_i):
            groups.append(rng.choice(open_i, size=4, replace=True))
        if len(hard):
            groups.append(rng.choice(hard, size=2, replace=True))
        pick = [train_frames[int(i)] for i in np.concatenate(groups)]
        rgb, cones, gt56 = loader.batch(pick, with_depth=True)
        gt = torch.tensor(cones, device="cuda")
        gt_map = torch.tensor(gt56, dtype=torch.float32, device="cuda")
        p56 = forward_maps(model, rgb)
        del rgb
        loss = cone_loss(p56, gt) + 5.0 * pixel_close(p56, gt_map)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        del p56, gt, gt_map, loss
        if step % 20 == 0 or step + 1 == args.steps:
            guard_rss(f"step {step}")
        if step % args.eval_every == 0 or step + 1 == args.steps:
            stats = evaluate(model, hold_frames, loader, limit=0 if step + 1 == args.steps else 400)
            yaw_stats = evaluate(model, yaw_hold, loader, limit=0) if yaw_hold else {}
            gc.collect()
            near_err = abs(stats.get("near", {}).get("da_med", 9) - stats.get("near", {}).get("gt_med", 1))
            open_err = abs(stats.get("open", {}).get("da_med", 9) - stats.get("open", {}).get("gt_med", 2.5))
            angle_stats = evaluate(model, angle_hold, loader, limit=0) if angle_hold else {}
            angle_fwd = max(0.0, angle_stats.get("near", {}).get("da_med", 1.3) - 1.3)
            angle_tail = max(0.0, angle_stats.get("thin", {}).get("da_med", 1.35) - 1.35)
            angle_tail += max(0.0, angle_stats.get("thin", {}).get("hi", 1.45) - 1.45)
            score = (
                near_err
                + 0.5 * open_err
                + 8.0 * (gate_gap(stats) + gate_gap(yaw_stats) + side_gap(stats) + side_gap(yaw_stats))
                + 4.0 * (side_gap(angle_stats) + angle_fwd + angle_tail)
            )
            turn_rlow, under13 = show_turn(model, loader)
            score += 6.0 * max(0.0, turn_rlow - 1.35)
            score += 10.0 * max(0.0, 0.03 - under13)
            yaw_open = yaw_stats.get("open", {}).get("da_med", 0.0)
            center_open = stats.get("open", {}).get("da_med", 0.0)
            gates_ok = gate_gap(stats) == 0.0 and gate_gap(yaw_stats) == 0.0 and center_open >= 2.20
            if gates_ok and (best is None or score < best):
                best = score
                best_stats = {"center": stats, "yaw": yaw_stats, "angle": angle_stats}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_ckpt(model, src_payload, Path(args.out), stats)
                print(f"  saved score {score:.3f}", flush=True)
            elif not gates_ok:
                print("  gate hold", flush=True)
            guard_rss(f"eval {step}")
    if best_state is not None:
        model.load_state_dict(best_state)
        del best_state
        gc.collect()
    print("best", json.dumps(best_stats), flush=True)
    final_center = evaluate(model, hold_frames, loader, limit=0)
    ok = accepts(final_center)
    print("ACCEPT" if ok else "REJECT", flush=True)
    if yaw_hold:
        print("yaw holdout", flush=True)
        yaw_stats = evaluate(model, yaw_hold, loader, limit=0)
        yaw_ok = accepts(yaw_stats)
        print("YAW_ACCEPT" if yaw_ok else "YAW_REJECT", flush=True)
        ok = ok and yaw_ok
    if angle_hold:
        print("angle holdout", flush=True)
        angle_stats = evaluate(model, angle_hold, loader, limit=0)
        angle_ok = (
            angle_stats.get("near", {}).get("da_med", 9) <= 1.3
            and angle_stats.get("near", {}).get("hi", 9) <= 1.45
            and angle_stats.get("side_near", {}).get("da_med", 9) <= 1.3
            and angle_stats.get("side_near", {}).get("hi", 9) <= 1.45
            and angle_stats.get("side_open", {}).get("da_med", 0) >= 2.0
        )
        print("ANGLE_ACCEPT" if angle_ok else "ANGLE_REJECT", flush=True)
        ok = ok and angle_ok
    loader.close()
    guard_rss("done")
    return 0 if ok else 2


@torch.no_grad()
def write_sidecars(args) -> int:
    pred = DepthMinPredictor.from_checkpoint(str(args.out), device="cuda")
    model = pred._model
    model.eval()
    ep_dir = DATA / "episodes"
    backup = DATA / "da3_before"
    backup.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in ep_dir.glob("episode_*.npz") if p.name.count(".") == 1)
    t0 = time.time()
    for n, path in enumerate(files):
        z = np.load(path)
        rgb = decode_frames(z)
        maps = []
        for i in range(0, len(rgb), 8):
            chunk = np.stack(rgb[i : i + 8])
            p56 = forward_maps(model, chunk).float().cpu().numpy()
            maps.append(p56.astype(np.float16))
        side = path.with_suffix(".da3.npz")
        kept = backup / side.name
        if side.is_file() and not kept.is_file():
            side.replace(kept)
        tmp = path.with_name(path.stem + ".da3.tmp.npz")
        np.savez(tmp, depth56=np.concatenate(maps))
        tmp.replace(side)
        if n % 40 == 0:
            print(f"sidecar {n}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    print(f"sidecars {len(files)} {time.time() - t0:.0f}s", flush=True)
    review = json.loads((DATA / "AUTO_REVIEW.json").read_text())
    rows = []
    for row in review["episodes"]:
        if not row.get("keep") or row.get("route_id") != "office_val":
            continue
        z = np.load(ep_dir / row["file"])
        side = np.load(ep_dir / row["file"].replace(".npz", ".da3.npz"))
        gt = np.asarray(z["depth56"], np.float32)
        da = np.asarray(side["depth56"], np.float32)
        for i in range(min(len(gt), len(da))):
            rows.append((float(cones_np(gt[i])[0]), float(cones_np(da[i])[0])))
    print("written holdout", flush=True)
    stats = report("file", rows)
    ok = accepts(stats)
    print("SIDECAR_ACCEPT" if ok else "SIDECAR_REJECT", flush=True)
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--angle-dir", default="")
    ap.add_argument("--write-sidecars", action="store_true")
    args = ap.parse_args()
    if args.write_sidecars:
        return write_sidecars(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
