#!/usr/bin/env python3
"""Fine-tune the DA3 depth decoder on SimWorld dog frames.

The encoder stays frozen. Routes in ``--val-routes`` are held out. The score is
how well the forward clearance matches the simulator on those routes, including
the contact frame of each collision. Original episode sidecars are not modified;
``--sidecar-dir`` receives symlinked episodes plus new ``*.da3.npz`` files.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("finetune_dog_depth")


def _detach(feats):
    import torch

    if torch.is_tensor(feats):
        return feats.detach()
    if isinstance(feats, dict):
        return {k: _detach(v) for k, v in feats.items()}
    if isinstance(feats, (list, tuple)):
        return type(feats)(_detach(v) for v in feats)
    return feats


def pool_min_t(depth, out: int = 56):
    b, h, w = depth.shape
    fh, fw = h // out, w // out
    d = depth[:, : fh * out, : fw * out].reshape(b, out, fh, out, fw)
    return d.min(dim=2).values.min(dim=3).values


def forward_cone(depth56) -> "np.ndarray":
    from experiments.aerial.rl.consequence_model import cones3_from_depth56

    return np.array([cones3_from_depth56(d)[0] for d in depth56], np.float64)


def decode_frame(z, i: int):
    import cv2

    off = z["jpeg_offsets"]
    bgr = cv2.imdecode(z["jpeg"][off[i]:off[i + 1]], cv2.IMREAD_COLOR)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def index_frames(episodes: List[Path], review_rows: Dict[str, dict], val: set, split: str):
    rows = []
    for p in episodes:
        meta = review_rows.get(p.name)
        if meta is None:
            continue
        if (meta["route_id"] in val) != (split == "val"):
            continue
        z = np.load(p)
        n = len(z["jpeg_offsets"]) - 1
        coll = np.flatnonzero(np.asarray(z["collided"]).astype(bool))
        gt = np.asarray(z["depth56"])
        want = set(int(i) for i in coll if i < n)
        # spread the rest across the episode
        step = max(1, n // 30)
        want.update(range(0, n, step))
        for i in sorted(want):
            if i >= len(gt):
                continue
            rows.append((p, i, meta["route_id"] in val and i in set(int(x) for x in coll)))
        z.close()
    return rows


def load_batch(rows, model, device):
    import cv2
    import torch

    size = int(model.image_size)
    rgbs, gts = [], []
    for path, i, _contact in rows:
        z = np.load(path)
        rgb = decode_frame(z, i)
        gt = np.asarray(z["depth56"][i], np.float32)
        z.close()
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        rgbs.append(rgb)
        gts.append(gt)
    rgb_t = torch.from_numpy(np.stack(rgbs)).to(device)
    gt_t = torch.from_numpy(np.stack(gts)).to(device)
    return rgb_t, gt_t


def predict_pooled(model, rgb_nhwc):
    import torch

    window = rgb_nhwc.unsqueeze(1)
    x = model._prep_last_frame(window)
    with torch.no_grad():
        feats, _aux = model.encoder(x.unsqueeze(1), cam_token=None, export_feat_layers=[])
    feats = _detach(feats)
    out = model.decoder(feats, x.shape[-2], x.shape[-1], patch_start_idx=0)
    depth = out["depth"][:, 0]
    if depth.shape[-2:] != (model.image_size, model.image_size):
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(model.image_size, model.image_size),
            mode="bilinear", align_corners=False)[:, 0]
    return pool_min_t(depth.clamp(min=1e-3))


def cone_stats(pred56, gt56) -> Dict[str, float]:
    p = forward_cone(pred56)
    g = forward_cone(gt56)
    if len(p) < 3 or np.std(p) < 1e-6 or np.std(g) < 1e-6:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(np.log(np.clip(p, 0.05, None)), np.log(np.clip(g, 0.05, None)))[0, 1])
    return {
        "n": int(len(p)),
        "corr": corr,
        "mae": float(np.mean(np.abs(p - g))),
        "med_p": float(np.median(p)),
        "med_g": float(np.median(g)),
    }


def evaluate(model, rows, device, batch: int) -> Dict[str, float]:
    import torch

    model.eval()
    pred, gt, contact = [], [], []
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        rgb, gt_t = load_batch(chunk, model, device)
        with torch.no_grad():
            pr = predict_pooled(model, rgb)
        pred.append(pr.float().cpu().numpy())
        gt.append(gt_t.float().cpu().numpy())
        contact.extend(bool(r[2]) for r in chunk)
    pred_a = np.concatenate(pred)
    gt_a = np.concatenate(gt)
    contact = np.array(contact, bool)
    out = {"all": cone_stats(pred_a, gt_a)}
    if contact.any():
        out["contact"] = cone_stats(pred_a[contact], gt_a[contact])
    return out


def write_sidecars(model, episodes: List[Path], dest: Path, device: str, batch: int) -> None:
    import torch

    ep_dir = dest / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for p in episodes:
        link = ep_dir / p.name
        if not link.exists():
            link.symlink_to(p)
        side = ep_dir / f"{p.stem}.da3.npz"
        z = np.load(p)
        n = len(z["jpeg_offsets"]) - 1
        frames = [(p, i, False) for i in range(n)]
        z.close()
        maps = []
        for s in range(0, len(frames), batch):
            rgb, _gt = load_batch(frames[s:s + batch], model, device)
            with torch.no_grad():
                pr = predict_pooled(model, rgb)
            maps.append(pr.float().cpu().numpy().astype(np.float16))
        np.savez(side, depth56=np.concatenate(maps))
        logger.info("sidecar %s frames %d", p.name, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--val-routes", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sidecar-dir", default="")
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    from experiments.aerial.rl.dynamics_torch import build_depth_head

    data = Path(args.data_dir)
    if not data.is_absolute():
        data = root / data
    review = json.loads((data / "AUTO_REVIEW.json").read_text())
    rows_by = {e["file"]: e for e in review["episodes"] if e.get("keep") and e.get("file")}
    val = set(r for r in args.val_routes.split(",") if r)
    episodes = sorted(p for p in (data / "episodes").glob("episode_*.npz")
                      if re.fullmatch(r"episode_\d+\.npz", p.name) and p.name in rows_by)
    train_rows = index_frames(episodes, rows_by, val, "train")
    val_rows = index_frames(episodes, rows_by, val, "val")
    logger.info("frames train %d val %d (contact %d)", len(train_rows), len(val_rows),
                sum(1 for r in val_rows if r[2]))

    payload = torch.load(args.init_ckpt, map_location="cpu")
    model = build_depth_head(payload)
    model.load_state_dict(payload["model"], strict=True)
    model.to(args.device)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.encoder.eval()
    opt = torch.optim.AdamW([p for p in model.decoder.parameters() if p.requires_grad], lr=args.lr)
    rng = np.random.default_rng(0)
    base = evaluate(model, val_rows, args.device, args.batch)
    logger.info("baseline %s", json.dumps(base))
    best = base.get("contact", base["all"]).get("corr", -1.0)
    if best != best:
        best = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return _train(args, model, opt, payload, train_rows, val_rows, base, best, best_state, rng, episodes)


def _train(args, model, opt, payload, train_rows, val_rows, base, best, best_state, rng, episodes) -> int:
    import torch

    for step in range(1, args.steps + 1):
        model.decoder.train()
        idx = rng.choice(len(train_rows), size=min(args.batch, len(train_rows)), replace=False)
        rgb, gt = load_batch([train_rows[int(i)] for i in idx], model, args.device)
        pred = predict_pooled(model, rgb).clamp(min=0.05)
        gt_c = gt.clamp(min=0.05)
        log_p = torch.log(pred)
        log_g = torch.log(gt_c)
        # forward middle band, same columns as cones3
        band_p = pred[:, 14:42, 14:42].amin(dim=(1, 2)).clamp(min=0.05)
        band_g = gt_c[:, 14:42, 14:42].amin(dim=(1, 2)).clamp(min=0.05)
        near = (band_g < 3.0).float()
        w = 1.0 + 3.0 * near
        loss_map = (torch.nn.functional.smooth_l1_loss(log_p, log_g, reduction="none").mean(dim=(1, 2)) * w).mean()
        loss_fwd = (torch.nn.functional.smooth_l1_loss(torch.log(band_p), torch.log(band_g), reduction="none") * w).mean()
        loss = loss_map + loss_fwd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            stats = evaluate(model, val_rows, args.device, args.batch)
            corr = stats.get("contact", stats["all"]).get("corr", float("nan"))
            logger.info("step %d loss %.4f %s", step, float(loss.detach()), json.dumps(stats))
            if corr == corr and corr > best:
                best = corr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload = dict(payload)
    payload["model"] = best_state
    out = Path(args.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    logger.info("saved %s best_contact_corr %.3f baseline %s", out, best, json.dumps(base.get("contact", base["all"])))
    model.load_state_dict(best_state)
    model.to(args.device)
    if args.sidecar_dir and best > base.get("contact", base["all"]).get("corr", -1.0) + 0.05:
        dest = Path(args.sidecar_dir)
        if not dest.is_absolute():
            dest = Path(__file__).resolve().parents[3] / dest
        review_src = Path(args.data_dir)
        if not review_src.is_absolute():
            review_src = Path(__file__).resolve().parents[3] / review_src
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / "AUTO_REVIEW.json"
        if not target.exists():
            target.symlink_to(review_src / "AUTO_REVIEW.json")
        write_sidecars(model, episodes, dest, args.device, args.batch)
        logger.info("sidecars %s", dest)
    else:
        logger.info("sidecars skipped (contact corr did not clear the baseline by 0.05)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
