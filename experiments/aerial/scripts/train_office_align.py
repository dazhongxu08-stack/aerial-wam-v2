#!/usr/bin/env python3
"""Align the car depth head with the sim office head, then refit consequence.

Both sides use the chassis hide line and the upper-cone loss. Real frames
are supervised by RealSense. Sim wall frames are supervised by simulator
depth. A new depth file and a new consequence file are written. The live
car head, the wall head, and the sim gap head stay in place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_office_train as col  # noqa: E402
import train_office_da3_far as far  # noqa: E402
from experiments.aerial.rl.consequence_model import (  # noqa: E402
    depth_input,
    goal_features,
    load_checkpoint,
    save_checkpoint,
)
from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402
from experiments.aerial.rl.env.obs import Observation  # noqa: E402
from office_bands import cones3_forward  # noqa: E402

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
DEPTH_INIT = ART / "da3_office_real_20261103/depth_office_cone_real.pt"
DEPTH_OUT = ART / "da3_office_align_20261104/depth_office_cone_align.pt"
CONS_INIT = ART / "indoor_office_gap_20261104/c_s218/consequence_best.pt"
CONS_OUT = ART / "indoor_office_align_20261104/c_s218"
REAL_DIR = Path("/data/simworld/run/keep_20261104")
SIM_LIVE = Path("/data/simworld/run/office_walldepth_20261104/live.npz")
CACHE = Path("/data/simworld/run/office_align_20261104/real.npz")
YAW = 0.3141592653589793
DEPTH_STEPS = 60
CONS_STEPS = 80


def load_real():
    if CACHE.is_file():
        z = np.load(CACHE)
        return z["rgb"], z["gt56"]
    rgb, gt56 = [], []
    for path in sorted(REAL_DIR.glob("frame_*.npz")):
        z = np.load(path)
        real = z["depth"].astype(np.float32)
        real[real <= 0] = np.nan
        seen = z["seen"].astype(bool)
        gt56.append(np.asarray(col.pool_min(col.hide(real, seen)), np.float32))
        rgb.append(np.ascontiguousarray(z["rgb"]))
    rgb = np.stack(rgb)
    gt56 = np.stack(gt56)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, rgb=rgb, gt56=gt56)
    return rgb, gt56


def cone_rows(da56: np.ndarray, gt56: np.ndarray) -> tuple[float, float, float]:
    short, over, side = [], [], []
    for dmap, gmap in zip(da56, gt56):
        d = cones3_forward(dmap)
        g = cones3_forward(gmap)
        if g[0] >= 4.0:
            short.append(max(0.0, np.log(g[0]) - np.log(max(d[0], 0.05))))
        if g[0] <= 1.2:
            over.append(max(0.0, np.log(max(d[0], 0.05)) - np.log(max(g[0], 0.05))))
        for i in (1, 2):
            if g[i] < 1.0 and g[0] > 2.0:
                side.append(max(0.0, np.log(max(d[i], 0.05)) - np.log(max(g[i], 0.05))))
    return (
        float(np.mean(short) if short else 0.0),
        float(np.mean(over) if over else 0.0),
        float(np.mean(side) if side else 0.0),
    )


def predict_56(model, rgb: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(rgb), 4):
        pred = far.forward_maps(model, rgb[i:i + 4]).detach().float().cpu().numpy()
        out.append(pred)
    return np.concatenate(out)


def domain_score(model, rgb, gt56, tag: str) -> tuple[float, tuple[float, float, float]]:
    model.eval()
    da56 = predict_56(model, rgb)
    parts = cone_rows(da56, gt56)
    score = parts[0] + parts[1] + parts[2]
    print(f"{tag} short {parts[0]:.3f} over {parts[1]:.3f} side {parts[2]:.3f} sum {score:.3f}", flush=True)
    model.encoder.eval()
    model.decoder.train()
    return score, parts


def side_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    h, w = pred.shape[-2:]
    hi = min(int(far.HZ56), h - 1)
    lo = max(h // 8, 0)

    def p20(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        k = max(1, int(round(0.20 * (flat.shape[1] - 1))))
        return flat.kthvalue(k + 1, dim=-1).values.clamp(0.05, far.FAR)

    fwd_g = p20(gt[:, lo:hi, w // 4 : w - w // 4])
    loss = pred.new_zeros(())
    n = 0
    for sl in (slice(0, w // 3), slice(2 * w // 3, w)):
        pg = p20(gt[:, lo:hi, sl])
        pp = p20(pred[:, lo:hi, sl])
        close = (pg < 1.0) & (fwd_g > 2.0)
        if bool(close.any()):
            loss = loss + torch.relu(torch.log(pp[close]) - torch.log(pg[close])).mean()
            n += 1
    return loss / max(n, 1)


def train_depth() -> None:
    real_rgb, real_gt = load_real()
    sim = np.load(SIM_LIVE)
    sim_rgb, sim_gt = sim["rgb"], sim["gt56"].astype(np.float32)
    print(f"real {len(real_rgb)} sim {len(sim_rgb)} hide {col.HIDE_ROW}", flush=True)
    wrap = DepthMinPredictor.from_checkpoint(str(DEPTH_INIT), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    model = wrap._model
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    model.encoder.eval()
    src = torch.load(str(DEPTH_INIT), map_location="cpu", weights_only=False)
    real0, _ = domain_score(model, real_rgb, real_gt, "real before")
    sim0, _ = domain_score(model, sim_rgb, sim_gt, "sim before")
    opt = torch.optim.AdamW((p for p in model.decoder.parameters() if p.requires_grad), lr=3e-6)
    rng = np.random.default_rng(4)
    best = real0 + sim0
    best_real = real0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for step in range(1, DEPTH_STEPS + 1):
        model.decoder.train()
        ri = rng.choice(len(real_rgb), size=4, replace=False)
        si = rng.choice(len(sim_rgb), size=4, replace=False)
        batch_rgb = np.concatenate([real_rgb[ri], sim_rgb[si]])
        batch_gt = np.concatenate([real_gt[ri], sim_gt[si]])
        pred = far.forward_maps(model, batch_rgb)
        gt_t = torch.as_tensor(batch_gt, dtype=torch.float32, device="cuda")
        loss = far.far_near_loss(pred[:4], gt_t[:4]) + far.far_near_loss(pred[4:], gt_t[4:])
        loss = loss + side_loss(pred[:4], gt_t[:4]) + side_loss(pred[4:], gt_t[4:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == DEPTH_STEPS:
            real_s, _ = domain_score(model, real_rgb, real_gt, f"real step {step}")
            sim_s, _ = domain_score(model, sim_rgb, sim_gt, f"sim step {step}")
            total = real_s + sim_s
            far.save_ckpt(model, src, DEPTH_OUT.with_name("depth_office_cone_align_last.pt"), {"step": step})
            if total < best and real_s <= real0 + 0.05:
                best = total
                best_real = real_s
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                far.save_ckpt(model, src, DEPTH_OUT, {"step": step, "real": real_s, "sim": sim_s})
                print(f"DEPTH_SAVE {step} real {real_s:.3f} sim {sim_s:.3f}", flush=True)
    if not DEPTH_OUT.is_file():
        model.load_state_dict(best_state)
        far.save_ckpt(model, src, DEPTH_OUT, {"step": DEPTH_STEPS, "fallback": True, "real": best_real})
        print("DEPTH_SAVE_LAST", flush=True)
    print(f"DEPTH_BEST real0 {real0:.3f} sim0 {sim0:.3f} OUT {DEPTH_OUT}", flush=True)


def pack(depth, goal, device):
    d = torch.as_tensor(depth, dtype=torch.float32, device=device)
    g = torch.as_tensor(goal, dtype=torch.float32, device=device)
    prev = torch.zeros(d.shape[0], 4, device=device)
    return d, g, prev


def logits(model, depth, goal, prev, dx, dyaw):
    actions = torch.zeros(depth.shape[0], 10, 4, device=depth.device)
    actions[:, 0, 0] = dx
    actions[:, 0, 3] = dyaw
    return model(depth, goal, prev, actions)["hazard_logit"][:, 0]


def encode_split(depth_model, rgb, gt56, limit: int | None = None):
    idx = range(len(rgb) if limit is None else min(limit, len(rgb)))
    depth, cones = [], []
    state = np.array([0.0, 0.0, 0.34, 0, 0, 0, 0.0], np.float64)
    goal_v = goal_features(state, np.array([8.0, 0.0, 0.34], np.float64))
    for i in idx:
        depth_model.reset()
        pred = depth_model.predict_depth(Observation(
            rgb=np.ascontiguousarray(rgb[i]), state=np.zeros(7, np.float32), collided=False,
        ))
        pred = np.asarray(pred, np.float32)
        if pred.shape[:2] != (224, 224):
            import cv2
            pred = cv2.resize(pred, (224, 224), interpolation=cv2.INTER_LINEAR)
        seen = rgb[i].sum(axis=-1) > 12
        da56 = col.pool_min(col.hide(pred, seen))
        depth.append(depth_input(np.asarray(da56, np.float32)))
        cones.append(np.asarray(cones3_forward(gt56[i]), np.float32))
    return (
        np.stack(depth).astype(np.float32),
        np.stack(cones).astype(np.float32),
        np.repeat(goal_v[None], len(depth), 0).astype(np.float32),
    )


def train_cons() -> None:
    wrap = DepthMinPredictor.from_checkpoint(str(DEPTH_OUT), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    real_rgb, real_gt = load_real()
    sim = np.load(SIM_LIVE)
    real_d, real_c, real_g = encode_split(wrap, real_rgb, real_gt)
    sim_d, sim_c, sim_g = encode_split(wrap, sim["rgb"], sim["gt56"].astype(np.float32))
    del wrap
    torch.cuda.empty_cache()
    dev = torch.device("cuda")
    splits = {
        "real": pack(real_d, real_g, dev) + (torch.as_tensor(real_c, device=dev),),
        "sim": pack(sim_d, sim_g, dev) + (torch.as_tensor(sim_c, device=dev),),
    }
    masks = {}
    for name, (depth, goal, prev, cones) in splits.items():
        fwd, left, right = cones[:, 0], cones[:, 1], cones[:, 2]
        side_l = (left < 1.0) & (fwd > 2.0) & (right > 1.2)
        side_r = (right < 1.0) & (fwd > 2.0) & (left > 1.2)
        masks[name] = {
            "safe": (fwd > 2.0) & ~(side_l | side_r),
            "open": (fwd > 2.5) & (left > 1.5) & (right > 1.5),
            "hit": fwd < 1.15,
            "side_l": side_l,
            "side_r": side_r,
        }
        m = masks[name]
        print(
            f"{name} n {len(fwd)} open {int(m['open'].sum())} hit {int(m['hit'].sum())} "
            f"side_l {int(side_l.sum())} side_r {int(side_r.sum())}",
            flush=True,
        )
    model = load_checkpoint(str(CONS_INIT), "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    def report(step: int):
        model.eval()
        stats = {}
        with torch.no_grad():
            for name, (depth, goal, prev, _cones) in splits.items():
                zf = logits(model, depth, goal, prev, 0.8, 0.0)
                zl = logits(model, depth, goal, prev, 0.0, -YAW)
                zr = logits(model, depth, goal, prev, 0.0, YAW)
                sf = 1.0 - torch.sigmoid(zf)
                sl = 1.0 - torch.sigmoid(zl)
                sr = 1.0 - torch.sigmoid(zr)
                m = masks[name]
                away = []
                if bool(m["side_l"].any()):
                    away.append(sr[m["side_l"]])
                if bool(m["side_r"].any()):
                    away.append(sl[m["side_r"]])
                side = m["side_l"] | m["side_r"]
                row = {
                    "open": float(sf[m["open"]].min()) if bool(m["open"].any()) else 0.0,
                    "spin": float(torch.maximum(sl, sr)[m["open"]].max()) if bool(m["open"].any()) else 1.0,
                    "hit": float(sf[m["hit"]].max()) if bool(m["hit"].any()) else 1.0,
                    "side": float(sf[side].max()) if bool(side.any()) else 1.0,
                    "away": float(torch.cat(away).min()) if away else 0.0,
                }
                stats[name] = row
                print(
                    f"step {step} {name} open {row['open']:.3f} spin {row['spin']:.3f} "
                    f"hit {row['hit']:.3f} side {row['side']:.3f} away {row['away']:.3f}",
                    flush=True,
                )
        return stats

    def ok(stats) -> bool:
        for row in stats.values():
            if not (row["open"] >= 0.95 and row["spin"] <= 0.5 and row["hit"] <= 0.2 and row["side"] <= 0.3 and row["away"] >= 0.95):
                return False
        return True

    CONS_OUT.mkdir(parents=True, exist_ok=True)
    report(0)
    best = None
    for step in range(1, CONS_STEPS + 1):
        model.train()
        loss = torch.zeros((), device=dev)
        for name, (depth, goal, prev, _cones) in splits.items():
            m = masks[name]
            z_fwd = logits(model, depth, goal, prev, 0.8, 0.0)
            z_left = logits(model, depth, goal, prev, 0.0, -YAW)
            z_right = logits(model, depth, goal, prev, 0.0, YAW)
            if bool(m["safe"].any()):
                loss = loss + torch.relu(z_fwd[m["safe"]] + 2.0).pow(2).mean()
            if bool(m["open"].any()):
                loss = loss + torch.relu(z_fwd[m["open"]] + 3.0).pow(2).mean()
                loss = loss + torch.relu(2.0 - z_left[m["open"]]).pow(2).mean()
                loss = loss + torch.relu(2.0 - z_right[m["open"]]).pow(2).mean()
            if bool(m["hit"].any()):
                loss = loss + 4.0 * torch.relu(2.0 - z_fwd[m["hit"]]).pow(2).mean()
            side = m["side_l"] | m["side_r"]
            if bool(side.any()):
                loss = loss + 4.0 * torch.relu(2.0 - z_fwd[side]).pow(2).mean()
            if bool(m["side_l"].any()):
                loss = loss + 4.0 * torch.relu(z_right[m["side_l"]] + 2.0).pow(2).mean()
            if bool(m["side_r"].any()):
                loss = loss + 4.0 * torch.relu(z_left[m["side_r"]] + 2.0).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == CONS_STEPS:
            stats = report(step)
            save_checkpoint(CONS_OUT / "consequence_last.pt", model, {"step": step})
            if ok(stats) and (best is None or stats["real"]["open"] > best):
                best = stats["real"]["open"]
                save_checkpoint(CONS_OUT / "consequence_best.pt", model, {"step": step, "real_open": best})
                print("CONS_SAVE", step, flush=True)
    if best is None:
        save_checkpoint(CONS_OUT / "consequence_best.pt", model, {"step": CONS_STEPS, "fallback": True})
        print("CONS_SAVE_LAST", flush=True)
    print("CONS_BEST", best, "OUT", CONS_OUT, flush=True)


def main() -> int:
    train_depth()
    train_cons()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
