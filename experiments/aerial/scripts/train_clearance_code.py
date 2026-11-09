"""Train the posterior so its mode carries forward, left, and right clearance.

Depth is a training label only. The actor is not updated. Nothing is flown.
The readout is linear in the 32-by-32 mode, which is the code the policy sees.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.train_rl import load_torch_dynamics
from experiments.aerial.scripts.train_posterior_gate import (
    _batch,
    _clear,
    _embed_scaled,
)


def _targets(frames) -> np.ndarray:
    rows = []
    for tr in frames:
        fwd, left, right = _clear(tr.obs)
        vals = []
        for v in (fwd, left, right):
            if not np.isfinite(v) or v <= 0:
                v = 80.0
            vals.append(np.log1p(min(float(v), 80.0)))
        rows.append(vals)
    return np.asarray(rows, dtype=np.float32)


def _code(model, rgb, prop, scale: float, *, pose_off: bool = False) -> torch.Tensor:
    embed = _embed_scaled(model, rgb, prop, scale)
    if pose_off:
        width = int(model.proprio_mlp[0].out_features)
        embed = embed.clone()
        embed[:, -width:] = 0
    h = model.rssm.initial_h(rgb.shape[0], model.device, model.torch_dtype)
    logits = model.rssm.post_net(torch.cat([h, embed], dim=-1))
    probs = model.rssm._logits_to_probs(logits)
    hard = F.one_hot(probs.argmax(-1), probs.shape[-1]).to(probs.dtype)
    straight = hard + probs - probs.detach()
    return straight.reshape(probs.shape[0], -1)


def _mirror_batch(rgb, prop, target):
    """Show both sides. A horizontal flip swaps the left and right labels.

    World xy is jittered so the posterior cannot index the route by position.
    Yaw flips with the picture. The held-out canyon is never passed in here.
    """
    rgb = rgb.clone()
    prop = prop.clone().float()
    target = target.clone()
    flip = torch.rand(rgb.shape[0]) < 0.5
    if bool(flip.any()):
        rgb[flip] = torch.flip(rgb[flip], dims=[-1])
        swapped = target[flip, 1].clone()
        target[flip, 1] = target[flip, 2]
        target[flip, 2] = swapped
        prop[flip, 3] = -prop[flip, 3]
    prop[:, 0:2] = prop[:, 0:2] + torch.randn(prop.shape[0], 2) * 30.0
    return rgb, prop, target


def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    meter_gap = torch.expm1(target[:, 1]) - torch.expm1(target[:, 2])
    big = meter_gap.abs() >= 8.0
    if not bool(big.any()):
        return mse
    sign = torch.where(
        meter_gap[big] >= 0, torch.ones_like(meter_gap[big]), -torch.ones_like(meter_gap[big])
    )
    side = pred[big, 1] - pred[big, 2]
    hinge = F.relu(0.35 - sign * side).mean()
    return mse + hinge


def _metrics(pred: np.ndarray, raw: np.ndarray) -> tuple[float, float, int]:
    pred_m = np.expm1(pred)
    sign_ok = 0
    sign_n = 0
    for p, t in zip(pred_m, raw):
        gap = float(t[1] - t[2])
        if abs(gap) < 8.0:
            continue
        sign_n += 1
        if (float(p[1] - p[2])) * gap > 0:
            sign_ok += 1
    acc = float(sign_ok) / float(sign_n) if sign_n else 0.0
    if len(raw) > 2 and np.std(raw[:, 0]) > 1e-6 and np.std(pred_m[:, 0]) > 1e-6:
        corr = float(np.corrcoef(pred_m[:, 0], raw[:, 0])[0, 1])
    else:
        corr = 0.0
    return acc, corr, sign_n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--street", required=True)
    p.add_argument("--passage", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    model, _ = load_torch_dynamics(
        cfg.get("world_model") or {}, args.wm, device="cuda", freeze=False
    )
    street_ep = load_episode(Path(args.street))
    train, held = [], []
    for tr in street_ep:
        if float(np.asarray(tr.obs.position)[1]) <= 50.0:
            held.append(tr)
        else:
            train.append(tr)
    train.extend(load_episode(Path(args.passage)))
    held_keys = []
    for tr in held:
        pos = np.asarray(tr.obs.position)
        held_keys.append((float(pos[0]), float(pos[1]), float(tr.obs.yaw)))
    extra_n = 0
    for item in args.extra:
        path = Path(item)
        files = sorted(path.glob("episode_*.npz")) if path.is_dir() else [path]
        for ep_path in files:
            for tr in load_episode(ep_path):
                if getattr(tr.obs, "depth", None) is None:
                    continue
                pos = np.asarray(tr.obs.position)
                yaw = float(tr.obs.yaw)
                leak = False
                for hx, hy, hyaw in held_keys:
                    dyaw = (yaw - hyaw + np.pi) % (2 * np.pi) - np.pi
                    if (float(pos[0]) - hx) ** 2 + (float(pos[1]) - hy) ** 2 < 4.0 and abs(dyaw) < 0.25:
                        leak = True
                        break
                if leak:
                    continue
                train.append(tr)
                extra_n += 1
    print(f"train={len(train)} extra={extra_n} held_canyon={len(held)}")
    if len(train) < 20 or len(held) < 10:
        raise SystemExit("refuse: not enough frames")

    for param in model.parameters():
        param.requires_grad_(False)
    for module in (model.encoder, model.rssm.post_net):
        for param in module.parameters():
            param.requires_grad_(True)
    readout = nn.Linear(model.rssm.stoch_dim * model.rssm.stoch_classes, 3).to(model.device)
    opt = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad]
        + list(readout.parameters()),
        lr=float(args.lr),
    )
    rgb, prop = _batch(train, None)
    target = torch.as_tensor(_targets(train), device=model.device)
    with torch.no_grad():
        pose_ref = model.proprio_mlp(prop.to(model.device, model.torch_dtype)).float()
        scale = float(pose_ref.std().clamp_min(1.0))
    print(f"pose_scale={scale:.1f}")

    model.train()
    batch = max(1, int(args.batch))
    for step in range(int(args.steps)):
        opt.zero_grad()
        pick = torch.randint(0, rgb.shape[0], (min(batch, rgb.shape[0]),))
        rgb_s, prop_s, target_s = _mirror_batch(rgb[pick], prop[pick], target[pick].cpu())
        pred = readout(_code(model, rgb_s, prop_s, scale, pose_off=True).float())
        loss = _loss(pred, target_s.to(pred.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad]
            + list(readout.parameters()),
            5.0,
        )
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 50 == 0:
            print(f"step={step} loss={float(loss.detach()):.4f}")

    width = int(model.proprio_mlp[0].out_features)
    with torch.no_grad():
        model.rssm.post_net[0].weight[:, -width:] = 0
    model.eval()
    readout.eval()

    def _eval(frames):
        preds, raws = [], []
        for start in range(0, len(frames), 64):
            chunk = frames[start : start + 64]
            rgb_e, prop_e = _batch(chunk, None)
            with torch.no_grad():
                pred = readout(_code(model, rgb_e, prop_e, 1.0, pose_off=True).float())
            preds.append(pred.cpu().numpy())
            raws.append(np.stack([_clear(tr.obs) for tr in chunk]))
        return np.concatenate(preds), np.concatenate(raws)

    pred, raw = _eval(held)
    acc, corr, sign_n = _metrics(pred, raw)
    mae = float(np.mean(np.abs(np.expm1(pred[:, 0]) - raw[:, 0])))
    gap = raw[:, 1] - raw[:, 2]
    pred_gap = np.expm1(pred[:, 1]) - np.expm1(pred[:, 2])
    big = np.abs(gap) >= 8.0
    left = big & (gap > 0)
    right = big & (gap < 0)
    left_ok = int(((pred_gap > 0) & left).sum())
    right_ok = int(((pred_gap < 0) & right).sum())
    train_pred, train_raw = _eval(train)
    train_acc, train_corr, train_n = _metrics(train_pred, train_raw)
    print(
        f"train sign_acc={train_acc:.2f} n={train_n} fwd_corr={train_corr:.2f}"
    )
    print(
        f"held sign_acc={acc:.2f} n={sign_n} "
        f"left={left_ok}/{int(left.sum())} right={right_ok}/{int(right.sum())} "
        f"fwd_corr={corr:.2f} fwd_mae_m={mae:.1f}"
    )
    ok = acc >= 0.8 and corr >= 0.5 and sign_n >= 10
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if ok:
        model.save_checkpoint(str(out), step=int(args.steps))
        torch.save({"readout": readout.state_dict()}, str(out.with_name("clearance_readout.pt")))
        print(f"PASS saved {out}")
    else:
        print("FAIL clearance code")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
