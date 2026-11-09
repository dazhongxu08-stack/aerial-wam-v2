"""Sharpen the world-model posterior so a facade and an open street get different codes.

The actor is not updated. Wall frames are the route-1 approach that keeps the
building in front (forward depth falling through 8–28 m with one side still
open). The recorded west leg looks down an empty street, so it is not this set.
Separation is measured twice: with each frame's own pose, and again with pose
held at zero, so a code that only reads coordinates does not pass.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.aerial.rl.dataset import load_episode
from experiments.aerial.rl.depth_geometry import approach_clearances
from experiments.aerial.rl.gap_passage import passage_phase
from experiments.aerial.rl.train_rl import load_torch_dynamics


def _clear(obs):
    depth = np.asarray(obs.depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[0]
    return approach_clearances(depth, center_frac=0.12)


def _facade(tr) -> bool:
    fwd, left, right = _clear(tr.obs)
    longer = max(float(left), float(right))
    shorter = min(float(left), float(right))
    return bool(
        8.0 < float(fwd) <= 28.0
        and longer >= 35.0
        and longer >= float(fwd) + 12.0
        and shorter > 3.0
    )


def _look(transitions) -> np.ndarray:
    rows = []
    for tr in transitions:
        image = np.asarray(tr.obs.rgb, dtype=np.float32)[::4, ::4].reshape(-1)
        image = image - image.mean()
        rows.append(image / max(float(np.linalg.norm(image)), 1e-6))
    return np.stack(rows)


def _goal(transitions):
    info = getattr(transitions[0].obs, "info", None) or {}
    return np.asarray(info["goal"], dtype=np.float64).reshape(3)


def _select(street_ep, facade_eps, canyon_eps, held_eps):
    street, wall, hold_street = [], [], []
    for tr in street_ep:
        y = float(np.asarray(tr.obs.position, dtype=np.float64)[1])
        fwd = float(_clear(tr.obs)[0])
        if y <= 50.0:
            hold_street.append(tr)
        elif fwd > 20.0 and not _facade(tr):
            street.append(tr)
    wall = [tr for ep in facade_eps for tr in ep if _facade(tr)]
    street_look = _look(street)
    wall_look = _look(wall)
    # South leg of the passage is a street between buildings. It is the visual
    # neighbor of route 0's canyon, and it is not part of that held-out flight.
    extra = []
    for ep in canyon_eps:
        goal = _goal(ep)
        for tr in ep:
            if _facade(tr) or float(_clear(tr.obs)[0]) <= 18.0:
                continue
            pos = np.asarray(tr.obs.position, dtype=np.float64)
            _yaw, phase = passage_phase(pos, goal)
            if phase != "south":
                continue
            extra.append(tr)
    if extra:
        extra_look = _look(extra)
        near_wall = (extra_look @ wall_look.T).max(axis=1)
        near_street = (extra_look @ street_look.T).max(axis=1)
        extra = [
            tr
            for tr, nw, ns in zip(extra, near_wall, near_street)
            if float(nw) < float(ns) and float(nw) < 0.40
        ]
    print(f"extra_street={len(extra)}")
    street = street + extra
    street_look = _look(street)
    hold_wall = []
    for ep in held_eps:
        for tr in ep:
            view = _look([tr])[0]
            near_wall = float((wall_look @ view).max())
            near_street = float((street_look @ view).max())
            fwd = float(_clear(tr.obs)[0])
            yaw = float(tr.obs.yaw)
            # Spawn heading faces the facade. Frames that have already turned
            # west or north are a different picture and do not belong here.
            heading = abs((yaw + 2.0944 + np.pi) % (2.0 * np.pi) - np.pi)
            square = heading < 0.55 and 20.0 < fwd < 40.0
            # Drop exact copies of a training frame.
            # 0.50 keeps a second look that still shows this facade. Weaker
            # matches are a different viewpoint and are not this test.
            if square and 0.50 < near_wall <= 0.98 and near_wall > near_street:
                hold_wall.append(tr)
    return street, wall, hold_street, hold_wall


def _batch(transitions, proprio: np.ndarray | None):
    rgb = torch.as_tensor(np.stack([tr.obs.rgb for tr in transitions]))
    if proprio is None:
        prop = torch.as_tensor(np.stack([tr.obs.proprio4() for tr in transitions]))
    else:
        prop = torch.as_tensor(np.repeat(proprio.reshape(1, -1), len(transitions), axis=0))
    return rgb, prop


def _embed_scaled(model, rgb, prop, scale: float):
    """Match ``_embed``, optionally dividing the pose half by ``scale``.

    The frozen pose MLP emits values near 150 while the image encoder emits
    values near 0.07, so the posterior otherwise ignores the picture. Training
    divides the pose half; ``_bake_pose_scale`` folds that division into the
    existing linear weights so the saved model uses the normal forward.
    """
    rgb = rgb.to(model.device)
    prop = prop.to(model.device, model.torch_dtype)
    if rgb.dim() == 4 and rgb.shape[-1] in (1, 3):
        rgb = rgb.permute(0, 3, 1, 2).to(model.torch_dtype) / 255.0
    pose = model.proprio_mlp(prop)
    if float(scale) != 1.0:
        pose = pose / float(scale)
    return torch.cat([model.encoder(rgb), pose], dim=-1)


def _logits(model, rgb, prop, scale: float):
    embed = _embed_scaled(model, rgb, prop, scale)
    h = model.rssm.initial_h(rgb.shape[0], model.device, model.torch_dtype)
    raw = model.rssm.post_net(torch.cat([h, embed], dim=-1))
    return raw.view(rgb.shape[0], model.rssm.stoch_dim, model.rssm.stoch_classes).float()


def _bake_pose_scale(model, scale: float) -> None:
    linear = model.rssm.post_net[0]
    width = int(model.proprio_mlp[0].out_features)
    if linear.weight.shape[1] < width:
        raise RuntimeError("post_net input is shorter than the pose embedding")
    with torch.no_grad():
        linear.weight[:, -width:] /= float(scale)


def _probs_from_logits(model, logits):
    return model.rssm._logits_to_probs(logits.reshape(logits.shape[0], -1)).float()


def _modes(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=-1).astype(np.int16)


def _nearest(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    dist = np.stack([np.not_equal(query, row).sum(axis=1) for row in bank], axis=1)
    return dist.min(axis=1)


def _fail_indices(query: np.ndarray, prefer: np.ndarray, other: np.ndarray) -> list[int]:
    to_prefer = _nearest(query, prefer)
    to_other = _nearest(query, other)
    return [int(i) for i in np.nonzero(to_prefer >= to_other)[0]]


def _report(title, street_p, wall_p, hold_s, hold_w):
    sm, wm = _modes(street_p), _modes(wall_p)
    hm_s, hm_w = _modes(hold_s), _modes(hold_w)
    overlap = len({tuple(r) for r in wm} & {tuple(r) for r in sm})
    canyon_to_street = _nearest(hm_s, sm)
    canyon_to_wall = _nearest(hm_s, wm)
    wall_to_street = _nearest(hm_w, sm)
    wall_to_wall = _nearest(hm_w, wm)
    canyon_ok = int(np.sum(canyon_to_street < canyon_to_wall))
    held_wall_ok = int(np.sum(wall_to_wall < wall_to_street))
    maxes = [float(p.max(axis=-1).mean()) for p in (street_p, wall_p, hold_s, hold_w)]
    mean_max = float(np.mean(maxes))
    print(
        f"{title} overlap={overlap} canyon_near_street={canyon_ok}/{len(hm_s)} "
        f"held_wall_near_wall={held_wall_ok}/{len(hm_w)} "
        f"maxp={mean_max:.3f} ham_canyon_s={canyon_to_street.mean():.1f} "
        f"ham_canyon_w={canyon_to_wall.mean():.1f} "
        f"ham_wall_w={wall_to_wall.mean():.1f} ham_wall_s={wall_to_street.mean():.1f}"
    )
    ok = (
        overlap == 0
        and canyon_ok == len(hm_s)
        and held_wall_ok == len(hm_w)
        and mean_max >= 0.15
    )
    if canyon_ok != len(hm_s) or held_wall_ok != len(hm_w):
        print(
            f"{title} fail canyon={_fail_indices(hm_s, sm, wm)} "
            f"held_wall={_fail_indices(hm_w, wm, sm)}"
        )
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--street", required=True)
    p.add_argument("--facade", nargs="+", required=True)
    p.add_argument("--street-canyon", nargs="+", required=True)
    p.add_argument("--held-wall", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-5)
    args = p.parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    model, _ = load_torch_dynamics(
        cfg.get("world_model") or {}, args.wm, device="cuda", freeze=False
    )
    street_ep = load_episode(Path(args.street))
    facade_eps = [load_episode(Path(path)) for path in args.facade]
    canyon_eps = [load_episode(Path(path)) for path in args.street_canyon]
    held_eps = [load_episode(Path(path)) for path in args.held_wall]
    street, wall, hold_s, hold_w = _select(street_ep, facade_eps, canyon_eps, held_eps)
    print(
        "hold_wall",
        " ".join(
            f"{float(tr.obs.position[1]):.0f}/{float(tr.obs.yaw):.2f}" for tr in hold_w
        ),
    )
    print(
        f"train street={len(street)} wall={len(wall)} "
        f"hold_street={len(hold_s)} hold_wall={len(hold_w)}"
    )
    print(
        "canyon_y",
        " ".join(f"{float(tr.obs.position[1]):.0f}" for tr in hold_s),
    )
    if len(street) < 10 or len(wall) < 8 or len(hold_s) < 5 or len(hold_w) < 5:
        raise SystemExit("refuse: not enough frames")
    # Held-out wall frames are the 27 m approach, not the contact frames.
    # The held-out misses are the spawn view: building ahead, both sides still open.
    # The later approach has already closed one side, so it must not define the class.
    open_wall = [
        tr
        for tr in wall
        if min(float(_clear(tr.obs)[1]), float(_clear(tr.obs)[2])) >= 30.0
    ]
    wall_fit = wall + open_wall * 8
    band = [
        tr
        for tr in street
        if float(np.asarray(tr.obs.position)[1]) <= 42.0
    ]
    street_fit = street + band * 2
    print(
        f"wall_fit={len(wall_fit)} open={len(open_wall)} "
        f"street_fit={len(street_fit)} band={len(band)}"
    )

    for param in model.parameters():
        param.requires_grad_(False)
    for module in (model.encoder, model.rssm.post_net):
        for param in module.parameters():
            param.requires_grad_(True)
    opt = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=float(args.lr)
    )
    rgb_s, prop_s = _batch(street_fit, None)
    rgb_w, prop_w = _batch(wall_fit, None)
    zeros = np.zeros(4, dtype=np.float32)
    rgb_sz, prop_sz = _batch(street_fit, zeros)
    rgb_wz, prop_wz = _batch(wall_fit, zeros)
    with torch.no_grad():
        pose_ref = model.proprio_mlp(prop_s.to(model.device, model.torch_dtype)).float()
        scale = float(pose_ref.std().clamp_min(1.0))
    print(f"pose_scale={scale:.1f}")

    def _center(rgb: torch.Tensor) -> torch.Tensor:
        _n, height, width, _c = rgb.shape
        dh, dw = max(1, int(height * 0.5)), max(1, int(width * 0.5))
        r0, c0 = (height - dh) // 2, (width - dw) // 2
        crop = rgb[:, r0 : r0 + dh, c0 : c0 + dw].float()
        image = F.interpolate(
            crop.permute(0, 3, 1, 2),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return image.permute(0, 2, 3, 1)

    def _ce(rgb, prop, class_id: int):
        logits = _logits(model, rgb, prop, scale)
        target = torch.full(
            (logits.shape[0], logits.shape[1]),
            int(class_id),
            device=logits.device,
            dtype=torch.long,
        )
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        acc = float((logits.argmax(-1) == target).float().mean().detach())
        return loss, acc

    model.train()
    for step in range(int(args.steps)):
        opt.zero_grad()
        loss_s, acc_s = _ce(rgb_s, prop_s, 0)
        loss_w, acc_w = _ce(rgb_w, prop_w, 1)
        loss_sz, acc_sz = _ce(rgb_sz, prop_sz, 0)
        loss_wz, acc_wz = _ce(rgb_wz, prop_wz, 1)
        loss_sc, _acc_sc = _ce(_center(rgb_sz), prop_sz, 0)
        loss_wc, _acc_wc = _ce(_center(rgb_wz), prop_wz, 1)
        loss = loss_s + loss_w + loss_sz + loss_wz + 0.5 * (loss_sc + loss_wc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad], 5.0
        )
        opt.step()
        if step == 0 or step + 1 == int(args.steps) or (step + 1) % 50 == 0:
            print(
                f"step={step} loss={float(loss.detach()):.4f} "
                f"acc_pose={acc_s:.2f}/{acc_w:.2f} acc_rgb={acc_sz:.2f}/{acc_wz:.2f}"
            )

    _bake_pose_scale(model, scale)
    model.eval()

    def _eval(proprio_fixed: bool):
        fixed_prop = zeros if proprio_fixed else None
        groups = []
        with torch.no_grad():
            for group in (street, wall, hold_s, hold_w):
                rgb, prop = _batch(group, fixed_prop)
                logits = _logits(model, rgb, prop, 1.0)
                groups.append(
                    _probs_from_logits(model, logits).detach().float().cpu().numpy()
                )
        return groups

    ok_pose = _report("pose", *_eval(False))
    ok_fixed = _report("rgb_only", *_eval(True))
    if ok_pose and ok_fixed:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(str(out), step=int(args.steps))
        print(f"PASS saved {out}")
    else:
        print("FAIL posterior gate")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
