"""A body-frame map of features from depth views already seen.

Each view keeps the spatial features of its depth image. The pose at capture
places those features in the world, and the current pose brings them into the
body. A cell keeps the feature of the nearest surface, not a range. The map is
not a survival score: the frozen head still reads only the current depth, and
nothing here deletes a candidate.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import nn

from experiments.aerial.rl.body_surround import (
    CAM_DZ,
    CELL_M,
    HALF_M,
    HFOV_DEG,
    N,
    VFOV_DEG,
    body_from_world,
)

# The camera sits 1 m below the body. Keep that slice and leave out the ground.
Z_LO = -1.5
Z_HI = 1.0
MAX_RANGE_M = HALF_M * 2


def spatial_features(model, depth) -> "np.ndarray":
    """Feature map from the frozen encoder, before it is flattened.

    ``depth`` is ``[56,56]`` or ``[B,56,56]`` in ``depth_input`` units.
    Returns ``[C,14,14]`` or ``[B,C,14,14]``. Running this does not change the
    survival head.
    """
    import torch

    single = depth.dim() == 2
    x = depth.unsqueeze(0).unsqueeze(0) if single else depth.unsqueeze(1)
    was = model.training
    model.eval()
    with torch.no_grad():
        for layer in model.enc[:4]:
            x = layer(x.to(next(model.parameters()).dtype))
    if was:
        model.train()
    out = x.detach().cpu().numpy()
    return out[0] if single else out


def latent_map(depths: Sequence[np.ndarray], feats: Sequence[np.ndarray],
               states: Sequence[np.ndarray], here: np.ndarray | None = None) -> np.ndarray:
    """Features of every view, in the body frame of ``here`` or the last pose.

    ``depths[t]`` is metric depth ``[56,56]``. ``feats[t]`` is ``[C,hf,wf]``
    for that same image. Returns ``[C,32,32]``. An empty input is all zeros.
    """
    if len(depths) == 0:
        raise ValueError("latent map needs at least one view")
    if not (len(depths) == len(feats) == len(states)):
        raise ValueError("depths, features and poses must be the same length")
    channels = int(np.asarray(feats[0]).shape[0])
    grid = np.zeros((channels, N, N), np.float32)
    dist = np.full((N, N), np.inf, np.float64)
    here = np.asarray(states[-1] if here is None else here, np.float64)
    for depth, feat, state in zip(depths, feats, states):
        fwd, left, dz, vec = _place(np.asarray(depth, np.float64), np.asarray(feat, np.float32),
                                    np.asarray(state, np.float64), here)
        _paint(grid, dist, fwd, left, dz, vec)
    return grid


def _place(depth: np.ndarray, feat: np.ndarray, state: np.ndarray, here: np.ndarray):
    """One feature per spatial cell, at the nearest depth pixel of that cell."""
    feat = np.asarray(feat, np.float32)
    channels, hf, wf = feat.shape
    h, w = depth.shape
    ys = np.linspace(0, h, hf + 1).astype(np.int64)
    xs = np.linspace(0, w, wf + 1).astype(np.int64)
    fx = (w / 2) / math.tan(math.radians(HFOV_DEG / 2))
    fy = (h / 2) / math.tan(math.radians(VFOV_DEG / 2))
    fwd_l, left_l, dz_l, vec_l = [], [], [], []
    yaw = float(state[6])
    c, s = math.cos(yaw), math.sin(yaw)
    for bi in range(hf):
        for bj in range(wf):
            block = depth[ys[bi]:ys[bi + 1], xs[bj]:xs[bj + 1]]
            ok = np.isfinite(block) & (block > 0.05) & (block < MAX_RANGE_M)
            if not np.any(ok):
                continue
            local = np.argwhere(ok)
            pick = local[int(np.argmin(block[ok]))]
            v = int(ys[bi] + pick[0])
            u = int(xs[bj] + pick[1])
            dist_m = float(block[pick[0], pick[1]])
            right = (u + 0.5 - w / 2) / fx * dist_m
            down = (v + 0.5 - h / 2) / fy * dist_m
            # Camera origin is CAM_DZ below the body. Image right is -left.
            world = np.array([
                state[0] + dist_m * c - right * s,
                state[1] + dist_m * s + right * c,
                state[2] - 1.0 - down,
            ], np.float64)
            fwd, left, dz = body_from_world(here, world.reshape(1, 3))
            fwd_l.append(float(fwd[0]))
            left_l.append(float(left[0]))
            dz_l.append(float(dz[0]))
            vec_l.append(feat[:, bi, bj])
    if not fwd_l:
        empty = np.zeros((0,), np.float64)
        return empty, empty, empty, np.zeros((0, channels), np.float32)
    return (np.asarray(fwd_l), np.asarray(left_l), np.asarray(dz_l), np.stack(vec_l))


def _paint(grid: np.ndarray, dist: np.ndarray, fwd: np.ndarray, left: np.ndarray,
           dz: np.ndarray, vec: np.ndarray) -> None:
    use = (dz >= Z_LO) & (dz <= Z_HI)
    if not np.any(use):
        return
    fwd, left, dz, vec = fwd[use], left[use], dz[use], vec[use]
    i = np.floor((fwd + HALF_M) / CELL_M).astype(np.int64)
    j = np.floor((left + HALF_M) / CELL_M).astype(np.int64)
    inside = (i >= 0) & (i < N) & (j >= 0) & (j < N)
    if not np.any(inside):
        return
    i, j, fwd, left, vec = i[inside], j[inside], fwd[inside], left[inside], vec[inside]
    rng = np.hypot(fwd, left)
    order = np.argsort(rng, kind="stable")
    flat = i[order] * N + j[order]
    _, first = np.unique(flat, return_index=True)
    pick = order[first]
    closer = rng[pick] < dist[i[pick], j[pick]]
    pick = pick[closer]
    if len(pick) == 0:
        return
    dist[i[pick], j[pick]] = rng[pick]
    grid[:, i[pick], j[pick]] = np.moveaxis(vec[pick], -1, 0)


class MapEncoder(nn.Module):
    """Depth image to a 14 by 14 feature map, and a depth read back from it."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = int(dim)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, self.dim, 1),
        )
        self.depth_head = nn.Linear(self.dim, 1)

    def features(self, depth_in: torch.Tensor) -> torch.Tensor:
        """``depth_in`` ``[B,56,56]`` in ``depth_input`` units. Returns ``[B,C,14,14]``."""
        x = depth_in.unsqueeze(1) if depth_in.dim() == 3 else depth_in
        return self.enc(x)

    def depth_of(self, feat: torch.Tensor) -> torch.Tensor:
        """``feat`` ``[B,C,H,W]`` -> ``[B,H,W]`` log-depth."""
        return self.depth_head(feat.permute(0, 2, 3, 1)).squeeze(-1)


def outside_view(fwd: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    """True when a body point is outside the current 90° camera.

    Same test as ``body_surround.in_view``: more than half the horizontal
    field off the nose, or not in front of the body.
    """
    ang = torch.atan2(left.abs(), fwd.clamp(min=1e-6))
    return (fwd <= 0.05) | (ang > math.radians(HFOV_DEG / 2))


def block_body(depth_m: torch.Tensor, state: torch.Tensor, here: torch.Tensor,
               hf: int = 14, wf: int = 14):
    """Body cell and body position of each feature block.

    Depth is meters, ``[B,56,56]``. Returns ``cell``, ``rng``, ``valid``,
    ``dist_m``, ``fwd``, ``left``, ``dz``, each ``[B,hf,wf]`` except ``cell``.
    """
    b, h, w = depth_m.shape
    bh, bw = h // hf, w // wf
    block = depth_m.reshape(b, hf, bh, wf, bw).permute(0, 1, 3, 2, 4).reshape(b, hf, wf, bh * bw)
    dist_m, which = block.min(dim=-1)
    dv, du = which // bw, which % bw
    v = torch.arange(hf, device=depth_m.device)[:, None] * bh + dv
    u = torch.arange(wf, device=depth_m.device)[None, :] * bw + du
    fx = (w / 2) / math.tan(math.radians(HFOV_DEG / 2))
    fy = (h / 2) / math.tan(math.radians(VFOV_DEG / 2))
    right = (u + 0.5 - w / 2) / fx * dist_m
    down = (v + 0.5 - h / 2) / fy * dist_m
    yaw = state[:, 6]
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    world_x = state[:, 0, None, None] + dist_m * c - right * s
    world_y = state[:, 1, None, None] + dist_m * s + right * c
    world_z = state[:, 2, None, None] + CAM_DZ - down
    hy, hx = here[:, 6], here
    cy, sy = torch.cos(hy)[:, None, None], torch.sin(hy)[:, None, None]
    dx = world_x - hx[:, 0, None, None]
    dy = world_y - hx[:, 1, None, None]
    fwd = cy * dx + sy * dy
    left = sy * dx - cy * dy
    dz = world_z - hx[:, 2, None, None]
    cell_i = torch.floor((fwd + HALF_M) / CELL_M).long()
    cell_j = torch.floor((left + HALF_M) / CELL_M).long()
    rng = torch.hypot(fwd, left)
    valid = ((dz >= Z_LO) & (dz <= Z_HI) & (dist_m > 0.05) & (dist_m < MAX_RANGE_M)
             & (cell_i >= 0) & (cell_i < N) & (cell_j >= 0) & (cell_j < N))
    cell = (cell_i.clamp(0, N - 1) * N + cell_j.clamp(0, N - 1)).long()
    return cell, rng, valid, dist_m, fwd, left, dz


def block_rays(depth_m: torch.Tensor, state: torch.Tensor, here: torch.Tensor,
               hf: int = 14, wf: int = 14):
    """Body cell of each feature block. Depth is meters, ``[B,56,56]``.

    Returns ``cell`` ``[B,hf,wf]``, ``rng`` and ``valid``.
    """
    cell, rng, valid, dist_m, _, _, _ = block_body(depth_m, state, here, hf, wf)
    return cell, rng, valid, dist_m


def _depth_units(depth_m: torch.Tensor) -> torch.Tensor:
    """Match ``depth_input``: log from 0.5 m to 100 m."""
    d = depth_m.clamp(0.5, 100.0)
    return torch.log(d / 0.5) / math.log(100.0 / 0.5)


def _masked_mean(err: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return err[mask].mean()
    return err.sum() * 0.0


def _weighted_mean(err: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return err.sum() * 0.0
    ww = weight[mask]
    return (err[mask] * ww).sum() / ww.sum().clamp(min=1e-6)


def scene_loss(enc: MapEncoder, depth_m: torch.Tensor, state: torch.Tensor):
    """Past views go into the current body. Each filled cell must read back
    the depth that was seen there.

    The current frame is only the pose. It is not written into the map, so a
    surface that has left the camera can stay in the cell. In-view cells and
    cells outside the current camera are averaged separately, so the leaving
    surfaces are not washed out. Nothing here is a survival score.
    """
    b, steps, _, _ = depth_m.shape
    if steps < 2:
        raise ValueError("the scene needs a past view and the current pose")
    flat = enc.features(_depth_units(depth_m.reshape(b * steps, 56, 56)))
    channels = flat.shape[1]
    feat = flat.reshape(b, steps, channels, 14, 14)
    here = state[:, -1]
    cells, rngs, valids, seen, outs = [], [], [], [], []
    for k in range(steps - 1):
        cell, rng, valid, dist, fwd, left, _dz = block_body(depth_m[:, k], state[:, k], here)
        cells.append(cell)
        rngs.append(rng)
        valids.append(valid)
        seen.append(dist.clamp(min=0.05).log())
        outs.append(outside_view(fwd, left))
    past = steps - 1
    hw = (b, past * 14, 14)
    past_f = feat[:, :past].permute(0, 2, 1, 3, 4).reshape(b, channels, past * 14, 14)
    past_c = torch.stack(cells, 1).reshape(hw)
    past_r = torch.stack(rngs, 1).reshape(hw)
    past_v = torch.stack(valids, 1).reshape(hw)
    past_s = torch.stack(seen, 1).reshape(b, 1, past * 14, 14)
    past_o = torch.stack(outs, 1).reshape(b, 1, past * 14, 14).to(past_f.dtype)
    grid, dist = fuse_nearest(past_f, past_c, past_r, past_v)
    target, _ = fuse_nearest(past_s, past_c, past_r, past_v)
    gone, _ = fuse_nearest(past_o, past_c, past_r, past_v)
    pred = enc.depth_of(grid)
    err = (pred - target[:, 0]).pow(2)
    written = dist < 1e8
    left_view = written & (gone[:, 0] > 0.5)
    still = written & ~left_view
    pred_m = pred.clamp(-3.0, 3.0).exp()
    true_m = target[:, 0].clamp(-3.0, 3.0).exp()
    meter = (pred_m - true_m).abs()
    # Closer surfaces decide whether one step enters them. The weight is detached
    # so the loss cannot improve by pushing the depth farther away.
    close = (1.0 / true_m.clamp(min=0.5)).clamp(max=4.0).detach()
    loss = (
        _masked_mean(err, left_view) + _masked_mean(err, still)
        + _weighted_mean(meter, left_view, close) + _weighted_mean(meter, still, close)
    )
    with torch.no_grad():
        mae = meter.detach()
        near = left_view & (true_m <= 2.0)
        filled = grid.permute(0, 2, 3, 1)[written]
        feat_std = float(filled.std()) if int(written.sum()) > 1 else 0.0
    return loss, {
        "out_mae": float(mae[left_view].mean()) if bool(left_view.any()) else None,
        "in_mae": float(mae[still].mean()) if bool(still.any()) else None,
        "near_mae": float(mae[near].mean()) if bool(near.any()) else None,
        "out_n": int(left_view.sum()),
        "in_n": int(still.sum()),
        "near_n": int(near.sum()),
        "feat_std": feat_std,
    }


def fuse_nearest(feat: torch.Tensor, cell: torch.Tensor, rng: torch.Tensor,
                 valid: torch.Tensor) -> tuple:
    """Keep the nearest feature in each body cell.

    ``feat`` is ``[B,C,H,W]``. Returns ``grid`` ``[B,C,32,32]`` and ``dist``
    ``[B,32,32]``.
    """
    b, channels, h, w = feat.shape
    flat_f = feat.permute(0, 2, 3, 1).reshape(b, h * w, channels)
    flat_c = cell.reshape(b, h * w)
    flat_r = rng.reshape(b, h * w)
    flat_v = valid.reshape(b, h * w)
    flat_c = torch.where(flat_v, flat_c, torch.full_like(flat_c, N * N))
    flat_r = torch.where(flat_v, flat_r, torch.full_like(flat_r, 1e9))
    order = flat_r.argsort(dim=1, descending=True)
    src = torch.gather(flat_f, 1, order.unsqueeze(-1).expand_as(flat_f))
    cells = torch.gather(flat_c, 1, order)
    ranges = torch.gather(flat_r, 1, order)
    grid = feat.new_zeros(b, N * N + 1, channels)
    dist = feat.new_full((b, N * N + 1), 1e9)
    grid.scatter_(1, cells.unsqueeze(-1).expand_as(src), src)
    dist.scatter_(1, cells, ranges)
    return grid[:, : N * N].permute(0, 2, 1).reshape(b, channels, N, N), dist[:, : N * N].reshape(b, N, N)
