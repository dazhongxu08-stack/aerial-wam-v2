"""Synthetic keep-out objects for training the consequence model's semantic head.

The simulator has no "red stool", and real Tello flights have no metric xy.
Instead, each training window can get a virtual object: a box pasted into the
current depth frame at its true range, its pixels marked in the ``sem`` mask,
and a keep-out label for every future step whose logged position comes within
``object radius + keep radius`` of it (horizontal only: keep-out means fly
around, not over).

Objects are placed near the logged path (offset sideways, so some windows pass
and some enter) or anywhere in view. Windows without an object keep an empty
mask, and real obstacles never carry the mask, so the model has to read the
mask channel rather than "anything near in depth" to predict keep-out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from experiments.aerial.rl.consequence_model import DEPTH_MAX_M, DEPTH_MIN_M

SIM_HFOV_RAD = math.radians(90.0)


@dataclass
class KeepoutSynthConfig:
    p_object: float = 0.5
    p_offpath: float = 0.3
    lateral_m: float = 1.5
    obj_radius_m: Tuple[float, float] = (0.15, 0.40)
    obj_height_m: Tuple[float, float] = (0.3, 1.2)
    keep_radius_m: Tuple[float, float] = (0.3, 1.0)
    range_m: Tuple[float, float] = (0.8, 10.0)
    hfov_rad: float = SIM_HFOV_RAD
    min_visible_px: int = 3


def _to_metric(x: torch.Tensor) -> torch.Tensor:
    return DEPTH_MIN_M * torch.exp(x * math.log(DEPTH_MAX_M / DEPTH_MIN_M))


def _to_input(d: torch.Tensor) -> torch.Tensor:
    d = d.clamp(DEPTH_MIN_M, DEPTH_MAX_M)
    return torch.log(d / DEPTH_MIN_M) / math.log(DEPTH_MAX_M / DEPTH_MIN_M)


def _u(gen: torch.Generator, n: int, lo: float, hi: float, dev: torch.device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, generator=gen, device=dev)


def synth_keepout(batch: Dict[str, torch.Tensor], gen: torch.Generator,
                  cfg: Optional[KeepoutSynthConfig] = None) -> Dict[str, torch.Tensor]:
    """Return a copy of ``batch`` with ``sem``, ``keep_radius``, ``ko_hit``, ``ko_alive``, ``ko_obj``.

    Needs ``depth`` ([B,56,56] or [B,K,56,56], channel 0 = current frame),
    ``fut_xy`` [B,H,2] body frame (x forward, y right) and ``alive`` [B,H].
    """
    cfg = cfg or KeepoutSynthConfig()
    dev = batch["depth"].device
    depth = batch["depth"]
    cur = depth if depth.dim() == 3 else depth[:, 0]
    B, Hh, Ww = cur.shape
    fut = batch["fut_xy"].to(cur.dtype)
    alive = batch["alive"]
    H = alive.shape[1]

    keep_r = _u(gen, B, *cfg.keep_radius_m, dev)
    obj_r = _u(gen, B, *cfg.obj_radius_m, dev)
    has = torch.rand(B, generator=gen, device=dev) < cfg.p_object
    n_alive = alive.sum(-1)
    k_star = (torch.rand(B, generator=gen, device=dev) * n_alive).long().clamp(0, H - 1)
    anchor = fut[torch.arange(B, device=dev), k_star]
    a_norm = anchor.norm(dim=-1, keepdim=True)
    fwd = torch.where(a_norm > 0.3, anchor / a_norm.clamp(min=1e-6), torch.tensor([1.0, 0.0], device=dev))
    perp = torch.stack([-fwd[:, 1], fwd[:, 0]], dim=-1)
    lat = _u(gen, B, -cfg.lateral_m, cfg.lateral_m, dev)
    c_path = anchor + lat[:, None] * perp
    off = (torch.rand(B, generator=gen, device=dev) < cfg.p_offpath) | (n_alive < 1)
    rr = _u(gen, B, *cfg.range_m, dev)
    bb = _u(gen, B, -0.45 * cfg.hfov_rad, 0.45 * cfg.hfov_rad, dev)
    c_off = torch.stack([rr * torch.cos(bb), rr * torch.sin(bb)], dim=-1)
    c = torch.where(off[:, None], c_off, c_path)

    r = c.norm(dim=-1)
    b = torch.atan2(c[:, 1], c[:, 0])
    has = has & (r > obj_r + keep_r + 0.2) & (r < cfg.range_m[1]) & (b.abs() < cfg.hfov_rad / 2)

    tan_half = math.tan(cfg.hfov_rad / 2)
    cols = torch.atan(((torch.arange(Ww, device=dev) + 0.5) / Ww - 0.5) * 2.0 * tan_half)
    half_w = torch.atan(obj_r / r.clamp(min=1e-3))
    colmask = (cols[None, :] - b[:, None]).abs() <= half_w[:, None]
    f_px = (Hh / 2) / tan_half
    h_px = (_u(gen, B, *cfg.obj_height_m, dev) / r.clamp(min=1e-3) * f_px).clamp(2, Hh)
    row_bot = _u(gen, B, Hh / 2 + 2, Hh, dev)
    row_top = row_bot - h_px
    rows = torch.arange(Hh, device=dev, dtype=cur.dtype) + 0.5
    rowmask = (rows[None, :] >= row_top[:, None]) & (rows[None, :] <= row_bot[:, None])
    region = rowmask[:, :, None] & colmask[:, None, :] & has[:, None, None]

    front = (r - obj_r).clamp(min=DEPTH_MIN_M)
    visible = region & (_to_metric(cur.float()) > front[:, None, None])
    has = has & (visible.flatten(1).sum(-1) >= cfg.min_visible_px)
    visible = visible & has[:, None, None]

    new_cur = torch.where(visible, _to_input(front)[:, None, None].to(cur.dtype), cur)
    out = dict(batch)
    if depth.dim() == 3:
        out["depth"] = new_cur
    else:
        out["depth"] = torch.cat([new_cur[:, None], depth[:, 1:]], dim=1)
    out["sem"] = visible.to(cur.dtype)
    out["keep_radius"] = keep_r

    dist = (fut - c[:, None, :]).norm(dim=-1)
    viol = (dist < (obj_r + keep_r)[:, None]) & (alive > 0) & has[:, None]
    cum = torch.cumsum(viol.float(), dim=-1)
    out["ko_hit"] = (viol & (cum == 1)).float()
    out["ko_alive"] = alive * ((cum - viol.float()) == 0).float()
    out["ko_obj"] = has.float()
    out["ko_center"] = c
    out["ko_reach"] = obj_r + keep_r
    return out


def keepout_side_check(depth: torch.Tensor, fut_xy: torch.Tensor, hit: torch.Tensor,
                       hfov_rad: float = SIM_HFOV_RAD, max_step: int = 4) -> Dict[str, float]:
    """Does body +y really map to the right half of the image in this data?

    For windows that collide within ``max_step``, the obstacle is where the
    drone ends up. Compare the depth at that bearing's column with the mirrored
    column: the hit side should read nearer. ``agree`` well above 0.5 confirms
    the convention ``synth_keepout`` uses.
    """
    cur = depth if depth.dim() == 3 else depth[:, 0]
    B, Hh, Ww = cur.shape
    k = hit[:, :max_step].float().argmax(-1)
    did = hit[:, :max_step].sum(-1) > 0
    p = fut_xy[torch.arange(B), k]
    b = torch.atan2(p[:, 1], p[:, 0])
    ok = did & (b.abs() > math.radians(8)) & (b.abs() < hfov_rad / 2 * 0.9) & (p.norm(dim=-1) > 0.2)
    if not bool(ok.any()):
        return {"n": 0, "agree": float("nan")}
    u = 0.5 + torch.tan(b[ok]) / (2.0 * math.tan(hfov_rad / 2))
    col = (u * Ww).long().clamp(0, Ww - 1)
    band = cur[ok][:, Hh // 4:, :].float()
    near = band.gather(2, col[:, None, None].expand(-1, band.shape[1], 1)).amin(dim=(1, 2))
    mir = band.gather(2, (Ww - 1 - col)[:, None, None].expand(-1, band.shape[1], 1)).amin(dim=(1, 2))
    return {"n": int(ok.sum()), "agree": float((near < mir).float().mean())}
