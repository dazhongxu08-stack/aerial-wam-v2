"""Yaw-consequence prior: after a turn, the three cones must match the new view.

Current cones come from the frozen depth head. The trained piece is the yaw
update. ``wm_soft_init`` and the signed depth checkpoint are read only. A
``cone_readout`` file is written only when the held-out turns pass.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

YAW_LIMIT = 0.3141592653589793
CENTER_FRAC = 0.12
SIDE_RATIO = 1.25
NEAR_M = 20.0
GROW_M = 3.0
SPREAD_M = 1.0
N_YAW = 18
ALT_Z = 24.0


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def split_yaw(total: float, limit: float = YAW_LIMIT) -> List[float]:
    """Body-limit yaw steps that sum to ``total`` radians."""
    steps: List[float] = []
    remaining = float(total)
    for _ in range(12):
        if abs(remaining) <= 1e-4:
            break
        step = float(max(-limit, min(limit, remaining)))
        steps.append(step)
        remaining -= step
    return steps


def toward_open(left: float, right: float, dyaw: float) -> bool:
    """True when ``dyaw`` turns toward the farther side and the sides differ.

    Positive body yaw turns the nose toward image right (AirSim NED).
    """
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    small = min(left, right)
    if small <= 1e-3:
        return False
    if max(left, right) / small < SIDE_RATIO:
        return False
    return float(dyaw) * (math.log(right) - math.log(left)) > 0.0


def gate_pass(records: Sequence[dict]) -> bool:
    """Held-out turns: sides match, and a longer street is predicted longer."""
    if len(records) < 4:
        return False
    wide = [r for r in records if float(r["spread_m"]) >= SPREAD_M]
    if len(wide) < int(math.ceil(0.8 * len(records))):
        return False
    clear = [r for r in records if r["dest_clear"]]
    same = [r for r in clear if r["dest_same_side"]]
    if len(clear) < 3 or len(same) < int(math.ceil(0.8 * len(clear))):
        return False
    grew = [r for r in records if r["gt_lengthens"]]
    if len(grew) < 2 or not all(r["pred_lengthens"] for r in grew):
        return False
    longer = [r for r in records if r.get("open_longer")]
    if len(longer) < int(math.ceil(0.8 * len(records))):
        return False
    return True


def sample_route_xy(points: np.ndarray, spacing: float = 8.0) -> np.ndarray:
    """Walk the polyline and keep a point every ``spacing`` metres."""
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] < 2:
        raise ValueError("route needs at least two points")
    kept = [xy[0]]
    walked = 0.0
    prev = xy[0]
    for point in xy[1:]:
        seg = float(np.linalg.norm(point - prev))
        walked += seg
        if walked >= spacing:
            kept.append(point)
            walked = 0.0
        prev = point
    if float(np.linalg.norm(kept[-1] - xy[-1])) > 1.0:
        kept.append(xy[-1])
    return np.stack(kept, axis=0)


def _band_crop(depth: np.ndarray) -> np.ndarray:
    image = np.asarray(depth, dtype=np.float64)
    height, width = image.shape
    dh = max(1, int(height * CENTER_FRAC))
    dw = max(1, int(width * CENTER_FRAC))
    r0 = (height - dh) // 2
    c0 = (width - dw) // 2
    return image[r0 : r0 + dh, c0 : c0 + dw]


def forward_band_mean(depth: np.ndarray) -> float:
    crop = _band_crop(depth)
    valid = crop[np.isfinite(crop) & (crop > 0.5) & (crop < 80.0)]
    if valid.size < 4:
        return float("nan")
    return float(np.mean(valid))


def side_pair(depth: np.ndarray) -> Tuple[float, float]:
    from experiments.aerial.rl.depth_geometry import cone_clearances

    cones = cone_clearances(np.asarray(depth, dtype=np.float64), center_frac=CENTER_FRAC)
    return float(cones["left"]), float(cones["right"])


def _side_clear(left: float, right: float) -> bool:
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    small = min(left, right)
    if small <= 1e-3:
        return False
    return max(left, right) / small >= SIDE_RATIO


def _same_side(gt_left: float, gt_right: float, pred_left: float, pred_right: float) -> bool:
    return (gt_left >= gt_right) == (pred_left >= pred_right)


def load_route(annotation: Path, route_index: int) -> Tuple[np.ndarray, np.ndarray]:
    payload = json.loads(annotation.read_text())
    episodes = payload.get("episodes") or payload.get("routes")
    episode = episodes[int(route_index)]
    pos = np.asarray(episode["pos"], dtype=np.float64)
    return pos[:, :2], pos[-1]


def assert_safe_output(path: Path, sources: Sequence[Path]) -> None:
    target = path.resolve()
    for source in sources:
        if target == Path(source).resolve():
            raise SystemExit(f"refusing to write over {target}")


def collect_fans(
    route_xy: np.ndarray,
    goal_xyz: np.ndarray,
    out_path: Path,
    *,
    camera: str,
    z_up: float,
) -> None:
    import time

    import airsim
    import cv2

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    yaws = np.linspace(0.0, 2.0 * math.pi, N_YAW, endpoint=False)
    rgbs: List[np.ndarray] = []
    depths: List[np.ndarray] = []
    positions: List[np.ndarray] = []
    try:
        for p_i, xy in enumerate(route_xy):
            views_rgb = []
            views_depth = []
            for y_i, yaw in enumerate(yaws):
                q = airsim.to_quaternion(0.0, 0.0, float(yaw))
                ned = airsim.Vector3r(float(xy[0]), float(xy[1]), float(-z_up))
                client.simSetVehiclePose(airsim.Pose(ned, q), ignore_collision=True)
                time.sleep(0.45 if y_i == 0 else 0.22)
                rgb, depth = _grab(client, airsim, cv2, camera)
                views_rgb.append(rgb)
                views_depth.append(depth)
            stack_d = np.stack(views_depth)
            means = [forward_band_mean(stack_d[i]) for i in range(N_YAW)]
            finite = [m for m in means if math.isfinite(m)]
            print(
                f"pos {p_i} xy=({xy[0]:.1f},{xy[1]:.1f}) "
                f"fwd_min={min(finite) if finite else float('nan'):.1f} "
                f"fwd_max={max(finite) if finite else float('nan'):.1f}",
                flush=True,
            )
            rgbs.append(np.stack(views_rgb))
            depths.append(stack_d)
            positions.append(np.array([xy[0], xy[1], z_up], dtype=np.float32))
    finally:
        try:
            client.enableApiControl(False)
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rgb=np.stack(rgbs),
        depth=np.stack(depths),
        pos=np.stack(positions),
        yaws=yaws.astype(np.float32),
        goal=np.asarray(goal_xyz, dtype=np.float32),
    )
    print(f"collected {len(rgbs)} positions -> {out_path}", flush=True)


def _grab(client, airsim, cv2, camera: str):
    rs = client.simGetImages(
        [
            airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(camera, airsim.ImageType.DepthPlanar, True, False),
        ]
    )
    if rs[0].height < 2 or rs[1].width < 2:
        raise SystemExit(f"camera {camera} returned an empty image")
    bgr = np.frombuffer(rs[0].image_data_uint8, dtype=np.uint8).reshape(
        rs[0].height, rs[0].width, 3
    )
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    depth = np.array(
        airsim.list_to_2d_float_array(rs[1].image_data_float, rs[1].width, rs[1].height),
        dtype=np.float32,
    )
    depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0).astype(np.float32)
    depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_AREA)
    if float(rgb.mean()) < 2.0:
        raise SystemExit(f"camera {camera} image is black")
    return np.ascontiguousarray(rgb, dtype=np.uint8), np.ascontiguousarray(depth, dtype=np.float32)


def _events_at(depth: np.ndarray, yaws: np.ndarray) -> int:
    count = 0
    n = int(depth.shape[0])
    step = 2  # 20° * 2 = 40°
    for src in range(n):
        src_fwd = forward_band_mean(depth[src])
        left, right = side_pair(depth[src])
        if not (math.isfinite(src_fwd) and src_fwd < NEAR_M):
            continue
        for sign, dst in ((1, (src + step) % n), (-1, (src - step) % n)):
            dyaw = wrap_angle(float(yaws[dst] - yaws[src]))
            if sign < 0 and dyaw > 0:
                dyaw -= 2.0 * math.pi
            if sign > 0 and dyaw < 0:
                dyaw += 2.0 * math.pi
            dst_fwd = forward_band_mean(depth[dst])
            if (
                toward_open(left, right, dyaw)
                and math.isfinite(dst_fwd)
                and dst_fwd > src_fwd + GROW_M
            ):
                count += 1
    return count


def split_positions(depth: np.ndarray, yaws: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
    events = [_events_at(depth[i], yaws) for i in range(depth.shape[0])]
    print("events_per_pos", events, flush=True)
    rich = [i for i, n in enumerate(events) if n >= 1]
    if sum(events) < 8 or len(rich) < 4:
        raise SystemExit(
            f"not enough near-wall turns that lengthen the street: events={events}"
        )
    hold = rich[-2:]
    train = [i for i in range(len(events)) if i not in hold]
    if sum(events[i] for i in train) < 6:
        raise SystemExit(f"training split lost the lengthening turns: hold={hold}")
    return train, hold, events


def _iter_pairs(depth_pos: np.ndarray, yaws: np.ndarray, pos_index: int, offset: int):
    n = int(yaws.shape[0])
    for src in range(n):
        dst = (src + offset) % n
        dyaw = wrap_angle(float(yaws[dst] - yaws[src]))
        if offset < 0 and dyaw > 0:
            dyaw -= 2.0 * math.pi
        if offset > 0 and dyaw < 0:
            dyaw += 2.0 * math.pi
        yield pos_index, src, dst, float(dyaw)


def _holdout_score(records: Sequence[dict]) -> float:
    if not records:
        return -1.0
    spread = sum(float(r["spread_m"]) for r in records) / len(records)
    if spread < SPREAD_M:
        return float(spread)
    clear = [r for r in records if r["dest_clear"]]
    same = sum(1 for r in clear if r["dest_same_side"]) / float(max(len(clear), 1))
    grew = [r for r in records if r["gt_lengthens"]]
    pred = sum(1 for r in grew if r["pred_lengthens"]) / float(max(len(grew), 1))
    return float(spread + 20.0 * same + 20.0 * pred)


def train_prior(
    data: Dict[str, np.ndarray],
    train_idx: Sequence[int],
    steps: int,
    hold_idx: Sequence[int],
) -> Tuple[bool, List[dict], object]:
    import torch
    import torch.nn.functional as F

    from experiments.aerial.rl.dynamics_torch import _symexp, _symlog

    depth = data["depth"]
    yaws = data["yaws"]
    sources = [(int(p), int(s)) for p in train_idx for s in range(N_YAW)]
    print(f"train_views {len(sources)}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    readout = build_readout(device)
    opt = torch.optim.AdamW(readout.parameters(), lr=1e-3, betas=(0.9, 0.95))
    print(f"trainable {sum(p.numel() for p in readout.parameters() if p.requires_grad)}", flush=True)
    rng = np.random.default_rng(0)
    readout.train()
    da3 = data["da3"]
    best_score = -1.0
    best_state = None
    best_records: List[dict] = []
    passed = False
    for step in range(int(steps)):
        take = min(8, len(sources))
        choice = rng.choice(len(sources), size=take, replace=False)
        batch = [sources[int(i)] for i in choice]
        loss, side_hit, now_std, now_mae = _cone_roll_loss(
            readout, da3, depth, yaws, batch, torch, F, _symlog, _symexp
        )
        if not bool(torch.isfinite(loss)):
            opt.zero_grad(set_to_none=True)
            print(f"step {step} nonfinite loss, skip", flush=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(readout.parameters()), 10.0)
        opt.step()
        if step % 25 == 0 or step == int(steps) - 1:
            print(
                f"step {step} loss={float(loss.detach()):.3f} side_hit={side_hit:.2f} "
                f"now_std={now_std:.2f} now_mae={now_mae:.2f}",
                flush=True,
            )
        if step % 50 == 0 or step == int(steps) - 1:
            records = evaluate_holdout(data, hold_idx, readout)
            readout.train()
            score = _holdout_score(records)
            ok = gate_pass(records)
            mean_spread = (
                sum(float(r["spread_m"]) for r in records) / len(records) if records else 0.0
            )
            n_clear = sum(1 for r in records if r["dest_clear"])
            n_same = sum(1 for r in records if r["dest_clear"] and r["dest_same_side"])
            n_grew = sum(1 for r in records if r["gt_lengthens"])
            n_pred = sum(1 for r in records if r["gt_lengthens"] and r["pred_lengthens"])
            n_longer = sum(1 for r in records if r.get("open_longer"))
            min_spread = min(float(r["spread_m"]) for r in records)
            print(
                f"hold step {step} score={score:.2f} spread={mean_spread:.2f} "
                f"min={min_spread:.2f} same={n_same}/{n_clear} grew={n_pred}/{n_grew} "
                f"longer={n_longer}/{len(records)} passed={ok} n={len(records)}",
                flush=True,
            )
            if ok:
                best_records = records
                passed = True
                break
            if score > best_score:
                best_score = score
                best_records = records
                best_state = {k: v.detach().cpu().clone() for k, v in readout.state_dict().items()}
    if not passed and best_state is not None:
        readout.load_state_dict(best_state)
    return passed, best_records, readout


def _st_code(dyn, h, embed, torch, F, prior: bool):
    """One-hot mode, with a gradient into the logits. This is the code eval reads."""
    if prior:
        logits = dyn.rssm.prior_net(h)
    else:
        logits = dyn.rssm.post_net(torch.cat([h, embed], dim=-1))
    grouped = logits.view(logits.shape[0], dyn.rssm.stoch_dim, dyn.rssm.stoch_classes)
    probs = torch.softmax(grouped, dim=-1)
    index = probs.argmax(dim=-1)
    one = F.one_hot(index, grouped.shape[-1]).to(dtype=probs.dtype)
    code = (one + probs - probs.detach()).reshape(logits.shape[0], -1)
    entropy = -(probs.clamp_min(1e-8).log() * probs).sum(dim=-1).mean()
    return code, entropy


def _mode_z(dyn, probs, torch, F, hard: bool = True):
    if not hard:
        return probs.reshape(probs.shape[0], -1)
    index = probs.argmax(dim=-1)
    one = F.one_hot(index, probs.shape[-1]).to(dtype=probs.dtype)
    sample = one + probs - probs.detach()
    return sample.reshape(probs.shape[0], -1)


def _decode(dyn, h, z, torch, F):
    pred = dyn.depth_decoder(torch.cat([h, z], dim=-1)).squeeze(1)
    return pred


def _band_mean_torch(depth_m, torch):
    batch, height, width = depth_m.shape
    dh = max(1, int(height * CENTER_FRAC))
    dw = max(1, int(width * CENTER_FRAC))
    r0 = (height - dh) // 2
    c0 = (width - dw) // 2
    crop = depth_m[:, r0 : r0 + dh, c0 : c0 + dw]
    valid = torch.isfinite(crop) & (crop > 0.5) & (crop < 80.0)
    means = []
    for i in range(int(batch)):
        values = crop[i][valid[i]]
        if int(values.numel()) < 4:
            means.append(depth_m.new_zeros(()))
        else:
            means.append(values.mean())
    return torch.stack(means)


def _prior_loss(dyn, rgb, depth, pos, yaws, batch, steps, rng, torch, F, symlog, symexp):
    from experiments.aerial.rl.depth_geometry import cone_mins_torch

    device = dyn.device
    src_rgb = []
    src_depth = []
    dst_depth = []
    proprio = []
    flip = rng.random(len(batch)) < 0.0
    for i, (p_i, s_i, d_i, _dyaw) in enumerate(batch):
        image = rgb[p_i, s_i]
        src_d = depth[p_i, s_i]
        dst_d = depth[p_i, d_i]
        yaw = float(yaws[s_i])
        if flip[i]:
            image = np.ascontiguousarray(image[:, ::-1])
            src_d = np.ascontiguousarray(src_d[:, ::-1])
            dst_d = np.ascontiguousarray(dst_d[:, ::-1])
        src_rgb.append(image)
        src_depth.append(src_d)
        dst_depth.append(dst_d)
        xyz = pos[p_i].astype(np.float32).copy()
        xyz[0] += float(rng.normal(0.0, 20.0))
        xyz[1] += float(rng.normal(0.0, 20.0))
        proprio.append([xyz[0], xyz[1], xyz[2], yaw])
    xb = torch.from_numpy(np.stack(src_rgb)).to(device)
    sb = torch.from_numpy(np.stack(src_depth)).to(device, dyn.torch_dtype)
    db = torch.from_numpy(np.stack(dst_depth)).to(device, dyn.torch_dtype)
    pb = torch.from_numpy(np.asarray(proprio, dtype=np.float32)).to(device, dyn.torch_dtype)
    h = dyn.rssm.initial_h(xb.shape[0], device, dyn.torch_dtype)
    embed = dyn._embed(xb, pb)
    z, entropy = _st_code(dyn, h, embed, torch, F, prior=False)
    pred_now = _decode(dyn, h, z, torch, F)
    signed_steps = list(steps)
    if bool(flip.any()):
        # Flipped images swap left and right, so the yaw steps flip with them.
        # Applied per-sample below.
        pass
    step_sign = torch.ones(xb.shape[0], device=device, dtype=dyn.torch_dtype)
    step_sign[torch.from_numpy(flip).to(device)] = -1.0
    for raw in signed_steps:
        action = torch.zeros(xb.shape[0], 4, device=device, dtype=dyn.torch_dtype)
        action[:, 3] = float(raw) * step_sign
        h = dyn.rssm.advance_h(h, z, action)
        z, prior_entropy = _st_code(dyn, h, None, torch, F, prior=True)
        entropy = entropy + prior_entropy
    pred = _decode(dyn, h, z, torch, F)
    if pred.shape[-2:] != db.shape[-2:]:
        pred = F.interpolate(pred.unsqueeze(1), size=db.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        pred_now = F.interpolate(
            pred_now.unsqueeze(1), size=sb.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
    loss_side = _order_loss(pred, db, torch, F, symexp, cone_mins_torch)
    loss_now = _order_loss(pred_now, sb, torch, F, symexp, cone_mins_torch)
    fwd_next = _band_mean_torch(symexp(pred), torch)
    fwd_now = _band_mean_torch(symexp(pred_now), torch)
    gt_next = _band_mean_torch(db, torch)
    gt_now = _band_mean_torch(sb, torch)
    delta = F.mse_loss(symlog(fwd_next.clamp_min(0.5)) - symlog(fwd_now.detach().clamp_min(0.5)),
                       (symlog(gt_next.clamp_min(0.5)) - symlog(gt_now.clamp_min(0.5))).detach())
    gt_std = gt_next.detach().float().std().clamp_min(1.0)
    spread_loss = torch.relu(gt_std.to(fwd_next.device) - fwd_next.float().std())
    loss = loss_side + delta + 0.25 * loss_now + spread_loss + 0.05 * entropy
    with torch.no_grad():
        fwd_std = float(fwd_next.detach().float().std())
        pred_m = symexp(pred)
        _pf, pl, pr = cone_mins_torch(pred_m, center_frac=CENTER_FRAC)
        _gf, gl, gr = cone_mins_torch(db, center_frac=CENTER_FRAC)
        gdiff = torch.log(gl.clamp_min(1e-3)) - torch.log(gr.clamp_min(1e-3))
        pdiff = torch.log(pl.clamp_min(1e-3)) - torch.log(pr.clamp_min(1e-3))
        usable = gdiff.abs() > math.log(SIDE_RATIO)
        hit = float((torch.sign(gdiff[usable]) == torch.sign(pdiff[usable])).float().mean()) if bool(usable.any()) else 0.0
    return loss, hit, fwd_std


def _order_loss(pred_sym, gt_m, torch, F, symexp, cone_mins_torch):
    from experiments.aerial.rl.dynamics_torch import _symlog

    pred_m = symexp(pred_sym)
    _pf, pl, pr = cone_mins_torch(pred_m, center_frac=CENTER_FRAC)
    gf, gl, gr = cone_mins_torch(gt_m, center_frac=CENTER_FRAC)
    gdiff = torch.log(gl.clamp_min(1e-3)) - torch.log(gr.clamp_min(1e-3))
    pdiff = torch.log(pl.clamp_min(1e-3)) - torch.log(pr.clamp_min(1e-3))
    ok = torch.isfinite(gdiff) & torch.isfinite(pdiff) & torch.isfinite(gf)
    side = ok & (gdiff.abs() > math.log(SIDE_RATIO))
    if bool(side.any()):
        order = torch.relu(-pdiff[side] * torch.sign(gdiff[side])).mean()
    else:
        order = pred_sym.new_zeros(())
    pred_fwd = _band_mean_torch(pred_m, torch)
    gt_fwd = _band_mean_torch(gt_m, torch)
    mag = F.mse_loss(_symlog(pred_fwd.clamp_min(0.5)), _symlog(gt_fwd.clamp_min(0.5)))
    return order + 0.25 * mag


def build_readout(device: str = "cuda"):
    """Yaw update of the three cones. The current cones come from the depth head."""
    import torch
    from torch import nn

    cap = math.log1p(80.0)

    class ConeReadout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.delta = nn.Sequential(
                nn.Linear(4, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 3),
            )

        def after(self, cones, signed_steps):
            if isinstance(signed_steps, (list, tuple)):
                dyaw = cones.new_full((cones.shape[0],), float(sum(float(s) for s in signed_steps)))
            else:
                dyaw = signed_steps.reshape(-1).to(device=cones.device, dtype=cones.dtype)
            return self.after_yaw(cones, dyaw)

        def after_yaw(self, cones, dyaw):
            state = cones
            n_steps = 3
            yaw = dyaw.reshape(-1).to(device=cones.device, dtype=cones.dtype)
            for _ in range(n_steps):
                step = (yaw / float(n_steps)).unsqueeze(-1)
                skip = torch.stack(
                    (
                        0.28 * step.squeeze(-1) * torch.tanh(state[:, 2] - state[:, 1]),
                        -0.18 * step.squeeze(-1),
                        0.18 * step.squeeze(-1),
                    ),
                    dim=-1,
                )
                state = (state + self.delta(torch.cat([state, step], dim=-1)) + skip).clamp(-0.5, cap)
            return state

    net = ConeReadout().to(device=device)
    torch.nn.init.zeros_(net.delta[-1].weight)
    torch.nn.init.zeros_(net.delta[-1].bias)
    return net


def _log_cones(meters: np.ndarray, torch_mod, device):
    arr = np.asarray(meters, dtype=np.float32)
    tensor = torch_mod.from_numpy(np.maximum(arr, 0.5)).to(device)
    return torch_mod.log1p(tensor)


def _gt_cones(depth: np.ndarray) -> np.ndarray:
    fwd = forward_band_mean(depth)
    left, right = side_pair(depth)
    return np.array([fwd, left, right], dtype=np.float32)


def _pair_dyaw(yaws: np.ndarray, src: int, offset: int) -> Tuple[int, float]:
    dst = (src + offset) % N_YAW
    dyaw = wrap_angle(float(yaws[dst] - yaws[src]))
    if offset < 0 and dyaw > 0:
        dyaw -= 2.0 * math.pi
    if offset > 0 and dyaw < 0:
        dyaw += 2.0 * math.pi
    return dst, dyaw


def _cone_roll_loss(readout, da3, depth, yaws, batch, torch, F, symlog, symexp):
    device = next(readout.parameters()).device
    src_da3 = []
    src_m = []
    plus_m = []
    minus_m = []
    plus_yaw = []
    minus_yaw = []
    for p_i, s_i in batch:
        src_da3.append(da3[p_i, s_i])
        src_m.append(_gt_cones(depth[p_i, s_i]))
        d_plus, y_plus = _pair_dyaw(yaws, s_i, 2)
        d_minus, y_minus = _pair_dyaw(yaws, s_i, -2)
        plus_m.append(_gt_cones(depth[p_i, d_plus]))
        minus_m.append(_gt_cones(depth[p_i, d_minus]))
        plus_yaw.append(y_plus)
        minus_yaw.append(y_minus)
    now = _log_cones(np.stack(src_da3), torch, device)
    gt_now = torch.from_numpy(np.stack(src_m)).to(device)
    gt_plus = torch.from_numpy(np.stack(plus_m)).to(device)
    gt_minus = torch.from_numpy(np.stack(minus_m)).to(device)
    yaw_plus = torch.tensor(plus_yaw, device=device, dtype=now.dtype)
    yaw_minus = torch.tensor(minus_yaw, device=device, dtype=now.dtype)
    pred_plus = readout.after_yaw(now, yaw_plus)
    pred_minus = readout.after_yaw(now, yaw_minus)

    def _one(pred, gt_next):
        valid = torch.isfinite(gt_now).all(dim=-1) & torch.isfinite(gt_next).all(dim=-1)
        valid = valid & (gt_now[:, 0] > 0.5) & (gt_next[:, 0] > 0.5)
        if not bool(valid.any()):
            return pred.sum() * 0.0, valid, pred.new_zeros(()), pred.new_zeros(())
        gdiff = torch.log(gt_next[:, 1].clamp_min(1e-3)) - torch.log(gt_next[:, 2].clamp_min(1e-3))
        pdiff = pred[:, 1] - pred[:, 2]
        side = valid & (gdiff.abs() > math.log(SIDE_RATIO))
        order = (
            torch.relu(-pdiff[side] * torch.sign(gdiff[side])).mean()
            if bool(side.any())
            else pred.new_zeros(())
        )
        loss = F.mse_loss(pred[valid], symlog(gt_next[valid])) + 2.0 * order
        loss = loss + F.mse_loss(
            pred[valid, 0] - now[valid, 0].detach(),
            (symlog(gt_next[valid, 0]) - symlog(gt_now[valid, 0])).detach(),
        )
        hit = (
            (torch.sign(pdiff[side]) == torch.sign(gdiff[side])).float().mean()
            if bool(side.any())
            else pred.new_zeros(())
        )
        return loss, valid, hit, pdiff

    loss_p, valid_p, hit_p, _pdiff = _one(pred_plus, gt_plus)
    loss_m, valid_m, hit_m, _mdiff = _one(pred_minus, gt_minus)
    valid = valid_p | valid_m
    right_open = valid & (gt_now[:, 2] > gt_now[:, 1] * SIDE_RATIO)
    left_open = valid & (gt_now[:, 1] > gt_now[:, 2] * SIDE_RATIO)
    sided = right_open | left_open
    fwd_plus = symexp(pred_plus[:, 0])
    fwd_minus = symexp(pred_minus[:, 0])
    open_fwd = torch.where(right_open, fwd_plus, fwd_minus)
    shut_fwd = torch.where(right_open, fwd_minus, fwd_plus)
    if bool(sided.any()):
        margin = torch.relu(shut_fwd[sided] - open_fwd[sided] + 3.0).mean()
    else:
        margin = pred_plus.new_zeros(())
    loss = loss_p + loss_m + margin
    with torch.no_grad():
        now_m = symexp(now[valid, 0]) if bool(valid.any()) else now[:, 0]
        now_std = float(now_m.detach().float().std()) if int(valid.sum()) > 1 else 0.0
        now_mae = (
            float((now_m - gt_now[valid, 0]).abs().mean()) if bool(valid.any()) else 0.0
        )
        hit = float(((hit_p + hit_m) * 0.5).detach())
    return loss, hit, now_std, now_mae


def evaluate_holdout(data: Dict[str, np.ndarray], hold_idx: Sequence[int], readout) -> List[dict]:
    import torch
    import torch.nn.functional as F

    from experiments.aerial.rl.dynamics_torch import _symexp

    readout.eval()
    depth = data["depth"]
    yaws = data["yaws"]
    da3 = data["da3"]
    records = []
    with torch.no_grad():
        for p_i in hold_idx:
            for src in range(N_YAW):
                plus = _predict_view(da3, yaws, p_i, src, 2, torch, F, _symexp, readout)
                minus = _predict_view(da3, yaws, p_i, src, -2, torch, F, _symexp, readout)
                spread = abs(plus["pred_fwd"] - minus["pred_fwd"])
                src_fwd = forward_band_mean(depth[p_i, src])
                src_l, src_r = side_pair(depth[p_i, src])
                for pred, other in ((plus, minus), (minus, plus)):
                    if not (math.isfinite(src_fwd) and src_fwd < NEAR_M):
                        continue
                    if not toward_open(src_l, src_r, pred["dyaw"]):
                        continue
                    dst = depth[p_i, pred["dst"]]
                    dst_fwd = forward_band_mean(dst)
                    dst_l, dst_r = side_pair(dst)
                    dest_clear = _side_clear(dst_l, dst_r)
                    records.append(
                        {
                            "pos": int(p_i),
                            "src": int(src),
                            "spread_m": float(spread),
                            "dest_clear": bool(dest_clear),
                            "dest_same_side": bool(
                                dest_clear and _same_side(dst_l, dst_r, pred["left"], pred["right"])
                            ),
                            "gt_lengthens": bool(math.isfinite(dst_fwd) and dst_fwd > src_fwd + GROW_M),
                            "pred_lengthens": bool(pred["pred_fwd"] > pred["now_fwd"] + 1.0),
                            "open_longer": bool(pred["pred_fwd"] > other["pred_fwd"]),
                            "src_fwd": round(float(src_fwd), 2),
                            "gt_fwd": round(float(dst_fwd), 2) if math.isfinite(dst_fwd) else None,
                            "pred_fwd": round(float(pred["pred_fwd"]), 2),
                            "now_fwd": round(float(pred["now_fwd"]), 2),
                        }
                    )
    return records


def _predict_view(da3, yaws, p_i, src, offset, torch, F, symexp, readout):
    dst = (src + offset) % N_YAW
    dyaw = wrap_angle(float(yaws[dst] - yaws[src]))
    if offset < 0 and dyaw > 0:
        dyaw -= 2.0 * math.pi
    if offset > 0 and dyaw < 0:
        dyaw += 2.0 * math.pi
    device = next(readout.parameters()).device
    now_c = _log_cones(da3[p_i, src], torch, device).view(1, 3)
    pred_c = readout.after_yaw(now_c, now_c.new_tensor([dyaw]))
    now = symexp(now_c[0])
    pred = symexp(pred_c[0])
    return {
        "dst": dst,
        "dyaw": float(dyaw),
        "pred_fwd": float(pred[0]),
        "now_fwd": float(now[0]),
        "left": float(pred[1]),
        "right": float(pred[2]),
    }


def cache_da3_cones(rgb: np.ndarray, depth_ckpt: Path, out_path: Path) -> np.ndarray:
    import torch

    from experiments.aerial.rl.depth_predictor import DepthMinPredictor

    if out_path.is_file():
        loaded = np.load(out_path)["da3"]
        if loaded.shape[:2] == rgb.shape[:2]:
            print(f"loaded da3 cones {loaded.shape} from {out_path}", flush=True)
            return loaded
    pred = DepthMinPredictor.from_checkpoint(str(depth_ckpt), device="cuda")
    model = pred._model
    model.eval()
    n_pos, n_yaw = rgb.shape[0], rgb.shape[1]
    flat = np.ascontiguousarray(rgb.reshape(n_pos * n_yaw, rgb.shape[2], rgb.shape[3], rgb.shape[4]))
    maps = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], 4):
            window = torch.from_numpy(flat[i : i + 4]).cuda().unsqueeze(1)
            depth, _ = model.predict_from_window(window)
            dmap = depth.detach().float().cpu().numpy()
            scale = float(getattr(pred, "forward_scale", 1.0))
            exponent = float(getattr(pred, "forward_exponent", 1.0))
            if scale != 1.0 or exponent != 1.0:
                dmap = scale * np.power(np.clip(dmap, 0.0, None), exponent)
            maps.append(dmap)
    stacked = np.concatenate(maps, axis=0)
    cones = np.stack([_gt_cones(stacked[i]) for i in range(stacked.shape[0])]).reshape(n_pos, n_yaw, 3)
    for p_i in range(n_pos):
        print(f"da3 pos {p_i} fwd {cones[p_i, :, 0].min():.1f}-{cones[p_i, :, 0].max():.1f}", flush=True)
    np.savez_compressed(out_path, da3=cones)
    del pred, model
    print(f"saved da3 cones {out_path}", flush=True)
    return cones


def train_actor(dyn, ac, data, train_idx: Sequence[int], steps: int) -> None:
    import torch

    from experiments.aerial.rl.dynamics_torch import _symlog
    from experiments.aerial.rl.goal_features import goal_rel_body

    for param in dyn.parameters():
        param.requires_grad_(False)
    dyn.eval()
    rgb = data["rgb"]
    pos = data["pos"]
    yaws = data["yaws"]
    goal = np.asarray(data["goal"], dtype=np.float64)
    rng = np.random.default_rng(1)
    sources = [(int(p), int(s)) for p in train_idx for s in range(N_YAW)]
    for step in range(int(steps)):
        choice = rng.choice(len(sources), size=min(8, len(sources)), replace=False)
        batch = [sources[int(i)] for i in choice]
        loss = _actor_loss(dyn, ac, rgb, pos, yaws, goal, batch, torch, _symlog, goal_rel_body)
        ac._actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(ac._actor.parameters()) + [ac._log_std], 100.0)
        ac._actor_opt.step()
        if step % 25 == 0 or step == int(steps) - 1:
            print(f"actor step {step} loss={float(loss):.3f}", flush=True)


def _actor_loss(dyn, ac, rgb, pos, yaws, goal, batch, torch, symlog, goal_rel_body):
    images = []
    proprios = []
    goals = []
    for p_i, s_i in batch:
        images.append(rgb[p_i, s_i])
        yaw = float(yaws[s_i])
        proprios.append([pos[p_i, 0], pos[p_i, 1], pos[p_i, 2], yaw])
        goals.append(goal_rel_body(pos[p_i], yaw, goal))
    xb = torch.from_numpy(np.stack(images)).to(dyn.device)
    pb = torch.from_numpy(np.asarray(proprios, dtype=np.float32)).to(dyn.device, dyn.torch_dtype)
    with torch.no_grad():
        h = dyn.rssm.initial_h(xb.shape[0], dyn.device, dyn.torch_dtype)
        z0 = _mode_z(dyn, dyn.rssm.post_probs(h, dyn._embed(xb, pb)), torch, torch.nn.functional, hard=False)
        now = _decode(dyn, h, z0, torch, torch.nn.functional)
        from experiments.aerial.rl.dynamics_torch import _symexp

        fwd_now = _band_mean_torch(_symexp(now), torch)
    z_np = z0.detach().float().cpu().numpy()
    g_np = np.stack(goals).astype(np.float32)
    feat = ac._feat_tensor(z_np, g_np)
    mean, _std = ac._pre_dist(feat)
    action = ac._limits * torch.tanh(mean)
    h = h.detach()
    z = z0.detach()
    dyaw = action[:, 3]
    for _ in range(3):
        rolled = torch.zeros(xb.shape[0], 4, device=dyn.device, dtype=dyn.torch_dtype)
        rolled[:, 3] = dyaw.to(dyn.torch_dtype)
        h = dyn.rssm.advance_h(h, z, rolled)
        z = _mode_z(dyn, dyn.rssm.prior_probs(h), torch, torch.nn.functional, hard=False)
    from experiments.aerial.rl.dynamics_torch import _symexp

    fwd_next = _band_mean_torch(_symexp(_decode(dyn, h, z, torch, torch.nn.functional)), torch)
    gain = symlog(fwd_next.clamp_min(0.5)) - symlog(fwd_now.clamp_min(0.5))
    gain = gain.to(action.device)
    dx_pos = action[:, 0].clamp(min=0.0)
    bearing = torch.atan2(
        torch.as_tensor(g_np[:, 1], device=action.device),
        torch.as_tensor(g_np[:, 0], device=action.device).clamp_min(1e-3),
    )
    heading = bearing - 3.0 * dyaw
    heading = torch.atan2(torch.sin(heading), torch.cos(heading))
    return -(gain + dx_pos * gain.detach()).mean() + 0.3 * heading.abs().mean()


def actor_sign_rate(dyn, ac, data, hold_idx: Sequence[int]) -> Tuple[int, int]:
    import torch

    from experiments.aerial.rl.goal_features import goal_rel_body

    dyn.eval()
    ac_device = ac._device
    rgb = data["rgb"]
    depth = data["depth"]
    pos = data["pos"]
    yaws = data["yaws"]
    goal = np.asarray(data["goal"], dtype=np.float64)
    hit = 0
    total = 0
    with torch.no_grad():
        for p_i in hold_idx:
            for src in range(N_YAW):
                src_fwd = forward_band_mean(depth[p_i, src])
                left, right = side_pair(depth[p_i, src])
                if not (math.isfinite(src_fwd) and src_fwd < NEAR_M and _side_clear(left, right)):
                    continue
                image = torch.from_numpy(rgb[p_i, src]).unsqueeze(0).to(dyn.device)
                yaw = float(yaws[src])
                proprio = torch.tensor(
                    [[pos[p_i, 0], pos[p_i, 1], pos[p_i, 2], yaw]],
                    device=dyn.device,
                    dtype=dyn.torch_dtype,
                )
                h = dyn.rssm.initial_h(1, dyn.device, dyn.torch_dtype)
                z = _mode_z(
                    dyn,
                    dyn.rssm.post_probs(h, dyn._embed(image, proprio)),
                    torch,
                    torch.nn.functional,
                )
                g = goal_rel_body(pos[p_i], yaw, goal)
                action = ac.act_latent(z.float().cpu().numpy()[0], goal_rel=g, deterministic=True)
                open_sign = 1.0 if left > right else -1.0
                total += 1
                if float(action[3]) * open_sign > 0.0:
                    hit += 1
    _ = ac_device
    return hit, total


def build_stick(device: str = "cpu"):
    """Stick from the current cones and the goal. Its yaw is scored by the readout."""
    import torch
    from torch import nn

    class ConeStick(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(6, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 3),
            )

        def forward(self, cones, goal):
            feat = torch.cat(
                [
                    cones,
                    goal[:, 0:1] / 50.0,
                    goal[:, 1:2] / 50.0,
                    torch.log1p(goal[:, 3:4].clamp_min(0.0)) / 5.0,
                ],
                dim=-1,
            )
            raw = self.net(feat)
            dx = (torch.tanh(raw[:, 0]) + 1.0) * 0.5
            near = torch.sigmoid((math.log1p(NEAR_M) - cones[:, 0]) * 6.0)
            dyaw = (near * torch.tanh(raw[:, 1]) + (1.0 - near) * torch.tanh(raw[:, 2])) * YAW_LIMIT
            return dx, dyaw

    return ConeStick().to(device)


def train_stick(readout, data, train_idx, hold_idx, steps: int):
    """Choose a yaw by how much the readout lengthens the street, then face the goal."""
    import torch
    import torch.nn.functional as F

    from experiments.aerial.rl.dynamics_torch import _symexp
    from experiments.aerial.rl.goal_features import goal_rel_body

    device = next(readout.parameters()).device
    readout.eval()
    for param in readout.parameters():
        param.requires_grad_(False)
    stick = build_stick(str(device))
    opt = torch.optim.AdamW(stick.parameters(), lr=1e-3, betas=(0.9, 0.95))
    da3 = data["da3"]
    pos = data["pos"]
    yaws = data["yaws"]
    goal = np.asarray(data["goal"], dtype=np.float64)
    sources = [(int(p), int(s)) for p in train_idx for s in range(N_YAW)]
    rng = np.random.default_rng(1)
    for step in range(int(steps)):
        choice = rng.choice(len(sources), size=min(16, len(sources)), replace=False)
        batch = [sources[int(i)] for i in choice]
        now = _log_cones(np.stack([da3[p, s] for p, s in batch]), torch, device)
        goals = np.stack(
            [goal_rel_body(pos[p], float(yaws[s]), goal) for p, s in batch]
        ).astype(np.float32)
        g = torch.from_numpy(goals).to(device)
        _dx, dyaw = stick(now, g)
        pred = readout.after_yaw(now, dyaw)
        now_m = _symexp(now[:, 0])
        gain = _symexp(pred[:, 0]) - now_m
        wall = (now_m < NEAR_M).float().detach()
        street = 1.0 - wall
        bearing = torch.atan2(g[:, 1], g[:, 0])
        heading = torch.atan2(torch.sin(bearing - dyaw), torch.cos(bearing - dyaw))
        fwd = _symexp(pred[:, 0])
        align = torch.cos(heading).clamp(min=0.0)
        opened = (gain.detach() > 1.0).float()
        target_dx = (fwd.detach() / 25.0).clamp(0.0, 1.0) * align.detach()
        target_dx = target_dx * (street + wall * opened)
        wsum = wall.sum().clamp_min(1.0)
        ssum = street.sum().clamp_min(1.0)
        loss = (
            -(gain * wall).sum() / wsum
            + 4.0 * (heading.abs() * street).sum() / ssum
            + F.mse_loss(_dx, target_dx)
        )
        if not bool(torch.isfinite(loss)):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(stick.parameters()), 10.0)
        opt.step()
        if step % 50 == 0 or step == int(steps) - 1:
            print(
                f"stick step {step} loss={float(loss.detach()):.3f} "
                f"gain={float(gain.detach().mean()):.2f} "
                f"head={float(heading.detach().abs().mean()):.2f}",
                flush=True,
            )
    hit, total, grew = stick_sign_rate(stick, readout, data, hold_idx)
    ok = total >= 4 and hit / float(total) >= 0.7 and grew / float(total) >= 0.7
    print(f"HOLD stick sign {hit}/{total} grew={grew} passed={ok}", flush=True)
    return ok, stick, hit, total, grew


def stick_sign_rate(stick, readout, data, hold_idx: Sequence[int]) -> Tuple[int, int, int]:
    import torch

    from experiments.aerial.rl.dynamics_torch import _symexp
    from experiments.aerial.rl.goal_features import goal_rel_body

    device = next(readout.parameters()).device
    stick.eval()
    readout.eval()
    depth = data["depth"]
    da3 = data["da3"]
    pos = data["pos"]
    yaws = data["yaws"]
    goal = np.asarray(data["goal"], dtype=np.float64)
    hit = total = grew = 0
    with torch.no_grad():
        for p_i in hold_idx:
            for src in range(N_YAW):
                src_fwd = forward_band_mean(depth[p_i, src])
                left, right = side_pair(depth[p_i, src])
                if not (math.isfinite(src_fwd) and src_fwd < NEAR_M and _side_clear(left, right)):
                    continue
                now = _log_cones(da3[p_i, src], torch, device).view(1, 3)
                g = torch.from_numpy(
                    goal_rel_body(pos[p_i], float(yaws[src]), goal)
                ).to(device).view(1, 4)
                _dx, dyaw = stick(now, g)
                total += 1
                if toward_open(left, right, float(dyaw.reshape(-1)[0])):
                    hit += 1
                pred = readout.after_yaw(now, dyaw.reshape(-1))
                if float(_symexp(pred[0, 0])) > float(_symexp(now[0, 0])) + 1.0:
                    grew += 1
    return hit, total, grew


class YawPriorDeploy:
    """Current frame sets the cones. The readout scores the yaw. That yaw is the stick."""

    def __init__(self, predictor, readout, stick, device: str) -> None:
        self.predictor = predictor
        self.readout = readout
        self.stick = stick
        self.device = device
        self.last_logit = None
        self._last_action = np.zeros(4, dtype=np.float32)

    @classmethod
    def load(cls, spec: str, device: str) -> "YawPriorDeploy":
        import torch

        from experiments.aerial.rl.depth_predictor import DepthMinPredictor

        depth_path, readout_path, stick_path = spec.split(",")
        predictor = DepthMinPredictor.from_checkpoint(depth_path, device=device)
        readout = build_readout(device)
        readout.load_state_dict(torch.load(readout_path, map_location=device))
        readout.eval()
        stick = build_stick(device)
        stick.load_state_dict(torch.load(stick_path, map_location=device))
        stick.eval()
        return cls(predictor, readout, stick, device)

    def act(self, obs) -> np.ndarray:
        import torch

        from experiments.aerial.rl.dynamics_torch import _symexp
        from experiments.aerial.rl.goal_features import goal_rel_from_obs

        self.predictor.reset()
        depth = self.predictor.predict_depth(obs)
        cones = None if depth is None else _gt_cones(np.asarray(depth, dtype=np.float32))
        goal_np = goal_rel_from_obs(obs)
        if cones is None or not np.isfinite(cones).all() or not np.isfinite(goal_np).all():
            return self._last_action.copy()
        now = _log_cones(cones, torch, self.device).view(1, 3)
        goal = torch.as_tensor(goal_np, dtype=torch.float32, device=self.device).view(1, 4)
        with torch.no_grad():
            _dx, dyaw = self.stick(now, goal)
            if not (torch.isfinite(_dx).all() and torch.isfinite(dyaw).all()):
                return self._last_action.copy()
            pred = self.readout.after_yaw(now, dyaw.reshape(-1))
            fwd = float(_symexp(pred[0, 0]))
            now_f = float(_symexp(now[0, 0]))
            yaw = float(dyaw.reshape(-1)[0])
            if not math.isfinite(fwd):
                return self._last_action.copy()
            dx = float(_dx.reshape(-1)[0])
            if math.isfinite(now_f) and now_f < NEAR_M and fwd < now_f + 1.0:
                dx = 0.0
        action = np.array([dx, 0.0, 0.0, yaw], dtype=np.float32)
        if not np.isfinite(action).all():
            return self._last_action.copy()
        self._last_action = action
        self.last_logit = {
            "now": round(now_f, 2) if math.isfinite(now_f) else None,
            "fwd": round(fwd, 2),
            "dyaw": round(yaw, 3),
        }
        return action


def save_actor(ac, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": ac._actor.state_dict(),
            "critic": ac._critic.state_dict(),
            "log_std": ac._log_std.detach().cpu(),
            "config": ac.config.__dict__,
        },
        path,
    )


def run(args) -> int:
    import torch

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    wm_path = Path(args.wm_ckpt)
    if not wm_path.is_absolute():
        wm_path = root / wm_path
    depth_ckpt = Path(getattr(args, "depth_ckpt", "") or "")
    if not depth_ckpt.is_absolute():
        depth_ckpt = root / depth_ckpt
    signed = root / "experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"
    anno = Path(args.annotation)
    if not anno.is_absolute():
        anno = root / anno
    pairs_path = out_dir / "pairs.npz"
    cone_path = out_dir / "cone_readout.pt"
    assert_safe_output(cone_path, [wm_path, depth_ckpt, signed])

    if not pairs_path.is_file() or bool(args.collect):
        route_xy, goal_pos = load_route(anno, int(args.route))
        sampled = sample_route_xy(route_xy, spacing=float(args.spacing))
        goal = np.array([goal_pos[0], goal_pos[1], ALT_Z], dtype=np.float64)
        print(f"collect {len(sampled)} positions goal={goal.tolist()}", flush=True)
        collect_fans(sampled, goal, pairs_path, camera=str(args.camera), z_up=ALT_Z)

    blob = np.load(pairs_path)
    data = {key: blob[key] for key in blob.files}
    data["da3"] = cache_da3_cones(data["rgb"], depth_ckpt, out_dir / "da3_cones.npz")
    train_idx, hold_idx, events = split_positions(data["depth"], data["yaws"])
    print(f"train_pos={train_idx} hold_pos={hold_idx}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reuse = cone_path.is_file() and not bool(getattr(args, "retrain_prior", False))
    if reuse:
        readout = build_readout(device)
        readout.load_state_dict(torch.load(cone_path, map_location=device))
        readout.eval()
        for param in readout.parameters():
            param.requires_grad_(False)
        print(f"loaded readout {cone_path}", flush=True)
    else:
        passed, records, readout = train_prior(data, train_idx, int(args.steps), hold_idx)
        passed = bool(passed and gate_pass(records))
        report = {
            "passed": bool(passed),
            "events": events,
            "train_pos": train_idx,
            "hold_pos": hold_idx,
            "n_records": len(records),
            "n_clear": int(sum(1 for r in records if r["dest_clear"])),
            "n_same": int(sum(1 for r in records if r["dest_clear"] and r["dest_same_side"])),
            "n_gt_lengthens": int(sum(1 for r in records if r["gt_lengthens"])),
            "n_pred_lengthens": int(sum(1 for r in records if r["gt_lengthens"] and r["pred_lengthens"])),
            "records": records,
            "depth_ckpt": str(depth_ckpt),
        }
        (out_dir / "holdout_prior.json").write_text(json.dumps(report, indent=2))
        print(
            f"HOLD prior passed={passed} records={len(records)} "
            f"clear={report['n_clear']} same={report['n_same']} "
            f"grew={report['n_gt_lengthens']} pred_grew={report['n_pred_lengthens']}",
            flush=True,
        )
        if not passed:
            print("prior gate failed; weights not saved", flush=True)
            return 1
        torch.save(readout.state_dict(), cone_path)
        print(f"saved {cone_path}", flush=True)

    stick_path = out_dir / "cone_stick.pt"
    assert_safe_output(stick_path, [wm_path, depth_ckpt, signed, cone_path])
    stick_steps = int(getattr(args, "stick_steps", 400))
    ok, stick, hit, total, grew = train_stick(readout, data, train_idx, hold_idx, stick_steps)
    stick_report = {
        "passed": bool(ok),
        "sign_hit": int(hit),
        "sign_total": int(total),
        "grew": int(grew),
        "hold_pos": list(hold_idx),
    }
    (out_dir / "holdout_stick.json").write_text(json.dumps(stick_report, indent=2))
    if not ok:
        print("stick gate failed; weights not saved", flush=True)
        return 1
    torch.save(stick.state_dict(), stick_path)
    print(f"saved {stick_path}", flush=True)
    return 0
