#!/usr/bin/env python3
"""Train the DA3 decoder on street-level camera yaws rendered in AirSim.

Poses come from the frozen street corners at 14 m. Routes 2, 6 and 10 stay
out of training. Physics is paused after each teleport so the vehicle cannot
fall between the pose command and the grab. The encoder stays frozen. A sky
half is pinned at 80 m. Left/right order is supervised only when one half is
a wall and the other is past the opening; when both halves are already past
it, both predictions must stay past it and their order is not trained.
The flight checkpoint is left in place.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

HOLDOUT_ROUTES = (2, 6, 10)
SKY_M = 80.0
OPEN_M = 24.0
MARGIN = math.log(1.5)
OFFSETS = (-40, 0, 40)
POSE_K = 12
HOLD_K = 8


def training_routes(route_ids) -> list[int]:
    hold = {int(i) for i in HOLDOUT_ROUTES}
    return [int(i) for i in route_ids if int(i) not in hold]


def sample_pose_indices(n: int, k: int = POSE_K) -> list[int]:
    n = int(n)
    k = int(k)
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    out: list[int] = []
    for value in np.linspace(0, n - 1, k):
        idx = int(round(float(value)))
        if idx not in out:
            out.append(idx)
    return out


def side_target_m(median_m: float, sky_m: float = SKY_M) -> float:
    """Map a half's true median onto the training target.

    Sky stays at ``sky_m`` instead of the old 40 m cap or the raw 3000 m fill.
    """
    value = float(median_m)
    if not math.isfinite(value) or value <= 0.05 or value > float(sky_m):
        return float(sky_m)
    return value


def _half_median_np(depth: np.ndarray) -> tuple[float, float]:
    d = np.asarray(depth, dtype=np.float64)
    h, w = d.shape
    dh = max(1, int(h * 0.5))
    r0 = (h - dh) // 2
    band = d[r0 : r0 + dh]
    mid = w // 2

    def med(crop: np.ndarray) -> float:
        valid = crop[np.isfinite(crop) & (crop > 0.05)]
        if valid.size == 0:
            return float("inf")
        return float(np.median(valid))

    return med(band[:, :mid]), med(band[:, mid:])


def _yaw_side_loss(pred_l, pred_r, gt_l, gt_r):
    import torch

    sky = float(SKY_M)
    open_m = float(OPEN_M)
    log_l = pred_l.clamp(min=0.5).log()
    log_r = pred_r.clamp(min=0.5).log()
    tl = torch.where(gt_l > sky, torch.full_like(gt_l, sky), gt_l.clamp(min=0.5))
    tr = torch.where(gt_r > sky, torch.full_like(gt_r, sky), gt_r.clamp(min=0.5))
    gap_g = tl.log() - tr.log()
    gap_p = log_l - log_r
    wall_open = ((gt_l <= open_m) & (gt_r > open_m)) | ((gt_r <= open_m) & (gt_l > open_m))
    clear = wall_open & (gap_g.abs() >= float(MARGIN))
    hinge = torch.relu(float(MARGIN) - gap_g.detach().sign() * gap_p)
    loss_gap = hinge[clear].mean() if bool(clear.any()) else pred_l.sum() * 0.0

    far = math.log(sky)
    loss_sky = pred_l.sum() * 0.0
    if bool((gt_l > sky).any()):
        loss_sky = loss_sky + (log_l - far).square()[gt_l > sky].mean()
    if bool((gt_r > sky).any()):
        loss_sky = loss_sky + (log_r - far).square()[gt_r > sky].mean()

    # Both halves already clear the opening: keep both predictions clear,
    # and do not train which one is farther.
    floor = math.log(open_m)
    both = (gt_l > open_m) & (gt_r > open_m)
    loss_open = pred_l.sum() * 0.0
    if bool(both.any()):
        loss_open = loss_open + torch.relu(floor - log_l)[both].mean()
        loss_open = loss_open + torch.relu(floor - log_r)[both].mean()

    loss_cap = pred_l.sum() * 0.0
    if bool((gt_l <= sky).any()):
        loss_cap = loss_cap + torch.relu(log_l - far)[gt_l <= sky].mean()
    if bool((gt_r <= sky).any()):
        loss_cap = loss_cap + torch.relu(log_r - far)[gt_r <= sky].mean()
    return loss_gap + loss_sky + loss_open + loss_cap


def _half_median_torch(depth):
    import torch

    b, h, w = depth.shape
    dh = max(1, int(h * 0.5))
    r0 = (h - dh) // 2
    band = depth[:, r0 : r0 + dh]
    mid = w // 2

    def med(crop):
        ordered, _ = torch.sort(crop.reshape(b, -1), dim=-1)
        return ordered[:, ordered.shape[1] // 2]

    return med(band[:, :, :mid]), med(band[:, :, mid:])


def _detach_feats(feats):
    out = []
    for feat in feats:
        if isinstance(feat, (list, tuple)):
            head = feat[0].detach()
            out.append(type(feat)((head, *feat[1:])))
        else:
            out.append(feat.detach())
    return out


def _root() -> Path:
    return Path(os.environ.get("AERIAL_ROOT", "/data/aerial-wam-v2"))


def _paths():
    root = _root()
    out = root / "experiments/aerial/rl/artifacts/da3_yaw_street_20261005"
    anno = root / "experiments/aerial/phase3_unified/annotations/street_corners_frozen.json"
    ckpt = (
        root
        / "experiments/aerial/rl/artifacts/depth_ckpt_p45mid_s8j_20260825/depth_best_holdout_da3_ft_head.pt"
    )
    return out, anno, ckpt


def _load_routes():
    _out, anno, _ckpt = _paths()
    payload = json.loads(anno.read_text())
    return payload["routes"]


def _wrap_deg(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _quat_yaw_deg(q) -> float:
    siny = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.degrees(math.atan2(siny, cosy))


def _connect():
    import airsim

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


def _lock_pose(client, x, y, z_up, yaw_deg):
    import airsim

    quat = airsim.to_quaternion(0.0, 0.0, math.radians(float(yaw_deg)))
    pose = airsim.Pose(airsim.Vector3r(float(x), float(y), -float(z_up)), quat)

    def _seat():
        client.simPause(True)
        client.simSetVehiclePose(pose, True)
        time.sleep(0.05)
        got_pose = client.simGetVehiclePose()
        z_err = abs(float(got_pose.position.z_val) - (-float(z_up)))
        yaw_err = abs(_wrap_deg(_quat_yaw_deg(got_pose.orientation) - float(yaw_deg)))
        return got_pose, z_err, yaw_err

    got, z_err, yaw_err = _seat()
    if z_err > 1.0 or yaw_err > 3.0:
        # A paused pose set can be ignored. Seat once while physics is running,
        # then pause again before the depth grab.
        client.simPause(False)
        client.simSetVehiclePose(pose, True)
        time.sleep(0.05)
        got, z_err, yaw_err = _seat()
    if z_err > 1.0 or yaw_err > 3.0:
        raise RuntimeError(
            f"pose did not lock z_err={z_err:.2f} yaw_err={yaw_err:.1f} "
            f"cmd=({x:.1f},{y:.1f},{z_up:.1f},{yaw_deg:.1f})"
        )
    return got


def _grab(client, x, y, z_up, yaw_deg):
    import airsim

    _lock_pose(client, x, y, z_up, yaw_deg)
    scene, depth_resp = client.simGetImages(
        [
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False),
        ]
    )
    bgr = np.frombuffer(scene.image_data_uint8, dtype=np.uint8).reshape(
        scene.height, scene.width, 3
    )
    depth = airsim.list_to_2d_float_array(
        depth_resp.image_data_float, depth_resp.width, depth_resp.height
    )
    return bgr[:, :, ::-1].copy(), np.asarray(depth, dtype=np.float32)


def _views(routes, route_ids, k: int):
    wanted = {int(i) for i in route_ids}
    for route in routes:
        rid = int(route["route_idx"])
        if rid not in wanted:
            continue
        idxs = sample_pose_indices(len(route["pos"]), k)
        for idx in idxs:
            x, y, z = (float(v) for v in route["pos"][idx])
            yaw = float(route["yaw"][idx])
            for off in OFFSETS:
                yield rid, idx, off, x, y, z, yaw + float(off)


def _stable(first, second) -> bool:
    for u, v in zip(first, second):
        if not (math.isfinite(u) and math.isfinite(v)):
            return False
        if u > SKY_M and v > SKY_M:
            continue
        if abs(math.log(max(u, 0.5)) - math.log(max(v, 0.5))) > MARGIN:
            return False
    return True


def _save_views(client, routes, route_ids, k: int, path: Path) -> None:
    import cv2

    rgbs, med_l, med_r, route_ids_out, pose_ids, offs = [], [], [], [], [], []
    for rid, idx, off, x, y, z, yaw in _views(routes, route_ids, k):
        if not 13.0 <= z <= 15.0:
            raise RuntimeError(f"route {rid} pose {idx} is at z={z:.1f}, want 14 m")
        rgb, gt = _grab(client, x, y, z, yaw)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        left, right = _half_median_np(gt)
        rgbs.append(rgb)
        med_l.append(left)
        med_r.append(right)
        route_ids_out.append(rid)
        pose_ids.append(idx)
        offs.append(off)
        if len(rgbs) % 12 == 0:
            print(f"collect {path.name} n={len(rgbs)} route {rid}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rgb=np.stack(rgbs).astype(np.uint8),
        med_l=np.asarray(med_l, dtype=np.float32),
        med_r=np.asarray(med_r, dtype=np.float32),
        route=np.asarray(route_ids_out, dtype=np.int32),
        pose=np.asarray(pose_ids, dtype=np.int32),
        offset=np.asarray(offs, dtype=np.int32),
    )
    print(f"wrote {path} n={len(rgbs)}", flush=True)


def _lock_check(client, routes) -> None:
    route = routes[0]
    x, y, z = (float(v) for v in route["pos"][0])
    yaw = float(route["yaw"][0])
    _lock_pose(client, x, y, z, yaw)
    _, gt_a = _grab(client, x, y, z, yaw)
    _, gt_b = _grab(client, x, y, z, yaw)
    med_a, med_b = _half_median_np(gt_a), _half_median_np(gt_b)
    print(
        f"lock z={z:.1f} yaw={yaw:.1f} med {med_a[0]:.1f}/{med_a[1]:.1f} "
        f"repeat {med_b[0]:.1f}/{med_b[1]:.1f}",
        flush=True,
    )
    if not _stable(med_a, med_b):
        raise RuntimeError(f"locked pose medians moved {med_a} -> {med_b}")


def collect() -> None:
    out, _anno, _ckpt = _paths()
    out.mkdir(parents=True, exist_ok=True)
    routes = _load_routes()
    ids = [int(r["route_idx"]) for r in routes]
    client = _connect()
    try:
        _lock_check(client, routes)
        _save_views(client, routes, training_routes(ids), POSE_K, out / "yaw_street.npz")
        _save_views(client, routes, HOLDOUT_ROUTES, HOLD_K, out / "holdout_street.npz")
    finally:
        client.simPause(False)


def _load_model():
    import torch
    from experiments.aerial.rl.dynamics_torch import build_depth_head

    _out, _anno, ckpt = _paths()
    payload = torch.load(str(ckpt), map_location="cpu")
    model = build_depth_head(payload)
    model.load_state_dict(payload["model"], strict=True)
    model.cuda().eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    return model, payload


def _predict_depth(model, rgb_uint8):
    window = rgb_uint8.unsqueeze(1)
    x = model._prep_last_frame(window)
    with __import__("torch").no_grad():
        feats, _aux = model.encoder(
            x.unsqueeze(1), cam_token=None, export_feat_layers=[]
        )
    feats = _detach_feats(feats)
    out = model.decoder(feats, int(x.shape[-2]), int(x.shape[-1]), patch_start_idx=0)
    return out["depth"][:, 0]


def _score_split(model, rgb, med_l, med_r) -> tuple[int, int, int, int]:
    import torch

    model.eval()
    rank_ok = rank_n = far_ok = far_n = 0
    with torch.no_grad():
        for i in range(0, rgb.shape[0], 4):
            batch = torch.from_numpy(rgb[i : i + 4]).cuda()
            depth = _predict_depth(model, batch)
            pl, pr = _half_median_torch(depth)
            gl = torch.from_numpy(med_l[i : i + 4]).cuda()
            gr = torch.from_numpy(med_r[i : i + 4]).cuda()
            tl = torch.where(gl > SKY_M, torch.full_like(gl, SKY_M), gl.clamp(min=0.5))
            tr = torch.where(gr > SKY_M, torch.full_like(gr, SKY_M), gr.clamp(min=0.5))
            gap_g = tl.log() - tr.log()
            gap_p = pl.clamp(min=0.5).log() - pr.clamp(min=0.5).log()
            wall_open = ((gl <= OPEN_M) & (gr > OPEN_M)) | ((gr <= OPEN_M) & (gl > OPEN_M))
            clear = wall_open & (gap_g.abs() >= MARGIN)
            agree = clear & (gap_g.sign() == gap_p.sign())
            both = (gl > OPEN_M) & (gr > OPEN_M)
            far_hit = both & (pl > OPEN_M) & (pr > OPEN_M)
            rank_n += int(clear.sum())
            rank_ok += int(agree.sum())
            far_n += int(both.sum())
            far_ok += int(far_hit.sum())
    return rank_ok, rank_n, far_ok, far_n


def train(npz_path: Path) -> Path:
    import torch

    data = np.load(npz_path)
    rgb = data["rgb"]
    med_l = data["med_l"]
    med_r = data["med_r"]
    model, payload = _load_model()
    print("train before", _score_split(model, rgb, med_l, med_r), flush=True)
    opt = torch.optim.AdamW(model.decoder.parameters(), lr=1e-5, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    n = rgb.shape[0]
    for step in range(400):
        model.decoder.train()
        pick = rng.integers(0, n, size=4)
        batch = torch.from_numpy(rgb[pick].copy()).cuda()
        gl = torch.from_numpy(med_l[pick].copy()).cuda()
        gr = torch.from_numpy(med_r[pick].copy()).cuda()
        depth = _predict_depth(model, batch)
        pl, pr = _half_median_torch(depth)
        loss = _yaw_side_loss(pl, pr, gl, gr)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 40 == 0 or step == 399:
            model.eval()
            print(
                f"step {step} loss {float(loss.detach()):.4f} "
                f"acc {_score_split(model, rgb, med_l, med_r)}",
                flush=True,
            )
    out, _anno, _ckpt = _paths()
    out.mkdir(parents=True, exist_ok=True)
    payload["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload["yaw_street_finetune"] = True
    payload["holdout_routes"] = list(HOLDOUT_ROUTES)
    payload["open_m"] = OPEN_M
    payload["sky_m"] = SKY_M
    dest = out / "depth_yaw_street.pt"
    torch.save(payload, dest)
    print(f"wrote {dest}", flush=True)
    return dest


def _judge(gl: float, gr: float, pl: float, pr: float):
    if gl > OPEN_M and gr > OPEN_M:
        return "both_far", bool(pl > OPEN_M and pr > OPEN_M)
    wall_open = (gl <= OPEN_M < gr) or (gr <= OPEN_M < gl)
    if not wall_open:
        return "skip", None
    tl = SKY_M if gl > SKY_M else max(gl, 0.5)
    tr = SKY_M if gr > SKY_M else max(gr, 0.5)
    gap = math.log(tl) - math.log(tr)
    if abs(gap) < MARGIN:
        return "skip", None
    pred_gap = math.log(max(pl, 0.5)) - math.log(max(pr, 0.5))
    return "rank", (gap > 0) == (pred_gap > 0)


def score_holdout(ckpt: Path, tag: str) -> None:
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation

    out, _anno, _flight = _paths()
    data = np.load(out / "holdout_street.npz")
    pred = DepthMinPredictor.from_checkpoint(ckpt, device="cuda")
    rank_ok = rank_n = far_ok = far_n = 0
    frames = []
    for i in range(data["rgb"].shape[0]):
        pred.reset()
        hat = pred.predict_depth(
            Observation(rgb=data["rgb"][i], state=np.zeros(7, dtype=np.float32))
        )
        pl, pr = _half_median_np(hat)
        gl, gr = float(data["med_l"][i]), float(data["med_r"][i])
        kind, ok = _judge(gl, gr, pl, pr)
        if kind == "rank":
            rank_n += 1
            rank_ok += int(bool(ok))
        elif kind == "both_far":
            far_n += 1
            far_ok += int(bool(ok))
        row = {
            "route": int(data["route"][i]),
            "pose": int(data["pose"][i]),
            "offset": int(data["offset"][i]),
            "kind": kind,
            "ok": ok,
            "da3_med": [round(pl, 2), round(pr, 2)],
            "gt_med": [round(gl, 2), round(gr, 2)],
        }
        frames.append(row)
        if kind != "skip" and not ok:
            print(
                f"{tag} route {row['route']} pose {row['pose']} {row['offset']:+d} "
                f"{kind} da3 L{pl:.1f} R{pr:.1f} gt L{gl:.1f} R{gr:.1f}",
                flush=True,
            )
    summary = {
        "tag": tag,
        "rank": f"{rank_ok}/{rank_n}",
        "both_far": f"{far_ok}/{far_n}",
        "frames": frames,
    }
    print(f"{tag} RANK {summary['rank']} BOTH_FAR {summary['both_far']}", flush=True)
    dest = out / f"holdout_{tag}.json"
    dest.write_text(json.dumps(summary, indent=2))
    del pred


def main() -> None:
    out, _anno, flight = _paths()
    if os.environ.get("YAW_LOCK_CHECK") == "1":
        routes = _load_routes()
        client = _connect()
        try:
            _lock_check(client, routes)
        finally:
            client.simPause(False)
        return
    train_npz = out / "yaw_street.npz"
    hold_npz = out / "holdout_street.npz"
    if not train_npz.is_file() or not hold_npz.is_file():
        collect()
    dest = train(train_npz)
    score_holdout(flight, "flight")
    score_holdout(dest, "street")


if __name__ == "__main__":
    main()
