#!/usr/bin/env python3
"""Collect 90° corner demonstrations and train forward, inverse, and policy heads.

The image encoder stays frozen. Hold out corners 2, 6, and 10. Route 10 is not
a demonstration. Flight weights of the depth model are not overwritten.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.aerial.rl.corner90 import (
    CHUNK,
    EXEC,
    HOLDOUT_ROUTES,
    Corner90Heads,
    chunk_policy_loss,
    corner90_losses,
    demo_routes,
    pool_tokens,
)
from experiments.aerial.rl.depth_geometry import approach_clearances
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.goal_features import goal_rel_body

ROOT = Path("/data/aerial-wam-v2")
TRAJ = ROOT / "experiments/aerial/rl/artifacts/eval_street_corners_frozen_20261005/polyline/traj"
ANN = ROOT / "experiments/aerial/phase3_unified/annotations/street_corners_frozen.json"
DEPTH = ROOT / "experiments/aerial/rl/artifacts/da3_yaw_street_20261005/depth_yaw_street.pt"
OUT = ROOT / "experiments/aerial/rl/artifacts/corner90_20261006"
Z_FALLBACK = 14.0


def _yaw_of(q) -> float:
    import airsim

    return math.degrees(airsim.to_eularian_angles(q)[2])


def _place(client, x, y, z, yaw_deg):
    import airsim

    quat = airsim.to_quaternion(0.0, 0.0, math.radians(float(yaw_deg)))
    pose = airsim.Pose(airsim.Vector3r(float(x), float(y), -float(z)), quat)
    client.simPause(False)
    client.simSetVehiclePose(pose, True, "drone_1")
    time.sleep(0.05)
    client.simPause(True)
    client.simSetVehiclePose(pose, True, "drone_1")
    time.sleep(0.03)
    got = client.simGetVehiclePose("drone_1")
    xy = math.hypot(got.position.x_val - x, got.position.y_val - y)
    yaw_err = abs((_yaw_of(got.orientation) - yaw_deg + 180.0) % 360.0 - 180.0)
    if xy > 1.5 or yaw_err > 4.0:
        raise RuntimeError(f"pose miss xy={xy:.2f} yaw_err={yaw_err:.1f}")


def _grab(client, image_size: int):
    import airsim
    import cv2

    scene = client.simGetImages(
        [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)]
    )[0]
    bgr = np.frombuffer(scene.image_data_uint8, dtype=np.uint8).reshape(
        scene.height, scene.width, 3
    )
    rgb = cv2.resize(bgr[:, :, ::-1], (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb)


def _encode(model, rgb: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(rgb).to(device)
    window = x.view(1, 1, *x.shape)
    frame = model._prep_last_frame(window)
    with torch.no_grad():
        feats, _aux = model.encoder(
            frame.unsqueeze(1), cam_token=None, export_feat_layers=[]
        )
        return pool_tokens(feats[-1]).squeeze(0).cpu().numpy().astype(np.float32)


def _cone_fwd(model, rgb: np.ndarray, device: str) -> float:
    x = torch.from_numpy(rgb).to(device)
    window = x.view(1, 1, *x.shape)
    frame = model._prep_last_frame(window)
    with torch.no_grad():
        depth, _log_s = model.forward(frame)
    fwd, _left, _right = approach_clearances(
        depth.squeeze(0).cpu().numpy(), center_frac=0.12
    )
    return float(fwd)


def body_step(row, nxt) -> np.ndarray:
    """One demonstrated step from the flown poses, in the body frame."""
    p0 = np.asarray(row["pos"], dtype=np.float64)
    p1 = np.asarray(nxt["pos"], dtype=np.float64)
    yaw = math.radians(float(row["yaw_deg"]))
    c, s = math.cos(yaw), math.sin(yaw)
    wx, wy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
    dx = c * wx + s * wy
    dy = -s * wx + c * wy
    dyaw = (float(nxt["yaw_deg"]) - float(row["yaw_deg"]) + 180.0) % 360.0 - 180.0
    return np.array([dx, dy, 0.0, math.radians(dyaw)], dtype=np.float32)


def kept_pairs(rows):
    for row, nxt in zip(rows, rows[1:]):
        dyaw = (float(nxt["yaw_deg"]) - float(row["yaw_deg"]) + 180.0) % 360.0 - 180.0
        if abs(dyaw) < 8.0 and int(row["step"]) % 4 != 0:
            continue
        yield row, nxt


def relabel() -> None:
    """Replace zeroed dx/dy with the displacement of the flown step."""
    for idx in list(demo_routes()) + [r for r in HOLDOUT_ROUTES if r != 10]:
        dest = OUT / f"route{idx:02d}.npz"
        rows = [
            json.loads(line)
            for line in (TRAJ / f"route{idx:02d}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        actions = [body_step(row, nxt) for row, nxt in kept_pairs(rows)]
        pack = np.load(dest)
        if len(actions) != len(pack["action"]):
            raise RuntimeError(f"route {idx} pairs {len(actions)} != npz {len(pack['action'])}")
        action = np.stack(actions)
        np.savez(
            dest,
            h=pack["h"],
            h_next=pack["h_next"],
            action=action,
            goal_rel=pack["goal_rel"],
            fwd=pack["fwd"],
            fwd_next=pack["fwd_next"],
            route=pack["route"],
        )
        turn = np.abs(action[:, 3]) > 0.05
        straight = ~turn
        print(
            f"route {idx} dx {action[straight, 0].mean():.2f} dy {action[straight, 1].mean():.2f} "
            f"straight_dyaw {np.degrees(action[straight, 3]).mean():.2f} "
            f"turn_dyaw {np.degrees(action[turn, 3]).mean() if turn.any() else 0:.1f}",
            flush=True,
        )


def collect() -> None:
    import airsim

    OUT.mkdir(parents=True, exist_ok=True)
    pred = DepthMinPredictor.from_checkpoint(DEPTH, device="cuda")
    model = pred._model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    routes = json.loads(ANN.read_text())["routes"]
    goals = {int(r["route_idx"]): np.asarray(r["pos"][-1], dtype=np.float64) for r in routes}
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, "drone_1")
    client.armDisarm(True, "drone_1")
    try:
        for idx in list(demo_routes()) + [r for r in HOLDOUT_ROUTES if r != 10]:
            dest = OUT / f"route{idx:02d}.npz"
            if dest.exists():
                print(f"skip route {idx}", flush=True)
                continue
            rows = [
                json.loads(line)
                for line in (TRAJ / f"route{idx:02d}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            hs, h_next, acts, goals_rel, fwds, fwd_next = [], [], [], [], [], []

            def _frame(row):
                pos = [float(v) for v in row["pos"]]
                z = pos[2] if pos[2] > 0.0 else Z_FALLBACK
                _place(client, pos[0], pos[1], z, float(row["yaw_deg"]))
                rgb = _grab(client, model.image_size)
                return (
                    _encode(model, rgb, "cuda"),
                    _cone_fwd(model, rgb, "cuda"),
                    np.array([pos[0], pos[1], z]),
                )

            for row, nxt in zip(rows, rows[1:]):
                dyaw = (float(nxt["yaw_deg"]) - float(row["yaw_deg"]) + 180.0) % 360.0 - 180.0
                if abs(dyaw) < 8.0 and int(row["step"]) % 4 != 0:
                    continue
                h, fwd, pos = _frame(row)
                hn, fwdn, _pos_n = _frame(nxt)
                action = body_step(row, nxt)
                grel = goal_rel_body(
                    pos,
                    math.radians(float(row["yaw_deg"])),
                    goals[idx],
                ).astype(np.float32)
                hs.append(h)
                h_next.append(hn)
                acts.append(action)
                goals_rel.append(grel)
                fwds.append(fwd)
                fwd_next.append(fwdn)
            np.savez(
                dest,
                h=np.stack(hs),
                h_next=np.stack(h_next),
                action=np.stack(acts),
                goal_rel=np.stack(goals_rel),
                fwd=np.asarray(fwds, np.float32),
                fwd_next=np.asarray(fwd_next, np.float32),
                route=np.full(len(hs), idx, np.int32),
            )
            print(f"route {idx} pairs {len(hs)}", flush=True)
    finally:
        client.simPause(False)
        client.armDisarm(False, "drone_1")
        client.enableApiControl(False, "drone_1")


def _load_split():
    hs, h_next, actions, goals, fwds, fwd_next, routes, chunks = [], [], [], [], [], [], [], []
    for path in sorted(OUT.glob("route*.npz")):
        pack = np.load(path)
        n = len(pack["h"])
        hs.append(pack["h"])
        h_next.append(pack["h_next"])
        actions.append(pack["action"])
        goals.append(pack["goal_rel"])
        fwds.append(pack["fwd"])
        fwd_next.append(pack["fwd_next"])
        routes.append(pack["route"])
        for i in range(n - CHUNK + 1):
            chunks.append(pack["action"][i : i + CHUNK])
    # Chunks are built per file, so they line up with the start index of each
    # file only after we record those starts. Rebuild from the per-file loop.
    del chunks
    h = np.concatenate(hs)
    h_next = np.concatenate(h_next)
    action = np.concatenate(actions)
    goal = np.concatenate(goals)
    fwd = np.concatenate(fwds)
    fwd_next = np.concatenate(fwd_next)
    route = np.concatenate(routes)
    chunk = []
    own = []
    offset = 0
    for path in sorted(OUT.glob("route*.npz")):
        pack = np.load(path)
        n = len(pack["h"])
        for i in range(n - CHUNK + 1):
            chunk.append(pack["action"][i : i + CHUNK])
            own.append(offset + i)
        offset += n
    chunk_arr = np.stack(chunk).astype(np.float32)
    own_arr = np.asarray(own, dtype=np.int64)
    train = own_arr[np.isin(route[own_arr], demo_routes())]
    hold = own_arr[np.isin(route[own_arr], HOLDOUT_ROUTES)]
    # Map start index -> row in chunk_arr
    row_of = {int(s): i for i, s in enumerate(own_arr)}
    return h, h_next, action, goal, fwd, fwd_next, chunk_arr, row_of, train, hold


def _batches(index, batch, rng):
    order = index.copy()
    rng.shuffle(order)
    for start in range(0, len(order), batch):
        yield order[start : start + batch]


def _turn_indices(own, chunk_arr, yaw_min: float = 0.4):
    sums = np.abs(chunk_arr[:, :, 3].sum(axis=1))
    return own[sums > yaw_min]


def train() -> None:
    h, h_next, action, goal, fwd, fwd_next, chunk_arr, row_of, train_idx, hold_idx = _load_split()
    device = "cuda"
    heads = Corner90Heads(h.shape[1]).to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-3)
    ht = torch.as_tensor(h, device=device)
    hn = torch.as_tensor(h_next, device=device)
    at = torch.as_tensor(action, device=device)
    gt = torch.as_tensor(goal, device=device)
    ft = torch.log(torch.as_tensor(fwd, device=device).clamp(min=0.5))
    fn = torch.log(torch.as_tensor(fwd_next, device=device).clamp(min=0.5))
    ct = torch.as_tensor(chunk_arr, device=device)
    train_rows = np.array([row_of[int(i)] for i in train_idx])
    rng = np.random.default_rng(0)
    best = -1.0
    best_deg = 1.0e9
    best_state = None
    for epoch in range(40):
        heads.train()
        total = 0.0
        n = 0
        for batch in _batches(train_rows, 64, rng):
            if len(batch) < 2:
                continue
            b = torch.as_tensor(batch, device=device)
            inv = {v: k for k, v in row_of.items()}
            starts = torch.as_tensor([inv[int(r)] for r in batch.tolist()], device=device)
            opt.zero_grad()
            step = corner90_losses(heads, ht[starts], hn[starts], at[starts], gt[starts], fn[starts])
            pol = chunk_policy_loss(heads, ht[starts], ct[b], gt[starts])
            loss = step["loss"] + pol
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1
        if epoch % 5 == 0 or epoch == 39:
            metrics = _eval_chunks(heads, ht, gt, ct, hold_idx, row_of, device)
            print(
                f"epoch {epoch} loss {total / max(n, 1):.4f} "
                f"hold_chunk_sign {metrics['chunk_sign']:.2f} n {metrics['chunk_n']} "
                f"hold_chunk_deg {metrics['chunk_deg']:.1f} "
                f"straight_deg {metrics['straight_deg']:.1f} "
                f"straight_step {metrics['straight_step']:.1f} "
                f"straight_dx {metrics['straight_dx']:.2f}",
                flush=True,
            )
            better = metrics["chunk_sign"] > best or (
                metrics["chunk_sign"] == best and metrics["straight_step"] < best_deg
            )
            if metrics["chunk_n"] and better:
                best = metrics["chunk_sign"]
                best_deg = metrics["chunk_deg"]
                best_state = {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}
    if best_state is not None:
        heads.load_state_dict(best_state)
    payload = {
        "dim": int(h.shape[1]),
        "heads": heads.state_dict(),
        "holdout": list(HOLDOUT_ROUTES),
        "chunk": CHUNK,
        "exec_horizon": EXEC,
    }
    torch.save(payload, OUT / "corner90_rate.pt")
    metrics = _eval_chunks(heads, ht, gt, ct, hold_idx, row_of, device)
    (OUT / "holdout_street.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"HOLDOUT chunk_sign {metrics['chunk_sign']:.3f} n {metrics['chunk_n']} "
        f"deg {metrics['chunk_deg']:.1f} straight_deg {metrics['straight_deg']:.1f} "
        f"straight_step {metrics['straight_step']:.1f} straight_dx {metrics['straight_dx']:.2f}",
        flush=True,
    )


def _eval_chunks(heads, ht, gt, ct, hold_idx, row_of, device) -> dict:
    heads.eval()
    starts_l, rows_l = [], []
    for i in hold_idx:
        if int(i) not in row_of:
            continue
        starts_l.append(int(i))
        rows_l.append(row_of[int(i)])
    if not rows_l:
        return {"chunk_sign": 0.0, "chunk_n": 0, "chunk_deg": 0.0, "straight_deg": 0.0}
    starts = torch.as_tensor(starts_l, device=device)
    rows = torch.as_tensor(rows_l, device=device)
    with torch.no_grad():
        pred = heads.chunk(ht[starts], gt[starts])
    true = ct[rows]
    yaw_true = true[:, :, 3].sum(dim=-1)
    yaw_pred = pred[:, :, 3].sum(dim=-1)
    straight = yaw_true.abs() < 0.15
    straight_deg = (
        float(torch.rad2deg(yaw_pred[straight].abs()).mean()) if int(straight.sum()) else 0.0
    )
    straight_step = (
        float(torch.rad2deg(pred[straight, :, 3].abs()).mean()) if int(straight.sum()) else 0.0
    )
    straight_dx = float(pred[straight, :, 0].mean()) if int(straight.sum()) else 0.0
    mask = yaw_true.abs() > 0.4
    if int(mask.sum()) == 0:
        return {
            "chunk_sign": 0.0,
            "chunk_n": 0,
            "chunk_deg": 0.0,
            "straight_deg": straight_deg,
            "straight_step": straight_step,
            "straight_dx": straight_dx,
        }
    sign = float((torch.sign(yaw_pred[mask]) == torch.sign(yaw_true[mask])).float().mean())
    deg = float(torch.rad2deg((yaw_pred[mask] - yaw_true[mask]).abs()).mean())
    return {
        "chunk_sign": sign,
        "chunk_n": int(mask.sum()),
        "chunk_deg": deg,
        "straight_deg": straight_deg,
        "straight_step": straight_step,
        "straight_dx": straight_dx,
    }


def train_bearing() -> None:
    """Fit the bearing threshold on the L demonstrations. Encoder stays unused."""
    from experiments.aerial.rl.corner90 import BearingTurnHead, demo_routes

    actions, goals, routes = [], [], []
    for path in sorted(OUT.glob("route*.npz")):
        pack = np.load(path)
        actions.append(pack["action"])
        goals.append(pack["goal_rel"])
        routes.append(pack["route"])
    action = np.concatenate(actions)
    goal = np.concatenate(goals)
    route = np.concatenate(routes)
    train = np.isin(route, demo_routes())
    hold = np.isin(route, HOLDOUT_ROUTES)
    device = "cpu"
    head = BearingTurnHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    gt = torch.as_tensor(goal, dtype=torch.float32)
    at = torch.as_tensor(action, dtype=torch.float32)
    tr = torch.as_tensor(train)
    for epoch in range(400):
        opt.zero_grad()
        pred = head(gt[tr])
        loss = (pred[:, 3] - at[tr, 3]).abs().mean() + 0.05 * (pred[:, 0] - at[tr, 0]).abs().mean()
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == 399:
            metrics = _bearing_metrics(head, gt, at, hold)
            print(
                f"epoch {epoch} loss {float(loss.detach()):.4f} "
                f"bias {math.degrees(float(head.bias.detach())):.1f} "
                f"gain {float(head.gain.detach()):.2f} "
                f"hold_straight {metrics['straight_deg']:.2f} "
                f"hold_turn_sign {metrics['turn_sign']:.2f}",
                flush=True,
            )
    metrics = _bearing_metrics(head, gt, at, hold)
    torch.save(
        {"kind": "bearing", "head": head.state_dict(), "metrics": metrics},
        OUT / "corner90_bearing.pt",
    )
    print(
        f"HOLDOUT straight_deg {metrics['straight_deg']:.2f} "
        f"turn_sign {metrics['turn_sign']:.2f} n {metrics['turn_n']} "
        f"bias {math.degrees(float(head.bias.detach())):.1f}",
        flush=True,
    )


def _bearing_metrics(head, gt, at, hold) -> dict:
    head.eval()
    with torch.no_grad():
        pred = head(gt[hold])
    yaw_t = at[hold, 3]
    yaw_p = pred[:, 3]
    straight = yaw_t.abs() < math.radians(2.0)
    turn = yaw_t.abs() > math.radians(5.0)
    straight_deg = float(torch.rad2deg(yaw_p[straight].abs()).mean()) if int(straight.sum()) else 0.0
    if int(turn.sum()) == 0:
        sign = 0.0
        n = 0
    else:
        sign = float((torch.sign(yaw_p[turn]) == torch.sign(yaw_t[turn])).float().mean())
        n = int(turn.sum())
    return {"straight_deg": straight_deg, "turn_sign": sign, "turn_n": n}


def _eval_holdout(heads, ht, hn, at, ft, fn, hold_idx) -> dict:
    heads.eval()
    if len(hold_idx) == 0:
        return {"yaw_sign": 0.0, "yaw_n": 0, "street": 0.0, "street_n": 0}
    b = torch.as_tensor(hold_idx, device=ht.device)
    with torch.no_grad():
        inv = heads.predict_action(ht[b], hn[b])
        _h_hat, cone_hat = heads.predict_next(ht[b], at[b])
    yaw_true = at[b, 3]
    mask = yaw_true.abs() > 0.05
    if int(mask.sum()) == 0:
        yaw_sign = 0.0
        yaw_n = 0
    else:
        yaw_sign = float((torch.sign(inv[mask, 3]) == torch.sign(yaw_true[mask])).float().mean())
        yaw_n = int(mask.sum())
    grew = fn[b] > ft[b] + math.log(1.25)
    if int(grew.sum()) == 0:
        street = 0.0
        street_n = 0
    else:
        street = float((cone_hat[grew] > ft[b][grew]).float().mean())
        street_n = int(grew.sum())
    return {"yaw_sign": yaw_sign, "yaw_n": yaw_n, "street": street, "street_n": street_n}


if __name__ == "__main__":
    import sys

    if "--relabel" in sys.argv:
        relabel()
    elif "--bearing" in sys.argv:
        train_bearing()
    elif "--train" in sys.argv:
        train()
    else:
        collect()
