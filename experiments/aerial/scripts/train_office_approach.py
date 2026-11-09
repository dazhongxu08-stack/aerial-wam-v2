#!/usr/bin/env python3
"""Teach the next forward step to stay safe until the barrel face is actually close.

The far head's 10-step survival collapses while the nose cone is still 7 m,
because the barrel lip is already in the frame. The office loop then stops
or spins, and a hand creep finishes the approach. This finetune labels the
next forward step from the true forward cone: open cone means the step is
safe, a cone inside 1 m means it hits. Yaw is not relabeled. The far
checkpoint and the car weights are left in place.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from office_bands import cones3_forward  # noqa: E402
from experiments.aerial.rl.consequence_model import (  # noqa: E402
    consequence_loss,
    depth_input,
    episode_windows,
    goal_features,
    load_checkpoint,
    load_episode,
    mirror_batch,
    save_checkpoint,
    survival,
)

MIX = Path("/data/simworld/run/indoor_office_mix")
ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
INIT = ART / "indoor_office_mix_far_20261103/c_s218/consequence_best.pt"
OUT = ART / "indoor_office_approach_20261103/c_s218"
ANCHOR = Path("/data/simworld/run/office_approach_20261103/anchors.npz")
OPEN_M = 2.0
HIT_M = 1.0
STEPS = 80


def grab_anchors() -> None:
    import run_office_cons_loop as loop
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor
    from experiments.aerial.rl.env.obs import Observation

    # Wall approach, plus the aisle and gap starts so forward stays legal there.
    poses = [(x, -250.0, 0.0, 80.0, -250.0) for x in (
        -520, -500, -480, -460, -444, -430, -415, -398, -377, -356,
        -335, -330, -325, -314, -305, -300,
    )]
    poses += [(-480.0, -200.0, 0.0, 200.0, -200.0), (-160.0, -200.0, 0.0, 200.0, -200.0)]
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cams = str(loop.col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    loop.restore_boxes(ucv)
    depth_model = DepthMinPredictor.from_checkpoint(loop.DEPTH, device="cuda")
    depths, gt_fwd, goals, states = [], [], [], []
    for x, y, yaw, gx, gy in poses:
        loop.col.place(ucv, x, y, yaw)
        loop.col.look(ucv, cam, x, y, yaw)
        packed = loop.col.grab(ucv, cam)
        if packed is None:
            raise RuntimeError(f"grab failed at {x},{y}")
        rgb, depth = packed
        image, seen, warped = loop.col.warp_pair(rgb, depth)
        gt56 = loop.col.pool_min(loop.col.hide(warped, seen))
        depth_model.reset()
        pred = np.asarray(depth_model.predict_depth(Observation(
            rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False,
        )), np.float32)
        da56 = loop.col.pool_min(loop.col.hide(pred, seen))
        fwd = float(cones3_forward(gt56)[0])
        state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, np.radians(yaw)], np.float64)
        goal = np.array([gx / 100.0, gy / 100.0, 0.52], np.float64)
        depths.append(depth_input(da56))
        gt_fwd.append(fwd)
        goals.append(goal_features(state, goal))
        states.append(state.astype(np.float32))
        print(f"anchor x={x:.0f} y={y:.0f} gt_fwd={fwd:.2f} da={float(cones3_forward(da56)[0]):.2f}", flush=True)
    ANCHOR.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        ANCHOR,
        depth=np.stack(depths).astype(np.float32),
        gt_fwd=np.asarray(gt_fwd, np.float32),
        goal=np.stack(goals).astype(np.float32),
        state=np.stack(states).astype(np.float32),
    )
    del depth_model
    torch.cuda.empty_cache()


def stack(paths, horizon=10):
    parts = {}
    gt_fwd = []
    for path in paths:
        ep = load_episode(path)
        if "da3_depth56" not in ep or len(ep["state"]) < 2:
            continue
        ep.pop("env", None)
        ep.pop("env_flip", None)
        windows = episode_windows(ep, horizon)
        if not len(windows["depth"]):
            continue
        gt = ep["depth56"]
        for t in range(len(ep["state"]) - 1):
            gt_fwd.append(float(cones3_forward(gt[t])[0]))
        for key, value in windows.items():
            parts.setdefault(key, []).append(value)
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data["gt_fwd"] = np.asarray(gt_fwd, np.float32)
    if len(data["gt_fwd"]) != len(data["depth"]):
        raise RuntimeError(f"gt_fwd {len(data['gt_fwd'])} != windows {len(data['depth'])}")
    return data


def step0_hazard(model, depth, goal, prev, actions):
    out = model(depth, goal, prev, actions)
    return 1.0 - survival(out["hazard_logit"])[:, 0]


def forward_actions(actions):
    alt = actions.clone()
    alt[:, 0, :] = 0
    alt[:, 0, 0] = 0.8
    return alt


def main() -> int:
    if not ANCHOR.is_file():
        grab_anchors()
    anchors = np.load(ANCHOR)
    review = json.loads((MIX / "AUTO_REVIEW.json").read_text())
    route_of = {e["file"]: e["route_id"] for e in review["episodes"] if e.get("keep") and e.get("file")}
    eps = sorted(
        p for p in (MIX / "episodes").glob("episode_*.npz")
        if p.name.count(".") == 1 and p.name in route_of
    )
    tr_paths = [p for p in eps if route_of[p.name] not in ("office_val", "office_yaw_val")]
    va_paths = [p for p in eps if route_of[p.name] == "office_val"]
    print(f"paths tr {len(tr_paths)} va {len(va_paths)}", flush=True)
    tr = stack(tr_paths)
    va = stack(va_paths)
    print(
        f"windows tr {len(tr['depth'])} open {int((tr['gt_fwd'] > 4).sum())} "
        f"face {int((tr['gt_fwd'] < HIT_M).sum())} va {len(va['depth'])}",
        flush=True,
    )
    dev = torch.device("cuda")
    model = load_checkpoint(str(INIT), "cuda")
    horizon = int(tr["actions"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal")
    rng = np.random.default_rng(218)

    def take(data, idx):
        picked = {k: torch.as_tensor(data[k][idx], dtype=torch.float32, device=dev) for k in keys}
        picked["gt_fwd"] = torch.as_tensor(data["gt_fwd"][idx], dtype=torch.float32, device=dev)
        return picked

    a_depth = torch.as_tensor(anchors["depth"], dtype=torch.float32, device=dev)
    a_goal = torch.as_tensor(anchors["goal"], dtype=torch.float32, device=dev)
    a_fwd = torch.as_tensor(anchors["gt_fwd"], dtype=torch.float32, device=dev)
    a_prev = torch.zeros(a_depth.shape[0], 4, device=dev)
    a_act = torch.zeros(a_depth.shape[0], horizon, 4, device=dev)
    a_act[:, 0, 0] = 0.8
    safe_m = a_fwd > OPEN_M
    hit_m = a_fwd < HIT_M
    print(f"anchors {len(a_fwd)} safe {int(safe_m.sum())} hit {int(hit_m.sum())}", flush=True)

    @torch.no_grad()
    def rank():
        model.eval()
        hazard = step0_hazard(model, a_depth, a_goal, a_prev, a_act)
        safe = float(hazard[safe_m].mean()) if bool(safe_m.any()) else 1.0
        face = float(hazard[hit_m].mean()) if bool(hit_m.any()) else 0.0
        model.train()
        return safe, face

    OUT.mkdir(parents=True, exist_ok=True)
    best = None
    safe_h, face_h = rank()
    print(f"step 0 safe_hazard {safe_h:.3f} face_hazard {face_h:.3f}", flush=True)
    model.train()
    n = len(tr["depth"])
    for step in range(1, STEPS + 1):
        idx = rng.choice(n, size=128)
        batch = take(tr, idx)
        if rng.random() < 0.5:
            flipped = mirror_batch(batch)
            flipped["gt_fwd"] = batch["gt_fwd"]
            batch = flipped
        out = model(batch["depth"], batch["goal"], batch["prev_action"], batch["actions"])
        loss, _parts = consequence_loss(out, batch)
        fwd_h = step0_hazard(model, batch["depth"], batch["goal"], batch["prev_action"], forward_actions(batch["actions"]))
        open_w = batch["gt_fwd"] > 4.0
        face_w = batch["gt_fwd"] < HIT_M
        if bool(open_w.any()):
            loss = loss + 2.0 * fwd_h[open_w].mean()
        if bool(face_w.any()):
            loss = loss + 2.0 * (1.0 - fwd_h[face_w]).mean()
        anchor_h = step0_hazard(model, a_depth, a_goal, a_prev, a_act)
        loss = loss + 6.0 * anchor_h[safe_m].mean() + 6.0 * (1.0 - anchor_h[hit_m]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == STEPS:
            safe_h, face_h = rank()
            gap = face_h - safe_h
            print(
                f"step {step} safe_hazard {safe_h:.3f} face_hazard {face_h:.3f} gap {gap:.3f}",
                flush=True,
            )
            save_checkpoint(OUT / "consequence_last.pt", model, {"step": step, "safe": safe_h, "face": face_h})
            if safe_h < 0.15 and (best is None or gap > best[0]):
                best = (gap, step, safe_h, face_h)
                save_checkpoint(OUT / "consequence_best.pt", model, {"step": step, "safe": safe_h, "face": face_h})
                print("SAVE", step, flush=True)
    if best is None:
        save_checkpoint(OUT / "consequence_best.pt", model, {"step": STEPS, "safe": safe_h, "face": face_h})
        print("SAVE_LAST", STEPS, flush=True)
    print("BEST", best, "OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
