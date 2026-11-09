#!/usr/bin/env python3
"""Continue the office avoidance head after the wall-depth finetune.

Next-step forward stays safe while the true forward cone is still open, and
becomes a hit once that cone is inside 1 m. A side cone inside 1 m, with the
forward cone still open, is not a reason to keep driving: yaw away from that
side. The previous approach checkpoint is the init. Far, wall-depth, and the
car weights stay where they are.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_office_cons_loop as loop  # noqa: E402
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
from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402
from experiments.aerial.rl.env.obs import Observation  # noqa: E402
from office_bands import cones3_forward  # noqa: E402

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
DEPTH = ART / "da3_office_wall_20261104/depth_office_cone_wall.pt"
INIT = ART / "indoor_office_approach_20261104/c_s218/consequence_best.pt"
OUT = ART / "indoor_office_approach_20261104/c_s218"
ANCHOR = Path("/data/simworld/run/office_approach_20261104/anchors.npz")
MIX = Path("/data/simworld/run/indoor_office_mix")
YAW = 0.3141592653589793
OPEN_M = 2.0
FACE_M = 1.0
SIDE_M = 1.0
STEPS = 80


def grab() -> None:
    poses = [(float(x), -250.0, 0.0, 80.0, -250.0) for x in range(-520, -299, 8)]
    poses += [(float(x), -200.0, 0.0, 200.0, -200.0) for x in (-480, -160)]
    poses += [(float(x), -170.0, 0.0, 220.0, -170.0) for x in (-480, -400, -360, -333, -319, -305)]
    ucv = loop.col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(loop.col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    loop.col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    loop.restore_boxes(ucv)
    depth_model = DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    depth, cones, goals = [], [], []
    for x, y, yaw, gx, gy in poses:
        loop.col.place(ucv, x, y, yaw)
        loop.col.look(ucv, cam, x, y, yaw)
        packed = loop.col.grab(ucv, cam)
        if packed is None:
            raise RuntimeError(f"grab failed {x},{y}")
        image, seen, warped = loop.col.warp_pair(*packed)
        gt56 = loop.col.pool_min(loop.col.hide(warped, seen))
        depth_model.reset()
        pred = np.asarray(depth_model.predict_depth(Observation(
            rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False,
        )), np.float32)
        if pred.shape[:2] != warped.shape[:2]:
            import cv2
            pred = cv2.resize(pred, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
        da56 = loop.col.pool_min(loop.col.hide(pred, seen))
        g = cones3_forward(gt56)
        d = cones3_forward(da56)
        state = np.array([x / 100.0, y / 100.0, 0.52, 0, 0, 0, np.radians(yaw)], np.float64)
        goal = np.array([gx / 100.0, gy / 100.0, 0.52], np.float64)
        depth.append(depth_input(da56))
        cones.append(np.asarray(g, np.float32))
        goals.append(goal_features(state, goal))
        print(
            f"anchor x={x:.0f} y={y:.0f} gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f} da={d[0]:.2f},{d[1]:.2f},{d[2]:.2f}",
            flush=True,
        )
    ANCHOR.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        ANCHOR,
        depth=np.stack(depth).astype(np.float32),
        cones=np.stack(cones).astype(np.float32),
        goal=np.stack(goals).astype(np.float32),
    )
    del depth_model
    torch.cuda.empty_cache()


def stack(paths, horizon=10):
    import json
    parts = {}
    for path in paths:
        ep = load_episode(path)
        if "da3_depth56" not in ep or len(ep["state"]) < 2:
            continue
        ep.pop("env", None)
        ep.pop("env_flip", None)
        windows = episode_windows(ep, horizon)
        if not len(windows["depth"]):
            continue
        for key, value in windows.items():
            parts.setdefault(key, []).append(value)
    return {k: np.concatenate(v) for k, v in parts.items()}


def hazard(model, depth, goal, prev, actions):
    out = model(depth, goal, prev, actions)
    return 1.0 - survival(out["hazard_logit"])[:, 0]


def logit0(model, depth, goal, prev, actions):
    """Step-0 hazard logit. A probability loss vanishes once the step already looks safe."""
    out = model(depth, goal, prev, actions)
    return out["hazard_logit"][:, 0]


def with_action(actions, dx: float, dyaw: float):
    alt = torch.zeros_like(actions)
    alt[:, 0, 0] = dx
    alt[:, 0, 3] = dyaw
    return alt


def main() -> int:
    if not DEPTH.is_file():
        print("MISSING_DEPTH", DEPTH, flush=True)
        return 1
    if not ANCHOR.is_file():
        grab()
    anchors = np.load(ANCHOR)
    import json
    review = json.loads((MIX / "AUTO_REVIEW.json").read_text())
    route_of = {e["file"]: e["route_id"] for e in review["episodes"] if e.get("keep") and e.get("file")}
    tr_paths = [
        p for p in sorted((MIX / "episodes").glob("episode_*.npz"))
        if p.name.count(".") == 1 and route_of.get(p.name) not in (None, "office_val", "office_yaw_val")
    ]
    print(f"paths tr {len(tr_paths)}", flush=True)
    tr = stack(tr_paths)
    dev = torch.device("cuda")
    model = load_checkpoint(str(INIT), "cuda")
    horizon = int(tr["actions"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal")
    a_depth = torch.as_tensor(anchors["depth"], dtype=torch.float32, device=dev)
    a_goal = torch.as_tensor(anchors["goal"], dtype=torch.float32, device=dev)
    cones = torch.as_tensor(anchors["cones"], dtype=torch.float32, device=dev)
    a_prev = torch.zeros(a_depth.shape[0], 4, device=dev)
    fwd_gt, left_gt, right_gt = cones[:, 0], cones[:, 1], cones[:, 2]
    face = fwd_gt < FACE_M
    # Open forward cone, including the lip the old depth called a wall.
    safe = (fwd_gt > OPEN_M) & (left_gt > SIDE_M) & (right_gt > SIDE_M)
    left_block = (left_gt < SIDE_M) & (fwd_gt > OPEN_M) & (right_gt > SIDE_M)
    right_block = (right_gt < SIDE_M) & (fwd_gt > OPEN_M) & (left_gt > SIDE_M)
    print(
        f"anchors {len(fwd_gt)} safe {int(safe.sum())} face {int(face.sum())} "
        f"left {int(left_block.sum())} right {int(right_block.sum())}",
        flush=True,
    )

    def act(dx, dyaw):
        a = torch.zeros(a_depth.shape[0], horizon, 4, device=dev)
        a[:, 0, 0] = dx
        a[:, 0, 3] = dyaw
        return a

    @torch.no_grad()
    def rank():
        model.eval()
        h_fwd = hazard(model, a_depth, a_goal, a_prev, act(0.8, 0.0))
        h_right = hazard(model, a_depth, a_goal, a_prev, act(0.0, YAW))
        h_left = hazard(model, a_depth, a_goal, a_prev, act(0.0, -YAW))
        def mean(mask, values):
            return float(values[mask].mean()) if bool(mask.any()) else float("nan")
        model.train()
        return {
            "safe": mean(safe, h_fwd),
            "face": mean(face, h_fwd),
            "left_fwd": mean(left_block, h_fwd),
            "left_yaw": mean(left_block, h_right),
            "right_fwd": mean(right_block, h_fwd),
            "right_yaw": mean(right_block, h_left),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    stats = rank()
    print("step 0", {k: round(v, 3) for k, v in stats.items()}, flush=True)
    best = None
    rng = np.random.default_rng(218)
    n = len(tr["depth"])
    model.train()
    for step in range(1, STEPS + 1):
        idx = rng.choice(n, size=64)
        batch = {k: torch.as_tensor(tr[k][idx], dtype=torch.float32, device=dev) for k in keys}
        if rng.random() < 0.5:
            batch = mirror_batch(batch)
        out = model(batch["depth"], batch["goal"], batch["prev_action"], batch["actions"])
        loss, _parts = consequence_loss(out, batch)
        z_fwd = logit0(model, a_depth, a_goal, a_prev, act(0.8, 0.0))
        z_right = logit0(model, a_depth, a_goal, a_prev, act(0.0, YAW))
        z_left = logit0(model, a_depth, a_goal, a_prev, act(0.0, -YAW))
        if bool(safe.any()):
            loss = loss + 2.0 * torch.relu(z_fwd[safe] + 2.0).mean()
        if bool(face.any()):
            loss = loss + 4.0 * torch.relu(2.0 - z_fwd[face]).mean()
        if bool(left_block.any()):
            loss = loss + 8.0 * torch.relu(2.0 - z_fwd[left_block]).mean()
            loss = loss + 4.0 * torch.relu(z_right[left_block] + 2.0).mean()
        if bool(right_block.any()):
            loss = loss + 8.0 * torch.relu(2.0 - z_fwd[right_block]).mean()
            loss = loss + 4.0 * torch.relu(z_left[right_block] + 2.0).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == STEPS:
            stats = rank()
            gap = stats["face"] - stats["safe"]
            print(f"step {step}", {k: round(v, 3) for k, v in stats.items()}, "gap", round(gap, 3), flush=True)
            save_checkpoint(OUT / "consequence_last.pt", model, {"step": step, **stats})
            side_ok = (stats["left_fwd"] > 0.7 or np.isnan(stats["left_fwd"])) and stats["safe"] < 0.2
            if side_ok and stats["face"] > 0.8 and (best is None or gap > best[0]):
                best = (gap, step, stats)
                save_checkpoint(OUT / "consequence_best.pt", model, {"step": step, **stats})
                print("SAVE", step, flush=True)
    if best is None:
        save_checkpoint(OUT / "consequence_best.pt", model, {"step": STEPS, **stats})
        print("SAVE_LAST", flush=True)
    print("BEST", best, "OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
