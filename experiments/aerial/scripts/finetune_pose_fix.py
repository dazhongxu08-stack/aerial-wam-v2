#!/usr/bin/env python3
"""Move only the three live depth errors, and keep the gap reading.

Encoder stays frozen. A step is kept only when the gap stays open and the
west wall, the left yaw, and the north close face all move toward the GT.
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")

spec = importlib.util.spec_from_file_location("ft", "/data/simworld/run/finetune_office_da3_cone.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

import apply_office_daylight as day
import collect_office_train as col

SRC = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt"
OUT = "/data/aerial-wam-v2/experiments/aerial/rl/artifacts/da3_office_cone_yaw_q_20261102/depth_office_cone_yaw.pt"

# name, x, y, yaw, kind
POSES = (
    ("west", -520.0, -250.0, 0.0, "open"),
    ("yaw_left", -170.0, -200.0, 18.0, "lift"),
    ("north_a", -128.0, -150.0, 0.0, "close"),
    ("north_b", -114.0, -150.0, 0.0, "close"),
    ("north_c", -100.0, -150.0, 0.0, "close"),
    ("gap", -160.0, -200.0, 0.0, "anchor"),
    ("east", -180.0, -450.0, 180.0, "open"),
    ("back", 140.0, -200.0, 180.0, "open"),
    ("yaw_right", -170.0, -200.0, -18.0, "anchor"),
)


def grab(ucv, cam, x, y, yaw):
    col.place(ucv, x, y, yaw)
    col.look(ucv, cam, x, y, yaw)
    rgb, depth = col.grab(ucv, cam)
    image, seen, warped = col.warp_pair(rgb, depth)
    gt = col.pool_min(col.hide(warped, seen)).astype(np.float32)
    return np.ascontiguousarray(image), gt


def cones_of(model, image: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        pred = ft.forward_maps(model, image[None]).detach().float().cpu().numpy()[0]
    return col.cones3(pred)


def pixel_loss(pred: torch.Tensor, gt: torch.Tensor, kind: str) -> torch.Tensor:
    p = pred.clamp(0.05, 30.0)
    g = gt.clamp(0.05, 30.0)
    too_close = (g >= 2.0) & (p < 2.0)
    lift = torch.relu(torch.log(torch.minimum(g, p.new_full((), 2.2))) - torch.log(p))
    too_far = (g < 1.4) & (p > g + 0.1)
    pull = torch.relu(torch.log(p) - torch.log(g))
    if kind == "open":
        err = lift * too_close
    elif kind == "close":
        err = pull * too_far
    elif kind == "lift":
        err = lift * too_close + pull * too_far
    else:
        err = 4.0 * lift * too_close + pull * too_far
    return err.mean()


def main() -> int:
    ft.start_rss_watch()
    cfg = day.load_cfg()
    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    day.apply(ucv, cfg)
    cams = str(col.req(ucv, "vget /cameras")).split()
    cam = cams.index("FusionCameraSensor")
    frames = []
    for name, x, y, yaw, kind in POSES:
        image, gt = grab(ucv, cam, x, y, yaw)
        frames.append((name, kind, image, gt))
        print(f"grab {name} gt {col.cones3(gt)}", flush=True)
    wrapper = ft.DepthMinPredictor.from_checkpoint(SRC, device="cuda")
    model = wrapper._model
    model.eval()
    base = {name: cones_of(model, image) for name, _k, image, _gt in frames}
    for name, _k, _i, _g in frames:
        da = base[name]
        print(f"base {name} da {float(da[0]):.2f},{float(da[1]):.2f},{float(da[2]):.2f}", flush=True)

    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW((p for p in model.decoder.parameters() if p.requires_grad), lr=1e-6)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = score(base)
    print(f"base score {best_score:.3f}", flush=True)
    kept = False
    for step in range(1, 17):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for _name, kind, image, gt in frames:
            pred = ft.forward_maps(model, image[None])[0]
            gt_t = torch.from_numpy(gt).to(pred.device)
            loss = loss + pixel_loss(pred, gt_t, kind)
        loss = loss / len(frames)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        model.eval()
        now = {name: cones_of(model, image) for name, _k, image, _gt in frames}
        sc = score(now)
        gap = now["gap"]
        print(
            f"step {step} loss {float(loss.detach()):.4f} score {sc:.3f} "
            f"gap {float(gap[0]):.2f} west {float(now['west'][0]):.2f} "
            f"left {float(now['yaw_left'][0]):.2f} north {float(now['north_b'][0]):.2f}",
            flush=True,
        )
        if sc < best_score and hold(base, now):
            best_score = sc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            kept = True
            print("KEEP", flush=True)
        ft.guard_rss(f"step {step}")
    if not kept:
        print("NO_SAVE", flush=True)
        return 1
    model.load_state_dict(best_state)
    payload = torch.load(SRC, map_location="cpu")
    ft.save_ckpt(model, payload, __import__("pathlib").Path(OUT), {"score": best_score})
    print("SAVED", OUT, flush=True)
    return 0


def score(cones: dict) -> float:
    west = max(0.0, 2.2 - float(cones["west"][0]))
    left = max(0.0, 1.7 - float(cones["yaw_left"][0]))
    north = max(0.0, float(cones["north_b"][0]) - 1.4)
    return west + left + north


def hold(base: dict, now: dict) -> bool:
    gap = now["gap"]
    if not (2.15 <= float(gap[0]) <= 2.75):
        return False
    if float(gap[1]) > float(base["gap"][1]) + 0.15 or float(gap[2]) > float(base["gap"][2]) + 0.15:
        return False
    if float(now["east"][0]) < 2.0 or float(now["back"][0]) < 1.7:
        return False
    if abs(float(now["yaw_right"][0]) - float(base["yaw_right"][0])) > 0.25:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
