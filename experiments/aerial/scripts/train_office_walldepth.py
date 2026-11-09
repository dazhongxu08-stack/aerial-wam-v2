#!/usr/bin/env python3
"""Teach the office depth head the barrel approach.

Along the west wall the true forward cone stays near 7 m until the face, while
this head already reports about 0.8 m. The avoidance head then stops short.
Decoder only, starting from the far checkpoint. The far file is not replaced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_office_train as col  # noqa: E402
import train_office_da3_far as far  # noqa: E402
from experiments.aerial.rl.depth_predictor import DepthMinPredictor  # noqa: E402
from experiments.aerial.rl.env.obs import Observation  # noqa: E402
from office_bands import cones3_forward  # noqa: E402

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
CKPT = ART / "da3_office_far_20261103/depth_office_cone_far.pt"
OUT = ART / "da3_office_wall_20261104/depth_office_cone_wall.pt"
LIVE = Path("/data/simworld/run/office_walldepth_20261104/live.npz")
STEPS = 40


def poses():
    rows = [(float(x), -250.0, 0.0) for x in range(-520, -299, 8)]
    rows += [(float(x), -200.0, 0.0) for x in (-480, -400, -320, -160, -80)]
    rows += [(float(x), -170.0, 0.0) for x in (-480, -400, -333, -305, -291)]
    return rows


def grab(depth_model) -> None:
    import run_office_cons_loop as loop

    ucv = col.UnrealCV(port=9000, ip="127.0.0.1", resolution=(640, 480))
    cam = str(col.req(ucv, "vget /cameras")).split().index("FusionCameraSensor")
    col.ensure_dog(ucv)
    try:
        loop.apply_daylight(ucv)
    except Exception as exc:  # noqa: BLE001
        print(f"DAYLIGHT_SKIP {exc}", flush=True)
    loop.restore_boxes(ucv)
    rgb, gt56, xy = [], [], []
    for x, y, yaw in poses():
        col.place(ucv, x, y, yaw)
        col.look(ucv, cam, x, y, yaw)
        packed = col.grab(ucv, cam)
        if packed is None:
            raise RuntimeError(f"grab failed {x},{y}")
        image, seen, warped = col.warp_pair(*packed)
        gt = col.pool_min(col.hide(warped, seen))
        depth_model.reset()
        pred = np.asarray(depth_model.predict_depth(Observation(
            rgb=np.ascontiguousarray(image), state=np.zeros(7, np.float32), collided=False,
        )), np.float32)
        if pred.shape[:2] != warped.shape[:2]:
            import cv2
            pred = cv2.resize(pred, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_LINEAR)
        da = cones3_forward(col.pool_min(col.hide(pred, seen)))
        g = cones3_forward(gt)
        print(
            f"grab x={x:.0f} y={y:.0f} gt={g[0]:.2f},{g[1]:.2f},{g[2]:.2f} "
            f"da={da[0]:.2f},{da[1]:.2f},{da[2]:.2f} rgb={image.shape}",
            flush=True,
        )
        rgb.append(np.ascontiguousarray(image))
        gt56.append(gt.astype(np.float32))
        xy.append((x, y, yaw))
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(LIVE, rgb=np.stack(rgb), gt56=np.stack(gt56), xy=np.asarray(xy, np.float32))


def report(model, rgb: np.ndarray, gt56: np.ndarray, xy: np.ndarray, tag: str) -> float:
    model.eval()
    das = []
    with torch.no_grad():
        for i in range(0, len(rgb), 4):
            pred = far.forward_maps(model, rgb[i:i + 4]).detach().float().cpu().numpy()
            das.append(pred)
    da56 = np.concatenate(das)
    short, face, aisle = [], [], []
    print(tag, flush=True)
    for i in range(len(xy)):
        g = cones3_forward(gt56[i])
        d = cones3_forward(da56[i])
        wall = abs(float(xy[i, 1]) + 250.0) < 0.5
        if wall and g[0] >= 4.0:
            short.append(max(0.0, np.log(g[0]) - np.log(max(d[0], 0.05))))
        if wall and g[0] <= 1.2:
            face.append(max(0.0, np.log(max(d[0], 0.05)) - np.log(max(g[0], 0.05))))
        if abs(float(xy[i, 0]) + 480) < 0.5 and abs(float(xy[i, 1]) + 200) < 0.5:
            aisle.append(d[0])
        if wall and int(xy[i, 0]) % 16 == 0:
            print(f"  x={xy[i, 0]:.0f} y={xy[i, 1]:.0f} gt={g[0]:.2f} da={d[0]:.2f}", flush=True)
    # Keep the lip long. A low mean log-error was shrinking it back toward 1 m.
    score = float(np.mean(short) if short else 1.0) + float(np.mean(face) if face else 1.0)
    if aisle and min(aisle) < 4.0:
        score += 2.0
    print(f"  score {score:.3f} aisle {min(aisle) if aisle else -1:.2f}", flush=True)
    model.encoder.eval()
    model.decoder.train()
    return score


def main() -> int:
    wrap = DepthMinPredictor.from_checkpoint(str(CKPT), device="cuda")
    if not LIVE.is_file():
        grab(wrap)
        del wrap
        torch.cuda.empty_cache()
        wrap = DepthMinPredictor.from_checkpoint(str(CKPT), device="cuda")
    live = np.load(LIVE)
    rgb = live["rgb"]
    gt56 = live["gt56"]
    xy = live["xy"]
    model = wrap._model
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    model.encoder.eval()
    src = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    train_refs, _hold = far.index_mix()
    far_refs = [r for r in train_refs if r[2] >= far.FAR_GT][:24]
    near_refs = [r for r in train_refs if r[2] <= far.NEAR_GT][:24]
    loader = far.Loader()
    mix_far = loader.batch(far_refs)
    mix_near = loader.batch(near_refs)
    loader.close()
    before = report(model, rgb, gt56, xy, "before")
    opt = torch.optim.AdamW((p for p in model.decoder.parameters() if p.requires_grad), lr=3e-6)
    rng = np.random.default_rng(4)
    best = before
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n = len(rgb)
    for step in range(1, STEPS + 1):
        model.decoder.train()
        live_i = rng.choice(n, size=4, replace=False)
        fi = rng.choice(len(mix_far[0]), size=2, replace=False)
        ni = rng.choice(len(mix_near[0]), size=2, replace=False)
        batch_rgb = np.concatenate([rgb[live_i], mix_far[0][fi], mix_near[0][ni]])
        batch_gt = np.concatenate([gt56[live_i], mix_far[1][fi], mix_near[1][ni]])
        pred = far.forward_maps(model, batch_rgb)
        gt_t = torch.as_tensor(batch_gt, dtype=torch.float32, device="cuda")
        loss = far.far_near_loss(pred[:4], gt_t[:4]) * 6.0 + far.far_near_loss(pred[4:], gt_t[4:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == STEPS:
            err = report(model, rgb, gt56, xy, f"step {step} loss {float(loss):.3f}")
            far.save_ckpt(model, src, OUT.with_name("depth_office_cone_wall_last.pt"), {"step": step, "err": err})
            if err < best:
                best = err
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                far.save_ckpt(model, src, OUT, {"step": step, "err": err})
                print("SAVE", step, f"err {err:.3f}", flush=True)
    model.load_state_dict(best_state)
    report(model, rgb, gt56, xy, "best")
    print("OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
