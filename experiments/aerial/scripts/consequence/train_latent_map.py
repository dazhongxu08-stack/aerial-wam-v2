#!/usr/bin/env python3
"""Train the body-frame feature map to remember a surface across poses.

Past views are placed by their capture pose. The loss matches the feature in
that cell to the feature of the same surface in the current view, and the
feature must also predict the depth that was seen. Survival is not trained.
The frozen krs6 weights are not loaded and not written.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/yao/aerial-wam-v2")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = ROOT / "experiments/aerial/rl/artifacts"
DATA = ART / "consequence_data_20261004"
OUT = ART / "latentmap_20261103"
FLIGHT = ROOT / "artifacts/hard_f35_trim_unseen_complex_20261029"
VAL = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
PAST = 8
DIM = 16
STEPS = 4000
BATCH = 32
DEPTH_MIN = 0.5
DEPTH_MAX = 100.0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def depth_in(depth_m: torch.Tensor) -> torch.Tensor:
    d = depth_m.clamp(DEPTH_MIN, DEPTH_MAX)
    return torch.log(d / DEPTH_MIN) / np.log(DEPTH_MAX / DEPTH_MIN)


def episodes():
    review = json.loads((DATA / "AUTO_REVIEW.json").read_text(encoding="utf-8"))
    rows, val = [], []
    for e in review["episodes"]:
        if not e.get("keep") or not e.get("file"):
            continue
        rid = str(e["route_id"])
        if not rid.startswith("od_train_"):
            continue
        path = DATA / "episodes" / e["file"]
        if path.is_file():
            (val if rid in VAL else rows).append(path)
    return rows, val


def load_lib(paths, load_episode):
    lib = []
    for path in paths:
        ep = load_episode(path)
        if "da3_depth56" not in ep or len(ep["state"]) < 4:
            continue
        n = min(len(ep["state"]), len(ep["da3_depth56"]))
        if n < 4:
            continue
        lib.append({
            "depth": np.asarray(ep["da3_depth56"][:n], np.float32),
            "state": np.asarray(ep["state"][:n], np.float32),
        })
    log(f"episodes {len(lib)}")
    return lib


def draw(lib, rng, batch, dev):
    depth, state = [], []
    for _ in range(batch):
        ep = lib[int(rng.integers(0, len(lib)))]
        t = int(rng.integers(1, len(ep["depth"])))
        start = max(0, t - PAST)
        idx = list(range(start, t + 1))
        while len(idx) < PAST + 1:
            idx.insert(0, idx[0])
        depth.append(ep["depth"][idx])
        state.append(ep["state"][idx])
    return (torch.as_tensor(np.stack(depth), device=dev),
            torch.as_tensor(np.stack(state), device=dev))


def loss_of(enc, depth_m, state):
    from experiments.aerial.rl.latent_map import block_rays, fuse_nearest
    b, steps, _, _ = depth_m.shape
    flat = enc.features(depth_in(depth_m.reshape(b * steps, 56, 56)))
    flat = flat.reshape(b, steps, enc.dim, 14, 14)
    here = state[:, -1]
    cells, rngs, valids, dmins = [], [], [], []
    for k in range(steps):
        cell, rng, valid, dmin = block_rays(depth_m[:, k], state[:, k], here)
        cells.append(cell)
        rngs.append(rng)
        valids.append(valid)
        dmins.append(dmin)
    cur = flat[:, -1]
    if bool(valids[-1].any()):
        dloss = (enc.depth_of(cur) - dmins[-1].clamp(min=0.05).log()).pow(2)[valids[-1]].mean()
    else:
        dloss = cur.sum() * 0.0
    past_n = steps - 1
    past_f = flat[:, :past_n].permute(0, 2, 1, 3, 4).reshape(b, enc.dim, past_n * 14, 14)
    past_c = torch.stack(cells[:past_n], 1).reshape(b, past_n * 14, 14)
    past_r = torch.stack(rngs[:past_n], 1).reshape(b, past_n * 14, 14)
    past_v = torch.stack(valids[:past_n], 1).reshape(b, past_n * 14, 14)
    grid, dist = fuse_nearest(past_f, past_c, past_r, past_v)
    cur_cell = cells[-1]
    gathered = grid.reshape(b, enc.dim, -1).gather(
        2, cur_cell.reshape(b, 1, -1).expand(b, enc.dim, -1)).reshape(b, enc.dim, 14, 14)
    hit = dist.reshape(b, -1).gather(1, cur_cell.reshape(b, -1)).reshape(b, 14, 14) < 1e8
    mask = hit & valids[-1]
    if bool(mask.any()):
        cons = (gathered - cur.detach()).pow(2).mean(1)[mask].mean()
        cos = torch.nn.functional.cosine_similarity(gathered, cur.detach(), dim=1)[mask].mean()
    else:
        cons = cur.sum() * 0.0
        cos = cons.detach()
    return cons + dloss, cons.detach(), dloss.detach(), cos.detach(), int(mask.sum())


def c01(enc, dev):
    from experiments.aerial.rl.latent_map import latent_map
    steps = json.loads((FLIGHT / "steps_unseen_c_01.json").read_text(encoding="utf-8"))
    tr = np.load(FLIGHT / "traj_unseen_c_01.npz")
    fr = np.load(FLIGHT / "last_frames_unseen_c_01.npz")
    traj = np.asarray(tr["traj"], np.float64)
    depth = np.asarray(fr["da3_depth56"], np.float32)
    n = min(len(steps), len(traj), len(depth))
    states = np.zeros((n, 7), np.float64)
    states[:, :3] = traj[:n]
    states[:, 6] = [float(steps[i]["yaw"]) for i in range(n)]
    depth, states = depth[:n], states[:n]
    x = depth_in(torch.as_tensor(depth, device=dev))
    with torch.no_grad():
        feat = enc.features(x).detach().cpu().numpy()
    rows = []
    for t in (10, 11):
        grid = latent_map(depth[:t], [feat[i] for i in range(t)], states[:t], here=states[t])
        center = feat[9, :, 7, 7]
        occupied = np.linalg.norm(grid, axis=0) > 1e-4
        vec = grid[:, occupied].T
        cos = vec @ center / (np.linalg.norm(vec, axis=1) * (np.linalg.norm(center) + 1e-8) + 1e-8)
        order = np.argsort(-cos)[:8]
        cols = np.nonzero(occupied)[1][order]
        rows.append({
            "t": t,
            "best_cols": [int(c) for c in cols],
            "best_col_mean": round(float(cols.mean()), 2),
            "best_cos": round(float(cos[order].mean()), 3),
        })
        log(f"c01 t {t} best cols {rows[-1]['best_cols']} mean {rows[-1]['best_col_mean']} cos {rows[-1]['best_cos']}")
    (OUT / "C01.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    if "shorew03_krs6" in str(OUT):
        raise SystemExit("refusing to write the frozen directory")
    from experiments.aerial.rl.consequence_model import load_episode
    from experiments.aerial.rl.latent_map import MapEncoder
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"start device {dev}")
    tr_paths, va_paths = episodes()
    tr = load_lib(tr_paths, load_episode)
    va = load_lib(va_paths, load_episode)
    enc = MapEncoder(DIM).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=3e-4, weight_decay=0.0)
    rng = np.random.default_rng(218)
    t0 = time.time()
    enc.train()
    for step in range(1, STEPS + 1):
        depth_m, state = draw(tr, rng, BATCH, dev)
        loss, cons, dloss, cos, nmask = loss_of(enc, depth_m, state)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            g = float(sum(p.grad.abs().sum() for p in enc.parameters() if p.grad is not None))
            log(f"step 1 grad {g:.4f} cons {float(cons):.4f} depth {float(dloss):.4f} covered {nmask}")
            if g <= 0.0:
                raise SystemExit("map encoder got no gradient")
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == STEPS:
            enc.eval()
            with torch.no_grad():
                vd, vs = draw(va, rng, BATCH, dev)
                _, vcons, vdloss, vcos, vn = loss_of(enc, vd, vs)
            log(f"step {step} cons {float(vcons):.4f} depth {float(vdloss):.4f} "
                f"cos {float(vcos):.3f} covered {vn} t {time.time() - t0:.0f}")
            torch.save({"model": enc.state_dict(), "dim": DIM, "step": step}, OUT / f"map_step{step:05d}.pt")
            enc.train()
    torch.save({"model": enc.state_dict(), "dim": DIM, "step": STEPS}, OUT / "map_last.pt")
    enc.eval()
    c01(enc, dev)
    log("done")


if __name__ == "__main__":
    main()
