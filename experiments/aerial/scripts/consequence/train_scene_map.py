#!/usr/bin/env python3
"""Train the body-frame scene.

Past views are placed by their capture pose into the current body. A cell
keeps the feature of the nearest surface, and that feature has to read back
the depth that was seen. Surfaces that have left the current camera are scored
on their own. Survival is not trained. The frozen krs6 weights are not loaded
and not written.
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
OUT = ART / "scenemap_20261104_cont"
RESUME = ART / "scenemap_20261104" / "map_step04000.pt"
VAL = {"od_train_03", "od_train_17", "od_train_25", "od_train_33", "od_train_07", "od_train_21"}
PAST = 16
DIM = 16
STEPS = 8000
BATCH = 32
LR = 1e-4


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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


def _avg(rows, key, count_key):
    num, den = 0.0, 0
    for row in rows:
        if row[key] is None or row[count_key] <= 0:
            continue
        num += row[key] * row[count_key]
        den += row[count_key]
    if den == 0:
        return None, 0
    return num / den, den


def evaluate(enc, lib, rng, dev):
    from experiments.aerial.rl.latent_map import scene_loss
    rows = []
    was = enc.training
    enc.eval()
    with torch.no_grad():
        for _ in range(4):
            depth_m, state = draw(lib, rng, BATCH, dev)
            _loss, stats = scene_loss(enc, depth_m, state)
            rows.append(stats)
    if was:
        enc.train()
    out_mae, out_n = _avg(rows, "out_mae", "out_n")
    in_mae, in_n = _avg(rows, "in_mae", "in_n")
    near_mae, near_n = _avg(rows, "near_mae", "near_n")
    feat = sum(r["feat_std"] for r in rows) / len(rows)
    return {
        "out_mae": out_mae, "in_mae": in_mae, "near_mae": near_mae,
        "out_n": out_n, "in_n": in_n, "near_n": near_n, "feat_std": feat,
    }


def _fmt(value):
    return "none" if value is None else f"{value:.3f}"


def main() -> None:
    if "shorew03_krs6" in str(OUT):
        raise SystemExit("refusing to write the frozen directory")
    from experiments.aerial.rl.consequence_model import load_episode
    from experiments.aerial.rl.latent_map import MapEncoder, scene_loss
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"start device {dev}")
    tr_paths, va_paths = episodes()
    tr = load_lib(tr_paths, load_episode)
    va = load_lib(va_paths, load_episode)
    enc = MapEncoder(DIM).to(dev)
    if not RESUME.is_file():
        raise SystemExit(f"missing resume checkpoint {RESUME}")
    enc.load_state_dict(torch.load(RESUME, map_location=dev)["model"])
    log(f"resume {RESUME.name}")
    opt = torch.optim.AdamW(enc.parameters(), lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(218)
    t0 = time.time()
    best = None
    enc.train()
    for step in range(1, STEPS + 1):
        depth_m, state = draw(tr, rng, BATCH, dev)
        loss, stats = scene_loss(enc, depth_m, state)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            g = float(sum(p.grad.abs().sum() for p in enc.parameters() if p.grad is not None))
            log(f"step 1 grad {g:.4f} out {_fmt(stats['out_mae'])} in {_fmt(stats['in_mae'])} "
                f"near {_fmt(stats['near_mae'])} out_n {stats['out_n']} near_n {stats['near_n']}")
            if g <= 0.0:
                raise SystemExit("scene encoder got no gradient")
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == STEPS:
            ev = evaluate(enc, va, rng, dev)
            log(f"step {step} out {_fmt(ev['out_mae'])} in {_fmt(ev['in_mae'])} "
                f"near {_fmt(ev['near_mae'])} out_n {ev['out_n']} near_n {ev['near_n']} "
                f"std {ev['feat_std']:.3f} t {time.time() - t0:.0f}")
            torch.save({"model": enc.state_dict(), "dim": DIM, "step": step, "val": ev},
                       OUT / f"map_step{step:05d}.pt")
            score = ev["near_mae"] if ev["near_mae"] is not None else ev["out_mae"]
            if score is not None and (best is None or score < best):
                best = score
                torch.save({"model": enc.state_dict(), "dim": DIM, "step": step, "val": ev},
                           OUT / "map_best.pt")
                log(f"best step {step} near {_fmt(ev['near_mae'])}")
            (OUT / "SCENE.json").write_text(json.dumps({"step": step, **{
                k: (None if v is None else round(float(v), 4)) for k, v in ev.items()
            }}, indent=2), encoding="utf-8")
            enc.train()
    torch.save({"model": enc.state_dict(), "dim": DIM, "step": STEPS}, OUT / "map_last.pt")
    log("done")


if __name__ == "__main__":
    main()
