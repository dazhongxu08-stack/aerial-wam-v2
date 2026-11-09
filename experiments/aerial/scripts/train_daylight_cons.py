#!/usr/bin/env python3
"""Train a new consequence head on the current office depth.

Sidecars are rewritten only after the yaw_f copies are stored. angle_f is not
overwritten. A fast step inside 1.3 m is trained as a hit, and an open floor
near 2 m stays a safe window, but only when the depth cone agrees with GT.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")

spec = importlib.util.spec_from_file_location("ft", "/data/simworld/run/finetune_office_da3_cone.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

from experiments.aerial.rl.consequence_model import (  # noqa: E402
    DEPTH_MAX_M, DEPTH_MIN_M, consequence_loss, load_checkpoint, mirror_batch, save_checkpoint, survival,
)
from experiments.aerial.rl.train_consequence import stack_windows  # noqa: E402

MIX = Path("/data/simworld/run/indoor_office_mix")
ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
NEW_DEPTH = ART / "da3_office_cone_yaw_q_20261102/depth_office_cone_yaw.pt"
OLD_DEPTH = ART / "da3_office_cone_yaw_o_20261102/depth_office_cone_yaw.pt"
INIT = ART / "indoor_office_mix_angle_f_20261102/c_s218/consequence_best.pt"
OUT = ART / "indoor_office_mix_daylight_20261102/c_s218"
LOG_SPAN = math.log(DEPTH_MAX_M / DEPTH_MIN_M)


def encode(depth_path: Path) -> None:
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor

    ft.start_rss_watch()
    files = sorted(p for p in (MIX / "episodes").glob("episode_*.npz") if p.name.count(".") == 1)
    backup = MIX / "sidecars_yaw_f"
    backup.mkdir(exist_ok=True)
    pred = DepthMinPredictor.from_checkpoint(str(depth_path), device="cuda")
    model = pred._model
    model.eval()
    for n, path in enumerate(files):
        side = path.with_suffix(".da3.npz")
        kept = backup / side.name
        if side.is_file() and not kept.is_file():
            shutil.copy2(side, kept)
        z = np.load(path)
        rgb = ft.decode_frames(z)
        maps = []
        for i in range(0, len(rgb), 8):
            chunk = np.stack(rgb[i:i + 8])
            with torch.no_grad():
                maps.append(ft.forward_maps(model, chunk).detach().float().cpu().numpy().astype(np.float16))
        tmp = path.with_name(path.stem + ".da3.tmp.npz")
        np.savez(tmp, depth56=np.concatenate(maps))
        tmp.replace(side)
        if n % 50 == 0:
            print(f"sidecar {n}/{len(files)} rss {ft.rss_mb():.0f}", flush=True)
            ft.guard_rss("encode")
    print(f"SIDECARS {len(files)}", flush=True)
    del pred, model
    gc.collect()
    torch.cuda.empty_cache()


def da_forward(depth: torch.Tensor) -> torch.Tensor:
    meters = depth[:, 0] if depth.ndim == 4 else depth
    meters = DEPTH_MIN_M * torch.exp(meters.clamp(0, 1) * LOG_SPAN)
    h, w = meters.shape[-2:]
    band = meters[:, h // 4:h - h // 4]
    c0, c1 = w // 4, w - w // 4
    return band[:, :, c0:c1].amin(dim=(-2, -1))


def main() -> int:
    depth_path = NEW_DEPTH if NEW_DEPTH.is_file() else OLD_DEPTH
    print("DEPTH", depth_path, flush=True)
    encode(depth_path)
    review = json.loads((MIX / "AUTO_REVIEW.json").read_text())
    route_of = {e["file"]: e["route_id"] for e in review["episodes"] if e.get("keep") and e.get("file")}
    eps = sorted(
        p for p in (MIX / "episodes").glob("episode_*.npz")
        if p.name.count(".") == 1 and p.name in route_of
    )
    tr_paths = [p for p in eps if route_of[p.name] not in ("office_val", "office_yaw_val")]
    va_paths = [p for p in eps if route_of[p.name] == "office_val"]
    tr = stack_windows(tr_paths, 10, "da3", route_of=route_of)
    va = stack_windows(va_paths, 10, "da3")
    fwd = np.exp(tr["log_cones"][:, 0, 0])
    hit = tr["hit"].sum(-1) > 0
    weight = np.ones(len(fwd), np.float64)
    weight[hit & (fwd < 1.3)] = 6.0
    weight[(~hit) & (fwd >= 2.0)] = 2.0
    tr["sample_w"] = weight
    dev = torch.device("cuda")
    model = load_checkpoint(str(INIT), "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    keys = ("depth", "goal", "prev_action", "actions", "hit", "alive", "valid", "log_cones", "dgoal")
    rng = np.random.default_rng(218)
    prob = tr["sample_w"] / tr["sample_w"].sum()

    def batch(d, idx):
        return {k: torch.as_tensor(d[k][idx], dtype=torch.float32, device=dev) for k in keys if k in d}

    @torch.no_grad()
    def rates(d):
        model.eval()
        ps, ys = [], []
        for s in range(0, len(d["depth"]), 1024):
            idx = np.arange(s, min(len(d["depth"]), s + 1024))
            b = batch(d, idx)
            out = model(b["depth"], b["goal"], b["prev_action"], b["actions"])
            last = (b["alive"].sum(-1) - 1).clamp(min=0).long()
            p = (1.0 - survival(out["hazard_logit"]).gather(1, last[:, None]).squeeze(1)).cpu().numpy()
            ps.append(p)
            ys.append((b["hit"].sum(-1) > 0).cpu().numpy())
        model.train()
        p = np.concatenate(ps)
        y = np.concatenate(ys)
        return float(p[y].mean()) if y.any() else 0.0, float(p[~y].mean()) if (~y).any() else 0.0

    OUT.mkdir(parents=True, exist_ok=True)
    best = None
    model.train()
    hit_p, safe_p = rates(va)
    print(f"step 0 hit {hit_p:.3f} safe {safe_p:.3f}", flush=True)
    for step in range(1, 81):
        idx = rng.choice(len(prob), size=128, p=prob)
        b = batch(tr, idx)
        if rng.random() < 0.5:
            b = mirror_batch(b)
        out = model(b["depth"], b["goal"], b["prev_action"], b["actions"])
        loss, _ = consequence_loss(out, b)
        last = (b["alive"].sum(-1) - 1).clamp(min=0).long()
        p_h = 1.0 - survival(out["hazard_logit"]).gather(1, last[:, None]).squeeze(1)
        y = (b["hit"].sum(-1) > 0).float()
        gt_f = torch.exp(b["log_cones"][:, 0, 0])
        dx = b["actions"][:, 0, 0]
        seen_f = da_forward(b["depth"])
        agree = (seen_f - gt_f).abs() < 0.4
        close_fast = agree & (gt_f < 1.3) & (dx >= 0.4)
        open_safe = agree & (gt_f >= 2.0) & (y < 0.5)
        extra = p_h.new_zeros(())
        if bool(close_fast.any()):
            extra = extra + (1.0 - p_h[close_fast]).mean()
        if bool(open_safe.any()):
            extra = extra + p_h[open_safe].mean()
        loss = loss + 4.0 * extra
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == 80:
            hit_p, safe_p = rates(va)
            print(f"step {step} hit {hit_p:.3f} safe {safe_p:.3f}", flush=True)
            ft.guard_rss(f"cons {step}")
            if hit_p >= 0.85 and safe_p <= 0.06 and (best is None or safe_p < best[0]):
                best = (safe_p, hit_p, step)
                save_checkpoint(OUT / "consequence_best.pt", model, {"step": step, "safe": safe_p, "hit": hit_p})
                print("SAVE", step, flush=True)
    if best is None:
        print("NO_SAVE", flush=True)
        return 1
    print("BEST", best, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
