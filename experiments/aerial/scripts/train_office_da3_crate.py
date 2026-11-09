#!/usr/bin/env python3
"""Paint short crates in the horizon→hide strip. Keep far-aisle p20.

Init from the far checkpoint. Mix far/near terms stay. No consequence, no robot.
"""
from __future__ import annotations

import json
import math
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
from office_bands import short_clearance  # noqa: E402

CKPT = far.OUT
OUT = far.ART / "da3_office_crate_20261103/depth_office_cone_crate.pt"
PAIRS = Path("/data/simworld/run/indoor_office_lowbox/pairs.npz")
HIDE56 = max(far.HZ56 + 1, int(round(col.HIDE_ROW * 56 / 224)))


def crate_loss(p56: torch.Tensor, gt56: torch.Tensor) -> torch.Tensor:
    r0, r1 = int(far.HZ56), min(p56.shape[-2], HIDE56)
    p = p56[:, r0:r1].clamp(0.05, far.FAR)
    g = gt56[:, r0:r1]
    close = torch.isfinite(g) & (g > 0.2) & (g < 1.5)
    if not bool(close.any()):
        return p56.new_zeros(())
    err = torch.relu(torch.log(p) - torch.log(g.clamp(min=0.05)))
    n = close.flatten(1).sum(1).clamp(min=1)
    per = (err * close).flatten(1).sum(1) / n
    return per[close.flatten(1).any(1)].mean()


@torch.no_grad()
def pair_stats(model, rgb, gt56) -> dict:
    pred = far.forward_maps(model, rgb).detach().float().cpu().numpy()
    rows = []
    for i in range(len(pred)):
        # depth56 already pooled; short_clearance on 56 uses hz56→hide56
        g = short_clearance(gt56[i], horizon=far.HZ56, hide=HIDE56)
        d = short_clearance(pred[i], horizon=far.HZ56, hide=HIDE56)
        rows.append((g, d))
    def val(t, i):
        v = t[i]
        return float(v) if v is not None else far.FAR
    gt_c = np.array([min(val(g, 0), val(g, 1), val(g, 2)) for g, _ in rows])
    da_c = np.array([min(val(d, 0), val(d, 1), val(d, 2)) for _, d in rows])
    near = gt_c < 1.5
    return {
        "n": int(len(rows)),
        "near_n": int(near.sum()),
        "gt_med": float(np.median(gt_c[near])) if near.any() else 0.0,
        "da_med": float(np.median(da_c[near])) if near.any() else 0.0,
    }


def main() -> int:
    print("CKPT", CKPT, "OUT", OUT, "HIDE56", HIDE56, flush=True)
    pairs = np.load(PAIRS)
    rgb_p = np.asarray(pairs["rgb"])
    gt_p = np.asarray(pairs["depth56"], np.float32)
    split = np.asarray(pairs["split"]).astype(str)
    tr = np.where(split != "hold")[0]
    ho = np.where(split == "hold")[0]
    print(f"pairs train {len(tr)} hold {len(ho)}", flush=True)
    mix_tr, mix_ho = far.index_mix()
    far_i = [r for r in mix_tr if r[2] >= far.FAR_GT]
    near_i = [r for r in mix_tr if r[2] <= far.NEAR_GT]
    loader = far.Loader()
    wrap = DepthMinPredictor.from_checkpoint(str(CKPT), device="cuda")
    model = wrap._model
    model.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    src = torch.load(str(CKPT), map_location="cpu")
    print("before pairs", pair_stats(model, rgb_p[ho[:8] if len(ho) else tr[:8]], gt_p[ho[:8] if len(ho) else tr[:8]]), flush=True)
    print("before mix", flush=True)
    before = far.eval_refs(model, mix_ho, loader, limit=240)
    opt = torch.optim.AdamW((p for p in model.decoder.parameters() if p.requires_grad), lr=3e-6)
    rng = np.random.default_rng(5)
    best = None
    best_state = None
    for step in range(1, 121):
        mix_pick = [far_i[int(i)] for i in rng.integers(0, len(far_i), size=3)]
        mix_pick += [near_i[int(i)] for i in rng.integers(0, len(near_i), size=2)]
        rgb_m, gt_m = loader.batch(mix_pick)
        pi = rng.choice(tr, size=min(6, len(tr)), replace=True)
        rgb = np.concatenate([rgb_m, rgb_p[pi]], axis=0)
        gt56 = np.concatenate([gt_m, gt_p[pi]], axis=0)
        p56 = far.forward_maps(model, rgb)
        gt_t = torch.as_tensor(gt56, dtype=torch.float32, device="cuda")
        n_m = len(rgb_m)
        loss = far.far_near_loss(p56[:n_m], gt_t[:n_m]) + 6.0 * crate_loss(p56[n_m:], gt_t[n_m:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == 120:
            hold_idx = ho if len(ho) else tr[:8]
            st = pair_stats(model, rgb_p[hold_idx], gt_p[hold_idx])
            mix = far.eval_refs(model, mix_ho, loader, limit=200)
            print(f"step {step} loss {float(loss.detach()):.4f} crate {st} mix_far {mix['far_da']:.2f}", flush=True)
            score = st["da_med"] + max(0.0, 4.0 - mix["far_da"])
            far.save_ckpt(model, src, OUT.with_name("depth_office_cone_crate_last.pt"), {"pair": st, "mix": mix})
            if best is None or score < best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                far.save_ckpt(model, src, OUT, {"step": step, "pair": st, "mix": mix, "score": score})
    if best_state is not None:
        model.load_state_dict(best_state)
    after_mix = far.eval_refs(model, mix_ho, loader, limit=0)
    hold_idx = ho if len(ho) else tr[:8]
    after_pair = pair_stats(model, rgb_p[hold_idx], gt_p[hold_idx])
    print("AFTER mix", after_mix, "crate", after_pair, flush=True)
    ok = after_pair["da_med"] <= 1.5 and after_mix["far_da"] >= 4.0
    print("DONE", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
