#!/usr/bin/env python3
"""DA3: paint above-ground crates, unstick far smear, keep aisle floor.

Init from far. Does not overwrite far/crate ckpts.
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
from office_bands import GROUND_FRAC, floor_range_map, short_clearance  # noqa: E402

CKPT = far.OUT
OUT = far.ART / "da3_office_geom_20261103/depth_office_cone_geom.pt"
PAIRS = Path("/data/simworld/run/indoor_office_lowbox/pairs.npz")
HIDE56 = max(far.HZ56 + 1, int(round(col.HIDE_ROW * 56 / 224)))


def _floor_t(h: int, w: int, device) -> torch.Tensor:
    return torch.as_tensor(floor_range_map(h, w), dtype=torch.float32, device=device)


def crate_above_ground(p56: torch.Tensor, gt56: torch.Tensor) -> torch.Tensor:
    r0, r1 = int(far.HZ56), min(p56.shape[-2], HIDE56)
    p = p56[:, r0:r1].clamp(0.05, far.FAR)
    g = gt56[:, r0:r1]
    fl = _floor_t(p56.shape[-2], p56.shape[-1], p56.device)[r0:r1]
    crate = torch.isfinite(g) & (g > 0.15) & (g < GROUND_FRAC * fl) & (g < 1.4)
    floor = torch.isfinite(g) & (g >= 0.85 * fl)
    extra = p56.new_zeros(())
    if bool(crate.any()):
        err = torch.relu(torch.log(p) - torch.log(g.clamp(min=0.05)))
        n = crate.flatten(1).sum(1).clamp(min=1)
        extra = extra + ((err * crate).flatten(1).sum(1) / n)[crate.flatten(1).any(1)].mean()
    if bool(floor.any()):
        keep = torch.relu(torch.log((0.80 * fl).clamp(min=0.05)) - torch.log(p))
        n = floor.flatten(1).sum(1).clamp(min=1)
        extra = extra + ((keep * floor).flatten(1).sum(1) / n)[floor.flatten(1).any(1)].mean()
    return extra


def smear_loss(p56: torch.Tensor, gt56: torch.Tensor) -> torch.Tensor:
    pf = far.p20_fwd_t(p56).clamp(0.05, far.FAR)
    gf = far.p20_fwd_t(gt56).clamp(0.05, far.FAR)
    bad = (gf >= 4.0) & (pf < 0.45 * gf)
    if not bool(bad.any()):
        return p56.new_zeros(())
    return (torch.log(gf[bad]) - torch.log(pf[bad])).mean()


@torch.no_grad()
def pair_stats(model, rgb, gt56) -> dict:
    pred = far.forward_maps(model, rgb).detach().float().cpu().numpy()
    gt_c, da_c = [], []
    for i in range(len(pred)):
        g = short_clearance(gt56[i], horizon=far.HZ56, hide=HIDE56)
        d = short_clearance(pred[i], horizon=far.HZ56, hide=HIDE56)
        def mn(t):
            vals = [v for v in t if v is not None]
            return min(vals) if vals else far.FAR
        gt_c.append(mn(g))
        da_c.append(mn(d))
    gt_c, da_c = np.asarray(gt_c), np.asarray(da_c)
    near = gt_c < 1.5
    return {
        "n": int(len(gt_c)),
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
    hold_idx = ho if len(ho) else tr[:8]
    print("before pairs", pair_stats(model, rgb_p[hold_idx], gt_p[hold_idx]), flush=True)
    print("before mix", flush=True)
    far.eval_refs(model, mix_ho, loader, limit=240)
    opt = torch.optim.AdamW((p for p in model.decoder.parameters() if p.requires_grad), lr=3e-6)
    rng = np.random.default_rng(7)
    best = None
    best_state = None
    for step in range(1, 151):
        mix_pick = [far_i[int(i)] for i in rng.integers(0, len(far_i), size=4)]
        mix_pick += [near_i[int(i)] for i in rng.integers(0, len(near_i), size=2)]
        rgb_m, gt_m = loader.batch(mix_pick)
        pi = rng.choice(tr, size=min(5, len(tr)), replace=True)
        rgb = np.concatenate([rgb_m, rgb_p[pi]], axis=0)
        gt56 = np.concatenate([gt_m, gt_p[pi]], axis=0)
        p56 = far.forward_maps(model, rgb)
        gt_t = torch.as_tensor(gt56, dtype=torch.float32, device="cuda")
        n_m = len(rgb_m)
        loss = (
            far.far_near_loss(p56[:n_m], gt_t[:n_m])
            + 8.0 * smear_loss(p56[:n_m], gt_t[:n_m])
            + 5.0 * crate_above_ground(p56[n_m:], gt_t[n_m:])
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 150:
            st = pair_stats(model, rgb_p[hold_idx], gt_p[hold_idx])
            mix = far.eval_refs(model, mix_ho, loader, limit=200)
            print(f"step {step} loss {float(loss.detach()):.4f} crate {st} mix_far {mix['far_da']:.2f}", flush=True)
            score = max(0.0, 4.0 - mix["far_da"]) + (0.0 if st["near_n"] == 0 else max(0.0, st["da_med"] - 1.4))
            far.save_ckpt(model, src, OUT.with_name("depth_office_cone_geom_last.pt"), {"pair": st, "mix": mix})
            if best is None or score < best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                far.save_ckpt(model, src, OUT, {"step": step, "pair": st, "mix": mix, "score": score})
    if best_state is not None:
        model.load_state_dict(best_state)
    after = far.eval_refs(model, mix_ho, loader, limit=0)
    st = pair_stats(model, rgb_p[hold_idx], gt_p[hold_idx])
    print("AFTER mix", after, "crate", st, flush=True)
    ok = after["far_da"] >= 4.0 and after["near_da"] <= 1.5
    print("DONE", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
