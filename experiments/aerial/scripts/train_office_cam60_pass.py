#!/usr/bin/env python3
"""Drive past a close side once the forward cone is open.

consequence_cam60.pt turns away from the barrel, then stops because the
barrel is still beside the camera. Forward clearance above 2.5 m has to
stay drivable. A short side only forbids turning into that side. A nose
that is short, with one side open, still turns. Both sides short still stop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/simworld/run")
sys.path.insert(0, "/data/aerial-wam-v2")
sys.path.insert(0, "/data/aerial-wam-v2/experiments/aerial/scripts")

import train_office_align as align
from experiments.aerial.rl.consequence_model import load_checkpoint, save_checkpoint
from experiments.aerial.rl.depth_predictor import DepthMinPredictor

ART = Path("/data/aerial-wam-v2/experiments/aerial/rl/artifacts")
SIM_NPZ = Path("/data/simworld/run/office_cam60_20261105/sim.npz")
DEPTH = ART / "da3_office_cam60_20261105/depth_office_cone_cam60.pt"
INIT = ART / "indoor_office_cam60_20261105/c_s218/consequence_cam60.pt"
OUT = ART / "indoor_office_cam60_20261105/c_s218/consequence_cam60_pass.pt"
YAW = 0.3141592653589793
STEPS = 80


def masks_of(cones: torch.Tensor) -> dict[str, torch.Tensor]:
    fwd, left, right = cones[:, 0], cones[:, 1], cones[:, 2]
    return {
        "pass": fwd > 2.5,
        "into_l": (fwd > 2.5) & (left < 1.2) & (right > left + 0.5),
        "into_r": (fwd > 2.5) & (right < 1.2) & (left > right + 0.5),
        "nose_r": (fwd < 1.5) & (fwd > 0.35) & (right > 2.2) & (right > left),
        "nose_l": (fwd < 1.5) & (fwd > 0.35) & (left > 2.2) & (left > right),
        "shut": (fwd < 1.2) & (left < 1.5) & (right < 1.5),
        "hit": fwd < 1.15,
    }


def main() -> int:
    sim = np.load(SIM_NPZ)
    wrap = DepthMinPredictor.from_checkpoint(str(DEPTH), device="cuda")
    wrap.forward_scale = 1.0
    wrap.forward_exponent = 1.0
    real_rgb, real_gt = align.load_real()
    real_d, real_c, real_g = align.encode_split(wrap, real_rgb, real_gt)
    sim_d, sim_c, sim_g = align.encode_split(wrap, sim["rgb"], sim["gt56"].astype(np.float32))
    del wrap
    torch.cuda.empty_cache()
    dev = torch.device("cuda")
    splits = {
        "real": align.pack(real_d, real_g, dev) + (torch.as_tensor(real_c, device=dev),),
        "sim": align.pack(sim_d, sim_g, dev) + (torch.as_tensor(sim_c, device=dev),),
    }
    masks = {name: masks_of(cones) for name, (*_rest, cones) in splits.items()}
    for name, m in masks.items():
        print(
            f"{name} pass {int(m['pass'].sum())} nose {int((m['nose_l']|m['nose_r']).sum())} "
            f"shut {int(m['shut'].sum())} hit {int(m['hit'].sum())}",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(str(INIT), "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    def surv(depth, goal, prev, dx, dyaw):
        z = align.logits(model, depth, goal, prev, dx, dyaw)
        return 1.0 - torch.sigmoid(z)

    def report(step: int):
        model.eval()
        stats = {}
        with torch.no_grad():
            for name, (depth, goal, prev, _c) in splits.items():
                sf = surv(depth, goal, prev, 0.8, 0.0)
                sl = surv(depth, goal, prev, 0.0, -YAW)
                sr = surv(depth, goal, prev, 0.0, YAW)
                m = masks[name]
                row = {
                    "pass": float(sf[m["pass"]].min()) if bool(m["pass"].any()) else 1.0,
                    "nose": float(torch.cat([
                        t for t, k in ((sr[m["nose_r"]], "nose_r"), (sl[m["nose_l"]], "nose_l")) if bool(m[k].any())
                    ]).min()) if bool(m["nose_r"].any() or m["nose_l"].any()) else 1.0,
                    "shut": float(torch.maximum(torch.maximum(sf, sl), sr)[m["shut"]].max()) if bool(m["shut"].any()) else 0.0,
                    "hit": float(sf[m["hit"]].max()) if bool(m["hit"].any()) else 0.0,
                }
                stats[name] = row
                print(
                    f"step {step} {name} pass {row['pass']:.3f} nose {row['nose']:.3f} "
                    f"shut {row['shut']:.3f} hit {row['hit']:.3f}",
                    flush=True,
                )
        return stats

    def ok(stats) -> bool:
        sim = stats["sim"]
        real = stats["real"]
        return sim["pass"] >= 0.95 and sim["nose"] >= 0.95 and sim["shut"] <= 0.35 and real["hit"] <= 0.25 and real["pass"] >= 0.95

    report(0)
    saved = False
    for step in range(1, STEPS + 1):
        model.train()
        loss = torch.zeros((), device=dev)
        for name, (depth, goal, prev, _c) in splits.items():
            m = masks[name]
            zf = align.logits(model, depth, goal, prev, 0.8, 0.0)
            zl = align.logits(model, depth, goal, prev, 0.0, -YAW)
            zr = align.logits(model, depth, goal, prev, 0.0, YAW)
            if bool(m["pass"].any()):
                loss = loss + torch.relu(zf[m["pass"]] + 3.0).pow(2).mean()
            if bool(m["into_l"].any()):
                loss = loss + torch.relu(2.0 - zl[m["into_l"]]).pow(2).mean()
            if bool(m["into_r"].any()):
                loss = loss + torch.relu(2.0 - zr[m["into_r"]]).pow(2).mean()
            if bool(m["nose_r"].any()):
                loss = loss + 4.0 * torch.relu(zr[m["nose_r"]] + 2.0).pow(2).mean()
                loss = loss + 4.0 * torch.relu(2.0 - zf[m["nose_r"]]).pow(2).mean()
            if bool(m["nose_l"].any()):
                loss = loss + 4.0 * torch.relu(zl[m["nose_l"]] + 2.0).pow(2).mean()
                loss = loss + 4.0 * torch.relu(2.0 - zf[m["nose_l"]]).pow(2).mean()
            if bool(m["shut"].any()):
                loss = loss + 4.0 * torch.relu(2.0 - zf[m["shut"]]).pow(2).mean()
                loss = loss + 4.0 * torch.relu(2.0 - zl[m["shut"]]).pow(2).mean()
                loss = loss + 4.0 * torch.relu(2.0 - zr[m["shut"]]).pow(2).mean()
            if bool(m["hit"].any()):
                loss = loss + 4.0 * torch.relu(2.0 - zf[m["hit"]]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == STEPS:
            stats = report(step)
            save_checkpoint(OUT.with_name("consequence_cam60_pass_last.pt"), model.eval(), {"step": step})
            model.train()
            if ok(stats):
                save_checkpoint(OUT, model.eval(), {"step": step})
                print("PASS_SAVE", step, flush=True)
                saved = True
                break
    if not saved:
        save_checkpoint(OUT, model.eval(), {"step": STEPS, "fallback": True})
        print("PASS_SAVE_LAST", flush=True)
    print("PASS_OUT", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
